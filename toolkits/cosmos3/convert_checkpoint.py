# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Convert an RLinf Cosmos3 SFT checkpoint (FSDP2 ``full_weights.pt``) into a
diffusers component directory (``model_diffusers``) for SGLang evaluation.

The conversion is four steps; this script runs them in order:

  1. ``full_weights.pt``  -> ``model_safetensors``  (strip the ``omni.`` prefix)
  2. ``model_safetensors`` -> ``model_dcp``         (``torch.distributed.checkpoint``)
  3. ``model_dcp``         -> ``model_hf``           (``cosmos_framework.scripts.export_model``)
  4. ``model_hf``          -> ``model_diffusers``    (``cosmos_framework.scripts.convert_model_to_diffusers``)

Steps 3-4 shell out to cosmos-framework's CLI scripts, so the whole script must
run in the cosmos-framework environment (``transformers 4.57.x`` + ``diffusers
0.39.0`` with ``Cosmos3OmniTransformer``). Set ``LIBERO_ROOT`` before running
(the export step reads it).

Use --vit-checkpoint-path for a local Qwen directory in both export steps and
--vae-checkpoint-path for a local Wan Diffusers root containing vae/.
"""

import argparse
import json
import math
import os
import subprocess
import sys

DEFAULT_EXPERIMENT = "action_policy_libero_all_nano"
DEFAULT_CONFIG = "cosmos_framework/configs/base/config.py"
PIPELINE_CLASS = "Cosmos3OmniDiffusersPipeline"
# Match transformers' default max_shard_size so the DCP shards land near 5 GB.
SHARD_BYTES = 5 * 1024**3

# Training-only / default config nodes that cosmos export_model has no public alias for.
EXPORT_DROPPED_CONFIG_KEYS = ("lbl", "multiview_attention")
# argv[1] lists dropped nodes, argv[2] supplies a local processor, then native CLI args.
_EXPORT_MODEL_CODE = """
import sys

import cosmos_framework.scripts.export_model as export_model

dropped = sys.argv[1].split(",")
local_processor_path = sys.argv[2]
coerce_to_base_model = export_model._coerce_to_base_model


def coerce_and_drop(model_dict):
    coerce_to_base_model(model_dict)
    for key in dropped:
        model_dict["config"].pop(key, None)


export_model._coerce_to_base_model = coerce_and_drop
if local_processor_path:
    from cosmos_framework.configs.base.defaults import reasoner

    download_tokenizer_files = reasoner.download_tokenizer_files

    def download_local_tokenizer(model_name, config_variant):
        if model_name == "Qwen/Qwen3-VL-8B-Instruct" and config_variant == "hf":
            return local_processor_path
        return download_tokenizer_files(model_name, config_variant)

    reasoner.download_tokenizer_files = download_local_tokenizer
    bundle_processor = export_model.bundle_processor_from_tokenizer_node

    def bundle_local_processor(tokenizer_node, output_dir):
        return bundle_processor(
            tokenizer_node,
            output_dir,
            download=lambda checkpoint: local_processor_path,
        )

    export_model.bundle_processor_from_tokenizer_node = bundle_local_processor
sys.argv = ["export_model", *sys.argv[3:]]
if local_processor_path:
    from cosmos_framework.scripts._convert_model_to_diffusers import (
        _skip_source_tokenizer_load,
    )

    # Export does not encode media, so the source VAE needs metadata only.
    with _skip_source_tokenizer_load():
        export_model.main()
else:
    export_model.main()
"""


# argv[1:3] supply local Qwen/Wan directories, followed by the native CLI args.
_CONVERT_DIFFUSERS_CODE = """
import sys

import cosmos_framework.scripts.convert_model_to_diffusers as conversion
from cosmos_framework.scripts import _convert_model_to_diffusers as backend

local_qwen_path, local_vae_path = sys.argv[1:3]
if local_qwen_path:
    from cosmos_framework.configs.base.defaults import reasoner

    download_tokenizer_files = reasoner.download_tokenizer_files

    def download_local_tokenizer(model_name, config_variant):
        if model_name == "Qwen/Qwen3-VL-8B-Instruct" and config_variant == "hf":
            return local_qwen_path
        return download_tokenizer_files(model_name, config_variant)

    reasoner.download_tokenizer_files = download_local_tokenizer
    convert = backend.convert_model_to_diffusers

    def convert_with_local_vision(args):
        return convert(
            args.model_copy(update={"vision_encoder_model": local_qwen_path})
        )

    backend.convert_model_to_diffusers = convert_with_local_vision
    load_tokenizer = backend.AutoTokenizer.from_pretrained

    def load_local_tokenizer(source, *args, **kwargs):
        if str(source) == "Qwen/Qwen3-VL-8B-Instruct":
            source = local_qwen_path
            kwargs["local_files_only"] = True
        return load_tokenizer(source, *args, **kwargs)

    backend.AutoTokenizer.from_pretrained = load_local_tokenizer
    download = conversion.CheckpointDirHf.download

    def download_local_qwen(checkpoint, *args, **kwargs):
        if checkpoint.repository == "Qwen/Qwen3-VL-8B-Instruct":
            return local_qwen_path
        return download(checkpoint, *args, **kwargs)

    conversion.CheckpointDirHf.download = download_local_qwen

if local_vae_path:
    load_vae = backend.AutoencoderKLWan.from_pretrained

    def load_local_vae(source, *args, **kwargs):
        if str(source) == "Wan-AI/Wan2.2-TI2V-5B-Diffusers":
            source = local_vae_path
            kwargs["local_files_only"] = True
        return load_vae(source, *args, **kwargs)

    backend.AutoencoderKLWan.from_pretrained = load_local_vae

sys.argv = ["convert_model_to_diffusers", *sys.argv[3:]]
conversion.main()
"""


def step1_strip_omni_prefix(src: str, out: str) -> str:
    """full_weights.pt -> model_safetensors (strip ``omni.`` -> ``net.`` / ``net_ema.``)."""
    import torch
    from safetensors.torch import save_file

    out_dir = os.path.join(out, "model_safetensors")
    os.makedirs(out_dir, exist_ok=True)

    print(f"Loading {src} (~91GB)...")
    # weights_only=False: RLinf stores a state dict carrying metadata.
    sd = torch.load(src, map_location="cpu", weights_only=False)
    print(f"Loaded {len(sd)} keys")

    stripped = {
        (k[5:] if k.startswith("omni.") else k): v.contiguous() for k, v in sd.items()
    }
    print(f"Stripped to {len(stripped)} keys (e.g. {list(stripped)[:3]})")

    dst = os.path.join(out_dir, "model.safetensors")
    save_file(stripped, dst)
    print(f"Saved {os.path.getsize(dst) / 1e9:.1f} GB -> {dst}")
    return dst


def step2_save_dcp(safetensors_path: str, out: str) -> str:
    """model.safetensors -> model_dcp (export_model only reads DCP format)."""
    import torch
    import torch.distributed.checkpoint as dcp
    from cosmos_framework.checkpoint.dcp import CustomSavePlanner
    from safetensors.torch import load_file
    from torch.distributed.checkpoint.filesystem import FileSystemWriter

    out_dir = os.path.join(out, "model_dcp", "model")
    os.makedirs(out_dir, exist_ok=True)

    state_dict = load_file(safetensors_path)
    print(f"Loaded {len(state_dict)} keys")

    nbytes = sum(
        v.numel() * v.element_size()
        for v in state_dict.values()
        if isinstance(v, torch.Tensor)
    )
    nshards = max(1, math.ceil(nbytes / SHARD_BYTES))
    writer = FileSystemWriter(out_dir, thread_count=nshards)
    dcp.save(state_dict=state_dict, storage_writer=writer, planner=CustomSavePlanner())
    print(f"Saved DCP -> {out_dir}")
    return out_dir


def step3_export_to_hf(
    dcp_dir: str,
    out: str,
    config_file: str,
    experiment: str,
    no_ema: bool,
    *,
    vit_checkpoint_path: str | None = None,
) -> str:
    """model_dcp -> model_hf (cosmos export_model, DCP by training config)."""
    out_dir = os.path.join(out, "model_hf")
    if vit_checkpoint_path is not None:
        vit_checkpoint_path = os.path.abspath(vit_checkpoint_path)
    cmd = [
        sys.executable,
        "-c",
        _EXPORT_MODEL_CODE,
        ",".join(EXPORT_DROPPED_CONFIG_KEYS),
        vit_checkpoint_path or "",
        "--checkpoint-path",
        dcp_dir,
        "--config-file",
        config_file,
        "--experiment",
        experiment,
    ]
    if no_ema:
        cmd.append("--no-use-ema-weights")
    if vit_checkpoint_path is not None:
        cmd += ["--vit-checkpoint-path", vit_checkpoint_path]
    cmd += ["-o", out_dir]
    print(
        "DCP -> HF: cosmos_framework.scripts.export_model",
        " ".join(cmd[5:]),
        f"(without model.config.{{{','.join(EXPORT_DROPPED_CONFIG_KEYS)}}})",
    )
    subprocess.run(cmd, check=True)
    return out_dir


def step4_convert_to_diffusers(
    hf_dir: str,
    out: str,
    *,
    vit_checkpoint_path: str | None = None,
    vae_checkpoint_path: str | None = None,
) -> str:
    """model_hf -> model_diffusers, optionally loading local Qwen/Wan assets."""
    out_dir = os.path.join(out, "model_diffusers")
    native_args = ["--checkpoint-path", hf_dir, "-o", out_dir]
    cmd = [
        sys.executable,
        "-m",
        "cosmos_framework.scripts.convert_model_to_diffusers",
        *native_args,
    ]
    print("HF -> diffusers:", " ".join(cmd))
    if vit_checkpoint_path is not None or vae_checkpoint_path is not None:
        cmd = [
            sys.executable,
            "-c",
            _CONVERT_DIFFUSERS_CODE,
            os.path.abspath(vit_checkpoint_path)
            if vit_checkpoint_path is not None
            else "",
            os.path.abspath(vae_checkpoint_path)
            if vae_checkpoint_path is not None
            else "",
            *native_args,
        ]
    subprocess.run(cmd, check=True)

    # convert_model_to_diffusers writes a non-standard _class_name; set the
    # pipeline class diffusers expects.
    index = os.path.join(out_dir, "model_index.json")
    data = json.load(open(index))
    if data.get("_class_name") != PIPELINE_CLASS:
        data["_class_name"] = PIPELINE_CLASS
        json.dump(data, open(index, "w"), indent=2)
        print(f"Set _class_name={PIPELINE_CLASS} in {index}")
    return out_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--src",
        required=True,
        help="RLinf SFT output .../full_weights.pt (FSDP2, omni.net.* prefix).",
    )
    parser.add_argument(
        "--out",
        required=True,
        help="Conversion working dir; steps 1-4 write subdirs model_safetensors/"
        "model_dcp/model_hf/model_diffusers under here.",
    )
    parser.add_argument("--config-file", default=DEFAULT_CONFIG)
    parser.add_argument("--experiment", default=DEFAULT_EXPERIMENT)
    parser.add_argument(
        "--vit-checkpoint-path",
        help="Local Qwen Hugging Face directory for vision weights and processor/tokenizer files.",
    )
    parser.add_argument(
        "--vae-checkpoint-path",
        help="Local Wan Diffusers directory containing vae/config.json and VAE weights.",
    )
    parser.add_argument(
        "--use-ema-weights",
        action="store_true",
        help="Keep the EMA weights (default drops them).",
    )
    args = parser.parse_args()
    if args.vit_checkpoint_path is not None and not os.path.isdir(
        args.vit_checkpoint_path
    ):
        parser.error("--vit-checkpoint-path must point to a local Qwen directory")
    if args.vae_checkpoint_path is not None and not os.path.isfile(
        os.path.join(args.vae_checkpoint_path, "vae", "config.json")
    ):
        parser.error(
            "--vae-checkpoint-path must point to a local Wan Diffusers directory "
            "containing vae/config.json"
        )

    os.makedirs(args.out, exist_ok=True)
    no_ema = not args.use_ema_weights

    safetensors = step1_strip_omni_prefix(args.src, args.out)
    dcp_dir = step2_save_dcp(safetensors, args.out)
    hf_dir = step3_export_to_hf(
        dcp_dir,
        args.out,
        args.config_file,
        args.experiment,
        no_ema,
        vit_checkpoint_path=args.vit_checkpoint_path,
    )
    diffusers = step4_convert_to_diffusers(
        hf_dir,
        args.out,
        vit_checkpoint_path=args.vit_checkpoint_path,
        vae_checkpoint_path=args.vae_checkpoint_path,
    )

    print(f"Done. model_diffusers -> {diffusers}")
    print("Point rollout.model.model_path at this directory for eval.")


if __name__ == "__main__":
    main()
