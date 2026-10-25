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

"""Offline dependency routing for Cosmos checkpoint conversion."""

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest


@pytest.fixture
def converter():
    path = (
        Path(__file__).resolve().parents[2] / "toolkits/cosmos3/convert_checkpoint.py"
    )
    spec = importlib.util.spec_from_file_location("cosmos_checkpoint_conversion", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def native_helpers():
    helpers = pytest.importorskip("cosmos_framework.scripts._export_model_helpers")
    distillation = pytest.importorskip(
        "cosmos_framework.inference.common.distillation_export"
    )
    return helpers, distillation


@pytest.fixture
def framework_tokenizer_routes(monkeypatch):
    reasoner = pytest.importorskip("cosmos_framework.configs.base.defaults.reasoner")
    requests = []

    def download(model_name, config_variant):
        requests.append((model_name, config_variant))
        return f"original:{model_name}:{config_variant}"

    monkeypatch.setattr(reasoner, "download_tokenizer_files", download)
    return reasoner, requests


def framework_tokenizer_sources(reasoner):
    return [
        reasoner.download_tokenizer_files("Qwen/Qwen3-VL-8B-Instruct", "hf"),
        reasoner.download_tokenizer_files("another/tokenizer", "hf"),
        reasoner.download_tokenizer_files("Qwen/Qwen3-VL-8B-Instruct", "s3"),
    ]


def assert_framework_tokenizer_sources(sources, requests, local_path):
    assert sources == [
        local_path if local_path else "original:Qwen/Qwen3-VL-8B-Instruct:hf",
        "original:another/tokenizer:hf",
        "original:Qwen/Qwen3-VL-8B-Instruct:s3",
    ]
    assert requests == ([] if local_path else [("Qwen/Qwen3-VL-8B-Instruct", "hf")]) + [
        ("another/tokenizer", "hf"),
        ("Qwen/Qwen3-VL-8B-Instruct", "s3"),
    ]


@pytest.mark.parametrize("local", [False, True])
@pytest.mark.parametrize("no_ema", [False, True])
def test_export_command_routes_local_dependency_without_shell_expansion(
    converter, monkeypatch, tmp_path, local, no_ema
):
    commands = []

    def run(command, *, check):
        assert check is True
        commands.append(command)

    monkeypatch.setattr(converter.subprocess, "run", run)
    monkeypatch.chdir(tmp_path)
    local_arg = "Qwen snapshot with spaces" if local else None
    kwargs = {"vit_checkpoint_path": local_arg} if local else {}
    result = converter.step3_export_to_hf(
        "checkpoint/model",
        "output with spaces",
        "config.py",
        "libero",
        no_ema,
        **kwargs,
    )

    assert result == "output with spaces/model_hf"
    assert len(commands) == 1
    command = commands[0]
    assert command[:3] == [sys.executable, "-c", converter._EXPORT_MODEL_CODE]
    assert command[3] == ",".join(converter.EXPORT_DROPPED_CONFIG_KEYS)
    assert command[5:11] == [
        "--checkpoint-path",
        "checkpoint/model",
        "--config-file",
        "config.py",
        "--experiment",
        "libero",
    ]
    assert ("--no-use-ema-weights" in command) == no_ema
    assert command[-2:] == ["-o", result]
    if local:
        expected = str(tmp_path / local_arg)
        assert command[4] == expected
        assert command[command.index("--vit-checkpoint-path") + 1] == expected
    else:
        assert command[4] == ""
        assert "--vit-checkpoint-path" not in command


def test_native_vision_resolver_uses_local_path_without_downloading(native_helpers):
    _, distillation = native_helpers

    def forbidden_download(uri):
        pytest.fail(f"Local vision checkpoint unexpectedly downloaded: {uri}")

    local = "/models/local Qwen snapshot"
    assert (
        distillation.resolve_vision_checkpoint_path(
            local_path=local,
            configured_uri="s3://bucket/Qwen/Qwen3-VL-8B-Instruct",
            download_checkpoint=forbidden_download,
        )
        == local
    )


def test_native_vision_resolver_preserves_configured_source(native_helpers):
    _, distillation = native_helpers
    requested = []

    def download(uri):
        requested.append(uri)
        return "/cache/resolved"

    uri = "s3://bucket/Qwen/Qwen3-VL-8B-Instruct"
    assert (
        distillation.resolve_vision_checkpoint_path(
            local_path=None, configured_uri=uri, download_checkpoint=download
        )
        == "/cache/resolved"
    )
    assert requested == [uri]


@pytest.mark.parametrize("local", [False, True])
def test_export_shim_preserves_config_and_copies_native_processor_files(
    converter, native_helpers, framework_tokenizer_routes, monkeypatch, tmp_path, local
):
    helpers, distillation = native_helpers
    reasoner, tokenizer_requests = framework_tokenizer_routes
    scripts = pytest.importorskip("cosmos_framework.scripts")
    checkpoint_db = pytest.importorskip("cosmos_framework.utils.checkpoint_db")
    snapshot = tmp_path / "snapshots" / ("a" * 40)
    snapshot.mkdir(parents=True)
    files = {
        "tokenizer.json": '{"tokenizer": "local"}',
        "tokenizer_config.json": '{"model_max_length": 1024}',
        "preprocessor_config.json": '{"processor": "local"}',
        "chat_template.json": '{"chat_template": "local"}',
        "merges.txt": "# local merges\n",
    }
    for name, content in files.items():
        (snapshot / name).write_text(content)
    (snapshot / "config.json").write_text('{"source_config": true}')
    output = tmp_path / "model_hf"
    output.mkdir()
    exported_config = '{"exported_config": true}'
    (output / "config.json").write_text(exported_config)
    downloads = []

    def download(checkpoint):
        if local:
            pytest.fail("Local processor unexpectedly used the network download path")
        downloads.append(checkpoint.repository)
        return str(snapshot)

    monkeypatch.setattr(checkpoint_db.CheckpointDirHf, "download", download)
    export = ModuleType("cosmos_framework.scripts.export_model")
    observed = {}

    def coerce(model_dict):
        observed["coerce_called"] = True
        model_dict["config"]["base_projection"] = True

    export._coerce_to_base_model = coerce
    export.bundle_processor_from_tokenizer_node = (
        helpers.bundle_processor_from_tokenizer_node
    )

    def export_main():
        observed["argv"] = list(sys.argv)
        observed["framework_tokenizer_sources"] = framework_tokenizer_sources(reasoner)
        model_dict = {
            "config": {
                "lbl": {"method": "local"},
                "multiview_attention": {},
                "precision": "bfloat16",
            }
        }
        export._coerce_to_base_model(model_dict)
        observed["config"] = model_dict["config"]
        if local:
            resolved = distillation.resolve_vision_checkpoint_path(
                local_path=sys.argv[sys.argv.index("--vit-checkpoint-path") + 1],
                configured_uri="s3://bucket/Qwen/Qwen3-VL-8B-Instruct",
                download_checkpoint=lambda uri: pytest.fail(f"Downloaded ViT: {uri}"),
            )
            assert resolved == str(snapshot)
        observed["processor_source"] = export.bundle_processor_from_tokenizer_node(
            {"pretrained_model_name": "Qwen/Qwen3-VL-8B-Instruct"}, output
        )

    export.main = export_main
    monkeypatch.setitem(sys.modules, export.__name__, export)
    monkeypatch.setattr(scripts, "export_model", export, raising=False)
    native_argv = ["--checkpoint-path", "/dcp/model", "-o", str(output)]
    if local:
        native_argv.extend(["--vit-checkpoint-path", str(snapshot)])
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "-c",
            ",".join(converter.EXPORT_DROPPED_CONFIG_KEYS),
            str(snapshot) if local else "",
            *native_argv,
        ],
    )
    exec(converter._EXPORT_MODEL_CODE, {})

    assert observed["argv"] == ["export_model", *native_argv]
    assert observed["coerce_called"] is True
    assert observed["config"] == {
        "precision": "bfloat16",
        "base_projection": True,
    }
    assert observed["processor_source"] == {
        "repo": "Qwen/Qwen3-VL-8B-Instruct",
        "revision": "a" * 40,
        "bundled": True,
    }
    assert downloads == ([] if local else ["Qwen/Qwen3-VL-8B-Instruct"])
    assert_framework_tokenizer_sources(
        observed["framework_tokenizer_sources"],
        tokenizer_requests,
        str(snapshot) if local else None,
    )
    for name, content in files.items():
        assert (output / name).read_text() == content
    assert (output / "config.json").read_text() == exported_config


def test_cli_rejects_missing_local_dependency_before_loading_weights(
    converter, monkeypatch, tmp_path, capsys
):
    output = tmp_path / "output"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "convert_checkpoint.py",
            "--src",
            str(tmp_path / "absent_weights.pt"),
            "--out",
            str(output),
            "--vit-checkpoint-path",
            str(tmp_path / "absent_qwen"),
        ],
    )
    with pytest.raises(SystemExit) as error:
        converter.main()
    assert error.value.code == 2
    assert (
        "--vit-checkpoint-path must point to a local Qwen directory"
        in capsys.readouterr().err
    )
    assert not output.exists()


@pytest.mark.parametrize(
    "local_qwen,local_vae", [(False, False), (True, False), (False, True), (True, True)]
)
def test_diffusers_command_preserves_native_route_and_pipeline_metadata(
    converter, monkeypatch, tmp_path, local_qwen, local_vae
):
    monkeypatch.chdir(tmp_path)
    commands = []
    expected_output = tmp_path / "output" / "model_diffusers"

    def run(command, *, check):
        assert check is True
        commands.append(command)
        expected_output.mkdir(parents=True)
        (expected_output / "model_index.json").write_text(
            json.dumps(
                {
                    "_class_name": "NativePipeline",
                    "transformer": ["cosmos", "Transformer"],
                }
            )
        )

    monkeypatch.setattr(converter.subprocess, "run", run)
    result = converter.step4_convert_to_diffusers(
        "hf weights",
        "output",
        vit_checkpoint_path="Qwen local" if local_qwen else None,
        vae_checkpoint_path="Wan local" if local_vae else None,
    )
    assert result == "output/model_diffusers"
    native_args = ["--checkpoint-path", "hf weights", "-o", result]
    assert len(commands) == 1
    command = commands[0]
    if local_qwen or local_vae:
        assert command[:3] == [sys.executable, "-c", converter._CONVERT_DIFFUSERS_CODE]
        assert command[3] == (str(tmp_path / "Qwen local") if local_qwen else "")
        assert command[4] == (str(tmp_path / "Wan local") if local_vae else "")
        assert command[5:] == native_args
    else:
        assert command == [
            sys.executable,
            "-m",
            "cosmos_framework.scripts.convert_model_to_diffusers",
            *native_args,
        ]
    metadata = json.loads((expected_output / "model_index.json").read_text())
    assert metadata == {
        "_class_name": converter.PIPELINE_CLASS,
        "transformer": ["cosmos", "Transformer"],
    }


@pytest.mark.parametrize(
    "local_qwen,local_vae", [(False, False), (True, False), (False, True), (True, True)]
)
def test_diffusers_shim_redirects_only_selected_dependencies(
    converter, framework_tokenizer_routes, monkeypatch, local_qwen, local_vae
):
    reasoner, tokenizer_requests = framework_tokenizer_routes
    scripts = pytest.importorskip("cosmos_framework.scripts")
    pydantic = pytest.importorskip("pydantic")
    conversion = ModuleType("cosmos_framework.scripts.convert_model_to_diffusers")
    backend = ModuleType("cosmos_framework.scripts._convert_model_to_diffusers")
    observed = {"tokenizers": [], "vaes": [], "downloads": []}
    qwen_path = "/models/local Qwen"
    vae_path = "/models/local Wan"

    class Args(pydantic.BaseModel):
        vision_encoder_model: str
        output: str

    class AutoTokenizer:
        @classmethod
        def from_pretrained(cls, source, *args, **kwargs):
            observed["tokenizers"].append((source, args, kwargs))
            return "tokenizer"

    class AutoencoderKLWan:
        @classmethod
        def from_pretrained(cls, source, *args, **kwargs):
            observed["vaes"].append((source, args, kwargs))
            return "vae"

    class CheckpointDirHf:
        def __init__(self, repository):
            self.repository = repository

        def download(self, *, quiet=False):
            observed["downloads"].append((self.repository, quiet))
            return f"remote:{self.repository}"

    def convert(args):
        observed["args"] = args
        return "converted"

    backend.AutoTokenizer = AutoTokenizer
    backend.AutoencoderKLWan = AutoencoderKLWan
    backend.convert_model_to_diffusers = convert
    conversion.CheckpointDirHf = CheckpointDirHf
    original_args = Args(
        vision_encoder_model="Qwen/Qwen3-VL-8B-Instruct", output="output"
    )

    def conversion_main():
        observed["argv"] = list(sys.argv)
        observed["framework_tokenizer_sources"] = framework_tokenizer_sources(reasoner)
        assert backend.convert_model_to_diffusers(original_args) == "converted"
        for source in ["Qwen/Qwen3-VL-8B-Instruct", "another/tokenizer"]:
            assert (
                backend.AutoTokenizer.from_pretrained(source, "extra", revision="main")
                == "tokenizer"
            )
        for source in ["Wan-AI/Wan2.2-TI2V-5B-Diffusers", "another/vae"]:
            assert (
                backend.AutoencoderKLWan.from_pretrained(
                    source, subfolder="vae", revision="main"
                )
                == "vae"
            )
        observed["qwen_snapshot"] = conversion.CheckpointDirHf(
            "Qwen/Qwen3-VL-8B-Instruct"
        ).download(quiet=True)
        assert (
            conversion.CheckpointDirHf("another/checkpoint").download(quiet=True)
            == "remote:another/checkpoint"
        )

    conversion.main = conversion_main
    for module in [backend, conversion]:
        monkeypatch.setitem(sys.modules, module.__name__, module)
        monkeypatch.setattr(
            scripts, module.__name__.rsplit(".", 1)[-1], module, raising=False
        )
    native_args = ["--checkpoint-path", "/models/hf", "-o", "/models/diffusers"]
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "-c",
            qwen_path if local_qwen else "",
            vae_path if local_vae else "",
            *native_args,
        ],
    )
    exec(converter._CONVERT_DIFFUSERS_CODE, {})

    assert observed["argv"] == ["convert_model_to_diffusers", *native_args]
    assert observed["args"].output == "output"
    assert original_args.vision_encoder_model == "Qwen/Qwen3-VL-8B-Instruct"
    assert observed["args"].vision_encoder_model == (
        qwen_path if local_qwen else original_args.vision_encoder_model
    )
    token_options = {"revision": "main"}
    if local_qwen:
        token_options["local_files_only"] = True
    assert observed["tokenizers"] == [
        (
            qwen_path if local_qwen else "Qwen/Qwen3-VL-8B-Instruct",
            ("extra",),
            token_options,
        ),
        ("another/tokenizer", ("extra",), {"revision": "main"}),
    ]
    vae_options = {"subfolder": "vae", "revision": "main"}
    if local_vae:
        vae_options["local_files_only"] = True
    assert observed["vaes"] == [
        (vae_path if local_vae else "Wan-AI/Wan2.2-TI2V-5B-Diffusers", (), vae_options),
        ("another/vae", (), {"subfolder": "vae", "revision": "main"}),
    ]
    assert observed["qwen_snapshot"] == (
        qwen_path if local_qwen else "remote:Qwen/Qwen3-VL-8B-Instruct"
    )
    assert observed["downloads"] == (
        [] if local_qwen else [("Qwen/Qwen3-VL-8B-Instruct", True)]
    ) + [("another/checkpoint", True)]
    assert_framework_tokenizer_sources(
        observed["framework_tokenizer_sources"],
        tokenizer_requests,
        qwen_path if local_qwen else None,
    )


@pytest.mark.parametrize("stage", ["hf", "diffusers"])
def test_native_custom_tokenizer_factory_reads_local_vocab_through_shim(
    converter, monkeypatch, tmp_path, stage
):
    scripts = pytest.importorskip("cosmos_framework.scripts")
    reasoner = pytest.importorskip("cosmos_framework.configs.base.defaults.reasoner")
    monkeypatch.setattr(
        reasoner, "download_tokenizer_files", reasoner.download_tokenizer_files
    )
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    local = tmp_path / "qwen"
    local.mkdir()
    (local / "vocab.json").write_text(
        json.dumps(
            {
                "a": 0,
                "b": 1,
                "c": 2,
                "<|endoftext|>": 3,
                "<|im_start|>": 4,
                "<|im_end|>": 5,
            }
        )
    )
    (local / "merges.txt").write_text("#version: 0.2\n")
    (local / "tokenizer_config.json").write_text(
        json.dumps(
            {
                "tokenizer_class": "Qwen2Tokenizer",
                "model_max_length": 1024,
                "eos_token": "<|endoftext|>",
                "pad_token": "<|endoftext|>",
            }
        )
    )
    observed = {}

    def main():
        processor = reasoner.create_qwen2_tokenizer_with_download(
            "Qwen/Qwen3-VL-8B-Instruct", "hf"
        )
        observed["tokenizer_class"] = type(processor.tokenizer).__module__
        observed["tokens"] = processor.tokenizer.encode("abc")

    if stage == "hf":
        export = ModuleType("cosmos_framework.scripts.export_model")
        export._coerce_to_base_model = lambda data: None
        export.bundle_processor_from_tokenizer_node = lambda node, output: None
        export.main = main
        monkeypatch.setitem(sys.modules, export.__name__, export)
        monkeypatch.setattr(scripts, "export_model", export, raising=False)
        argv = ["-c", ",".join(converter.EXPORT_DROPPED_CONFIG_KEYS), str(local)]
        code = converter._EXPORT_MODEL_CODE
    else:
        conversion = ModuleType("cosmos_framework.scripts.convert_model_to_diffusers")
        backend = ModuleType("cosmos_framework.scripts._convert_model_to_diffusers")
        backend.convert_model_to_diffusers = lambda args: None
        backend.AutoTokenizer = SimpleNamespace(
            from_pretrained=lambda *args, **kwargs: None
        )
        conversion.CheckpointDirHf = type(
            "CheckpointDirHf", (), {"download": lambda self: None}
        )
        conversion.main = main
        for module in [backend, conversion]:
            monkeypatch.setitem(sys.modules, module.__name__, module)
            monkeypatch.setattr(
                scripts, module.__name__.rsplit(".", 1)[-1], module, raising=False
            )
        argv = ["-c", str(local), ""]
        code = converter._CONVERT_DIFFUSERS_CODE
    monkeypatch.setattr(sys, "argv", [*argv, "--checkpoint-path", "/dcp/model"])
    exec(code, {})
    assert (
        observed["tokenizer_class"]
        == "cosmos_framework.model.generator.tokenizers.tokenization_qwen2"
    )
    assert observed["tokens"] == [0, 1, 2]


def test_cli_rejects_wan_root_without_vae_config_before_loading_weights(
    converter, monkeypatch, tmp_path, capsys
):
    wan_root = tmp_path / "wan"
    wan_root.mkdir()
    output = tmp_path / "output"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "convert_checkpoint.py",
            "--src",
            str(tmp_path / "absent_weights.pt"),
            "--out",
            str(output),
            "--vae-checkpoint-path",
            str(wan_root),
        ],
    )
    with pytest.raises(SystemExit) as error:
        converter.main()
    assert error.value.code == 2
    assert "containing vae/config.json" in capsys.readouterr().err
    assert not output.exists()


@pytest.mark.parametrize("local", [False, True])
@pytest.mark.parametrize("fail", [False, True])
def test_export_uses_native_metadata_tokenizer_and_restores_source_setup(
    converter, framework_tokenizer_routes, monkeypatch, local, fail
):
    scripts = pytest.importorskip("cosmos_framework.scripts")
    backend = pytest.importorskip(
        "cosmos_framework.scripts._convert_model_to_diffusers"
    )
    source_tokenizer = {
        "_target_": "cosmos_framework.model.generator.tokenizers.wan2pt2_vae_4x16x16.Wan2pt2VAEInterface",
        "vae_path": "/unavailable/Wan2.2_VAE.pth",
        "spatial_compression_factor": 16,
        "temporal_compression_factor": 4,
        "chunk_duration": 93,
        "causal": True,
    }
    owner = SimpleNamespace(
        config=SimpleNamespace(
            tokenizer=source_tokenizer,
            state_ch=48,
            sound_gen=True,
        )
    )
    observed = {}

    def set_up_tokenizers(model):
        observed["tokenizer_config"] = dict(model.config.tokenizer)
        observed["sound_gen"] = model.config.sound_gen
        if local:
            metadata_kwargs = {
                name: value
                for name, value in model.config.tokenizer.items()
                if name != "_target_"
            }
            metadata = backend._MetadataVideoTokenizer(**metadata_kwargs)
            assert metadata.latent_ch == 48
            assert metadata.spatial_compression_factor == 16
            assert metadata.temporal_compression_factor == 4
            assert metadata.get_latent_num_frames(17) == 5
            assert metadata.get_pixel_num_frames(5) == 17
        if fail:
            raise RuntimeError("export failed after tokenizer setup")
        return "initialized"

    monkeypatch.setattr(backend.OmniMoTModel, "set_up_tokenizers", set_up_tokenizers)
    export = ModuleType("cosmos_framework.scripts.export_model")
    export._coerce_to_base_model = lambda data: None
    export.bundle_processor_from_tokenizer_node = lambda node, output: None

    def main():
        observed["main_called"] = True
        observed["result"] = backend.OmniMoTModel.set_up_tokenizers(owner)

    export.main = main
    monkeypatch.setitem(sys.modules, export.__name__, export)
    monkeypatch.setattr(scripts, "export_model", export, raising=False)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "-c",
            ",".join(converter.EXPORT_DROPPED_CONFIG_KEYS),
            "/local/qwen" if local else "",
            "--checkpoint-path",
            "/dcp/model",
        ],
    )
    if fail:
        with pytest.raises(RuntimeError, match="export failed after tokenizer setup"):
            exec(converter._EXPORT_MODEL_CODE, {})
    else:
        exec(converter._EXPORT_MODEL_CODE, {})
        assert observed["result"] == "initialized"

    assert observed["main_called"] is True
    assert owner.config.tokenizer is source_tokenizer
    assert owner.config.sound_gen is True
    assert backend.OmniMoTModel.set_up_tokenizers is set_up_tokenizers
    if local:
        assert (
            observed["tokenizer_config"]["_target_"]
            == f"{backend.__name__}._MetadataVideoTokenizer"
        )
        assert observed["sound_gen"] is False
        assert "vae_path" not in observed["tokenizer_config"]
    else:
        assert observed["tokenizer_config"] == source_tokenizer
        assert observed["sound_gen"] is True
