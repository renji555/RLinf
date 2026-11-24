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

"""Join PPO sample metrics to recorded frames and burn them into rollout MP4s."""

import json
import math
from pathlib import Path
from typing import Any


def sample_key(sample_id: list[int]) -> str:
    """Encode step, epoch, env rank, stage, chunk and local environment index."""
    if len(sample_id) != 6 or any(
        type(value) is not int or value < 0 for value in sample_id
    ):
        raise ValueError(
            f"Expected six nonnegative integer sample IDs, got {sample_id}."
        )
    return ":".join(map(str, sample_id))


def load_video_metrics(
    log_path: str | Path, step: int, expected_ranks: int | list[int]
) -> dict[str, dict[str, Any]]:
    """Load every actor shard; retain the final training encounter of each sample."""
    root = Path(log_path) / "video/train/ppo_metrics" / f"step_{step + 1:06d}"
    if isinstance(expected_ranks, int):
        expected_ranks = list(range(expected_ranks))
    latest, seen, owners = {}, set(), {}
    for rank in expected_ranks:
        path = root / f"rank_{rank:03d}.jsonl"
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                record = json.loads(line)
                key = sample_key(record["sample_id"])
                if record["sample_id"][0] != step or record["actor_rank"] != rank:
                    raise ValueError(
                        f"Metric identity does not match shard {path}: {key}."
                    )
                update = (record["update_epoch"], record["optimizer_step"])
                if any(type(value) is not int or value < 1 for value in update):
                    raise ValueError(
                        f"Expected one-based update/optimizer counters, got {update}."
                    )
                if (key, *update) in seen or (key in owners and owners[key] != rank):
                    raise ValueError(
                        f"Duplicate PPO sample/update identity: {key}, {update}."
                    )
                seen.add((key, *update))
                owners[key] = rank
                if key not in latest or update > (
                    latest[key]["update_epoch"],
                    latest[key]["optimizer_step"],
                ):
                    latest[key] = record
    return latest


def _format(value: Any) -> str:
    if value is None:
        return "N/A"
    result = float(value)
    return f"{result:+.5g}" if math.isfinite(result) else "N/A"


def _lines(record: dict[str, Any] | None, frame_index: int) -> list[str]:
    if record is None:
        return [
            f"Video frame {frame_index}",
            "Reset / boundary observation",
            "adv / logp / loss: N/A",
        ]
    step, epoch, rank, _stage, chunk, env = record["sample_id"]
    return [
        f"r={step + 1} ep={epoch} env={rank}:{env} c={chunk} f={frame_index}",
        f"adv={_format(record['advantage'])} ratio={_format(record['ratio'])}",
        f"logp old={_format(record['logprob_old'])}",
        f"logp new={_format(record['logprob_new'])}",
        f"policy loss={_format(record['policy_loss'])}",
        f"value loss={_format(record['value_loss'])} valid={record['valid']}",
        f"V(train)={_format(record['value'])} R={_format(record['return'])}",
        f"loss e{record['update_epoch']}/opt{record['optimizer_step']} (before step)",
    ]


def annotate_video(mp4_path: str | Path, metrics: dict[str, dict[str, Any]]) -> Path:
    """Atomically replace an MP4 with per-tile PPO footers, retaining its frames/FPS.

    Each manifest entry describes the existing rendered tiles. Reset frames have
    no action sample, and one chunk's metrics apply to its intermediate frames.
    The source MP4 stays intact if decoding, alignment or encoding fails.
    """
    import imageio.v2 as imageio
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont

    path = Path(mp4_path)
    manifest_path = path.with_suffix(".ppo_frames.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("ppo_annotated", False):
        return path
    frames = manifest["frames"]
    if not frames:
        raise ValueError(f"Expected nonempty frame manifest for {path}.")
    layout = frames[0]["tile_boxes"]
    if not layout:
        raise ValueError(f"Expected at least one video tile for {path}.")
    if any(
        len(box) != 4
        or any(type(v) is not int or v < 0 for v in box)
        or min(box[2:]) < 1
        for box in layout
    ):
        raise ValueError(
            f"Expected nonnegative integer tile boxes with positive size: {layout}."
        )
    widths, heights = {box[2] for box in layout}, {box[3] for box in layout}
    if len(widths) != 1 or len(heights) != 1:
        raise ValueError(
            "PPO video annotation requires equally sized environment tiles."
        )
    tile_width, tile_height = next(iter(widths)), next(iter(heights))
    if any(x % tile_width or y % tile_height for x, y, _, _ in layout):
        raise ValueError("Expected tile boxes on an aligned rectangular grid.")
    if len({(box[0], box[1]) for box in layout}) != len(layout):
        raise ValueError("Duplicate tile boxes in PPO video manifest.")
    rows = max(y // tile_height for _, y, _, _ in layout) + 1
    for frame in frames:
        if frame["tile_boxes"] != layout:
            raise ValueError("PPO video tile layout changed within one MP4.")
        ids = frame["sample_ids"]
        if ids is not None and len(ids) != len(layout):
            raise ValueError("Expected one sample ID per environment video tile.")
        for sample_id in ids or []:
            if sample_id is not None and sample_key(sample_id) not in metrics:
                raise ValueError(
                    f"Missing training metrics for video sample {sample_id}."
                )

    font = ImageFont.load_default()
    line_height, footer_height = 13, 112
    temporary = path.with_name(path.stem + ".ppo-overlay.tmp.mp4")
    reader, writer = None, None
    try:
        reader = imageio.get_reader(path)
        metadata = reader.get_meta_data()
        fps = float(metadata["fps"])
        if not math.isfinite(fps) or fps <= 0:
            raise ValueError(f"Expected positive finite video FPS, got {fps}.")
        writer = imageio.get_writer(temporary, fps=fps, macro_block_size=1)
        count = 0
        for index, image in enumerate(reader):
            if index >= len(frames):
                raise ValueError("MP4 has more frames than its PPO frame manifest.")
            height, width = image.shape[:2]
            if any(x + w > width or y + h > height for x, y, w, h in layout):
                raise ValueError("PPO tile boxes extend outside the recorded frame.")
            output_height = height + rows * footer_height
            canvas = Image.new(
                "RGB", (width + width % 2, output_height + output_height % 2)
            )
            ids = frames[index]["sample_ids"] or [None] * len(layout)
            for box, sample_id in zip(layout, ids):
                x, y, w, h = box
                top = y + (y // tile_height) * footer_height
                canvas.paste(Image.fromarray(image[y : y + h, x : x + w]), (x, top))
                record = (
                    metrics[sample_key(sample_id)] if sample_id is not None else None
                )
                panel = Image.new("RGB", (w, footer_height))
                drawing = ImageDraw.Draw(panel)
                for line_index, text in enumerate(_lines(record, index)):
                    drawing.text(
                        (4, 3 + line_index * line_height), text, font=font, fill="white"
                    )
                canvas.paste(panel, (x, top + h))
            writer.append_data(np.asarray(canvas))
            count += 1
        if count != len(frames):
            raise ValueError(
                f"MP4 has {count} frames but its manifest has {len(frames)}."
            )
        writer.close()
        writer = None
        reader.close()
        reader = None
        temporary.replace(path)
        manifest["ppo_annotated"] = True
        sidecar_temporary = manifest_path.with_suffix(".json.tmp")
        sidecar_temporary.write_text(json.dumps(manifest), encoding="utf-8")
        sidecar_temporary.replace(manifest_path)
    finally:
        if writer is not None:
            writer.close()
        if reader is not None:
            reader.close()
        temporary.unlink(missing_ok=True)
    return path


def make_sample_ids(key, size: int):
    """Create identities before routing splits a source's environment batch."""
    import torch

    prefix = [key.step_id, key.epoch_id, key.env_rank, key.stage_id, key.chunk_id]
    ids = torch.empty(size, 6, dtype=torch.long)
    ids[:, :5] = torch.tensor(prefix, dtype=torch.long)
    ids[:, 5] = torch.arange(size)
    return ids


def collect_video_metrics(
    sample_ids,
    sample_metrics: dict,
    *,
    actor_rank: int,
    update_epoch: int,
    optimizer_step: int,
) -> list[dict]:
    """Read detached per-chunk loss terms from the actual training forward."""
    import math

    import torch

    ids = sample_ids.detach().cpu()
    if ids.ndim != 2 or ids.shape[1] != 6 or ids.dtype != torch.long:
        raise ValueError(
            f"Expected video_sample_id int64 [B,6], got {ids.shape}/{ids.dtype}."
        )
    required = {
        "advantage",
        "logprob_old",
        "logprob_new",
        "ratio",
        "policy_loss",
        "value_loss",
        "value",
        "return",
        "loss_mask",
    }
    if not required.issubset(sample_metrics):
        raise ValueError(
            f"Missing PPO video metrics: {sorted(required - sample_metrics.keys())}."
        )
    columns = {}
    for name in required | {"raw_advantage", "rollout_value", "reward"}.intersection(
        sample_metrics
    ):
        values = sample_metrics[name].detach().reshape(-1).cpu()
        if len(values) != len(ids):
            raise ValueError(
                f"Expected one {name} per chunk, got {values.shape} for {ids.shape}."
            )
        columns[name] = values.tolist()
    rows = []
    for index, sample_id in enumerate(ids.tolist()):
        row = {
            "sample_id": sample_id,
            "actor_rank": actor_rank,
            "update_epoch": update_epoch,
            "optimizer_step": optimizer_step,
        }
        for name, values in columns.items():
            value = values[index]
            if name == "loss_mask":
                row["valid"] = bool(value)
            else:
                row[name] = float(value) if math.isfinite(value) else None
        rows.append(row)
    return rows


def write_video_metrics(
    log_path: str, step: int, actor_rank: int, rows: list[dict]
) -> None:
    """Atomically publish one actor shard after its rollout training is complete."""
    import json
    from pathlib import Path

    directory = Path(log_path) / "video/train/ppo_metrics" / f"step_{step + 1:06d}"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"rank_{actor_rank:03d}.jsonl"
    temporary = path.with_suffix(".jsonl.tmp")
    temporary.write_text(
        "".join(json.dumps(row, allow_nan=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    temporary.replace(path)


def load_video_metric_history(
    log_path: str | Path, step: int, expected_ranks: int | list[int]
) -> dict[str, list[dict[str, Any]]]:
    """Load every training encounter after validating all metric shard identities."""
    load_video_metrics(log_path, step, expected_ranks)
    ranks = range(expected_ranks) if isinstance(expected_ranks, int) else expected_ranks
    root = Path(log_path) / "video/train/ppo_metrics" / f"step_{step + 1:06d}"
    histories = {}
    for rank in ranks:
        with (root / f"rank_{rank:03d}.jsonl").open(encoding="utf-8") as stream:
            for line in stream:
                record = json.loads(line)
                histories.setdefault(sample_key(record["sample_id"]), []).append(record)
    for records in histories.values():
        records.sort(
            key=lambda record: (record["update_epoch"], record["optimizer_step"])
        )
    return histories


def build_video_details(
    mp4_path: str | Path,
    metrics: dict[str, dict[str, Any]],
    histories: dict[str, list[dict[str, Any]]] | None = None,
) -> dict[str, Any]:
    """Join recorded frame identities to exact, unrounded training metric records."""
    import imageio.v2 as imageio

    path = Path(mp4_path)
    manifest = json.loads(
        path.with_suffix(".ppo_frames.json").read_text(encoding="utf-8")
    )
    frames = manifest["frames"]
    if not frames or not frames[0]["tile_boxes"]:
        raise ValueError(f"Expected nonempty frames and tile layout for {path}.")
    layout = frames[0]["tile_boxes"]
    width, height = layout[0][2:]
    if width < 1 or height < 1:
        raise ValueError("Expected positive video tile dimensions.")
    if any(
        w != width or h != height or x % width or y % height for x, y, w, h in layout
    ):
        raise ValueError("Expected equal-sized, grid-aligned video tiles.")
    with imageio.get_reader(path) as reader:
        metadata = reader.get_meta_data()
        fps = float(metadata["fps"])
        count = metadata.get("nframes", float("inf"))
        # The annotation pass already checked every decoded frame against this
        # manifest. A second full decode is only needed for unannotated videos.
        if not math.isfinite(count) and not manifest.get("ppo_annotated", False):
            count = reader.count_frames()
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError(f"Expected positive finite video FPS, got {fps}.")
    if math.isfinite(count) and count != len(frames):
        raise ValueError(f"MP4 has {count} frames but its manifest has {len(frames)}.")

    output_frames, samples = [], {}
    for frame in frames:
        identities = frame["sample_ids"] or [None] * len(layout)
        if frame["tile_boxes"] != layout or len(identities) != len(layout):
            raise ValueError("Expected stable video tiles and one sample ID per tile.")
        keys = []
        for identity in identities:
            key = sample_key(identity) if identity is not None else None
            keys.append(key)
            if key is None or key in samples:
                continue
            if key not in metrics:
                raise ValueError(
                    f"Missing training metrics for video sample {identity}."
                )
            records = histories.get(key) if histories is not None else [metrics[key]]
            if not records or any(
                sample_key(record["sample_id"]) != key for record in records
            ):
                raise ValueError(
                    f"Missing or mismatched training history for video sample {identity}."
                )
            samples[key] = sorted(
                records,
                key=lambda record: (record["update_epoch"], record["optimizer_step"]),
            )
            if samples[key][-1] != metrics[key]:
                raise ValueError(
                    f"Latest history differs from the video overlay metrics for {identity}."
                )
        output_frames.append({**frame, "sample_ids": keys})
    return {
        "schema_version": 1,
        "video": {
            "name": path.name,
            "src": path.name,
            "fps": fps,
            "frame_count": len(frames),
            "ppo_annotated": bool(manifest.get("ppo_annotated", False)),
            "rows": max(y // height for _, y, _, _ in layout) + 1,
            "cols": max(x // width for x, _, _, _ in layout) + 1,
        },
        "frames": output_frames,
        "samples": samples,
        "csv_href": path.with_suffix(".ppo.csv").name,
        "json_href": path.with_suffix(".ppo_details.json").name,
    }


def _write_video_web_text(path: Path, text: str) -> None:
    """Replace a generated companion file only after its complete write."""
    import os
    import tempfile

    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=path.name + ".",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            stream.write(text)
        os.chmod(temporary, 0o644)
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def export_video_page(
    mp4_path: str | Path,
    metrics: dict[str, dict[str, Any]],
    histories: dict[str, list[dict[str, Any]]] | None = None,
    *,
    output_dir: str | Path | None = None,
) -> Path:
    """Save a detail page, full metric history and one latest-encounter CSV row per frame/tile."""
    import csv
    import io

    source = Path(mp4_path)
    details = build_video_details(source, metrics, histories)
    path = source if output_dir is None else Path(output_dir) / source.name
    path.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(
        details, ensure_ascii=False, allow_nan=False, separators=(",", ":")
    )
    template = (
        Path(__file__).with_name("ppo_video_detail.html").read_text(encoding="utf-8")
    )
    if template.count("__PPO_VIDEO_DATA__") != 1:
        raise ValueError(
            "Expected exactly one PPO video data placeholder in the detail template."
        )
    embedded = (
        serialized.replace("<", "\\u003c")
        .replace("\u2028", "\\u2028")
        .replace("\u2029", "\\u2029")
    )
    page = path.with_suffix(".ppo.html")
    _write_video_web_text(path.with_suffix(".ppo_details.json"), serialized)
    csv_buffer = io.StringIO(newline="")
    identity_fields = [
        "rollout_step",
        "rollout_epoch",
        "env_rank",
        "stage_id",
        "chunk_id",
        "env_slot",
    ]
    metric_fields = [
        "actor_rank",
        "update_epoch",
        "optimizer_step",
        "advantage",
        "raw_advantage",
        "reward",
        "logprob_old",
        "logprob_new",
        "ratio",
        "policy_loss",
        "value_loss",
        "value",
        "rollout_value",
        "return",
        "valid",
    ]
    frame_info_fields = {
        "frame_reward": "reward",
        "frame_termination": "termination",
        "frame_episode_len": "episode.episode_len",
        "frame_success_once": "episode.success_once",
    }
    writer = csv.DictWriter(
        csv_buffer,
        fieldnames=[
            "video_frame",
            "time_seconds",
            "video_tile",
            "frame_kind",
            "sample_key",
            *identity_fields,
            *frame_info_fields,
            *(
                "training_reward" if field == "reward" else field
                for field in metric_fields
            ),
        ],
    )
    writer.writeheader()
    for frame_index, frame in enumerate(details["frames"]):
        infos = frame.get("infos")
        if infos is not None and len(infos) != len(frame["sample_ids"]):
            raise ValueError(
                "Expected one recorded environment info dict per video tile."
            )
        for tile, key in enumerate(frame["sample_ids"]):
            row = {
                "video_frame": frame_index,
                "time_seconds": frame_index / details["video"]["fps"],
                "video_tile": tile,
                "frame_kind": "action_chunk" if key is not None else "boundary",
                "sample_key": key or "",
            }
            if infos is not None:
                row.update(
                    {
                        field: infos[tile].get(source)
                        for field, source in frame_info_fields.items()
                    }
                )
            if key is not None:
                record = details["samples"][key][-1]
                row.update(zip(identity_fields, record["sample_id"]))
                row.update(
                    {
                        "training_reward" if field == "reward" else field: record.get(
                            field
                        )
                        for field in metric_fields
                    }
                )
            writer.writerow(row)
    _write_video_web_text(path.with_suffix(".ppo.csv"), csv_buffer.getvalue())
    _write_video_web_text(page, template.replace("__PPO_VIDEO_DATA__", embedded))
    first = next(
        (records[-1]["sample_id"] for records in details["samples"].values()), None
    )
    summary = {
        "schema_version": 1,
        "name": path.name,
        "page": page.name,
        "video": path.name,
        "seed": path.parent.name,
        "step": first[0] if first is not None else None,
        "epoch": first[1] if first is not None else None,
        "env_rank": first[2] if first is not None else None,
        "fps": details["video"]["fps"],
        "frame_count": details["video"]["frame_count"],
        "sample_count": len(details["samples"]),
        "ppo_annotated": details["video"]["ppo_annotated"],
    }
    _write_video_web_text(
        path.with_suffix(".ppo_summary.json"),
        json.dumps(summary, ensure_ascii=False, allow_nan=False),
    )
    return page


def build_video_catalog(video_root: str | Path) -> Path:
    """Build a compact catalog without loading each video's full frame data."""
    root = Path(video_root)
    videos = []
    for summary_path in root.glob("seed_*/*.ppo_summary.json"):
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        page = summary_path.with_name(summary["page"])
        if not page.is_file():
            continue
        summary["href"] = page.relative_to(root).as_posix()
        videos.append(summary)
    videos.sort(
        key=lambda video: (-video["step"], video["seed"], video["epoch"], video["name"])
    )
    data = {"run_name": root.parent.parent.name, "videos": videos}
    serialized = json.dumps(data, ensure_ascii=False, allow_nan=False)
    embedded = (
        serialized.replace("<", "\\u003c")
        .replace("\u2028", "\\u2028")
        .replace("\u2029", "\\u2029")
    )
    template = (
        Path(__file__).with_name("ppo_video_catalog.html").read_text(encoding="utf-8")
    )
    root.mkdir(parents=True, exist_ok=True)
    path = root / "index.html"
    import tempfile

    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=root, suffix=".html.tmp", delete=False
    ) as stream:
        stream.write(template.replace("__PPO_CATALOG__", embedded))
        temporary = Path(stream.name)
    try:
        temporary.chmod(0o644)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    return path
