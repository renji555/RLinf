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

"""LeRobot archive schema and frame ownership contracts."""

import copy
import json

import numpy as np
import pytest

from rlinf.data.storage.lerobot.writer import LeRobotDatasetWriter


@pytest.mark.parametrize("with_images", [False, True])
def test_episode_roundtrip_preserves_scalar_schema_and_shared_frames(
    tmp_path, with_images
):
    pytest.importorskip("lerobot")
    parquet = pytest.importorskip("pyarrow.parquet")
    root = tmp_path / "archive"
    writer = LeRobotDatasetWriter()
    writer.create(
        repo_id=str(root),
        has_image=with_images,
        image_shape=(32, 32, 3),
        wrist_image_keys={"wrist_image": (32, 32, 3)} if with_images else None,
        has_segment_id=True,
        image_writer_processes=0,
        image_writer_threads=0,
    )
    frames = []
    for step in range(3):
        frame = {
            "state": np.arange(8, dtype=np.float32) + step,
            "actions": np.arange(7, dtype=np.float32) + step,
            "done": np.array([step == 2], dtype=bool),
            "is_success": np.array([True], dtype=bool),
            "intervene_flag": np.array([step == 1], dtype=bool),
            "segment_id": np.array([step], dtype=np.uint8),
            "task": "pick up the cube",
        }
        if with_images:
            frame["image"] = np.full((32, 32, 3), step * 32, dtype=np.uint8)
            frame["wrist_image"] = np.full((32, 32, 3), step * 16, dtype=np.uint8)
        frames.append(frame)
    before = copy.deepcopy(frames)
    try:
        writer.add_episode(frames)
        writer.add_episode(frames)
        assert writer.dataset.meta.total_episodes == 2
        assert writer.dataset.meta.total_frames == 6
    finally:
        writer.finalize()

    files = sorted(root.rglob("*.parquet"))
    assert files
    table = parquet.read_table(files)
    assert table.num_rows == 6
    assert str(table.schema.field("segment_id").type) == "uint8"
    assert table.column("segment_id").to_pylist() == [0, 1, 2] * 2
    assert table.column("done").to_pylist() == [False, False, True] * 2
    assert table.column("intervene_flag").to_pylist() == [False, True, False] * 2
    assert (
        table.column("actions").to_pylist()
        == [frame["actions"].tolist() for frame in before] * 2
    )
    assert (
        table.column("state").to_pylist()
        == [frame["state"].tolist() for frame in before] * 2
    )
    if with_images:
        assert {"image", "wrist_image"}.issubset(table.column_names)
        assert all(row["bytes"] for row in table.column("image").to_pylist())
    info = json.loads((root / "meta" / "info.json").read_text())
    assert info["total_episodes"] == 2
    assert info["features"]["segment_id"]["shape"] == [1]
    for actual, expected in zip(frames, before, strict=True):
        assert actual.keys() == expected.keys()
        for key in actual:
            if isinstance(expected[key], np.ndarray):
                np.testing.assert_array_equal(actual[key], expected[key])
                assert actual[key].shape == expected[key].shape
                assert actual[key].dtype == expected[key].dtype
            else:
                assert actual[key] == expected[key]
