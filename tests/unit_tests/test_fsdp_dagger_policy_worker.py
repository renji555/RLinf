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

"""DAgger supervised accumulation and optimizer contracts on CPU."""

import contextlib
import logging
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
from omegaconf import OmegaConf

from rlinf.workers.actor.fsdp_dagger_policy_worker import EmbodiedDAGGERFSDPPolicy


class SmallPolicy(torch.nn.Module):
    def __init__(self, dictionary_output):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(()))
        self.dictionary_output = dictionary_output
        self.steps = []

    def forward(self, batch):
        loss = (self.weight - batch["target"]).square().mean()
        if self.dictionary_output:
            return {"loss": loss, "flow_matching_loss_action": loss}
        return loss

    def set_global_step(self, step, **kwargs):
        self.steps.append((step, kwargs))


class CPUStrategy:
    """Replace distributed hardware while using the real manager/scaler step."""

    def clip_grad_norm_(self, *, model):
        return torch.nn.utils.clip_grad_norm_(model.parameters(), 1000)


class CPUWorker(EmbodiedDAGGERFSDPPolicy):
    @property
    def device(self):
        return torch.device("cpu")

    def worker_timer(self, *args, **kwargs):
        return contextlib.nullcontext()

    @contextlib.contextmanager
    def before_micro_batch(self, model, *, is_last_micro_batch):
        self.context_flags.append(is_last_micro_batch)
        self.inside_backward_context = True
        try:
            yield
        finally:
            self.inside_backward_context = False

    def forward_actor(self, batch):
        if not self.inside_backward_context:
            raise AssertionError(
                "Forward must be inside the FSDP accumulation context."
            )
        return self.model(batch)


def make_worker(*, model_type="cosmos3", dictionary_output=True, targets=(1.0, 3.0)):
    worker = object.__new__(CPUWorker)
    worker.cfg = SimpleNamespace(
        actor=SimpleNamespace(model=SimpleNamespace(model_type=model_type))
    )
    worker.model = SmallPolicy(dictionary_output)
    worker.optimizer = torch.optim.SGD(worker.model.parameters(), lr=0.1)
    worker.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
        worker.optimizer, lambda _: 1
    )
    worker.grad_scaler = torch.amp.GradScaler("cpu", enabled=True, init_scale=128)
    worker.amp_context = contextlib.nullcontext()
    worker._strategy = CPUStrategy()
    worker.optimizer_steps = 0
    worker.critic_warmup_steps = 0
    worker.update_step = 0
    worker._logger = logging.getLogger(__name__)
    worker._stacklevel = 1
    worker.enable_drq = False
    worker._data_epoch = 0
    worker._lerobot_loader_lock = threading.Lock()
    worker._lerobot_loader = torch.utils.data.DataLoader(
        [{"target": torch.tensor(value)} for value in targets], batch_size=1
    )
    worker._lerobot_iter = iter(worker._lerobot_loader)
    worker.inside_backward_context = False
    worker.context_flags = []
    return worker


class DaggerWorkerTests(unittest.TestCase):
    def test_scaled_accumulation_matches_global_batch_gradient(self):
        worker = make_worker()
        metrics = worker.update_lerobot_one_epoch()
        torch.testing.assert_close(worker.model.weight, torch.tensor(0.4))
        self.assertEqual(worker.context_flags, [False, True])
        self.assertEqual(worker.update_step, 1)
        self.assertEqual(worker.optimizer_steps, 1)
        self.assertEqual(worker.lr_scheduler.last_epoch, 1)
        self.assertEqual(worker.model.steps, [(1, {})])
        self.assertEqual(metrics["dagger/actor_loss"], 5)
        self.assertEqual(metrics["dagger/flow_matching_loss_action"], 5)
        self.assertIsNone(worker.model.weight.grad)
        # The iterator is reset for the next DAgger update epoch.
        worker.update_lerobot_one_epoch()
        torch.testing.assert_close(worker.model.weight, torch.tensor(0.72))
        self.assertEqual(worker.update_step, 2)

    def test_tensor_loss_preserves_openpi_updates(self):
        worker = make_worker(
            model_type="openpi", dictionary_output=False, targets=(1.0,)
        )
        metrics = worker.update_lerobot_one_epoch()
        torch.testing.assert_close(worker.model.weight, torch.tensor(0.2))
        self.assertEqual(worker.model.steps, [])
        self.assertEqual(metrics["dagger/actor_loss"], 1)

    def test_nonfinite_gradients_skip_parameters_scheduler_and_ema(self):
        worker = make_worker(targets=(float("nan"),))
        metrics = worker.update_lerobot_one_epoch()
        self.assertEqual(worker.model.weight.item(), 0)
        self.assertEqual(worker.update_step, 0)
        self.assertEqual(worker.lr_scheduler.last_epoch, 0)
        self.assertEqual(worker.model.steps, [])
        self.assertEqual(metrics["dagger/update_skipped"], 1)
        self.assertIsNone(worker.model.weight.grad)

    def test_rollout_version_does_not_advance_cosmos_optimizer_iteration(self):
        worker = make_worker()
        worker.update_step = 7
        worker.set_global_step(100)
        self.assertEqual(worker.version, 100)
        self.assertEqual(worker.model.steps, [(7, {"update_ema": False})])

    def test_loss_contract_rejects_missing_detached_and_vector_losses(self):
        for output in (
            {},
            {"loss": torch.tensor(1.0)},
            torch.ones(2, requires_grad=True),
        ):
            with self.subTest(output=output):
                with self.assertRaises(ValueError):
                    EmbodiedDAGGERFSDPPolicy._sft_loss_and_metrics(output)

    def test_training_horizon_is_independent_of_execution_chunk(self):
        with tempfile.TemporaryDirectory() as root:
            worker = make_worker()
            worker.cfg = OmegaConf.create(
                {
                    "actor": {"model": {"num_action_chunks": 10}},
                    "algorithm": {
                        "dagger": {
                            "online_lerobot": {"data_path": root, "min_frames": 1024}
                        }
                    },
                }
            )
            worker._build_lerobot_dataset()
            self.assertEqual(worker.dataset.chunk_size, 10)
            worker.cfg.algorithm.dagger.online_lerobot.training_action_horizon = 16
            worker._build_lerobot_dataset()
            self.assertEqual(worker.dataset.chunk_size, 16)
            self.assertEqual(worker.cfg.actor.model.num_action_chunks, 10)
            self.assertEqual(worker.dataset.min_frames, 1024)
            self.assertFalse(worker.dataset.is_ready())
            for invalid in (0, -1, 1.5, True, None, "16"):
                worker.cfg.algorithm.dagger.online_lerobot.training_action_horizon = (
                    invalid
                )
                with self.subTest(horizon=invalid):
                    with self.assertRaisesRegex(ValueError, "positive integer"):
                        worker._build_lerobot_dataset()

    def test_incomplete_archives_reserve_their_directory_names(self):
        with tempfile.TemporaryDirectory() as root:
            worker = make_worker()
            worker.cfg = OmegaConf.create(
                {
                    "actor": {"model": {"num_action_chunks": 10}},
                    "algorithm": {"dagger": {"online_lerobot": {"data_path": root}}},
                }
            )
            worker._rank = 3
            worker._next_lerobot_archive_id = 0
            worker._pending_archive_path = None
            worker._build_lerobot_dataset()
            rank_dir = Path(root) / "rank_3"
            for archive_id in (0, 7):
                (rank_dir / f"id_{archive_id}").mkdir(parents=True)
            self.assertEqual(worker._discover_lerobot_resume_shards(), [])
            next_path = Path(worker._current_archive_path())
            self.assertEqual(next_path, rank_dir / "id_8")
            self.assertFalse(next_path.exists())
            self.assertTrue((rank_dir / "id_0").is_dir())
            self.assertTrue((rank_dir / "id_7").is_dir())


if __name__ == "__main__":
    unittest.main()
