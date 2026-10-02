"""Exercise epoch and model transitions without downloading the dataset."""

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

import spot_checkpoint
from upstream import flowers_comparison as upstream


class SimulatedInterruption(Exception):
    pass


class SpotCheckpointTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / spot_checkpoint.CHECKPOINT_NAME
        patcher = mock.patch.object(spot_checkpoint, "checkpoint_path", return_value=self.path)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.args = SimpleNamespace(
            epochs=3,
            seed=42,
            batch_size=2,
            test_batch_size=2,
            lr=0.001,
            dry_run=False,
            save_model=False,
            log_interval=100,
        )
        data = torch.tensor(
            [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0],
             [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]
        )
        labels = torch.tensor([0, 1, 0, 1])
        dataset = TensorDataset(data, labels)
        self.train_loader = DataLoader(dataset, batch_size=2, shuffle=True)
        self.test_loader = DataLoader(dataset, batch_size=2, shuffle=False)
        self.device = torch.device("cpu")

    @staticmethod
    def model_builder(_num_classes):
        return nn.Linear(4, upstream.NUM_CLASSES)

    def train_model(self, name):
        builder = getattr(self, "real_builders", {}).get(name, self.model_builder)
        return spot_checkpoint.run_training(
            self.args,
            self.device,
            self.train_loader,
            self.test_loader,
            name,
            builder,
        )

    def train_pair(self):
        return [self.train_model(name) for name in spot_checkpoint.MODEL_NAMES]

    def interrupt_after(self, model_name, epoch):
        original = spot_checkpoint.write_checkpoint

        def write_then_interrupt(path, state):
            original(path, state)
            if state["model_name"] == model_name and state["epoch"] == epoch:
                raise SimulatedInterruption

        return mock.patch.object(
            spot_checkpoint, "write_checkpoint", side_effect=write_then_interrupt
        )

    def expected_result(self):
        upstream.set_fixed_seed(self.args.seed)
        result = self.train_pair()
        self.path.unlink()
        return result

    def test_resume_baseline_epoch(self):
        expected = self.expected_result()
        upstream.set_fixed_seed(self.args.seed)
        with self.interrupt_after(spot_checkpoint.MODEL_NAMES[0], 2):
            with self.assertRaises(SimulatedInterruption):
                self.train_model(spot_checkpoint.MODEL_NAMES[0])
        upstream.set_fixed_seed(self.args.seed)
        self.assertEqual(self.train_pair(), expected)

    def test_resume_perforated_epoch(self):
        expected = self.expected_result()
        upstream.set_fixed_seed(self.args.seed)
        self.train_model(spot_checkpoint.MODEL_NAMES[0])
        with self.interrupt_after(spot_checkpoint.MODEL_NAMES[1], 2):
            with self.assertRaises(SimulatedInterruption):
                self.train_model(spot_checkpoint.MODEL_NAMES[1])
        upstream.set_fixed_seed(self.args.seed)
        self.assertEqual(self.train_pair(), expected)

    def test_restart_between_models(self):
        expected = self.expected_result()
        upstream.set_fixed_seed(self.args.seed)
        self.train_model(spot_checkpoint.MODEL_NAMES[0])
        upstream.set_fixed_seed(self.args.seed)
        self.assertEqual(self.train_pair(), expected)

    def test_reject_mismatched_and_corrupt_state(self):
        upstream.set_fixed_seed(self.args.seed)
        self.train_model(spot_checkpoint.MODEL_NAMES[0])
        self.args.seed += 1
        with self.assertRaisesRegex(ValueError, "does not match"):
            self.train_model(spot_checkpoint.MODEL_NAMES[0])
        self.path.write_bytes(b"not a checkpoint")
        with self.assertRaisesRegex(RuntimeError, "Cannot read"):
            self.train_model(spot_checkpoint.MODEL_NAMES[0])

    def test_resume_real_pinned_models(self):
        """Reload both actual network formats after an epoch-one interruption."""
        from runner import build_pinned_perforated_model

        self.args.epochs = 2
        images = torch.rand(2, 3, 64, 64)
        labels = torch.tensor([0, 1])
        dataset = TensorDataset(images, labels)
        self.train_loader = DataLoader(dataset, batch_size=2, shuffle=True)
        self.test_loader = DataLoader(dataset, batch_size=2)
        self.real_builders = {
            spot_checkpoint.MODEL_NAMES[0]: upstream.build_torchvision_resnet18,
            spot_checkpoint.MODEL_NAMES[1]: build_pinned_perforated_model,
        }
        previous_threads = torch.get_num_threads()
        torch.set_num_threads(2)
        try:
            upstream.set_fixed_seed(self.args.seed)
            expected = self.train_pair()
            self.path.unlink()

            upstream.set_fixed_seed(self.args.seed)
            with self.interrupt_after(spot_checkpoint.MODEL_NAMES[0], 1):
                with self.assertRaises(SimulatedInterruption):
                    self.train_model(spot_checkpoint.MODEL_NAMES[0])
            upstream.set_fixed_seed(self.args.seed)
            baseline = self.train_model(spot_checkpoint.MODEL_NAMES[0])
            with self.interrupt_after(spot_checkpoint.MODEL_NAMES[1], 1):
                with self.assertRaises(SimulatedInterruption):
                    self.train_model(spot_checkpoint.MODEL_NAMES[1])
            upstream.set_fixed_seed(self.args.seed)
            perforated = self.train_model(spot_checkpoint.MODEL_NAMES[1])
            self.assertEqual([baseline, perforated], expected)
        finally:
            torch.set_num_threads(previous_threads)


if __name__ == "__main__":
    unittest.main()
