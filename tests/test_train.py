import argparse
import os
import shutil
import tempfile
import torch
import pytest

from scripts.train import Trainer, get_args


class TestTrainResume:
    @pytest.fixture(autouse=True)
    def setup_and_teardown(self):
        self.temp_dir = tempfile.mkdtemp()
        yield
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def _get_base_args(self, checkpoint_dir, resume_path=None):
        return argparse.Namespace(
            model="resnet18",
            pretrained=False,
            posenet_pretrained=False,
            resume=resume_path,
            num_scales=4,
            disp_bias_init=-4.5,
            dataset_dir="data/sample",
            split_dir="data/sample/splits",
            checkpoint_dir=checkpoint_dir,
            max_samples=2,
            height=192,
            width=640,
            frame_ids=[0, -1, 1],
            use_stereo=False,
            use_automask=True,
            batch_size=1,
            epochs=2,
            lr=1e-4,
            lr_step_size=15,
            lr_gamma=0.1,
            weight_decay=1e-4,
            min_depth=0.1,
            max_depth=100.0,
            smoothness_weight=0.001,
            warmup_steps=1000,
            dead_neuron_std_thresh=1e-6,
            max_consecutive_collapse=100,
            fail_on_collapse=False,
            num_workers=0,
            use_amp=False,
            no_cuda=True,
            log_freq=1,
            save_freq=1,
        )

    def test_resume_roundtrip(self):
        # 1. Initialize first trainer
        args = self._get_base_args(checkpoint_dir=self.temp_dir)
        trainer1 = Trainer(args)

        # Modify state to simulate active training
        trainer1.health_monitor.step_count = 137
        trainer1.health_monitor.consecutive_collapse_batches = 3
        trainer1.best_val_loss = 0.0425

        # Take an optimizer step so optimizer state contains momentum buffers
        dummy_loss = sum(p.sum() for p in trainer1.depth_net.parameters())
        dummy_loss.backward()
        trainer1.optimizer.step()

        # Save checkpoint at epoch 1
        trainer1.save_checkpoint(epoch=1, is_best=True)

        ckpt_file = os.path.join(self.temp_dir, "checkpoint_epoch_2.pth")
        best_file = os.path.join(self.temp_dir, "best_model.pth")
        assert os.path.isfile(ckpt_file), f"Checkpoint missing: {ckpt_file}"
        assert os.path.isfile(best_file), f"Best model checkpoint missing: {best_file}"

        # 2. Initialize second trainer resuming from checkpoint
        resume_args = self._get_base_args(checkpoint_dir=self.temp_dir, resume_path=ckpt_file)
        trainer2 = Trainer(resume_args)

        # Assert epoch resumed
        assert trainer2.start_epoch == 2, f"Expected start_epoch=2, got {trainer2.start_epoch}"
        # Assert health monitor state restored
        assert trainer2.health_monitor.step_count == 137, (
            f"Expected health step_count=137, got {trainer2.health_monitor.step_count}"
        )
        assert trainer2.health_monitor.consecutive_collapse_batches == 3
        # Assert best val loss restored
        assert abs(trainer2.best_val_loss - 0.0425) < 1e-6, (
            f"Expected best_val_loss=0.0425, got {trainer2.best_val_loss}"
        )

        # Assert weights match exactly
        for (n1, p1), (n2, p2) in zip(
            trainer1.depth_net.named_parameters(), trainer2.depth_net.named_parameters()
        ):
            assert torch.allclose(p1, p2), f"Weight mismatch in depth_net layer {n1}"

        for (n1, p1), (n2, p2) in zip(
            trainer1.pose_net.named_parameters(), trainer2.pose_net.named_parameters()
        ):
            assert torch.allclose(p1, p2), f"Weight mismatch in pose_net layer {n1}"

    def test_posenet_pretrained_arg_exists(self):
        # Confirm posenet_pretrained is exposed via get_args parser
        import sys
        test_argv = ["train.py", "--model", "resnet18", "--posenet_pretrained"]
        sys_argv_bak = sys.argv
        try:
            sys.argv = test_argv
            parsed = get_args()
            assert hasattr(parsed, "posenet_pretrained")
            assert parsed.posenet_pretrained is True
        finally:
            sys.argv = sys_argv_bak
