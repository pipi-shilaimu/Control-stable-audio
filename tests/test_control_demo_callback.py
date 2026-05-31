from __future__ import annotations

import importlib.util
import sys
import types
import unittest
from pathlib import Path

import torch


_STUB_MODULE_NAMES = [
    "pytorch_lightning",
    "pytorch_lightning.utilities",
    "pytorch_lightning.utilities.rank_zero",
    "torchaudio",
    "einops",
    "stable_audio_tools",
    "stable_audio_tools.inference",
    "stable_audio_tools.inference.sampling",
    "stable_audio_tools.training",
    "stable_audio_tools.training.diffusion",
    "stable_audio_tools.training.utils",
    "stable_audio_tools.interface",
    "stable_audio_tools.interface.aeiou",
]


def _install_import_stubs() -> None:
    lightning = types.ModuleType("pytorch_lightning")

    class Callback:
        pass

    lightning.Callback = Callback
    sys.modules.setdefault("pytorch_lightning", lightning)

    rank_zero = types.ModuleType("pytorch_lightning.utilities.rank_zero")
    rank_zero.rank_zero_only = lambda fn: fn
    sys.modules.setdefault("pytorch_lightning.utilities", types.ModuleType("pytorch_lightning.utilities"))
    sys.modules.setdefault("pytorch_lightning.utilities.rank_zero", rank_zero)

    torchaudio = types.ModuleType("torchaudio")
    sys.modules.setdefault("torchaudio", torchaudio)

    einops = types.ModuleType("einops")
    einops.rearrange = lambda tensor, pattern: tensor.reshape(tensor.shape[1], tensor.shape[0] * tensor.shape[2])
    sys.modules.setdefault("einops", einops)

    sampling = types.ModuleType("stable_audio_tools.inference.sampling")
    sampling.sample = lambda *args, **kwargs: args[1]
    training_diffusion = types.ModuleType("stable_audio_tools.training.diffusion")

    class DiffusionCondTrainingWrapper:
        pass

    training_diffusion.DiffusionCondTrainingWrapper = DiffusionCondTrainingWrapper
    training_utils = types.ModuleType("stable_audio_tools.training.utils")
    training_utils.log_audio = lambda *args, **kwargs: None
    training_utils.log_image = lambda *args, **kwargs: None
    training_utils.log_point_cloud = lambda *args, **kwargs: None
    aeiou = types.ModuleType("stable_audio_tools.interface.aeiou")
    aeiou.audio_spectrogram_image = lambda audio: audio

    for name in [
        "stable_audio_tools",
        "stable_audio_tools.inference",
        "stable_audio_tools.training",
        "stable_audio_tools.interface",
    ]:
        sys.modules.setdefault(name, types.ModuleType(name))
    sys.modules.setdefault("stable_audio_tools.inference.sampling", sampling)
    sys.modules.setdefault("stable_audio_tools.training.diffusion", training_diffusion)
    sys.modules.setdefault("stable_audio_tools.training.utils", training_utils)
    sys.modules.setdefault("stable_audio_tools.interface.aeiou", aeiou)


def _load_callback_module():
    saved_modules = {name: sys.modules.get(name) for name in _STUB_MODULE_NAMES}
    _install_import_stubs()
    repo_root = Path(__file__).resolve().parents[1]
    module_path = repo_root / "stable_audio_control" / "inference" / "control_demo_callback.py"
    spec = importlib.util.spec_from_file_location("control_demo_callback", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module spec from {module_path}")
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    finally:
        for name, saved_module in saved_modules.items():
            if saved_module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = saved_module
    return module


class ControlDemoCallbackTests(unittest.TestCase):
    def test_default_grid_sweeps_cfg_control_scale_and_variants(self) -> None:
        module = _load_callback_module()
        callback = module.ControlNetDemoCallback(demo_cfg_scales=[3, 6])

        combos = callback.iter_control_demo_combinations()

        self.assertEqual(len(combos), 2 * 5 * 3)
        self.assertEqual(
            combos[:3],
            [
                (3, 0.0, "correct"),
                (3, 0.0, "shuffled"),
                (3, 0.0, "zero"),
            ],
        )
        self.assertIn((6, 1.0, "zero"), combos)

    def test_single_control_scale_remains_backward_compatible(self) -> None:
        module = _load_callback_module()
        callback = module.ControlNetDemoCallback(
            demo_cfg_scales=[3],
            control_scale=0.7,
            control_scales=None,
            control_variants=["correct"],
        )

        self.assertEqual(callback.iter_control_demo_combinations(), [(3, 0.7, "correct")])

    def test_filename_and_logger_tags_include_diagnostic_dimensions(self) -> None:
        module = _load_callback_module()
        callback = module.ControlNetDemoCallback()

        stem = callback.format_demo_stem(cfg_scale=3, control_scale=0.3, variant="shuffled", step=1000)

        self.assertEqual(stem, "demo_cfg_3_control_0p3_shuffled_step_00001000")
        self.assertEqual(
            callback.format_audio_tag(cfg_scale=3, control_scale=0.3, variant="shuffled", step=1000),
            "demo_step_00001000_cfg_3_control_0p3_shuffled",
        )
        self.assertEqual(
            callback.format_melspec_tag(cfg_scale=3, control_scale=0.3, variant="shuffled", step=1000),
            "demo_melspec_step_00001000_cfg_3_control_0p3_shuffled",
        )

    def test_control_variants_prepare_distinct_batch_audio(self) -> None:
        module = _load_callback_module()
        callback = module.ControlNetDemoCallback()
        reals = torch.arange(2 * 1 * 6, dtype=torch.float32).reshape(2, 1, 6)

        correct = callback.make_variant_reals(reals, "correct")
        shuffled = callback.make_variant_reals(reals, "shuffled")
        zero = callback.make_variant_reals(reals, "zero")

        torch.testing.assert_close(correct, reals)
        torch.testing.assert_close(shuffled, torch.roll(reals, shifts=1, dims=0))
        torch.testing.assert_close(zero, torch.zeros_like(reals))
        self.assertIsNot(shuffled, reals)
        self.assertIsNot(zero, reals)

    def test_shuffled_single_item_batch_uses_time_reversal_fallback(self) -> None:
        module = _load_callback_module()
        callback = module.ControlNetDemoCallback()
        reals = torch.arange(6, dtype=torch.float32).reshape(1, 1, 6)

        shuffled = callback.make_variant_reals(reals, "shuffled")

        torch.testing.assert_close(shuffled, torch.flip(reals, dims=[-1]))
        self.assertIsNot(shuffled, reals)

    def test_train_batch_end_recomputes_conditioning_for_each_control_variant(self) -> None:
        module = _load_callback_module()
        sample_calls = []
        audio_tags = []
        image_tags = []
        noise_batch_sizes = []

        def fake_sample(model, noise, steps, eta, **kwargs):
            noise_batch_sizes.append(noise.shape[0])
            sample_calls.append(kwargs)
            return noise + 1.0

        module.sample = fake_sample
        module.sf.write = lambda *args, **kwargs: None
        module.log_audio = lambda logger, tag, filename, sample_rate: audio_tags.append(tag)
        module.log_image = lambda logger, tag, image: image_tags.append(tag)
        module.audio_spectrogram_image = lambda audio: audio

        class MelodyAugmenter:
            def __init__(self) -> None:
                self.calls = []

            def set_batch_audio(self, audio: torch.Tensor) -> None:
                self.calls.append(audio.clone())

        class Diffusion:
            io_channels = 1
            pretransform = None

            def __init__(self) -> None:
                self.conditioner_calls = 0

            def conditioner(self, cond, device):
                self.conditioner_calls += 1
                return {"call": self.conditioner_calls}

        diffusion = Diffusion()
        augmenter = MelodyAugmenter()
        module_under_test = types.SimpleNamespace(
            device=torch.device("cpu"),
            diffusion=diffusion,
            melody_augmenter=augmenter,
            eval=lambda: None,
            train=lambda: None,
        )
        trainer = types.SimpleNamespace(global_step=1, default_root_dir=".", logger=object())
        reals = torch.arange(2 * 1 * 6, dtype=torch.float32).reshape(2, 1, 6)
        metadata = [{"prompt": "a"}, {"prompt": "b"}]
        callback = module.ControlNetDemoCallback(
            demo_every=1,
            num_demos=1,
            sample_size=6,
            demo_steps=2,
            demo_cfg_scales=[3],
            control_scales=[0.0, 0.3],
            control_variants=["correct", "zero"],
        )

        callback.on_train_batch_end(trainer, module_under_test, None, (reals, metadata), 0)

        self.assertEqual(diffusion.conditioner_calls, 4)
        self.assertEqual(len(augmenter.calls), 4)
        demo_reals = reals[:1]
        torch.testing.assert_close(augmenter.calls[0], demo_reals)
        torch.testing.assert_close(augmenter.calls[1], torch.zeros_like(demo_reals))
        self.assertEqual(noise_batch_sizes, [1, 1, 1, 1])
        self.assertEqual([call["control_scale"] for call in sample_calls], [0.0, 0.0, 0.3, 0.3])
        self.assertEqual(
            audio_tags,
            [
                "demo_step_00000001_cfg_3_control_0_correct",
                "demo_step_00000001_cfg_3_control_0_zero",
                "demo_step_00000001_cfg_3_control_0p3_correct",
                "demo_step_00000001_cfg_3_control_0p3_zero",
            ],
        )
        self.assertEqual(
            image_tags,
            [
                "demo_melspec_step_00000001_cfg_3_control_0_correct",
                "demo_melspec_step_00000001_cfg_3_control_0_zero",
                "demo_melspec_step_00000001_cfg_3_control_0p3_correct",
                "demo_melspec_step_00000001_cfg_3_control_0p3_zero",
            ],
        )


if __name__ == "__main__":
    unittest.main()
