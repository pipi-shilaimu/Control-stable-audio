from __future__ import annotations

import csv
import importlib.util
import sys
import types
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

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
    torchaudio.load = lambda path: (_ for _ in ()).throw(RuntimeError(f"Unexpected torchaudio.load({path})"))
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
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
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

        self.assertEqual(len(combos), 2 * 5 * 4)
        self.assertEqual(
            combos[:4],
            [
                (3, 0.0, "correct"),
                (3, 0.0, "shuffled"),
                (3, 0.0, "zero"),
                (3, 0.0, "null"),
            ],
        )
        self.assertIn((6, 1.0, "zero"), combos)
        self.assertIn((6, 1.0, "null"), combos)

    def test_single_control_scale_remains_backward_compatible(self) -> None:
        module = _load_callback_module()
        callback = module.ControlNetDemoCallback(
            demo_cfg_scales=[3],
            control_scale=0.7,
            control_scales=None,
            control_variants=["correct"],
        )

        self.assertEqual(callback.iter_control_demo_combinations(), [(3, 0.7, "correct")])

    def test_default_demo_count_is_one(self) -> None:
        module = _load_callback_module()
        callback = module.ControlNetDemoCallback()

        self.assertEqual(callback.num_demos, 1)

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
        null = callback.make_variant_reals(reals, "null")

        torch.testing.assert_close(correct, reals)
        torch.testing.assert_close(shuffled, torch.roll(reals, shifts=1, dims=0))
        torch.testing.assert_close(zero, torch.zeros_like(reals))
        self.assertIsNone(null)
        self.assertIsNot(shuffled, reals)
        self.assertIsNot(zero, reals)

    def test_normalizes_disabled_control_variant_aliases_to_null(self) -> None:
        module = _load_callback_module()

        self.assertEqual(module.normalize_control_variant("disabled"), "null")
        self.assertEqual(module.normalize_control_variant("none"), "null")

    def test_audio_spectral_stats_report_low_mid_bias(self) -> None:
        module = _load_callback_module()
        sample_rate = 8000
        t = torch.arange(sample_rate, dtype=torch.float32) / sample_rate
        low_tone = torch.sin(2 * torch.pi * 100 * t).reshape(1, -1)
        high_tone = torch.sin(2 * torch.pi * 3000 * t).reshape(1, -1)

        low_stats = module.compute_audio_spectral_stats(low_tone, sample_rate=sample_rate)
        high_stats = module.compute_audio_spectral_stats(high_tone, sample_rate=sample_rate)

        self.assertGreater(low_stats["low_band_ratio"], 0.8)
        self.assertGreater(low_stats["low_mid_band_ratio"], high_stats["low_mid_band_ratio"])
        self.assertGreater(high_stats["high_band_ratio"], 0.8)

    def test_normalizes_silent_demo_audio_to_zero_int16_without_nan(self) -> None:
        module = _load_callback_module()
        silent = torch.zeros(2, 16, dtype=torch.float32)

        wav = module.normalize_audio_for_wav(silent)

        self.assertEqual(wav.dtype, torch.int16)
        self.assertEqual(tuple(wav.shape), (2, 16))
        self.assertTrue(torch.equal(wav, torch.zeros_like(wav)))

    def test_pairwise_delta_stats_detect_identical_correct_and_shuffled(self) -> None:
        module = _load_callback_module()
        correct = torch.tensor([1.0, 1.0, 0.0])
        shuffled = torch.tensor([1.0, 1.0, 0.0])

        stats = module.compute_pairwise_delta_stats(correct, shuffled)

        self.assertAlmostEqual(stats["cosine"], 1.0, places=6)
        self.assertAlmostEqual(stats["relative_l2_difference"], 0.0, places=6)

    def test_shuffled_single_item_batch_uses_time_reversal_fallback(self) -> None:
        module = _load_callback_module()
        callback = module.ControlNetDemoCallback()
        reals = torch.arange(6, dtype=torch.float32).reshape(1, 1, 6)

        shuffled = callback.make_variant_reals(reals, "shuffled")

        torch.testing.assert_close(shuffled, torch.flip(reals, dims=[-1]))
        self.assertIsNot(shuffled, reals)

    def test_prepares_external_demo_control_audio_as_stereo_padded_batch(self) -> None:
        module = _load_callback_module()
        audio = torch.tensor([[1.0, -1.0, 0.5]], dtype=torch.float32)

        prepared = module.prepare_demo_control_audio(
            audio,
            source_sample_rate=44_100,
            target_sample_rate=44_100,
            target_sample_size=5,
            device=torch.device("cpu"),
        )

        self.assertEqual(tuple(prepared.shape), (1, 2, 5))
        torch.testing.assert_close(prepared[0, 0], torch.tensor([1.0, -1.0, 0.5, 0.0, 0.0]))
        torch.testing.assert_close(prepared[0, 1], torch.tensor([1.0, -1.0, 0.5, 0.0, 0.0]))

    def test_train_batch_end_uses_external_control_audio_for_correct_variant(self) -> None:
        module = _load_callback_module()
        loaded_audio = torch.tensor([[0.25, 0.5, 0.75, 1.0]], dtype=torch.float32)
        module.torchaudio.load = lambda path: (loaded_audio, 44_100)
        module.sample = lambda model, noise, steps, eta, **kwargs: noise + 1.0
        module.sf.write = lambda *args, **kwargs: None
        module.log_audio = lambda *args, **kwargs: None
        module.log_image = lambda *args, **kwargs: None
        module.audio_spectrogram_image = lambda audio: audio

        class MelodyAugmenter:
            def __init__(self) -> None:
                self.calls = []

            def set_batch_audio(self, audio: torch.Tensor) -> None:
                self.calls.append(audio.clone())

        class Diffusion:
            io_channels = 1
            pretransform = None

            def conditioner(self, cond, device):
                return {"ok": True}

        augmenter = MelodyAugmenter()
        module_under_test = types.SimpleNamespace(
            device=torch.device("cpu"),
            diffusion=Diffusion(),
            melody_augmenter=augmenter,
            eval=lambda: None,
            train=lambda: None,
        )
        trainer = types.SimpleNamespace(global_step=1, default_root_dir=".", logger=object())
        batch_reals = torch.full((2, 1, 4), 9.0)
        metadata = [{"prompt": "a"}, {"prompt": "b"}]
        callback = module.ControlNetDemoCallback(
            demo_every=1,
            num_demos=2,
            sample_size=4,
            sample_rate=44_100,
            demo_cfg_scales=[3],
            control_scales=[1.0],
            control_variants=["correct"],
            demo_control_audio_path="melody.wav",
            demo_melody_similarity=False,
        )

        callback.on_train_batch_end(trainer, module_under_test, None, (batch_reals, metadata), 0)

        expected_single = torch.tensor(
            [[[0.25, 0.5, 0.75, 1.0], [0.25, 0.5, 0.75, 1.0]]],
            dtype=torch.float32,
        )
        torch.testing.assert_close(augmenter.calls[0], expected_single.repeat(2, 1, 1))

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
            control_variants=["correct", "zero", "null"],
            demo_melody_similarity=False,
        )

        callback.on_train_batch_end(trainer, module_under_test, None, (reals, metadata), 0)

        self.assertEqual(diffusion.conditioner_calls, 6)
        self.assertEqual(len(augmenter.calls), 4)
        demo_reals = reals[:1]
        torch.testing.assert_close(augmenter.calls[0], demo_reals)
        torch.testing.assert_close(augmenter.calls[1], torch.zeros_like(demo_reals))
        torch.testing.assert_close(augmenter.calls[2], demo_reals)
        torch.testing.assert_close(augmenter.calls[3], torch.zeros_like(demo_reals))
        self.assertEqual(noise_batch_sizes, [1, 1, 1, 1, 1, 1])
        self.assertEqual([call["control_scale"] for call in sample_calls], [0.0, 0.0, 0.0, 0.3, 0.3, 0.3])
        self.assertEqual(
            audio_tags,
            [
                "demo_step_00000001_cfg_3_control_0_correct",
                "demo_step_00000001_cfg_3_control_0_zero",
                "demo_step_00000001_cfg_3_control_0_null",
                "demo_step_00000001_cfg_3_control_0p3_correct",
                "demo_step_00000001_cfg_3_control_0p3_zero",
                "demo_step_00000001_cfg_3_control_0p3_null",
            ],
        )
        self.assertEqual(
            image_tags,
            [
                "demo_melspec_step_00000001_cfg_3_control_0_correct",
                "demo_melspec_step_00000001_cfg_3_control_0_zero",
                "demo_melspec_step_00000001_cfg_3_control_0_null",
                "demo_melspec_step_00000001_cfg_3_control_0p3_correct",
                "demo_melspec_step_00000001_cfg_3_control_0p3_zero",
                "demo_melspec_step_00000001_cfg_3_control_0p3_null",
            ],
        )

    def test_train_batch_end_writes_one_pairwise_collapse_row_per_control_scale(self) -> None:
        module = _load_callback_module()
        module.sample = lambda model, noise, steps, eta, **kwargs: noise + 1.0
        module.sf.write = lambda *args, **kwargs: None
        module.log_audio = lambda *args, **kwargs: None
        module.log_image = lambda *args, **kwargs: None
        module.audio_spectrogram_image = lambda audio: audio

        class DemoModel(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.weight = torch.nn.Parameter(torch.ones(()))

            def forward(self, x, t, **kwargs):
                control_input = kwargs.get("control_input")
                control_scale = float(kwargs.get("control_scale", 0.0))
                if control_input is None:
                    return x
                return x + control_input.to(dtype=x.dtype) * control_scale

        class MelodyAugmenter:
            def __init__(self) -> None:
                self.last_audio = None

            def set_batch_audio(self, audio: torch.Tensor) -> None:
                self.last_audio = audio.clone()

        class Diffusion:
            io_channels = 1
            pretransform = None

            def __init__(self, augmenter: MelodyAugmenter) -> None:
                self.model = DemoModel()
                self.augmenter = augmenter

            def conditioner(self, cond, device):
                if self.augmenter.last_audio is None:
                    return {"text": True}
                return {"melody_control": [self.augmenter.last_audio.mean(dim=1, keepdim=True), None]}

            def _extract_control_input(self, *, cond, target_len, dtype, device):
                if "melody_control" not in cond:
                    return None
                return torch.ones((1, 1, target_len), device=device, dtype=dtype)

        augmenter = MelodyAugmenter()
        diffusion = Diffusion(augmenter)
        module_under_test = types.SimpleNamespace(
            device=torch.device("cpu"),
            diffusion=diffusion,
            melody_augmenter=augmenter,
            eval=lambda: None,
            train=lambda: None,
        )
        reals = torch.arange(1 * 1 * 6, dtype=torch.float32).reshape(1, 1, 6)
        metadata = [{"prompt": "a"}]
        callback = module.ControlNetDemoCallback(
            demo_every=1,
            num_demos=1,
            sample_size=6,
            demo_steps=2,
            demo_cfg_scales=[3],
            control_scales=[0.3],
            control_variants=["correct", "shuffled", "zero", "null"],
            demo_melody_similarity=False,
            demo_control_diagnostics=True,
            demo_control_diagnostics_csv="diag.csv",
            stop_on_collapse=True,
            collapse_cosine_threshold=0.98,
        )

        with TemporaryDirectory() as tmp:
            trainer = types.SimpleNamespace(global_step=1, default_root_dir=tmp, logger=object(), should_stop=False)
            callback.on_train_batch_end(trainer, module_under_test, None, (reals, metadata), 0)
            with (Path(tmp) / "diag.csv").open("r", encoding="utf-8", newline="") as fp:
                rows = list(csv.DictReader(fp))

        pairwise_rows = [row for row in rows if row["row_type"] == "pairwise"]
        self.assertEqual(len(pairwise_rows), 1)
        self.assertEqual(pairwise_rows[0]["variant"], "correct_vs_shuffled")
        self.assertEqual(pairwise_rows[0]["variant_a"], "correct")
        self.assertEqual(pairwise_rows[0]["variant_b"], "shuffled")
        self.assertIn("correct/shuffled forward_delta collapse", pairwise_rows[0]["warning"])
        self.assertTrue(trainer.should_stop)

    def test_train_batch_end_writes_each_demo_without_concatenating_batch_time(self) -> None:
        module = _load_callback_module()
        written = []

        module.sample = lambda model, noise, steps, eta, **kwargs: noise + 1.0
        module.sf.write = lambda filename, audio, sample_rate: written.append((filename, audio.shape, sample_rate))
        module.log_audio = lambda *args, **kwargs: None
        module.log_image = lambda *args, **kwargs: None
        module.audio_spectrogram_image = lambda audio: audio

        class MelodyAugmenter:
            def set_batch_audio(self, audio: torch.Tensor) -> None:
                pass

        class Diffusion:
            io_channels = 1
            pretransform = None

            def conditioner(self, cond, device):
                return {"ok": True}

        module_under_test = types.SimpleNamespace(
            device=torch.device("cpu"),
            diffusion=Diffusion(),
            melody_augmenter=MelodyAugmenter(),
            eval=lambda: None,
            train=lambda: None,
        )
        trainer = types.SimpleNamespace(global_step=1, default_root_dir=".", logger=object())
        reals = torch.arange(2 * 1 * 6, dtype=torch.float32).reshape(2, 1, 6)
        metadata = [{"prompt": "a"}, {"prompt": "b"}]
        callback = module.ControlNetDemoCallback(
            demo_every=1,
            num_demos=2,
            sample_size=6,
            demo_steps=2,
            demo_cfg_scales=[3],
            control_scales=[1.0],
            control_variants=["correct"],
            demo_melody_similarity=False,
        )

        callback.on_train_batch_end(trainer, module_under_test, None, (reals, metadata), 0)

        self.assertEqual(len(written), 2)
        self.assertEqual([shape for _, shape, _ in written], [(6, 1), (6, 1)])
        self.assertTrue(written[0][0].endswith("_demo_00.wav"))
        self.assertTrue(written[1][0].endswith("_demo_01.wav"))

    def test_train_batch_end_overrides_demo_prompt_without_mutating_batch_metadata(self) -> None:
        module = _load_callback_module()
        seen_prompts = []

        module.sample = lambda model, noise, steps, eta, **kwargs: noise + 1.0
        module.sf.write = lambda *args, **kwargs: None
        module.log_audio = lambda *args, **kwargs: None
        module.log_image = lambda *args, **kwargs: None
        module.audio_spectrogram_image = lambda audio: audio

        class MelodyAugmenter:
            def set_batch_audio(self, audio: torch.Tensor) -> None:
                pass

        class Diffusion:
            io_channels = 1
            pretransform = None

            def conditioner(self, cond, device):
                seen_prompts.append([item["prompt"] for item in cond])
                return {"ok": True}

        module_under_test = types.SimpleNamespace(
            device=torch.device("cpu"),
            diffusion=Diffusion(),
            melody_augmenter=MelodyAugmenter(),
            eval=lambda: None,
            train=lambda: None,
        )
        trainer = types.SimpleNamespace(global_step=1, default_root_dir=".", logger=object())
        reals = torch.arange(2 * 1 * 6, dtype=torch.float32).reshape(2, 1, 6)
        metadata = [{"prompt": "batch prompt a"}, {"prompt": "batch prompt b"}]
        callback = module.ControlNetDemoCallback(
            demo_every=1,
            num_demos=2,
            sample_size=6,
            demo_steps=2,
            demo_cfg_scales=[3],
            control_scales=[1.0],
            control_variants=["correct"],
            demo_prompt="fixed diagnostic prompt",
            demo_melody_similarity=False,
        )

        callback.on_train_batch_end(trainer, module_under_test, None, (reals, metadata), 0)

        self.assertEqual(seen_prompts, [["fixed diagnostic prompt", "fixed diagnostic prompt"]])
        self.assertEqual(metadata, [{"prompt": "batch prompt a"}, {"prompt": "batch prompt b"}])

    def test_train_batch_end_scores_demo_melody_similarity_to_csv_and_logger(self) -> None:
        module = _load_callback_module()
        compare_calls = []
        logged_metrics = []

        def fake_compare_audio_tensors_melody_similarity(reference_audio, generated_audio, **kwargs):
            compare_calls.append((reference_audio.clone(), generated_audio.clone(), kwargs))
            return {
                "similarity": {
                    "metric_name": "cqt_topk_pitch_overlap_rate",
                    "score": 0.5,
                    "matched_tokens": 2,
                    "total_tokens": 4,
                    "compared_frames": 3,
                    "additional_metrics": {
                        "cqt_top1_accuracy": {
                            "metric_name": "cqt_top1_pitch_accuracy",
                            "score": 0.75,
                            "matched_tokens": 3,
                            "total_tokens": 4,
                            "compared_frames": 3,
                        }
                    },
                }
            }

        module.compare_audio_tensors_melody_similarity = fake_compare_audio_tensors_melody_similarity
        module.sample = lambda model, noise, steps, eta, **kwargs: noise + 1.0
        module.sf.write = lambda *args, **kwargs: None
        module.log_audio = lambda *args, **kwargs: None
        module.log_image = lambda *args, **kwargs: None
        module.audio_spectrogram_image = lambda audio: audio

        class MelodyAugmenter:
            extractor = object()

            def set_batch_audio(self, audio: torch.Tensor) -> None:
                pass

        class Diffusion:
            io_channels = 1
            pretransform = None

            def conditioner(self, cond, device):
                return {"ok": True}

        class Logger:
            def log_metrics(self, metrics, step=None):
                logged_metrics.append((metrics, step))

        module_under_test = types.SimpleNamespace(
            device=torch.device("cpu"),
            diffusion=Diffusion(),
            melody_augmenter=MelodyAugmenter(),
            eval=lambda: None,
            train=lambda: None,
        )
        reals = torch.arange(1 * 1 * 6, dtype=torch.float32).reshape(1, 1, 6)
        metadata = [{"prompt": "batch prompt"}]
        callback = module.ControlNetDemoCallback(
            demo_every=1,
            num_demos=1,
            sample_size=6,
            sample_rate=44_100,
            demo_cfg_scales=[3],
            control_scales=[0.3],
            control_variants=["correct"],
            demo_melody_similarity=True,
            demo_melody_similarity_csv="metrics.csv",
            demo_melody_similarity_feature="cqt",
            demo_melody_similarity_top_k=1,
        )

        with TemporaryDirectory() as tmp:
            trainer = types.SimpleNamespace(global_step=1, default_root_dir=tmp, logger=Logger())
            callback.on_train_batch_end(trainer, module_under_test, None, (reals, metadata), 0)
            with (Path(tmp) / "metrics.csv").open("r", encoding="utf-8", newline="") as fp:
                rows = list(csv.DictReader(fp))

        self.assertEqual(len(compare_calls), 2)
        self.assertEqual(compare_calls[0][2]["feature"], "cqt")
        self.assertEqual(compare_calls[0][2]["top_k"], 1)
        self.assertEqual(compare_calls[0][2]["sample_size"], 6)
        self.assertEqual(compare_calls[1][2]["feature"], "cqt")
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["step"], "1")
        self.assertEqual(rows[0]["variant"], "correct")
        self.assertEqual(rows[0]["control_scale"], "0.3")
        self.assertEqual(rows[0]["metric_name"], "cqt_topk_pitch_overlap_rate")
        self.assertEqual(rows[0]["cqt_top1_score"], "0.75")
        self.assertEqual(rows[0]["cqt_topk_score"], "0.5")
        self.assertEqual(rows[1]["metric_name"], "original_ref_cqt_topk_pitch_overlap_rate")
        self.assertEqual(rows[1]["cqt_top1_score"], "0.75")
        self.assertEqual(rows[1]["cqt_topk_score"], "0.5")
        self.assertEqual(logged_metrics[0][1], 1)
        self.assertIn(
            "demo_melody_similarity/cfg_3_control_0p3_correct/top1",
            logged_metrics[0][0],
        )
        self.assertIn(
            "demo_melody_similarity/cfg_3_control_0p3_correct/original_ref_top1",
            logged_metrics[1][0],
        )

    def test_train_batch_end_scores_original_reference_for_shuffled_and_zero_variants(self) -> None:
        module = _load_callback_module()
        compare_calls = []

        def fake_compare_audio_tensors_melody_similarity(reference_audio, generated_audio, **kwargs):
            compare_calls.append((reference_audio.clone(), generated_audio.clone(), kwargs))
            score = 0.1 * len(compare_calls)
            return {
                "similarity": {
                    "metric_name": "cqt_topk_pitch_overlap_rate",
                    "score": score,
                    "matched_tokens": len(compare_calls),
                    "total_tokens": 10,
                    "compared_frames": 5,
                    "additional_metrics": {
                        "cqt_top1_accuracy": {
                            "metric_name": "cqt_top1_pitch_accuracy",
                            "score": score + 0.01,
                            "matched_tokens": len(compare_calls),
                            "total_tokens": 10,
                            "compared_frames": 5,
                        }
                    },
                }
            }

        module.compare_audio_tensors_melody_similarity = fake_compare_audio_tensors_melody_similarity
        module.sample = lambda model, noise, steps, eta, **kwargs: noise + 1.0
        module.sf.write = lambda *args, **kwargs: None
        module.log_audio = lambda *args, **kwargs: None
        module.log_image = lambda *args, **kwargs: None
        module.audio_spectrogram_image = lambda audio: audio

        class MelodyAugmenter:
            extractor = object()

            def set_batch_audio(self, audio: torch.Tensor) -> None:
                pass

        class Diffusion:
            io_channels = 1
            pretransform = None

            def conditioner(self, cond, device):
                return {"ok": True}

        module_under_test = types.SimpleNamespace(
            device=torch.device("cpu"),
            diffusion=Diffusion(),
            melody_augmenter=MelodyAugmenter(),
            eval=lambda: None,
            train=lambda: None,
        )
        reals = torch.arange(1 * 1 * 6, dtype=torch.float32).reshape(1, 1, 6)
        metadata = [{"prompt": "batch prompt"}]
        callback = module.ControlNetDemoCallback(
            demo_every=1,
            num_demos=1,
            sample_size=6,
            sample_rate=44_100,
            demo_cfg_scales=[3],
            control_scales=[1.0],
            control_variants=["shuffled", "zero"],
            demo_melody_similarity=True,
            demo_melody_similarity_csv="metrics.csv",
            demo_melody_similarity_feature="cqt",
            demo_melody_similarity_top_k=1,
        )

        with TemporaryDirectory() as tmp:
            trainer = types.SimpleNamespace(global_step=1, default_root_dir=tmp, logger=object())
            callback.on_train_batch_end(trainer, module_under_test, None, (reals, metadata), 0)
            with (Path(tmp) / "metrics.csv").open("r", encoding="utf-8", newline="") as fp:
                rows = list(csv.DictReader(fp))

        self.assertEqual(len(compare_calls), 3)
        torch.testing.assert_close(compare_calls[0][0], torch.flip(reals, dims=[-1]))
        torch.testing.assert_close(compare_calls[1][0], reals)
        torch.testing.assert_close(compare_calls[2][0], reals)
        self.assertEqual(
            [(row["variant"], row["metric_name"], row["skipped_reason"]) for row in rows],
            [
                ("shuffled", "cqt_topk_pitch_overlap_rate", ""),
                ("shuffled", "original_ref_cqt_topk_pitch_overlap_rate", ""),
                ("zero", "", "zero_control_has_no_reference_melody"),
                ("zero", "original_ref_cqt_topk_pitch_overlap_rate", ""),
            ],
        )
        self.assertEqual(rows[1]["cqt_topk_score"], "0.2")
        self.assertEqual(rows[3]["cqt_top1_score"], "0.31000000000000005")


if __name__ == "__main__":
    unittest.main()
