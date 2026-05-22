if __name__ == "__main__":
    import sys, json, torch, random
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[0]))
    from stable_audio_control.audio_io import install_torchaudio_load_fallback
    install_torchaudio_load_fallback()
    from stable_audio_tools import get_pretrained_model
    from stable_audio_tools.data.dataset import SampleDataset, LocalDatasetConfig
    from stable_audio_control.models import build_control_wrapper
    from stable_audio_tools.inference.sampling import get_alphas_sigmas
    import importlib.util

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # Load model
    base_model, model_config = get_pretrained_model("stabilityai/stable-audio-open-1.0")
    sample_rate = int(model_config["sample_rate"])
    sample_size = int(model_config["sample_size"])

    control_model = build_control_wrapper(
        base_wrapper=base_model, num_control_layers=12, control_id="melody_control",
        default_control_scale=1.0, freeze_base=True,
        melody_channels=8, melody_num_pitch_bins=128,
        melody_embedding_dim=64, melody_hidden_dim=256,
        melody_conv_layers=2, use_melody_encoder=True,
    )
    dtype = next(control_model.parameters()).dtype
    control_model.to(device)
    dummy = torch.zeros(1, control_model.control_dim_in, device=device, dtype=dtype)
    _ = control_model.control_projector(dummy)
    print("Lazy params initialized.")
    control_model.eval()

    trainable = sum(p.numel() for p in control_model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in control_model.parameters())
    print(f"Trainable params: {trainable:,} / {total:,}")

    # Build dataset directly (no DataLoader, avoid spawn issues)
    cfg = json.loads(Path("stable_audio_control/data/song_describer/dataset_config_train.json").read_text())
    audio_dir_cfg = cfg["datasets"][0]
    spec = importlib.util.spec_from_file_location("md_module", audio_dir_cfg["custom_metadata_module"])
    md_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(md_module)

    ds = SampleDataset(
        [LocalDatasetConfig(audio_dir_cfg["id"], audio_dir_cfg["path"], md_module.get_custom_metadata)],
        sample_size=sample_size, sample_rate=sample_rate,
        random_crop=cfg.get("random_crop", True), force_channels="stereo",
    )
    print(f"Dataset size: {len(ds)}")

    # Get one sample
    idx = random.randrange(len(ds))
    reals, info = ds[idx]
    reals = reals.unsqueeze(0)  # [C,T] -> [1,C,T]
    metadata = [info]

    prompt = metadata[0].get("prompt", "N/A")
    print(f"Real audio shape: {reals.shape}")
    print(f"Prompt: {prompt[:80]}...")
    reals = reals.to(device)

    # Encode
    with torch.no_grad():
        if control_model.pretransform is not None:
            control_model.pretransform.to(device)
            diffusion_input = control_model.pretransform.encode(reals)
        else:
            diffusion_input = reals
    print(f"Latent shape (after VAE): {diffusion_input.shape}")

    # Conditioning
    conditioner = control_model.conditioner
    conditioner.to(device)

    # MelodyControlAugmenter wraps the conditioner; it expects set_batch_audio
    if hasattr(conditioner, "set_batch_audio"):
        conditioner.set_batch_audio(reals)

    with torch.no_grad():
        conditioning = conditioner(metadata, device)
    print(f"Conditioning keys: {list(conditioning.keys())}")
    for k, v in conditioning.items():
        if isinstance(v, (list, tuple)) and torch.is_tensor(v[0]):
            t = v[0]
            print(f"  {k}: shape={tuple(t.shape)}, mean={t.mean().item():.3f}, std={t.std().item():.3f}")

    # One forward pass at t=0.5
    t = torch.full((diffusion_input.shape[0],), 0.5, device=device)
    alphas, sigmas = get_alphas_sigmas(t)
    alphas = alphas[:, None, None]
    sigmas = sigmas[:, None, None]
    noise = torch.randn_like(diffusion_input)
    noised_inputs = diffusion_input * alphas + noise * sigmas
    targets = noise * alphas - diffusion_input * sigmas
    print(f"targets std: {targets.std().item():.4f}")

    with torch.no_grad(), torch.cuda.amp.autocast():
        output = control_model(noised_inputs, t, cond=conditioning, cfg_dropout_prob=0.0)

    mse = torch.nn.functional.mse_loss(output, targets)
    zero_mse = torch.nn.functional.mse_loss(torch.zeros_like(output), targets)
    print(f"Frozen backbone MSE: {mse.item():.4f}")
    print(f"Zero-output baseline: {zero_mse.item():.4f}")
    print(f"output std: {output.std().item():.4f}")

    if mse.item() < 0.5:
        print(">>> Backbone is fine. Problem is in ControlNet training dynamics.")
    else:
        print(">>> Backbone is bad. Testing original base model...")
        base_model = base_model.to(device).eval()
        with torch.no_grad(), torch.cuda.amp.autocast():
            output_base = base_model(noised_inputs, t, cond=conditioning)
        mse_base = torch.nn.functional.mse_loss(output_base, targets)
        print(f"Original base model MSE: {mse_base.item():.4f}")
        if mse_base < 0.5:
            print(">>> Original base model works! ControlNet wrapper degrades backbone.")
        else:
            print(">>> Even original base model fails. Conditioning/data issue.")
