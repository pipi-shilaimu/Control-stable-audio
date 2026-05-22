if __name__ == "__main__":
    import sys, json, torch, random
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[0]))
    from stable_audio_control.audio_io import install_torchaudio_load_fallback
    install_torchaudio_load_fallback()
    from stable_audio_tools import get_pretrained_model
    from stable_audio_tools.data.dataset import SampleDataset, LocalDatasetConfig
    from stable_audio_tools.inference.sampling import get_alphas_sigmas
    import importlib.util

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # ORIGINAL base model, NO ControlNet
    base_model, model_config = get_pretrained_model("stabilityai/stable-audio-open-1.0")
    base_model = base_model.to(device).eval()

    sample_rate = int(model_config["sample_rate"])
    sample_size = int(model_config["sample_size"])

    # Load ONE audio sample
    cfg = json.loads(Path("stable_audio_control/data/song_describer/dataset_config_train.json").read_text())
    audio_dir_cfg = cfg["datasets"][0]
    spec = importlib.util.spec_from_file_location("md_module", audio_dir_cfg["custom_metadata_module"])
    md_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(md_module)

    ds = SampleDataset(
        [LocalDatasetConfig(audio_dir_cfg["id"], audio_dir_cfg["path"], md_module.get_custom_metadata)],
        sample_size=sample_size, sample_rate=sample_rate,
        random_crop=False, force_channels="stereo",
    )
    idx = random.randrange(len(ds))
    reals, info = ds[idx]
    reals = reals.unsqueeze(0).to(device)  # [1,2,T]
    prompt = info.get("prompt", "N/A")
    print(f"Audio: {reals.shape}, Prompt: {prompt[:60]}...")

    # Fix metadata for conditioner
    info["padding_mask"] = [info["padding_mask"]]

    # Encode
    with torch.no_grad():
        if base_model.pretransform is not None:
            base_model.pretransform.to(device)
            latent = base_model.pretransform.encode(reals)
        else:
            latent = reals
    print(f"Latent: {latent.shape}")

    # Conditioning
    conditioner = base_model.conditioner.to(device)
    with torch.no_grad():
        cond = conditioner([info], device)

    # Test at FIVE timesteps
    print("t        | MSE_with_cond | MSE_uncond | zero_baseline")
    print("---------|---------------|------------|--------------")
    for t_val in [0.1, 0.25, 0.5, 0.75, 0.9]:
        t = torch.full((1,), t_val, device=device)
        alphas, sigmas = get_alphas_sigmas(t)
        alphas = alphas[:, None, None]
        sigmas = sigmas[:, None, None]
        noise = torch.randn_like(latent)
        noised = latent * alphas + noise * sigmas
        targets = noise * alphas - latent * sigmas

        with torch.no_grad(), torch.cuda.amp.autocast():
            out_cond = base_model(noised, t, cond=cond)
            # Unconditional: zero out cross-attention cond
            cond_uncond = dict(cond)
            for k in ["prompt", "seconds_start", "seconds_total"]:
                if k in cond_uncond:
                    v = cond_uncond[k]
                    if isinstance(v, (list, tuple)):
                        cond_uncond[k] = [torch.zeros_like(v[0]), v[1]]
            out_uncond = base_model(noised, t, cond=cond_uncond)

        mse_cond = torch.nn.functional.mse_loss(out_cond, targets).item()
        mse_uncond = torch.nn.functional.mse_loss(out_uncond, targets).item()
        zero_mse = torch.nn.functional.mse_loss(torch.zeros_like(targets), targets).item()
        print(f"{t_val:.1f}     | {mse_cond:.4f}        | {mse_uncond:.4f}     | {zero_mse:.4f}")
