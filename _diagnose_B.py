if __name__ == "__main__":
    """Trace the conditioning signal through the entire pipeline."""
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
    base_model, model_config = get_pretrained_model("stabilityai/stable-audio-open-1.0")
    base_model = base_model.to(device).eval()

    sample_rate = int(model_config["sample_rate"])
    sample_size = int(model_config["sample_size"])

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
    reals, info = ds[0]
    reals = reals.unsqueeze(0).to(device)
    info["padding_mask"] = [info["padding_mask"]]
    print(f"Audio: {reals.shape}")

    # Encode
    with torch.no_grad():
        base_model.pretransform.to(device)
        latent = base_model.pretransform.encode(reals)
    print(f"Latent: {latent.shape}")

    # 1. T5 conditioner raw output
    print("n=== STEP 1: Control conditioner raw outputs ===")
    conditioner = base_model.conditioner.to(device)
    with torch.no_grad():
        cond = conditioner([info], device)
    for k, v in cond.items():
        if isinstance(v, (list, tuple)) and torch.is_tensor(v[0]):
            t = v[0]
            print(f"  {k}: shape={tuple(t.shape)} mean={t.mean():.4f} std={t.std():.4f} min={t.min():.4f} max={t.max():.4f}")

    # 2. What get_conditioning_inputs produces
    print("n=== STEP 2: get_conditioning_inputs outputs ===")
    cond_inputs = base_model.get_conditioning_inputs(cond)
    for k, v in cond_inputs.items():
        if torch.is_tensor(v):
            print(f"  {k}: shape={tuple(v.shape)} mean={v.mean():.4f} std={v.std():.4f}")
        elif v is None:
            print(f"  {k}: None")

    # 3. What DiTWrapper.to_cond_embed outputs
    print("n=== STEP 3: to_cond_embed projection ===")
    cross_attn_cond = cond_inputs["cross_attn_cond"]
    if cross_attn_cond is not None:
        projected = base_model.model.to_cond_embed(cross_attn_cond)
        print(f"  cross_attn_cond projected: shape={tuple(projected.shape)} mean={projected.mean():.4f} std={projected.std():.4f}")
    
    global_cond = cond_inputs["global_cond"]
    if global_cond is not None:
        projected_g = base_model.model.to_global_embed(global_cond)
        print(f"  global_cond projected: shape={tuple(projected_g.shape)} mean={projected_g.mean():.4f} std={projected_g.std():.4f}")

    # 4. EXACT same test but going through the encoder wrapper
    print("n=== STEP 4: Full forward (conditioned vs unconditioned) at t=0.5 ===")
    t = torch.full((1,), 0.5, device=device)
    alphas, sigmas = get_alphas_sigmas(t)
    alphas = alphas[:, None, None]
    sigmas = sigmas[:, None, None]
    noise = torch.randn_like(latent)
    noised = latent * alphas + noise * sigmas
    targets = noise * alphas - latent * sigmas

    # Conditioned
    with torch.no_grad(), torch.cuda.amp.autocast():
        out_cond = base_model(noised, t, cond=cond)

    # Unconditioned (all zeros)
    cond_zero = {}
    for k, v in cond.items():
        if isinstance(v, (list, tuple)):
            cond_zero[k] = [torch.zeros_like(v[0]), v[1] if torch.is_tensor(v[1]) else None]
        else:
            cond_zero[k] = v
    with torch.no_grad(), torch.cuda.amp.autocast():
        out_uncond = base_model(noised, t, cond=cond_zero)

    mse_cond = torch.nn.functional.mse_loss(out_cond, targets).item()
    mse_uncond = torch.nn.functional.mse_loss(out_uncond, targets).item()
    mse_zero = torch.nn.functional.mse_loss(torch.zeros_like(targets), targets).item()
    print(f"  Conditioned MSE:     {mse_cond:.4f}")
    print(f"  Unconditioned MSE:   {mse_uncond:.4f}")
    print(f"  Zero-output baseline: {mse_zero:.4f}")
    print(f"  Output diff (cond vs uncond): {(out_cond - out_uncond).abs().mean().item():.6f}")

    # 5. Verify: are the DiT model weights actually loaded correctly?
    print("n=== STEP 5: Check if to_cond_embed has pretrained weights or random init ===")
    embed = base_model.model.to_cond_embed
    if hasattr(embed, "weight") or hasattr(embed, "parameters"):
        for name, p in embed.named_parameters():
            print(f"  to_cond_embed.{name}: mean={p.mean():.6f} std={p.std():.6f}")
            break  # Just first param
    
    # Also check final_cross_attn_ix
    print(f"  final_cross_attn_ix (in DiT): {base_model.model.model.final_cross_attn_ix}")
    print(f"  cross_attend (layers have cross-attn): {base_model.model.model.transformer.layers[0].cross_attend}")
