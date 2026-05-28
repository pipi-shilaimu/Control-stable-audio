from transformers import AutoModel, PreTrainedModel

# ── 兼容性补丁 ──────────────────────────────────────────
# transformers v5 新增了 all_tied_weights_keys 属性要求。
# oriyonay/musicnn-pytorch 的远程代码是为旧版 transformers 写的，
# 没有定义该属性。这里在基类上补上默认值。
if "all_tied_weights_keys" not in PreTrainedModel.__dict__:
    PreTrainedModel.all_tied_weights_keys = {}
# ──────────────────────────────────────────────────────

# Load the model (downloads automatically)
model = AutoModel.from_pretrained("oriyonay/musicnn-pytorch", trust_remote_code=True)

# Use the model
tags = model.predict_tags(r"C:\PROJECT\StableAudio\stable_audio_control\tool\musicnn\000000_track_id-1004034_caption_id-859.wav", top_k=10)
print(f"Top 5 tags: {tags}")