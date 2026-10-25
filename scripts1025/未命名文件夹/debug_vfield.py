#!/usr/bin/env python
"""对比训练路径 (model.forward) 与采样路径 (denoise_step+KV cache) 的速度场 v_t。
两者若不一致 → 推理采样循环有 bug; 一致 → 模型高 t 区速度场本来就弱。"""

import io
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "lerobot" / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts1025"))

from debug_offline_train import decode_frame, download_train_meta, load_episode  # noqa: E402
from demo_live_drag import load_policy  # noqa: E402
from lerobot.policies.smolvla.modeling_smolvla import make_att_2d_masks  # noqa: E402
from lerobot.utils.constants import OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS  # noqa: E402

ds_root = download_train_meta("h-siyuan/ScreenDrag")
ep = load_episode(ds_root, 5)
gt = np.stack(ep["action"].values)
img_t = decode_frame(ep.iloc[0]["observation.images.screen"])
policy, pre, post = load_policy("models/showui-pi", ds_root, "mps")

actions = gt.copy()
actions[:, 1] /= 1920
actions[:, 2] /= 1080
act_t = torch.tensor(actions[None], dtype=torch.float32, device="mps")
batch = {"observation.images.screen": img_t,
         "observation.state": torch.tensor([[0.0, -1.0, -1.0]]),
         "task": "Solve the drag-and-drop captcha."}
mb = pre(batch)
lang_tokens, lang_masks = mb[OBS_LANGUAGE_TOKENS], mb[OBS_LANGUAGE_ATTENTION_MASK]
images, img_masks = policy.prepare_images({"observation.images.screen": img_t.to("mps")})
state = policy.prepare_state({"observation.state": torch.tensor([[0.0, -1.0, -1.0]]).to("mps")})

mode = sys.argv[1] if len(sys.argv) > 1 else "train"   # train | sample | cmp
out_dir = Path("outputs/debug_vfield")
out_dir.mkdir(parents=True, exist_ok=True)

for t in (0.1, 0.5, 0.9):
    g = torch.Generator().manual_seed(0)
    noise = (torch.randn(act_t.shape, generator=g) * policy.config.noise_std).to("mps")
    x_t = (1 - t) * act_t + t * noise
    time = torch.tensor([t], device="mps")
    f = out_dir / f"v_{mode}_t{t}.pt"

    if mode == "train":
        with torch.no_grad():
            _, _, _, _, a_hat, _, _ = policy.model.forward(
                images, img_masks, lang_tokens, lang_masks, state, act_t, noise, time, is_grounding=None)
        v = ((x_t - a_hat[..., :3]) / t)[0]
        torch.save(v.cpu(), f)
        print("saved", f)

    elif mode == "sample":
        prefix_embs, prefix_pad_masks, prefix_att_masks = policy.model.embed_prefix(
            images, img_masks, lang_tokens, lang_masks, state=state)
        prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
        with torch.no_grad():
            _, past_key_values = policy.model.vlm_with_expert.forward(
                attention_mask=prefix_att_2d_masks,
                position_ids=prefix_position_ids,
                past_key_values=None,
                inputs_embeds=[prefix_embs, None],
                use_cache=policy.model.config.use_cache,
                fill_kv_cache=True,
            )
            v = policy.model.denoise_step(
                prefix_pad_masks, past_key_values, x_t, time.expand(1).clone())
        torch.save(v[0].cpu(), f)
        print("saved", f)

    else:
        a = torch.load(out_dir / f"v_train_t{t}.pt")
        b = torch.load(out_dir / f"v_sample_t{t}.pt")
        d = (a - b).abs()
        px = (d * torch.tensor([1.0, 1920.0, 1080.0])).sum(dim=-1)
        print(f"t={t}: |v_train - v_sample| px per-step: mean={px.mean():.1f} "
              f"max={px.max():.1f} | v_norm={a.norm():.3f} vs {b.norm():.3f}")
