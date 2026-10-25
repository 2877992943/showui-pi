#!/usr/bin/env python
"""ShowUI-pi 单任务实时屏幕测试 — 截屏 → 推理一笔拖拽 → pyautogui 执行 → 闭环。

用法（先跑 scripts/test_permissions.py 确认权限通过）:

  # 干跑: 只截屏+推理+保存轨迹可视化，不动鼠标（只需屏幕录制权限）
  PYTHONPATH=lerobot/src python scripts/demo_live_drag.py --dry-run \
      --task "Drag to write 'abc' in the drawing canvas."

  # 真实执行（需要 屏幕录制 + 辅助功能 权限）
  PYTHONPATH=lerobot/src python scripts/demo_live_drag.py \
      --task "Drag to write 'abc' in the drawing canvas." --strokes 6

说明:
  - 模型每次推理输出 21 步拖拽轨迹（一笔）。写 "abc" 这样的多笔画内容，
    靠闭环循环完成: 每笔执行后重新截屏、以上一步鼠标状态为 state 再规划下一笔。
  - 坐标空间: 模型固定按 1920x1080 归一化（policy_preprocessor.json 的
    smolvla_fixed_xy_normalizer）。本脚本把任意分辨率的截图 letterbox 到
    1920x1080 再喂给模型，预测坐标按逆映射换算回真实屏幕。
  - 急停: 把鼠标甩到屏幕左上角可触发 pyautogui 的 failsafe 中止。
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import numpy as np
import torch
from PIL import Image, ImageDraw

from huggingface_hub import HfApi, hf_hub_download

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "lerobot" / "src"))

from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata  # noqa: E402
from lerobot.policies.factory import (  # noqa: E402
    make_policy,
    make_policy_config,
    make_pre_post_processors,
)

MODEL_W, MODEL_H = 1920.0, 1080.0


# --------------- benchmark meta (仅 meta，不下载 parquet 数据) ---------------


def download_benchmark_meta(dataset_id: str) -> Path:
    """只下载 DEX benchmark 的 meta/ 目录（info.json、tasks.parquet 等，几 MB），
    用于构造 LeRobotDatasetMetadata 提供特征形状。"""
    api = HfApi()
    all_files = api.list_repo_files(repo_id=dataset_id, repo_type="dataset")
    wanted = [
        p for p in all_files
        if p.startswith("test/parquet/meta/")
    ]
    if not wanted:
        raise FileNotFoundError(f"{dataset_id} does not contain a test/parquet/meta/ tree")
    print(f"Downloading {len(wanted)} meta files from {dataset_id} ...", flush=True)
    local_root: Path | None = None
    for i, rel in enumerate(wanted, 1):
        local = Path(hf_hub_download(repo_id=dataset_id, filename=rel, repo_type="dataset"))
        if local_root is None:
            parts = local.parts
            for j, part in enumerate(parts):
                if part == "test":
                    local_root = Path(*parts[:j])
                    break
    if local_root is None:
        raise RuntimeError("Could not determine snapshot root from downloads")
    root = local_root / "test" / "parquet"
    if not (root / "meta" / "info.json").is_file():
        raise FileNotFoundError(f"Expected meta at {root}/meta/info.json")
    return root


def print_handwriting_tasks(ds_root: Path, n: int = 15) -> None:
    """打印 DEX handwriting 域的任务文本样例，帮助选择贴近训练分布的 --task 措辞。"""
    import pandas as pd

    with (ds_root / "meta" / "domain_map.json").open() as f:
        domain_map = {int(k): str(v) for k, v in json.load(f).items()}
    eps_df = pd.read_parquet(ds_root / "meta" / "episodes" / "chunk-000" / "file-000.parquet")
    tasks_df = pd.read_parquet(ds_root / "meta" / "tasks.parquet")
    tasks_by_id = {
        int(tasks_df.iloc[i]["task_index"]): str(tasks_df.index[i]) for i in range(len(tasks_df))
    }
    handwriting_tasks = {
        tasks_by_id[int(row["task_index"])]
        for _, row in eps_df.iterrows()
        if domain_map.get(int(row["episode_index"])) == "handwriting"
    }
    print(f"\nhandwriting 域任务文本样例（{len(handwriting_tasks)} 种）:")
    for t in sorted(handwriting_tasks)[:n]:
        print(f"  - {t}")
    print()


# --------------- policy 加载（同 eval_dex，但设备可配） ---------------


def load_policy(ckpt_dir: str, ds_root: Path, device: str):
    ckpt_path = Path(ckpt_dir)
    with (ckpt_path / "config.json").open() as f:
        raw_cfg = json.load(f)
    ptype = raw_cfg.pop("type", None)
    if not ptype:
        raise ValueError(f"Could not determine policy type from {ckpt_path}")

    policy_cfg = make_policy_config(ptype, **raw_cfg)
    policy_cfg.pretrained_path = str(ckpt_path)
    policy_cfg.device = device

    ds_meta = LeRobotDatasetMetadata("ScreenDrag", root=ds_root, revision="v3.0")
    policy = make_policy(cfg=policy_cfg, ds_meta=ds_meta)
    policy.eval()
    if device == "mps":
        # checkpoint 混有 bf16/fp32 参数，MPS 的矩阵乘法要求 dtype 一致，
        # 统一转 fp32（Apple Silicon 上无 bf16 matmul 支持问题）。
        policy = policy.to(torch.float32)

    # 整段取出一笔（21 步轨迹）: 默认 n_action_steps=1 会在弹出 1 个动作后
    # 丢弃 chunk 的其余步。设为 chunk_size 让队列保留整笔轨迹。
    policy.config.n_action_steps = policy.config.chunk_size

    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy_cfg,
        pretrained_path=str(ckpt_path),
        preprocessor_overrides={
            "device_processor": {"device": device},
        },
    )
    return policy, preprocessor, postprocessor


# --------------- 屏幕坐标 <-> 模型坐标 (1920x1080 letterbox) ---------------


def capture_model_frame() -> tuple[torch.Tensor, dict]:
    """截全屏并 letterbox 到 1920x1080。返回 ((1,3,1080,1920) float [0,1], 映射参数)。"""
    import pyautogui

    pil = pyautogui.screenshot().convert("RGB")
    # Retina 屏幕上截图是 2x 像素，而 pyautogui 鼠标坐标是逻辑分辨率，
    # 先统一缩到逻辑尺寸再 letterbox，保证逆映射出的坐标可直接喂给鼠标。
    logical_w, logical_h = pyautogui.size()
    if pil.size != (logical_w, logical_h):
        pil = pil.resize((logical_w, logical_h), Image.BILINEAR)
    W, H = pil.size
    scale = min(MODEL_W / W, MODEL_H / H)
    new_w, new_h = round(W * scale), round(H * scale)
    off_x = round((MODEL_W - new_w) / 2)
    off_y = round((MODEL_H - new_h) / 2)

    canvas = Image.new("RGB", (int(MODEL_W), int(MODEL_H)), (0, 0, 0))
    canvas.paste(pil.resize((new_w, new_h), Image.BILINEAR), (off_x, off_y))

    arr = np.asarray(canvas, dtype=np.uint8)
    img = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    mapping = {"scale": scale, "off_x": off_x, "off_y": off_y, "screen_w": W, "screen_h": H}
    return img, mapping


def model_to_screen(x: float, y: float, m: dict) -> tuple[float, float]:
    """模型空间 (1920x1080) → 真实屏幕坐标。"""
    rx = (x - m["off_x"]) / m["scale"]
    ry = (y - m["off_y"]) / m["scale"]
    return (
        float(np.clip(rx, 0, m["screen_w"] - 1)),
        float(np.clip(ry, 0, m["screen_h"] - 1)),
    )


# --------------- 一笔的推理与执行 ---------------


def predict_stroke(policy, preprocessor, postprocessor, img: torch.Tensor,
                   state: torch.Tensor, task_text: str) -> np.ndarray:
    """推理一整笔。返回 (chunk_size, 3) 的 [btn, x, y]，像素坐标（模型空间）。"""
    policy._queues["action"].clear()
    batch = {"observation.images.screen": img, "observation.state": state, "task": task_text}
    mb = preprocessor(batch)

    chunk = policy.config.chunk_size
    actions = []
    with torch.no_grad():
        for _ in range(chunk):
            act = policy.select_action(mb)
            act = postprocessor(act)
            actions.append(act.numpy()[0])
            # 队列耗尽时 select_action 会自动用新噪声重规划，为避免混入第二笔，
            # 只要 chunk 弹完即停。
    return np.asarray(actions, dtype=np.float32)


def execute_stroke(actions: np.ndarray, mapping: dict, step_duration: float,
                   force_pen_down: bool = True) -> None:
    """用 pyautogui 执行一笔拖拽。btn>0.5 视为鼠标按下（画笔触纸）。
    force_pen_down=True 时忽略 btn 通道（checkpoint 的 btn 输出不可靠，
    而 DEX 每条 episode 本就是一笔按住画的线），整笔按下拖拽后抬起。"""
    import pyautogui

    if force_pen_down:
        actions = actions.copy()
        actions[:, 0] = 1.0

    pressed = False
    prev = None
    for btn, mx, my in actions:
        btn_down = float(btn) > 0.5
        rx, ry = model_to_screen(mx, my, mapping)
        if prev is not None and (abs(rx - prev[0]) < 0.5 and abs(ry - prev[1]) < 0.5):
            prev = (rx, ry)
            continue  # 去重相邻重复点
        if btn_down and not pressed:
            pyautogui.moveTo(rx, ry, duration=min(step_duration, 0.1))
            pyautogui.mouseDown()
            pressed = True
        elif not btn_down and pressed:
            pyautogui.mouseUp()
            pressed = False
        pyautogui.moveTo(rx, ry, duration=step_duration)
        prev = (rx, ry)
    if pressed:
        pyautogui.mouseUp()


def visualize_stroke(img: torch.Tensor, actions: np.ndarray, out_path: Path) -> None:
    """在 letterbox 后的截图上画出预测轨迹，便于 dry-run 调试。"""
    canvas = Image.fromarray((img[0].permute(1, 2, 0).numpy() * 255).astype(np.uint8))
    draw = ImageDraw.Draw(canvas)
    pts = [(float(a[1]), float(a[2])) for a in actions]
    if len(pts) >= 2:
        draw.line(pts, fill=(0, 255, 0), width=4)
    draw.ellipse([pts[0][0] - 8, pts[0][1] - 8, pts[0][0] + 8, pts[0][1] + 8], fill=(255, 0, 0))
    draw.ellipse([pts[-1][0] - 8, pts[-1][1] - 8, pts[-1][0] + 8, pts[-1][1] + 8], fill=(0, 80, 255))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)


# --------------- main ---------------


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="models/showui-pi", help="pretrained_model 目录")
    ap.add_argument("--task", default="Write [abc] on the canvas.",
                    help="任务指令（DEX 训练格式为 Write [文本] on the canvas.，可用 --show-tasks 查看样例）")
    ap.add_argument("--strokes", type=int, default=6, help="规划的笔画数（一笔 = 一次截屏+推理+拖拽）")
    ap.add_argument("--device", default="auto", choices=["auto", "mps", "cpu", "cuda"])
    ap.add_argument("--dry-run", action="store_true",
                    help="只推理并在 outputs/ 保存轨迹可视化，不控制鼠标")
    ap.add_argument("--show-tasks", action="store_true",
                    help="打印 DEX handwriting 域任务文本样例后退出")
    ap.add_argument("--hf_dataset", default="h-siyuan/ScreenDrag")
    ap.add_argument("--step-duration", type=float, default=0.02, help="轨迹相邻点之间的鼠标移动时长(秒)")
    ap.add_argument("--pause", type=float, default=1.0, help="笔画之间的停顿(秒)")
    ap.add_argument("--output_dir", default="outputs/live_drag")
    ap.add_argument("--no-force-pen-down", dest="force_pen_down", action="store_false",
                    help="不强制落笔，按模型输出的 btn 通道判断按下/抬起")
    ap.add_argument("--num-steps", type=int, default=10, help="flow matching ODE 积分步数")
    ap.add_argument("--ensemble", type=int, default=1,
                    help=">1 时采样多条轨迹取逐点中值（更平滑，代价是推理时间×N）")
    args = ap.parse_args()

    device = args.device
    if device == "auto":
        device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"device = {device}")

    ds_root = download_benchmark_meta(args.hf_dataset)
    if args.show_tasks:
        print_handwriting_tasks(ds_root)
        return 0

    print("Loading policy ...", flush=True)
    policy, preprocessor, postprocessor = load_policy(args.ckpt, ds_root, device)
    if args.num_steps != 10:
        policy.config.num_steps = args.num_steps
    if args.ensemble > 1:
        policy.config.inference_ensemble = args.ensemble
    print(f"chunk_size = {policy.config.chunk_size} | num_steps = {getattr(policy.config, 'num_steps', 10)} "
          f"| ensemble = {getattr(policy.config, 'inference_ensemble', 1)}")

    if not args.dry_run:
        import pyautogui
        pyautogui.FAILSAFE = True  # 鼠标甩到屏幕角落急停
        print("\n*** 3 秒后开始控制鼠标，想中止请立即把鼠标甩到屏幕左上角 ***")
        time.sleep(3)

    out_dir = Path(args.output_dir)
    # 初始 state: [btn, x, y] = [0, -1, -1]（与 eval_dex 一致）
    state = torch.tensor([[0.0, -1.0, -1.0]], dtype=torch.float32)

    for stroke in range(args.strokes):
        img, mapping = capture_model_frame()
        t0 = time.time()
        actions = predict_stroke(policy, preprocessor, postprocessor, img, state, args.task)
        dt = time.time() - t0

        viz = out_dir / f"stroke_{stroke:02d}.png"
        visualize_stroke(img, actions, viz)
        start = model_to_screen(*actions[0][1:], mapping)
        end = model_to_screen(*actions[-1][1:], mapping)
        print(f"[stroke {stroke + 1}/{args.strokes}] 推理 {dt:.1f}s | "
              f"起点屏幕坐标 ({start[0]:.0f},{start[1]:.0f}) → 终点 ({end[0]:.0f},{end[1]:.0f}) "
              f"| btn ∈ [{actions[:, 0].min():.1f}, {actions[:, 0].max():.1f}] | 轨迹图 {viz}")

        if not args.dry_run:
            execute_stroke(actions, mapping, args.step_duration, args.force_pen_down)

        # state 反馈: 取这一笔最后一个点（像素空间, 模型坐标系），btn 取模型输出
        last = actions[-1]
        btn = 1.0 if float(last[0]) > 0.5 else 0.0
        if args.force_pen_down:
            btn = 1.0 if stroke + 1 < args.strokes else 0.0  # 收尾笔后抬笔
        state = torch.tensor(
            [[btn, float(np.clip(last[1], 0, MODEL_W - 1)), float(np.clip(last[2], 0, MODEL_H - 1))]],
            dtype=torch.float32,
        )
        if stroke + 1 < args.strokes:
            time.sleep(args.pause)

    print(f"\n完成。轨迹可视化已保存到 {out_dir}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
