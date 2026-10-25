# 本地环境与数据说明（macOS / Apple Silicon）

记录模型、数据、uv 环境的位置，以及如何复现 DEX benchmark 评测。

## 1. uv 环境

- 虚拟环境：项目根目录下的 **`.venv/`**
- Python：**3.10.20**（uv 托管的 cpython：`~/.local/share/uv/python/cpython-3.10-macos-aarch64-none`）
- uv 版本：0.11.8
- 依赖：按 `pyproject.toml` 安装（torch 2.14.1 / transformers / lerobot 等）
- 本机无 CUDA，有 **MPS**（`torch.backends.mps.is_available() == True`）

使用方式（项目根目录下）：

```bash
source .venv/bin/activate          # 或直接用 .venv/bin/python
PYTHONPATH=lerobot/src python ...  # lerobot 以源码方式挂在 lerobot/src
```

## 2. 模型

| 模型 | 位置 | 大小 | 说明 |
|---|---|---|---|
| ShowUI-π checkpoint | `models/showui-pi/` | 900MB | 本地目录，含 `model.safetensors` + `config.json` + `policy_preprocessor.json` / `policy_postprocessor.json`。与 HF 仓库 `showlab/ShowUI-pi` 内容完全一致（已核对文件清单），无需再下载 |
| 基座 VLM：SmolVLM2-500M-Video-Instruct | `hf_cache/hub/models--HuggingFaceTB--SmolVLM2-500M-Video-Instruct/` | 2.0GB | 实体 blob 在 `hf_cache/hub/blobs/`（所以 `du` 显示模型目录只有 3.4MB 是正常的——snapshot 里是符号链接）。首次加载 policy 时由 transformers 自动下载，**已下载完整** |

注意：
- `~/.cache/huggingface/hub/` 里只有同名 stub（无权重 blob），**实际权重都在项目内的 `hf_cache/`**。
- 基座仓库里的 `onnx/`（约 4GB 导出文件）和 `merges.txt`/`vocab.json`（慢速 tokenizer 的备用文件）**没有也不需要下载**，`from_pretrained` 用不到它们。`hf_cache` 里也不应有残留的 `.incomplete` 文件（已清理）。

### 2.1 两个模型分别如何加载使用

加载是串联的两步——2GB 基座负责"骨架 + 分词器"，900MB checkpoint 负责"训练好的权重"：

```
from_pretrained("models/showui-pi")
│
├─ ① 读 models/showui-pi/config.json → SmolVLAConfig
│     （vlm_model_name = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"）
│
├─ ② 实例化 SmolVLAPolicy → VLAFlowMatching.__init__（modeling_smolvla.py:604）
│     → SmolVLMWithExpertModel(model_id=基座repo)          ← 2GB 基座在此加载
│
└─ ③ safetensors.load_model(policy, "model.safetensors")   ← 900MB checkpoint 在此覆盖权重
      （lerobot/src/lerobot/policies/pretrained.py:72，strict=False）
```

**2GB 基座的用途**（加载点 `lerobot/src/lerobot/policies/smolvla/smolvlm_with_expert.py:137`）：

1. `AutoModelForImageTextToText.from_pretrained(基座, dtype=bfloat16)` 用基座权重初始化完整 VLM，随即截断为前 16 层（`num_vlm_layers: 16`）
2. `AutoProcessor.from_pretrained(基座)` 拿 tokenizer + image processor——**推理时一直真正在用**：preprocessor 用它编码 task 文本（`policy_preprocessor.json` 的 `tokenizer_name` 就指向该 repo）；`fake_image_token` / `global_image_token` id 也来自它（modeling_smolvla.py:652）
3. 基座 config（hidden_size 等）决定 expert 结构尺寸（`expert_width_multiplier: 0.75`）

即：基座提供架构 + 初始权重（随后被覆盖）+ tokenizer/processor + 结构超参。

**900MB checkpoint 的用途**（`models/showui-pi/model.safetensors`，500 个张量）：

- 345 个 VLM 骨干张量：把从基座继承的对应权重**全部覆盖**（训练真正更新的部分）
- 145 个 expert 张量：动作 expert，基座没有、纯新部件
- 10 个 action/state 投影张量（`action_in_proj` / `state_proj` 等）
- 同目录的 pre/postprocessor json 定义推理管线（tokenizer → 设备 → 坐标按 1920×1080 归一化 / 反归一化）

**推理时分工**：

```
截图+task → preprocessor（用基座的 tokenizer/processor 编码 + xy 归一化）
         → policy.select_action（VLM骨干+expert 做 flow matching 去噪，10 步；权重全部来自 checkpoint）
         → postprocessor（xy 反归一化回像素坐标）
```

一句话：**2GB 基座是"建模型的原料和工具（tokenizer）"，只在构建时用；900MB checkpoint 是"实际干活的权重"，每次推理用**。基座只在首次加载时下载，之后走 `hf_cache` 缓存。

## 3. 数据（DEX benchmark）

- HF 数据集：`h-siyuan/ScreenDrag`（LeRobot v3.0 格式，`test/parquet/` 子树）
- 本地位置：`hf_cache/hub/datasets--h-siyuan--ScreenDrag/snapshots/03765b0bff5da02a4a28ea66c0506583ba7a1dc6/test/parquet/`
- 已下载：
  - `meta/` 全部（info.json、domain_map.json、tasks.parquet、episodes/、source_episode_index.json）
  - `data/chunk-000/file-010.parquet`（123MB，**10 个 premiere episode**，每集 21 步）
- 全量 benchmark 共 11 个 data parquet、**约 4.4GB**、505 个 episode（5 个 domain × 101：captcha / premiere / desktop / powerpoint / handwriting，episode 是打散存储的，不是按 domain 连续分块）
- episode → domain 映射见 `meta/domain_map.json`

## 4. 关键环境变量

所有 HF 下载（模型 + 数据）都要落在项目目录内，运行任何脚本前设置：

```bash
export HF_HOME="$PWD/hf_cache"
```

## 5. 跑 DEX 评测（本地适配版）

`scripts1025/eval_dex.py` 是原版（默认 `--device cuda`、硬编码 `torch.autocast("cuda")`、首次运行自动下载数据），在 mac 上不能直接跑。
本地适配版为 `scripts1025/eval_dex_local.py`，改动：

- 不下载，直接读 `--ds_root` 下已存在的 parquet（可只评测子集）
- `--device` 默认 mps，autocast 仅在 cuda 下启用
- `--limit N` 限制每个 domain 评测的 episode 数
- **精度修复**：模型是混合 dtype（VLM 骨干 bf16 + 新模块 fp32），MPS 直接崩溃（`MPSNDArrayMatrixMultiplication` 断言失败）、CPU 报 dtype 不匹配。脚本对非 cuda 设备自动执行 `policy.float()` 统一为 fp32。**MPS 即使转 fp32 也可能不稳，实际验证可行的是 `--device cpu`**
- 速度参考：CPU fp32 约 1.6s/步，每集 21 步 ≈ 35s

命令：

```bash
HF_HOME="$PWD/hf_cache" PYTHONPATH=lerobot/src \
.venv/bin/python scripts1025/eval_dex_local.py \
  --ckpt models/showui-pi \
  --ds_root "hf_cache/hub/datasets--h-siyuan--ScreenDrag/snapshots/03765b0bff5da02a4a28ea66c0506583ba7a1dc6/test/parquet" \
  --output_dir outputs/eval_dex_local \
  --limit 1 --device cpu   # 每个 domain 测 1 集
```

每个 domain 测 10 集（即 6.1 节结果的命令）：

```bash
HF_HOME="$PWD/hf_cache" PYTHONPATH=lerobot/src \
.venv/bin/python scripts1025/eval_dex_local.py \
  --ckpt models/showui-pi \
  --ds_root "hf_cache/hub/datasets--h-siyuan--ScreenDrag/snapshots/03765b0bff5da02a4a28ea66c0506583ba7a1dc6/test/parquet" \
  --output_dir outputs/eval_dex_local \
  --limit 10 \
  --device cpu
```

全量 505 集：把 `--limit` 参数去掉即可（CPU 约 5 小时）。建议把 stdout 重定向到日志文件，例如 `> outputs/eval_dex_local/run_full.log 2>&1`。

## 6. 已跑结果（2026-10-25，每 domain 1 集，CPU fp32）

| domain | episode | err_best (px) | err_final (px) | success_best (≤20px) |
|---|---|---|---|---|
| captcha | 0 | 55.66 | 237.61 | ✗ |
| premiere | 77 | 4.07 | 263.28 | ✓ |
| desktop | 27 | 176.49 | 435.92 | ✗ |
| powerpoint | 366 | 16.52 | 186.49 | ✓ |
| handwriting | 265 | 74.89 | 116.93 | ✗ |

`endpoint_success_best` 2/5；`endpoint_success`（final）0/5。样本量太小（5/505），仅供参考流程验证；完整结论需要跑全量 505 集（CPU 上约 5 小时）。

### 6.1 每 domain 10 集（2026-10-25，共 50 集，CPU fp32，约 35 分钟）

| domain | endpoint_accuracy_best | endpoint_accuracy (final) |
|---|---|---|
| captcha | 30.00% (3/10) | 0.00% |
| premiere | 50.00% (5/10) | 0.00% |
| desktop | 50.00% (5/10) | 0.00% |
| powerpoint | 40.00% (4/10) | 0.00% |
| handwriting | 60.00% (6/10) | 0.00% |
| **平均** | **46.00%** | **0.00%** |

规律与 1 集/域时一致：模型能中途把轨迹压到终点 20px 内（best 指标正常），但最后一步几乎总是偏出（final 全为 0%），说明轨迹末端系统性拖偏。完整日志见 `outputs/eval_dex_local/run_10perdomain.log`，逐集结果在 `outputs/eval_dex_local/<domain>/per_episode.json`。

## 7. 指标

- 每集 21 步闭环推理，ground truth 是最后一帧 `action[1:3]`（终点 x, y，像素坐标）
- `endpoint_success`：最后一步预测距终点 ≤ 20px
- `endpoint_success_best`：整段中任意一步距终点 ≤ 20px（论文主指标 `endpoint_accuracy_best`）
