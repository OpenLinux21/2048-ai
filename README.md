# 2048-ai:遗传算法 + PyTorch CUDA 的 2048 AI

以经典的 [games/2048.c](games/2048.c)(Maurits van der Schee, v1.0.3,一行未改)为游戏环境,
用遗传算法(GA)进化 AI 的策略参数。训练时全部对局在 GPU 上向量化并行(RTX 5060 Laptop 实测
**单代 3.3 万局并行、约 210 万步/秒**);演示时 AI 通过**方向键**驱动真实终端游戏。

300 代(8192 种群 × 4 局 = 每代 32768 局)训练结果:**平均分 1.2 万,11% 的对局到达 2048,
反复出现 4096,单局最高 6 万+ 分**(见 [训练结果](#训练结果))。

## 架构

```
                    ┌────────────────────────────────────────────┐
                    │            games/2048.c(未改动)            │
                    │  slideArray / rotateBoard / moveX / gameEnded│
                    └──────┬─────────────────────┬───────────────┘
              #include 原样复用逻辑            │ 编译产物(演示)
                    │                        │
     ┌──────────────▼───────────┐    ┌───────▼────────────────┐
     │ csrc/shim2048.c → .so    │    │ games/2048 终端游戏      │
     │ 种子化 xorshift32 生成新块 │    └───────▲────────────────┘
     └──────────────┬───────────┘            │ pty:读 ANSI 帧→解析棋盘
                    │ ctypes (~2µs/步)       │ 注入 \x1b[A/B/C/D 方向键
     ┌──────────────▼────────────────────────┴──────┐
     │  ai2048/vec_env.py  批量 torch 环境(CPU/CUDA)│  ← 与 C 逐位一致
     │  行查找表 65536 项 + 逐局 xorshift32          │
     └──────────────┬───────────────────────────────┘
                    │  apply_move(前瞻) / step(移动+生成)
     ┌──────────────▼───────────┐   ┌──────────────────────────┐
     │ ai2048/policy.py 策略     │   │ ai2048/ga.py 遗传算法     │
     │ Linear / MLP,按局并行打分 │   │ 精英+锦标赛+均匀/BLX+变异 │
     └──────────────┬───────────┘   └──────────┬───────────────┘
                    └────────── ai2048/train.py ─┘  每代 P×K 局同时跑
```

三个环境后端共享同一套逻辑契约,**同种子轨迹逐位一致**(300 局差分验证):

| 后端 | 用途 | 实测吞吐 |
|---|---|---|
| C shim(ctypes) | 逻辑权威、复核、评估 | ~1.1M 步/秒(单线程) |
| VecEnv CPU(torch) | 小规模训练、无 GPU 环境 | ~5.6M 步/秒(B=16384) |
| VecEnv CUDA(torch) | 大规模并行自弈 | ~12.2M 步/秒(B=16384,纯环境) |

## 快速开始

```bash
# 1. 环境(已有 .venv: Python 3.12 + torch 2.11 cu128)
python3.12 -m venv .venv && .venv/bin/pip install -r requirements.txt

# 2. 编译 C shim + 运行全部测试(C 自测 13 向量 + pytest)
make test

# 3. 最小训练(CPU, ~5 分钟, 50 种群)
.venv/bin/python -m ai2048.train --config configs/minimal.yaml

# 4. GPU 大规模训练(8192 种群 × 4 局 = 每代 3.3 万并行对局)
.venv/bin/python -m ai2048.train --config configs/gpu.yaml

# 5. 评估检查点(200 局固定种子)
.venv/bin/python -m ai2048.evaluate --checkpoint runs/gpu/best.pt --games 200

# 6. 看 AI 用方向键玩真 2048.c(pty 驱动原版二进制)
.venv/bin/python -m ai2048.play_tui --checkpoint runs/gpu/best.pt          # 逐行看分数
.venv/bin/python -m ai2048.play_tui --checkpoint runs/gpu/best.pt --render # 实时重绘棋盘
```

## 模块说明

| 文件 | 说明 |
|---|---|
| `csrc/shim2048.c` | `#include "2048.c"` 原样复用移动/合并/判负逻辑;仅把新块生成换成可设种的 xorshift32(分布策略照抄原版:列主序空格表、先选位后定值、2:0.9 / 4:0.1) |
| `ai2048/shim_env.py` | ctypes 封装;棋盘 C 侧 `[列][行]` ↔ Python 侧 `[行][列]` 转换集中于此 |
| `ai2048/vec_env.py` | 批量 torch 环境:16 位行查找表(65536 项)、逐局 xorshift32(int64 模拟 uint32)、生成/判负/合法步掩码全向量化 |
| `ai2048/policy.py` | `LinearPolicy`(16~24 维可解释特征)与 `MLPPolicy`(577 维小网络);按局并行给 4 个候选走步的结果棋盘打分 |
| `ai2048/lookahead.py` | 可选 2-ply chance-sampling expectimax(生成感知,评估/演示用) |
| `ai2048/ga.py` | 精英保留 + 锦标赛/排名选择 + 均匀/BLX-α 交叉 + 高斯变异(可选自适配 σ)+ HOF;全张量化,CPU/CUDA 皆可 |
| `ai2048/train.py` | 训练循环:每代 P×K 局一个 batch;CRN(同代个体面对相同种子组);适应度形状化(纯得分 / 得分+最高块档位奖励 / log 得分);检查点续训;JSONL 指标 |
| `ai2048/play_tui.py` | pty 启动真游戏,解析 ANSI 渲染帧得到棋盘,注入方向键字节,实时观看 AI 对局 |
| `ai2048/validate.py` | C↔torch(CPU/CUDA)差分验证 + 各后端吞吐基准 |
| `ai2048/evaluate.py` | 固定种子批量评测:分数统计 + 最高块分布 |

## 配置要点(configs/*.yaml)

- `population` × `episodes_per_genome`:每代并行对局数(GPU 建议 ≥3 万)。
- `features`:特征集。默认 `value16 + empty`;GPU 版加 `corner_max, max_exp, monotonic4, smoothness`(共 24 维)。
- `fitness.mode = score_maxtile`:得分 + 超过 1024 的每个块档位奖励(`maxtile_tier_bonus`)。
- `ga.*`:精英数、锦标赛宽度、交叉方式(`uniform`/`blx`)、变异率/强度、`adaptive_sigma`。
- `seed`:主种子;每代评估种子 = f(seed, gen),同代所有个体共享(CRN 方差缩减)。
- `compile: true`:尝试 `torch.compile` 策略前向(需系统 python dev 头文件;缺头文件时自动回退 eager)。

## 训练结果

### GPU 主训练(configs/gpu.yaml,RTX 5060 Laptop 8GB,8192 种群 × 300 代)

- 每代 32768 局同时进行,每代约 9~13 秒(约 210 万步/秒,含策略前向)。
- 约 20 代出现首个 2048;60 代内出现 4096;第 300 代:**11% 的对局到达 2048**(3615/32768),
  单代 best 得分 6 万+,平均分 1.2 万;曲线见 `runs/gpu/curves.png`,指标见 `runs/gpu/metrics.jsonl`。
- 固定种子留出集评测(`ai2048.evaluate`,200 局):均分 11001 / 中位 8844 / 单局最高 35848,
  45% 到达 1024、7% 到达 2048(`runs/gpu/eval_200.json`)。
- 同一权重换 2-ply expectimax 推理(50 局):**2048 到达率 7% → 18%**(46% 到 1024)——
  深搜索确实能抬档,但每步开销约 500 倍,适合终评/演示而非训练内循环(`runs/gpu/eval_50_la2.json`)。

### M1 最小验证(configs/minimal.yaml,纯 CPU)

- 50 种群 × 200 代约 5 分钟:best 得分 12084、最高块 1024,平均分 860 → 4000。

### 吞吐与等价性

- C shim 1.1M 步/秒(单线程)→ torch CPU 5.6M → torch CUDA 12.2M(纯环境),
  训练回路(含策略)约 2.1M 步/秒 @ B=32768。
- 差分验证:300 局 × 平均 143 步,C shim / VecEnv-CPU / VecEnv-CUDA 三者棋盘与得分**逐位一致**。

## 设计要点

- **为什么不直接用 `rand()`**:`2048.c` 的 `addRandom` 首次调用会强制 `srand(time(NULL))`
  (函数内 static 标志),无法设种、不可复现、非线程安全。shim 用逐局 xorshift32 复刻其分布策略,
  torch 侧用相同算法(int64 模拟 uint32 位运算),保证三个环境序列一致。
- **转置警告**:C 侧 `board[x][y]` 中 x=列、y=行,与直觉相反。全部转换集中在
  `shim_env._write_board/board()`,且被 13 组 C 自测向量 + 差分测试双保险覆盖。
- **CRN(共同随机数)**:同代所有个体面对相同的 K 局种子组,适应度差异来自基因而非生成运气;
  跨代种子轮换避免过拟合固定牌局。
- **批量评估**:整个种群拼成一个 batch(P×K 局)锁步推进,每代只有极少的 CPU↔GPU 同步点
  (`sync_chunk=32`),评估结束后仅回读 P 个标量。

## 已知限制与后续方向

- `torch.compile` 需要系统 `python3-devel` 头文件(本机未装),当前自动回退 eager;
  安装后加 `compile: true` 即可启用。
- 新块生成的 RNG 流与原版 `rand()` 不同(分布一致、序列不同),因此与真人玩的原版游戏
  牌流不可逐帧对照(演示模式本来就是时间种子的随机牌流)。
- 棋盘打包按 4 bit 指数处理,合并出 65536(指数 16)时按 32768 封顶——4×4 实战中不可达。
- 后续可做:ntuple/元组网络权重(经典 2048 强化方案)、更深 expectimax、
  种群岛模型、C 环境多进程复核集群、W&B/TensorBoard 曲线。

## 测试

```bash
make test    # C 自测(13 向量)+ 73 项 pytest:
             # 移动/合并/得分语义、三环境逐位等价、批量一致性、
             # GA 算子(可复现/精英/选择压力/BLX/自适配σ)、
             # 策略掩码与按局权重、TUI 帧解析、迷你训练与断点续训
```
