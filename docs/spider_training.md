# SPIDER 机器人参考训练

实现与验收日期：2026-10-20。训练仓库为当前 `spider_maniptrans`；生成端修改位于
`/home/user/ylj/spider_repro/spider`。原方案文件保留不动。

## 行为与兼容性

`task.env.referenceSource=legacy` 为默认值，保留原数据工厂路由、人手目标、
`data/retargeting` 初始化及原来的提前三帧结束规则。

显式选择 `referenceSource=spider datasetType=spider` 后，使用
`data/retargeting_spider_native/<工具>/<数据集>/training/<任务>/manifest.json`。
训练包内的手部数组由对应正式 SPIDER PKL 与其 `reference_export_source` 逐字段
核对后导出。不会回退到旧 retarget、零 DOF 或 MANO 参考。手的复位、观测、
奖励、速度及接触距离均来自这套机器人参考；真实的 28 个 link 按名称对齐，
27 个非 palm link 直接打包到原网络兼容字段，避免 proximal link 合并。

默认 `objectReference=demonstration`，物体目标来自已对齐的源示范。
`spider_rollout` 显式跟踪 SPIDER 模拟中的物体，供消融实验使用。
两种策略各自统一物体复位、观测、奖励、速度和结束判断。

PPO 损失、基础策略与残差组合、动作含义和网络结构保持原实现。
基础策略仍使用单帧未来输入，因此训练入口要求 `obsFutureLength=1`；检查器另行
验证 `obsFutureLength=4` 的每个未来索引。不能直接把多帧观测交给原单帧权重。

完整但质量失败的参考在 `allowFailedReference=true` 时可加载，原始失败状态保持。
缺帧、非有限值、非法旋转、名称不匹配、资产校验和不一致、时钟不符均报错。
SPIDER 模式不会按 `episodeLength` 裁剪，1294 帧参考要求至少 1294 的配置。

legacy 回归还修复了两个输入读取问题：左右 OakInk2 类仅索引非隐藏的 PKL 标注，
并只读取实际选择的操纵物体网格，避免 `.DS_Store` 和无关场景物体缺失导致启动失败。
原任务选择、网格数值、手部目标和 retarget 路径未改变。

## 已有训练包

以下包已生成到当前仓库 `data/retargeting_spider_native`，均为完整 60 Hz 轨迹，
质量状态均为 `failed`。没有重新运行 SPIDER 优化。

| 数据集 | 任务 | 工具命名空间 | 帧数 | 当前验收 |
| --- | --- | --- | --- | --- |
| TACO | `sk_box_1104218` | `045` | 259 | 目标、模型、PhysX、两轮 PPO |
| TACO | `sk_box_1104218` | `153` | 259 | 目标与 `spider_rollout` 策略 |
| OakInk-v2 | `9fc3e@0` | `kni@1` | 1294 | 目标、PhysX、两轮 PPO，随机复位 |
| OakInk-v2 | `9b728@0` | `spoon@1` | 879 | 目标与完整长度 |

TACO 153 的右手 `obj_id=205, selected_asset_id=153`，左手保持 `143/143`；
工具命名空间同时包含两手，并不意味着两手都加载替换工具。

方案中旧的 `validation/native_9fc3e_kni1_relaxed/generation.json` 与当前正式 PKL
已不一致，转换器会拒绝混用。本次 kni@1 使用与当前 PKL 一致的
`example_datasets/processed/oakink2_native/artimano/bimanual/9fc3e@0/kni@1/0/generation.json`。
任务短名只用于 manifest 定位，正式 OakInk2 PKL 的完整文件名保留在元数据中。

## 转换已有生成结果

生成端新增 `spider/postprocess/export_maniptrans_training.py`，仅依赖 NumPy、SciPy
和标准库。正式 PKL 通过固定 NumPy 全局白名单读取；包内 NPZ 禁用 pickle。

```bash
cd /home/user/ylj/spider_repro/spider

.venv/native311/bin/python -m spider.postprocess.export_maniptrans_training \
  --generation-json example_datasets/validation/native_taco_sk_box_045/generation.json \
  --output-root /home/user/ylj/spider_maniptrans/data/retargeting_spider_native

.venv/native311/bin/python -m spider.postprocess.export_maniptrans_training \
  --generation-json 'example_datasets/processed/oakink2_native/artimano/bimanual/9fc3e@0/kni@1/0/generation.json' \
  --output-root /home/user/ylj/spider_maniptrans/data/retargeting_spider_native
```

生成端 `native_generation.py` 已接入完整正式参考导出后的自动打包，原生 YAML 新增
`export_training_bundle: true`。报告记录 `training_bundle` 和打包状态；质量失败不会
单独阻断完整包，格式错误仍会阻断。原正式 PKL 导出接口保持。

首次发布原子替换完整目录；再次生成时先写入不可变的、带校验和名称的 NPZ，
再原子切换 manifest。正在读取旧 manifest 的进程仍持有完整旧轨迹。
因此再次发布后的 `trajectory_file` 可以是 `trajectory.<sha256>.npz`，加载器以
manifest 为准，不应硬编码 NPZ 文件名。

## 训练前检查

当前 `spider` 环境是 Python 3.8 / Torch 1.13，数据与 PhysX 检查已在该环境通过，
但缺少 `rl_games`。当前已有的 `physgraph` 环境是 Python 3.8 / Torch 2.0，具备
训练依赖，本次模型严格加载与 PPO 验收使用此环境。没有安装、升级或替换环境依赖，
也没有将生成端 Python 3.11 / MuJoCo / Warp 导入训练进程。

```bash
cd /home/user/ylj/spider_maniptrans
export SPIDER_TRAIN_ENV=/home/user/miniconda3/envs/physgraph
export PATH="$SPIDER_TRAIN_ENV/bin:$PATH"
export LD_LIBRARY_PATH="$SPIDER_TRAIN_ENV/lib:${LD_LIBRARY_PATH:-}"

"$SPIDER_TRAIN_ENV/bin/python" scripts/check_spider_reference.py \
  --dataset TACO --task sk_box_1104218 --new-obj 045 \
  --check-models --simulator-check --report /tmp/spider_reference_check.json
```

OakInk2 使用 `--dataset OakInk-v2 --task '9fc3e@0' --new-obj 'kni@1'`。
`--reject-failed` 会拒绝质量失败包；默认保留用户允许的失败参考。
`--episode-length 1200` 会拒绝完整 1294 帧包。

CPU 检查使用真实 `pack_data/compute_observations_side/compute_reward_side`，
覆盖错误人手字段隔离、同目标零误差、两种结束规则、末帧、索引越过末帧和四个未来偏移。
`--check-models` 严格加载原左右模型并检查动作有限及观测形状。
`--simulator-check` 在独立的两环境实例中检查首、中、末帧 PhysX FK 和 done 后随机复位。
此静态检查临时关闭该实例的接触并清零速度，以隔离 FK；实际 PPO 保持原碰撞设置。
静态 FK 通过不能证明动力学跟踪成功。

## 两轮 PPO 命令

```bash
cd /home/user/ylj/spider_maniptrans
export SPIDER_TRAIN_ENV=/home/user/miniconda3/envs/physgraph
export PATH="$SPIDER_TRAIN_ENV/bin:$PATH"
export LD_LIBRARY_PATH="$SPIDER_TRAIN_ENV/lib:${LD_LIBRARY_PATH:-}"

OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 "$SPIDER_TRAIN_ENV/bin/python" -m main.rl.train \
  task=ResDexHand side=BiH bimanual_mode=united dexhand=artimano \
  headless=true test=false num_envs=16 max_iterations=2 \
  randomStateInit=false rolloutStateInit=false \
  'dataIndices=[sk_box_1104218]' \
  task.env.referenceSource=spider task.env.datasetType=spider \
  task.env.referenceRoot=/home/user/ylj/spider_maniptrans/data/retargeting_spider_native \
  task.env.referenceDataset=TACO task.env.newObj=045 \
  task.env.objectReference=demonstration task.env.allowFailedReference=true \
  task.env.episodeLength=1500 task.task.randomize=false \
  rl_train.params.config.horizon_length=32 rl_train.params.config.minibatch_size=256 \
  rh_base_model_checkpoint=/home/user/ylj/spider_maniptrans/assets/imitator_rh_artimano.pth \
  lh_base_model_checkpoint=/home/user/ylj/spider_maniptrans/assets/imitator_lh_artimano.pth \
  experiment=spider_taco_smoke
  
  
# train

# TACO
cd /home/user/ylj/spider_maniptrans

export SPIDER_TRAIN_ENV=/home/user/miniconda3/envs/physgraph
export PATH="$SPIDER_TRAIN_ENV/bin:$PATH"
export LD_LIBRARY_PATH="$SPIDER_TRAIN_ENV/lib:${LD_LIBRARY_PATH:-}"

OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
CUDA_VISIBLE_DEVICES=1 "$SPIDER_TRAIN_ENV/bin/python" -m main.rl.train \
  task=ResDexHand side=BiH bimanual_mode=united dexhand=artimano \
  headless=true test=false num_envs=4096 learning_rate=2e-4 \
  max_iterations=10000 \
  randomStateInit=false rolloutStateInit=false \
  'dataIndices=[bb_box_0928046]' \
  task.env.referenceSource=spider task.env.datasetType=spider \
  task.env.referenceRoot=/home/user/ylj/spider_maniptrans/data/retargeting_spider_native \
  task.env.referenceDataset=TACO task.env.newObj=004 \
  task.env.objectReference=demonstration task.env.allowFailedReference=true \
  task.env.episodeLength=1500 task.task.randomize=false \
  rl_train.params.config.horizon_length=32 \
  rl_train.params.config.minibatch_size=1024 \
  early_stop_epochs=1000 \
  rh_base_model_checkpoint=/home/user/ylj/spider_maniptrans/assets/imitator_rh_artimano.pth \
  lh_base_model_checkpoint=/home/user/ylj/spider_maniptrans/assets/imitator_lh_artimano.pth \
  experiment=SPIDER_Taco_bb_box_0928046_004_artimano


# OakInk-v2
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
CUDA_VISIBLE_DEVICES=0 "$SPIDER_TRAIN_ENV/bin/python" -m main.rl.train \
  task=ResDexHand side=BiH bimanual_mode=united dexhand=artimano \
  headless=true test=false num_envs=4096 learning_rate=2e-4 \
  max_iterations=10000 \
  randomStateInit=false rolloutStateInit=false \
  'dataIndices=[9b728@0]' \
  task.env.referenceSource=spider task.env.datasetType=spider \
  task.env.referenceRoot=/home/user/ylj/spider_maniptrans/data/retargeting_spider_native \
  task.env.referenceDataset=OakInk-v2 task.env.newObj=spoon@1 \
  task.env.objectReference=demonstration task.env.allowFailedReference=true \
  task.env.episodeLength=1500 task.task.randomize=false \
  rl_train.params.config.horizon_length=32 \
  rl_train.params.config.minibatch_size=1024 \
  early_stop_epochs=1000 \
  rh_base_model_checkpoint=/home/user/ylj/spider_maniptrans/assets/imitator_rh_artimano.pth \
  lh_base_model_checkpoint=/home/user/ylj/spider_maniptrans/assets/imitator_lh_artimano.pth \
  experiment=SPIDER_OakInk_9b728@0_spoon@1_artimano


# test

# OakInk-v2
python -m main.rl.train \
  task=ResDexHand side=BiH bimanual_mode=united dexhand=artimano \
  test=true headless=false num_envs=4 test_episodes=120 \
  randomStateInit=false rolloutStateInit=false \
  'dataIndices=[9b728@0]' \
  task.env.referenceSource=spider task.env.datasetType=spider \
  task.env.referenceRoot=data/retargeting_spider_native \
  task.env.referenceDataset=OakInk-v2 task.env.newObj=spoon@1 \
  task.env.objectReference=demonstration task.env.allowFailedReference=true \
  task.env.episodeLength=1500 task.task.randomize=false \
  rh_base_model_checkpoint=assets/imitator_rh_artimano.pth \
  lh_base_model_checkpoint=assets/imitator_lh_artimano.pth \
  checkpoint=runs/SPIDER_OakInk_9b728@0_spoon@1_artimano__10-26-19-08-20/nn/last_SPIDER_OakInk_9b728@0_spoon@1_artimano_ep_10000_rew__8167.7_.pth \
  experiment=test_oakink_9b728_spoon1 task.env.semanticContactEval.enabled=true

# TACO
python -m main.rl.train \
  task=ResDexHand side=BiH bimanual_mode=united dexhand=artimano \
  test=true headless=false num_envs=4 test_episodes=120 \
  randomStateInit=false rolloutStateInit=false \
  'dataIndices=[sk_box_1104218]' \
  task.env.referenceSource=spider task.env.datasetType=spider \
  task.env.referenceRoot=data/retargeting_spider_native \
  task.env.referenceDataset=TACO task.env.newObj=045 \
  task.env.objectReference=demonstration task.env.allowFailedReference=true \
  task.env.episodeLength=1500 task.task.randomize=false \
  rh_base_model_checkpoint=assets/imitator_rh_artimano.pth \
  lh_base_model_checkpoint=assets/imitator_lh_artimano.pth \
  checkpoint=runs/SPIDER_Taco_sk_box_1104218_045_artimano__10-23-10-13-30/nn/last_SPIDER_Taco_sk_box_1104218_045_artimano_ep_10000_rew__141.53_.pth \
  experiment=test_taco_sk_box_045 task.env.semanticContactEval.enabled=true

```

OakInk2 替换为 `dataIndices=[9fc3e@0]`、`referenceDataset=OakInk-v2`、`newObj=kni@1`、
`randomStateInit=true`、`experiment=spider_oakink_smoke`。正式训练可提高环境数和迭代数，
同时保持 `num_envs * horizon_length` 可被 `minibatch_size` 整除。

`newObj=null` 表示 `original` 命名空间。TACO 数字工具 ID 在拼路径前补齐三位；
OakInk2 名称原样保留。此字段用于 SPIDER 模式，原顶层 `new_obj` 用于 legacy，
两个字段冲突时报错。`referenceRoot`、权重路径与资产重映射在 Hydra 路径规范化后使用。

移动资产时通过 `assetRoot` 解释 manifest 中的相对路径，或通过 `assetPathMap`
显式将旧绝对根映射到新根，例如：

```text
'task.env.assetPathMap={/old/repo/data:/new/repo/data}'
```

加载器不会搜索同名网格，重映射后的网格、物体 URDF 和当前训练手 URDF 均核对校验和。

每次 SPIDER 实验目录中额外保存 `spider_references.json`，记录真实 manifest、
trajectory 校验和、失败状态、物体目标策略和资产信息，原 `config.yaml` 同时保留。

## 实测结果

| 验证 | 结果 |
| --- | --- |
| 当前仓库 CPU 回归 | 45 项通过：新增 12，原有 33 |
| 生成端打包器回归 | 新增 6 项通过，含生成结束前的自动打包接口 |
| 原有 PKL 导出回归 | 9 项通过 |
| 完整参考目标及 URDF FK | 四组两手均通过，最大位置误差小于 `1e-6 m` |
| TACO 045 静态 PhysX FK | 帧 0、129、258，两手误差小于 `1e-6 m` |
| OakInk2 kni@1 静态 PhysX FK | 帧 0、647、1293，两手误差小于 `1e-6 m` |
| Artimano 基础模型 | 严格加载通过，基础动作与残差动作均为 56 维双手接口 |
| 原网络观测 | proprioception 158，privileged 118，target 900 |
| TACO 045 PPO | 16 环境，两轮更新，loss 与所有已记录 scalar 有限 |
| OakInk2 kni@1 PPO | 16 环境，随机复位，两轮更新，loss 与所有已记录 scalar 有限 |
| legacy OakInk2 PPO | 原路由、原 retarget、默认 1200 帧处理，一轮更新通过 |

TACO 两轮 actor loss 为 `-0.05055, -0.03313`，critic loss 为 `1.34404, 0.34979`。
OakInk2 两轮 actor loss 为 `-0.05070, -0.05451`，critic loss 为 `1.97543, 0.69739`。

对应本机实验目录：

- `runs/spider_taco_smoke__10-20-18-00-02`
- `runs/spider_oakink_smoke__10-20-18-02-29`
- `runs/spider_legacy_regression__10-20-18-06-54`

复测命令：

```bash
/home/user/miniconda3/envs/spider/bin/python -m unittest discover -s tests -q

cd /home/user/ylj/spider_repro/spider
.venv/native311/bin/python -m unittest discover -s tests -p 'test_training_bundle.py' -q
.venv/native311/bin/python -m unittest discover -s tests -p 'test_maniptrans_reference.py' -q
```

本次验证证明参考接入、模型兼容和 PPO 更新可运行；没有运行长训练或多 seed 收敛实验。
两个 PPO 冒烟实验的短期成功率为 0，不能把它们表述为已经学会任务。
153 与 spoon@1 尚未进行 PPO 验收。`demonstration` 策略下失败 SPIDER 手部与原物体
可能不接触，这是参考质量问题，未通过改变任务物体目标来规避。
