# ManipTrans 的 checkpoint 语义接触评估

本实现仅在 `ResDexHandBiH` 的测试阶段读取物理状态，直接评估已保存 checkpoint。
默认 `task.env.semanticContactEval.enabled=false`，训练时即使将该项设为 true 也不会初始化检测器、采集状态或增加日志。
主指标使用完整计划的固定分母，提前失败不会因缩短 episode 而得到虚高的完成比例或语义接触覆盖率。
评估不增加策略观测，不修改网络、checkpoint 恢复、归一化、动作处理、参考轨迹、reward 或原任务成功/失败判定。
原有 `reward`、`steps` 的累计和打印方式保持不变。没有 affordance 输入的策略也可以使用本评估器；本对比仓库不需要新增策略语义开关。

## 使用

在原来可以正确恢复 checkpoint 的测试命令中追加：

```bash
test=true randomStateInit=false rolloutBegin=null rolloutLen=null \
task.env.semanticContactEval.enabled=true
```

正式跨方法比较还需追加同一个共享时间标记文件：

```bash
task.env.semanticContactEval.contactRequiredMaskFile=/absolute/path/to/common_contact_required.json
```

例如，使用项目原有的双手入口：

```bash
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
python main/rl/train.py task=ResDexHand side=BiH dexhand=artimano \
  headless=true num_envs=4 test=true randomStateInit=false \
  rolloutBegin=null rolloutLen=null 'dataIndices=[cs_plate_0917023]' \
  rh_base_model_checkpoint=assets/imitator_rh_artimano.pth \
  lh_base_model_checkpoint=assets/imitator_lh_artimano.pth \
  actionsMovingAverage=0.4 checkpoint=/absolute/path/to/checkpoint.pth \
  task.env.semanticContactEval.enabled=true
```

`dexhand`、网络配置、基础策略 checkpoint、PID、动作平滑等必须使用该 checkpoint 训练时的配置；示例值不能用于替代实际训练配置。
如需替换工具，使用 `new_obj` 替换 TACO 或 OakInk-V2 的右手工具，并选择对应数据；OakInk-V2 使用原生物体 ID（例如 `new_obj=O02@0015@00004`）。
对比方法仍通过原数据加载器读取 `data/retargeting` 中 checkpoint 训练时的原始参考。
公共修正 PKL 仅可提供评估标注，里面的腕部、手指和物体轨迹不会进入策略。

检测器默认从已加载右手工具的 `obj_mesh_path` 读取红黑标注，包括当前 `new_obj` 指向的工具。
每个顶点必须明确为红色（R >= 200、G/B <= 40）或黑色（RGB <= 40），且两种区域均非空。
未标注或标注不完整会报错，不能作为“无接触”数据继续统计。
若物理网格不含颜色，可在 `main/cfg/task/ResDexHand.yaml` 配置同一工具的标注副本：

```yaml
semanticContactEval:
  enabled: true
  semanticMeshPaths:
    "cs_plate_0917023": /absolute/path/to/same_tool_labeled.ply
```

键为 `dataIndices` 中的完整数据 ID，路径指向与物理工具具有相同几何、原点和方向的网格。
网格坐标必须是米，不能使用未换算的 TACO `_cm.obj`。该映射只提供评估标注，不替换物理资产或策略输入。
当前 `run_taco_train.sh` 的 `cs_plate_0917023`、`cs_plate_1104155` 分别使用工具 `033`、`181`。
本地这两个物理 PLY 没有顶点颜色，需通过上述映射提供人工确认过的红黑标注副本。
使用 `new_obj` 时，映射仍以数据 ID 为键，但必须指向替换后的工具标注。
红黑点集分别按顶点顺序用 `linspace` 确定性取 512 点，与共同 PhysGraph 数据语义点集的采样规则一致。
比较时要固定同一工具标注及采样规则，不可混用顶点点集、面中心点集或软语义特征。

## 片段、采集时序与事件

四项指标均使用 `dataIndices` 选定的完整操作片段，不另选抓握子区间。
本仓库 `compute_imitation_reward` 在参考帧 `t` 使用 `t + 1 + 3 >= seq_len` 判断正常结束，
因此评估的包含式终点为：

```text
end = max(min(right_seq_len, left_seq_len, episodeLength) - 4, 0)
planned_frames = end + 1
```

从帧 0 正常执行到 `end`，实际执行帧数为 `planned_frames`。提前失败、非零起点或截短 rollout 均不缩短该计划。
`post_physics_step` 原有的观测更新先刷新实时状态，原 reward 随后判断终止；
评估在这些操作后、`progress_buf` 递增前采集，包含终止帧。
终止快照由独立 tensor 副本组成，不会被下一个环境复位覆盖。
这里只能对本仓库使用上述终点规则；移植到其他环境必须重新核对计数和终止规则。

## 共同应接触时间标记

`contact_required[t]` 在初始化时固定，帧索引相对于 `dataIndices` 选中并实际加载的完整片段，
与原始 mocap 文件索引或随机初始化之后的局部 episode 计数不同。
加载优先级为：

1. `contactRequiredMaskFile` 中以完整数据 ID 为键的共享 JSON。
2. 公共修正 PKL 的 `contact_retarget.source_contact_intent.contact_frame_mask`。
3. 旧修正 PKL 的 `contact_retarget.grasp_interval`，作为粗粒度连续应接触区间。
4. checkpoint 原参考的 `tips_distance` 中任一指尖距离 `<= expectedContactDistance` 的帧。

逐帧源接触意图来自几何修正前的原工具。它使用 `grasp_interval` 的局部时间轴，
加载时会加上区间起点，再裁剪到完整有效计划。不会将它误置于帧 0。
局部布尔数组必须覆盖包含首尾的抓握区间；缺少时间偏移或标注格式错误会报错。
没有逐帧源意图的旧 PKL 保持抓握区间兼容。

共享 JSON 示例（以下 ID 和区间仅示意，实际实验须按任务确定）：

```json
{
  "sequence@0": {"required_intervals": [[100, 299], [350, 399]]},
  "another@0": [false, true, true, false]
}
```

区间包含首尾，并与完整有效计划取交集；空区间列表表示没有应接触标记。
布尔数组必须至少覆盖完整有效计划，超出计划的尾部会裁剪。
配置了共享 JSON 后，缺失数据 ID、短数组或非布尔数组均报错，不会静默回退到某方法的参考推断。

本基线的公共修正元数据默认在以下 canonical 目录查找（仅当 `new_obj` 非空）：

```text
task.env.semanticContactEval.contactRetargetRoot/
  <dataset>/mano2<dexhand>_rh/<data_id>/<new_obj>/*.pkl
```

`contactRetargetRoot` 默认为 `data/retargeting_contact`，可设为公共 PhysGraph 修正数据目录的绝对路径。
TACO 的工具 ID 会规范为三位数；同一目录出现多个 PKL 会报错，避免任意选取。
还可直接指定公共 PKL，适用于非 canonical 文件或没有 `new_obj` 的评估：

```yaml
semanticContactEval:
  contactRequiredMaskFile: null
  contactRetargetRoot: /absolute/path/to/PhysGraph/data/retargeting_contact
  contactMetadataPaths:
    "cs_plate_0917023": /absolute/path/to/common_corrected.pkl
```

共享 JSON 优先于这两种 PKL 配置。显式配置的 PKL 必须存在；PKL 无评估标注或 canonical 路径缺失时，才使用参考推断。
每份计划都会打印来源及应接触帧数，参考回退明确标为 `FALLBACK checkpoint reference tips_distance`。
不同参考可能推断出不同标记，正式比较应提供同一份共享 JSON，或确认公共 PKL 的标注完全一致。

## 接触检测

`collision_surface_proxy` 从右手 URDF 的全部碰撞 link 采样，包含手掌和指节，不限于策略中的指尖/contact_body_names。
采样器来自共同参考的 `URDFCollisionKinematics`，碰撞 origin、mesh scale 和采样算法保持一致。
代码位于 `main/semantic_contact_geometry.py`，独立于会自动注册数据集的 `main.dataset` 包；没有导入修正轨迹或优化器。
实时刚体姿态把碰撞样本从 link 局部变换到世界，再逆变换到工具局部；工具语义点集乘以仿真器 `get_actor_scale` 返回的实际缩放。

一个有效代理事件要求所属 link 的净接触力模长 `>= forceThreshold`，并且样本到红/黑点集的最小距离 `<= contactDistance`。
最近黑区距离 `<= 最近红区距离 + 1e-6` 时保守归黑，否则归红。
事件单位是“一个有效碰撞采样点在一个控制步的状态”，不是 PhysX 精确接触点数量或接触面积。
净接触力可能来自其他物体，几何筛选只能降低这个误差。

## 指标与汇总

每个环境独立累计红黑事件、实际执行片段帧、黑接触帧、有效接触帧，及应接触帧的执行情况。
合规应接触帧要求 `contact_required AND has_red AND NOT has_black`；同一帧同时有红黑接触仍归为黑区违规。

| 指标 | 定义 |
| --- | --- |
| `operation_completion_ratio` | 实际执行片段帧总数 / 完整计划帧总数 |
| `planned_semantic_contact_coverage` | 已完成合规应接触帧总数 / 完整计划应接触帧总数 |
| `black_violation_frame_ratio` | 实际黑接触帧总数 / 实际执行片段帧总数 |
| `semantic_task_success_rate` | 本批次语义成功数 / 本批次全部结束 episode 数 |

一个 episode 语义成功需要同时满足：原任务成功、从参考帧 0 连续执行到完整终点、有效接触非空、
有效接触帧数 `>= minContactFrames`、原红接触事件占比 `>= minRedContactRatio`、
黑接触事件占比 `<= maxBlackContactRatio`、黑区违规帧比例 `<= maxBlackViolationFrameRatio`，
且计划应接触标记非空、`planned_semantic_contact_coverage >= minPlannedSemanticContactCoverage`。
随机初始化（即使随机到 0）、非零起点、跳帧、回退或只执行截短片段不计完整片段成功，但仍可统计实际接触。

前两个指标的完整计划分母不会因提前失败缩短。正常接近、释放等未要求接触的帧不进入应接触分母。
提前失败统计实际执行部分并计入成功率分母。无接触时辅助红黑占比为 `NA`；有执行帧时黑区违规比例为 0。
有应接触计划但完全无接触时覆盖率为 0；没有任何应接触标记时覆盖率为 `NA`，语义成功仍失败，应检查标注。
没有执行片段帧时黑区违规比例也为 `NA`。
红黑事件占比分别为 `sum(N_R)/sum(N_R+N_B)`、`sum(N_B)/sum(N_R+N_B)`，保留为辅助统计。
所有比例都不使用原 `steps` 作分母；后者可能包含初始化参考索引。
并行环境只汇总当前结束的环境，按事件/帧总数加权，不直接平均 episode 的比例。
复位只清空相应环境，重复采集同一参考帧或已结束环境不会重复计数。

错误与未完成计划分别记录：

- `missing_contact_frames`：已执行且应接触，但红黑区都没有接触。
- `black_required_frames`：已执行应接触帧中有黑接触，保留在终止快照与汇总对象中。
- `unexecuted_required_frames`：尚未执行的应接触帧，包含提前失败、非零起点和跳帧造成的缺失。

每个 episode 的快照都满足：

```text
planned_required_frames = compliant_required_frames + black_required_frames
                        + missing_contact_frames + unexecuted_required_frames
```

未执行帧不会被虚构成实际黑接触。

## 配置与日志

以下初始阈值全部位于 `task.env.semanticContactEval`，正式比较前需固定；训练期间不使用这些阈值。

| 配置 | 默认值 |
| --- | --- |
| `forceThreshold` | 1.0 |
| `contactDistance` | 0.01 米 |
| `pointsPerLink` | 32 |
| `envChunkSize` | 16 |
| `minRedContactRatio` | 0.90 |
| `maxBlackContactRatio` | 0.05 |
| `maxBlackViolationFrameRatio` | 0.05 |
| `minContactFrames` | 3 |
| `minPlannedSemanticContactCoverage` | 0.80 |
| `expectedContactDistance` | 0.03 米（仅参考回退） |
| `contactRequiredMaskFile` | null |

日志在原 `reward`、`steps` 后追加四项指标与辅助统计，例如：

```text
reward: 454.82 steps: 36.0 operation_completion_ratio: 0.1200 planned_semantic_contact_coverage: 0.1500 black_violation_frame_ratio: 0.0000 semantic_task_success_rate: 0.0000 semantic_sr_total: 0.0000 eval_frames: 36 complete_episodes: 0/1 no_contact_episodes: 0 planned_frames: 300 required_frames: 200 compliant_required_frames: 30 missing_contact_frames: 6 unexecuted_required_frames: 164 red_contact_ratio: 1.0000 black_contact_ratio: 0.0000
```

`semantic_sr_total` 是本次 player 运行以来所有已结束 episode 的累计语义成功率，`eval_frames` 是当前结束批次的实际帧数之和。
`complete_episodes` 是从 0 连续到完整终点的数量（原任务是否成功另行要求），`no_contact_episodes` 是当前批次无接触数量。
`planned_frames`、`required_frames` 分别是本批次完整计划帧数、完整计划应接触帧数之和。
辅助红黑比例位于日志末尾，不再占用四项主指标的位置。
所有有效比例均为 0–1。关闭评估后只打印原 `reward`、`steps`。

## 验证和结论边界

CPU 回归测试：

```bash
PYTHONDONTWRITEBYTECODE=1 python -m unittest discover -s tests -v
```

测试覆盖力/距离边界、保守黑区分类、坐标变换与缩放、全碰撞 link、确定性采样、终止帧与重复采集、提前失败、无接触、
随机/非零/截短/不连续片段、并行选择性复位、快照、事件加权、批次/累计成功率，以及真实环境/player 方法在无 Isaac Gym 的模拟状态下的集成行为。
新增测试覆盖固定计划分母、接近/释放排除、应接触帧分类守恒、空标记、覆盖率下限、JSON 优先级/格式、源意图局部偏移、旧 PKL 兼容和 canonical 查找。
GPU 仿真仍需在 NVIDIA 驱动可用的环境恢复实际 checkpoint 验证。

实际仿真前还需检查所选工具的红黑标注；标注缺失或不完整时，初始化会明确报错。
CPU 测试不替代真实物理 rollout。

本次验证：33 项 CPU 回归测试通过，Python 3.8 语法、配置默认值和 `git diff --check` 检查通过。
实际 Inspire、Shadow、Allegro、Artimano 右手 URDF 分别采样 416、800、672、512 个点，均有限且确定。
碰撞采样相关函数/类与本地 PhysGraph `contact_trajectory_retarget.py` 的 AST 一致。
本次调整只修改环境的测试初始化；原 reward、终止、观测、动作和复位方法的 AST 一致，player 与碰撞采样器文件保持原样。
已读取本地真实旧版修正 PKL，确认抓握区间可正确裁剪到完整有效计划。
本次环境 NVIDIA 驱动不可用，尚未实跑真实 checkpoint。

各方法必须固定相同工具、机器人、标注/检测器/采样数、物理参数及阈值；各自保留训练时的参考轨迹与成功/失败标准。
所以 `semantic_task_success_rate` 表示“该方法原任务成功 + 共同语义接触约束合规”的联合结果。
不同参考可能改变原任务判定，不能将此指标称为完全统一的功能任务成功率。
计划覆盖率同时受执行长度和接触质量影响，应结合完成比例解释。
