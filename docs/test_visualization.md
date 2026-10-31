# Test 双手仿真显示配置

适用于 `task=ResDexHand side=BiH`，兼容 legacy 和 SPIDER reference。
这些选项只控制显示，不改变策略观测、奖励、参考轨迹或模型权重，无需重新训练。
单手任务和独立的 retarget 可视化脚本不读取这组配置。

在现有 test 命令末尾追加以下参数即可，无需 Hydra 的 `+` 前缀：

```bash
  'task.env.visualization.rightHandColor=[0.35,0.75,0.95]' \
  'task.env.visualization.leftHandColor=[0.75,0.45,0.95]' \
  'task.env.visualization.objectColor=[0.20,0.70,0.35]' \
  'task.env.visualization.toolColor=[1.00,0.45,0.08]' \
  task.env.visualization.showReferenceSkeleton=false \
  task.env.visualization.showContactHighlight=false
```

颜色分别对应右手 `dexhand_r`、左手 `dexhand_l`、左手侧目标物体
`manip_obj_lh` 和右手侧工具 `manip_obj_rh`。替换工具后仍按 actor 角色着色。
RGB 必须是三个 `[0, 1]` 范围内的有限数值；无效配置在创建仿真前报错。

默认值保持原显示行为：

| 配置（均位于 `task.env.visualization`） | 默认值 | 含义 |
| --- | --- | --- |
| `rightHandColor` / `leftHandColor` | `null` | 保留手部资产原颜色；接触高亮另行控制 |
| `objectColor` / `toolColor` | `null` | 保留目标物体 / 工具资产原颜色 |
| `showReferenceSkeleton` | `true` | 在 `headless=false` 时创建关节小球并绘制绿色骨架连线 |
| `showContactHighlight` | `true` | viewer 模式中接触部位为粉红色，其余部位为指定手色；未指定则沿用原白色 |
| `preserveTextures` | `false` | 为指定了自定义颜色的 actor 用白色纹理覆盖原纹理，保证统一设色；为 `true` 时保留原纹理并乘上指定 RGB |

`showContactHighlight=false` 时不进行逐帧接触染色。若手色为 `null`，保留原资产材质；
如果希望手部固定为白色，应显式指定 `[1,1,1]`。
`preserveTextures` 对颜色为 `null` 的 actor 不生效，不会修改原资产文件。
灯光和阴影仍会影响画面中实际显示的颜色。

骨架开关同时控制关节小球的创建、相关状态索引和逐帧更新，以及绿色连线绘制。
关闭骨架不会关闭接触高亮；二者可以独立组合。SPIDER 模式下，这套显示使用机器人参考
关节位置，兼容字段虽名为 `mano_joints`，并非原始 MANO 人手骨架。

actor 颜色同时作用于 viewer 和 camera sensor，因此 `capture_video=true` 时也可自定义颜色。
保持原 headless 行为：`headless=true` 不创建参考关节小球，也不执行逐帧接触高亮。
Isaac Gym 的 `add_lines` 只进入 viewer，不进入 camera sensor，
因此 `capture_video` 不包含绿色骨架连线；`headless=false` 时创建的关节小球则可被相机拍到。
本配置没有改变这一录制方式。

CPU 回归验证：

```bash
/home/user/miniconda3/envs/spider/bin/python -m unittest discover -s tests -q
```
