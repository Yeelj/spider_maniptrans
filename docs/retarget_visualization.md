# Retarget 轨迹可视化

在仓库根目录运行 `scripts/visualize_taco_retarget.py`。脚本根据 `--data_idx`
自动选择 TACO 或 OakInk-V2，显示双手、右手物体、左手物体和桌面。

## 当前机器的 physgraph 环境

当前 Isaac Gym 位于 `/home/user/ylj/isaacgym/python`，已安装在
`/home/user/miniconda3/envs/physgraph` 的 Python 3.8 中，无需重新安装。
项目提供启动脚本，固定使用该环境的 Python、动态库、Ninja 和系统 GCC/G++ 11。
即使已经 `conda activate physgraph`，也可以直接运行：

```bash
cd /home/user/ylj/spider_maniptrans
bash scripts/visualize_spider_retarget.sh --data_idx '9fc3e@0' --new_obj 'kni@1'
```

脚本自动添加 `--reference_source spider`，其他参数原样传给 Python 脚本。
可用 `RETARGET_ENV`、`RETARGET_CC`、`RETARGET_CXX` 显式指定其他环境和编译器。

手动启动时，同样需要明确解释器、动态库、Ninja 和编译器：

```bash
cd /home/user/ylj/spider_maniptrans
export RETARGET_ENV=/home/user/miniconda3/envs/physgraph
export PATH="$RETARGET_ENV/bin:$PATH"
export LD_LIBRARY_PATH="$RETARGET_ENV/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export CC=/usr/bin/gcc-11
export CXX=/usr/bin/g++-11

"$RETARGET_ENV/bin/python" scripts/visualize_taco_retarget.py --help
```

终端提示符中的 `(physgraph)` 不能单独证明 `python` 指向该环境。
若出现 `ModuleNotFoundError: No module named 'isaacgym'`，在报错的同一终端检查
`type -a python`、`python -c 'import sys; print(sys.executable)'` 和
`python -m pip show isaacgym`。`libpython3.8.so.1.0` 缺失应检查上面的
`LD_LIBRARY_PATH`；`Ninja is required` 应检查 `PATH`，当前环境已安装两者。

当前机器激活 `physgraph` 时，Conda 会设置
`CXX=/home/user/miniconda3/envs/physgraph/bin/x86_64-conda-linux-gnu-c++`。
该编译器的 sysroot 缺少 `crypt.h`，从空缓存编译 `gymtorch` 时可复现
`Python.h:44:10: fatal error: crypt.h: No such file or directory`，最后汇总为
`RuntimeError: Error building extension 'gymtorch'`。上面的系统编译器设置解决此问题。
已有可用缓存时可能暂时正常，不能只用缓存命中判断编译环境正确；无需删除整个
PyTorch 扩展缓存，也无需重装 Isaac Gym 或 PyTorch。

## SPIDER 生成的轨迹

显式添加 `--reference_source spider`，复用训练的 SPIDER loader，读取
`data/retargeting_spider_native/<工具>/<数据集>/training/<任务>/manifest.json`
及其指定的 NPZ。路径、校验和、DOF 顺序、坐标系、完整帧数与物体资产均由
现有 loader 验证，不再读取原始 `data/retargeting` PKL 或对轨迹再次变换。

```bash
# 接上面的环境设置：显示已有 9fc3e@0 / kni@1 的 SPIDER 结果
"$RETARGET_ENV/bin/python" scripts/visualize_taco_retarget.py \
  --reference_source spider --data_idx '9fc3e@0' --new_obj 'kni@1'

# TACO 的 SPIDER 结果
"$RETARGET_ENV/bin/python" scripts/visualize_taco_retarget.py \
  --reference_source spider --data_idx sk_box_1104218 --new_obj 045
```

SPIDER 模式下，`--new_obj` 是包的工具命名空间，例如 `kni@1` 或 `045`；
左右手各自的物体资产、缩放和轨迹取自该包。默认
`--object_reference spider_rollout` 显示 SPIDER 物理回放中实际记录的物体位置。
添加 `--object_reference demonstration` 可显示训练默认使用的已对齐示范物体目标，
手部始终来自 SPIDER。`--reference_root` 可指定其他训练包根目录；相对路径以仓库
根目录为准。播放速度由 `--playback_fps` 控制，不裁剪或重采样包内帧。

完整但质量失败的包也可查看，终端打印其真实 `validation` 状态。当前
`9fc3e@0 / kni@1` 包为 1294 帧，标记为 `failed`，查看轨迹不会改变此状态。

无桌面时可添加 `--headless`，不创建窗口，按顺序执行一遍后退出：

```bash
"$RETARGET_ENV/bin/python" scripts/visualize_taco_retarget.py \
  --reference_source spider --data_idx '9fc3e@0' --new_obj 'kni@1' --headless
```

## 原始 MANO retarget

默认 `--reference_source legacy` 使用原始 retarget 文件及数据集资产。
OakInk-V2 的此模式要求原生物体 ID，`kni@1` 这样的 SPIDER 工具命名空间
应使用上面的 SPIDER 模式。

```bash
# TACO：原始工具
python scripts/visualize_taco_retarget.py --data_idx db_teapot_0930049 --dexhand artimano

# TACO：替换右手工具
python scripts/visualize_taco_retarget.py --data_idx db_teapot_0930049 --dexhand artimano --new_obj 004

# OakInk-V2：原始物体（9b728 为序列 hash 前五位，0 为阶段号）
python scripts/visualize_taco_retarget.py --data_idx '9b728@0' --dexhand artimano

# OakInk-V2：替换右手物体
python scripts/visualize_taco_retarget.py --data_idx '9b728@0' --dexhand artimano --new_obj 'O02@0015@00004'
```

左右手都需预先完成 retarget，且必须使用相同的 dexhand 和阶段。TACO 的
PKL 位于 `data/retargeting/TACO/mano2<dexhand>_{rh,lh}/<data_idx>.pkl`；
OakInk-V2 的 PKL 位于
`data/retargeting/OakInk-v2/mano2<dexhand>_{rh,lh}/<完整标注文件名去掉.pkl>@<阶段号>.pkl`。
OakInk-V2 目前要求该阶段同时包含左右手。

`--new_obj` 只切换右手物体资产，保留原始手轨迹、左右物体轨迹和左手物体。
OakInk-V2 的训练和测试也可使用同一物体 ID，通过 Hydra 参数
`new_obj=O02@0015@00004` 指定（`side=BiH` 或 `side=RH`）。
不传此参数时使用原物体。替换物体与原物体沿用同一个局部坐标系，脚本不进行
自动配准；OakInk-V2 的物体缩放沿用训练环境中的 `oakink2_obj_scale` 配置。

TACO 的替换资产可用 `scripts/preprocess_taco_assets.py` 预处理。
OakInk-V2 使用其原生物体 ID，读取
`object_preview/align_ds/<物体ID>/` 下唯一的 `.obj` 或 `.ply` 文件，以及
`coacd_object_preview/align_ds/<物体ID>/` 下同名的 CoACD mesh 和 URDF；
准备方法见 README 中 OakInk-V2 的 prerequisites。

`--data_dir` 可指定数据集根目录；默认分别是 `data/TACO` 和 `data/OakInk-v2`。
脚本使用现有 dataset loader，因此仍需要 README 所列的 MANO/SMPL-X 等依赖。

默认以 60 FPS 循环回放。`Space` 暂停/继续，`R` 从起始帧重播，`Esc` 退出。
可用 `--once` 播放一遍后退出、`--start_frame 100` 指定起始帧、
`--playback_fps 30` 减慢播放。回放逐帧写入手和物体状态，用于检查 retarget
轨迹与替换几何的关系。

## 独立设置颜色和显示对象

legacy / SPIDER、TACO / OakInk-V2 均支持以下参数，启动脚本会原样转发：

| 对象 | 颜色参数及默认 RGB | 显示参数（默认 `true`） |
| --- | --- | --- |
| 左手 | `--left_hand_color 0.75,0.45,0.95` | `--show_left_hand true` |
| 右手 | `--right_hand_color 0.35,0.75,0.95` | `--show_right_hand true` |
| 目标物体（左手侧） | `--object_color 0.2,0.7,0.35` | `--show_object true` |
| 工具（右手侧） | `--tool_color 1.0,0.45,0.08` | `--show_tool true` |

颜色使用逗号分隔的三个有限数值，每个分量必须在 `[0,1]` 内。
显示参数明确接受 `true` 或 `false`，大小写均可；`false` 表示隐藏。
四个显示开关独立，物体与对应手的显示互不依赖。
默认保留四者全部显示、原有 RGB 数值、桌面及播放控制。

例如只查看右手和工具：

```bash
bash scripts/visualize_spider_retarget.sh \
  --data_idx '9fc3e@0' --new_obj 'kni@1' \
  --right_hand_color '0.35,0.75,0.95' \
  --tool_color '1.00,0.45,0.08' \
  --show_left_hand false --show_object false
```

仅显示目标物体和工具，可再加 `--show_right_hand false`，并将
`--show_object` 设为 `true`；四者都设为 `false` 时仍显示桌面。
相机中心由可见对象的完整轨迹范围确定，四者都隐藏时取桌面位置。

默认 `--preserve_textures false`，用一张共享白色纹理覆盖四类对象的原纹理，
确保设色不受导入纹理影响；灯光和阴影仍影响实际画面颜色。
此设置同时修正旧版 SDK 在原纹理干扰下 RGB 参数与实际手色不一致的问题。
添加 `--preserve_textures true` 则保留原纹理，并与指定 RGB 相乘。
资产文件和轨迹文件均不会因此改写。

显示开关在启动时生效。隐藏对象不创建 actor，手部 DOF 索引按实际场景查询；
两只手都隐藏时跳过 DOF 张量读写。仍按原流程加载并校验完整双手 retarget 数据，
隐藏左手不代表可以省略左手输入文件。坐标变换、替换物体缩放、播放帧数与速度、
`--object_reference` 的物体轨迹选择保持原逻辑。
`--headless` 保持原语义：不创建窗口，关闭图形设备，顺序回放一遍后退出。

CPU 回归（包含全部 16 种显示组合）：

```bash
/home/user/miniconda3/envs/spider/bin/python -m unittest discover -s tests -q
```
