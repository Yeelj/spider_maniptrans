# Retarget 轨迹可视化

在仓库根目录、已配置 Isaac Gym 的 `maniptrans` 环境中运行
`scripts/visualize_taco_retarget.py`。脚本根据 `--data_idx` 自动选择 TACO 或
OakInk-V2，显示双手、右手物体、左手物体和桌面。

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
