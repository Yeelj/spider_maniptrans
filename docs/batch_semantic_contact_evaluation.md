# 批量评测 checkpoint 并导出 Excel

从项目根目录运行 `scripts/batch_semantic_contact_eval.py`。它使用当前 Python 环境，
按 `--ids-file` 中的顺序为每条数据单独执行 `python -m main.rl.train`，
所有数据共用命令行指定的 checkpoint、替换物体和参考数据集。
需先激活原测试命令可用的 Isaac Gym 环境；Excel 导出仅使用 Python 标准库，无需安装额外依赖。

ID 文件为 UTF-8 文本，每行一个 ID，支持空行、`#` 注释和 UTF-8 BOM；保留原顺序及重复项。例如：

```text
# ids.txt
sk_box_1104218
# 在后面逐行添加其他数据 ID
```

```bash
cd /home/user/ylj/spider_maniptrans
python scripts/batch_semantic_contact_eval.py \
  --ids-file ids.txt \
  --checkpoint runs/SPIDER_Taco_sk_box_1104218_045_artimano__10-23-10-13-30/nn/last_SPIDER_Taco_sk_box_1104218_045_artimano_ep_10000_rew__141.53_.pth \
  --new-obj 045 \
  --reference-dataset TACO \
  --output add_data_evaluation/taco_sk_box_045.xlsx
```

OakInk-v2 示例：

```bash
python scripts/batch_semantic_contact_eval.py \
  --ids-file add_data_evaluation/OakInk/kind_1_ids.txt \
  --checkpoint runs/SPIDER_OakInk_9b728@0_spoon@1_artimano__10-26-19-08-20/nn/last_SPIDER_OakInk_9b728@0_spoon@1_artimano_ep_10000_rew__8167.7_.pth \
  --new-obj spoon@1 \
  --reference-dataset OakInk-v2 \
  --new-data-root /srv/spider_maniptrans/data \
  --output add_data_evaluation/OakInk/spoon1_results.xlsx
```

`--new-data-root PATH`（别名 `--new-data`）统一映射整个 `data` 根目录，
适用于 `--reference-dataset OakInk-v2` 和 `--reference-dataset TACO`。
PATH 应指向同时包含 `OakInk-v2/`、`TACO/` 子目录的 `data/`，例如：

```text
/srv/spider_maniptrans/data/
  OakInk-v2/
  TACO/
```

脚本默认把现有 manifest 中的两个旧 `data` 根目录都映射到 PATH：

```text
/media/user/89785a2b-45b3-43b5-99ad-a5558e7ad6484/spider_data/PhysGraph/data
/home/user/ylj/spider_repro/spider/data
```

因此旧路径下的 `OakInk-v2/...` 和 `TACO/...` 分别保留相同相对路径，落到新 `data/` 下。
无需修改 manifest。若只迁移本次测试的数据集，可只复制对应的子目录。
如果目标项目的 `data/` 就是新的资产目录，可直接使用 `--new-data-root data`。

其他来源的参考包可以通过 `--old-data-root` 指定 manifest 记录的旧绝对 `data` 根目录，
该选项可重复；显式指定后替代上述默认旧根目录。例如，在原批处理命令中追加：

```bash
  --old-data-root /old/server/project/data \
  --new-data-root /new/server/project/data
```

未传任何新根目录选项时不自动添加映射，继续使用 manifest 原有路径。
相对目标路径基于项目根目录，支持路径中的空格。源路径按 manifest 的字面路径匹配，
无需在新服务器存在，也不会在新服务器解析旧路径的软链接。
映射可与 `--override` 中的其他资产映射合并，相同源根目录以命名选项为准；
更具体的子目录映射由加载器优先匹配。
物体资产映射与 `--reference-root` 独立，后者仍用于定位 `retargeting_spider_native` 训练包。

保留旧选项 `--new-oakink PATH` / `--new-oakink-root PATH` 的原有含义：
PATH 是 `OakInk-v2/` 本身，只映射该数据集。它与 `--new-data-root` 互斥，
统一迁移时请使用新的 `--new-data-root`，无需设置任何旧、新路径环境变量。

`--newObj`、`--referenceDataset` 分别为 `--new-obj`、`--reference-dataset` 的别名。
TACO 数字物体 ID 自动补齐三位，例如 `45` 保存为文本 `045`；OakInk-v2 使用原生物体 ID。
`--ids-file`、`--output`、`--logs-dir` 的相对路径基于运行脚本时的目录；checkpoint、
基础策略 checkpoint 和 reference root 的相对路径基于项目根目录。

默认参数对应原测试命令：

```text
task=ResDexHand side=BiH bimanual_mode=united dexhand=artimano
test=true headless=false num_envs=4 randomStateInit=false rolloutStateInit=false
task.env.referenceSource=spider task.env.datasetType=spider
task.env.referenceRoot=data/retargeting_spider_native
task.env.objectReference=demonstration task.env.allowFailedReference=true
task.env.episodeLength=1500 task.task.randomize=false
rh_base_model_checkpoint=assets/imitator_rh_artimano.pth
lh_base_model_checkpoint=assets/imitator_lh_artimano.pth
task.env.semanticContactEval.enabled=true test_episodes=120
```

可使用 `--test-episodes 120`、`--num-envs 4`、`--headless true`、`--reference-root PATH`、
`--rh-checkpoint PATH`、`--lh-checkpoint PATH` 调整对应参数。
实验名由脚本自动生成，包含批次唯一标记、行号和数据 ID，避免读取历史或其他进程的日志。
每条测试的 stdout/stderr 保存至独立的 `process.log`，终端显示命令、日志路径和最终最佳指标。
每条 ID 完成后，终端还会单独打印一次 `[test_task_summary]`，包含该 ID 全部 N 个回合的 SR
及三项终止帧误差均值 E_t、E_j、E_ft（cm）；该行也保留在 `process.log` 中。
原始精度见对应测试目录的 `summary.json.task_metrics`，统计定义见 [固定数量测试说明](test_episodes.md)。
这些统计不参与最佳 episode 选择，也不修改 `batch_report.json.metrics` 的语义指标。
Excel 在原有五列后追加这四项指标；兼容没有 `task_metrics` 的旧汇总，此时新增四列填为 `NA`。

资产映射、标注网格、共享时间标记等附加 Hydra 参数通过可重复的 `--override` 传入。
例如，针对本机 `sk_box_1104218` / `045`，在上述脚本命令末尾追加：

```bash
  --override '+task.env.assetPathMap={/home/user/ylj/spider_repro/spider/data/TACO/object_models/045:data/spider_physical_assets/TACO/045}' \
  --override '+task.env.semanticContactEval.semanticMeshPaths={sk_box_1104218:data/TACO/object_models/045/045.ply}'
```

批量包含多个 ID 时，在 `semanticMeshPaths` 字典中配置各 ID 对应的标注副本。
具体前提见 [语义接触评估说明](checkpoint_semantic_contact_evaluation.md)。
脚本控制的测试模式、episode 上限、ID、checkpoint、物体、数据集、实验名和输出目录等参数不能通过 `--override` 改写。
追加 `--dry-run` 可只打印所有完整命令，不启动仿真，也不创建或覆盖 Excel 和日志。

Excel 包含以下九列，每条 ID 占一行：

| dataIndices | new_obj | operation_completion_ratio | planned_semantic_contact_coverage | black_violation_frame_ratio | SR | E_t | E_j | E_ft |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| sk_box_1104218 | 045 | 0.95 | 0.80 | 0.02 | 0.75 | 1.23 | 0.88 | 1.10 |

上表数值仅为格式示例。前三个语义指标读取自本次评测 `summary.json` 的 `best_episode`，
对应终端 `[best_steps]` 的完整最佳日志，保留 JSON 原始数值精度。
新增 `SR`、`E_t`、`E_j`、`E_ft` 读取自同一文件的 `task_metrics`，
对应该 ID 全部 N 回合的成功率和三项终止帧误差均值，误差单位为 cm；以数值写入 Excel，保留原始精度。
最佳记录沿用项目现有规则：steps 降序、完成比例降序、计划语义覆盖率降序、黑区违规比例升序；
完全相同则保留最早 episode。规则详见 [固定数量测试说明](test_episodes.md)。
Excel 中有效比例为 0–1 数值；原始值 `null` 写为文本 `NA`，物体 ID 保存为文本以保留前导零。

脚本在每条结束后原子更新 `--output` 文件；再次运行会覆盖该文件。
评测失败、汇总缺失、未完成指定 episode 数、缺少必需语义指标或已有任务汇总无效时，该行七项指标写为 `ERROR`，
继续运行后续 ID，最终退出码为 1；全部成功为 0，Ctrl+C 中断为 130。
尚未运行的行指标为空，中断时保留此前结果。
默认批次日志目录为 `<output文件名去扩展名>_logs/<批次标记>/`，可用 `--logs-dir PATH` 更改父目录。
其中 `batch_report.json` 按输入顺序记录每条命令、状态、失败原因及原始日志路径。
项目原有 `dumps/test_<实验名>__<时间>/` 中的逐 episode 日志和汇总文件继续保留。

CPU 测试（无需 Isaac Gym）：

```bash
PYTHONDONTWRITEBYTECODE=1 python -m unittest discover -s tests -p 'test_batch_semantic_contact_eval.py' -v
```
