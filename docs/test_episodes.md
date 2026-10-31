# 固定数量的测试 episode

在原有能够恢复 checkpoint 的 test 命令末尾追加：

```bash
test_episodes=100 num_envs=4 randomStateInit=false
```

`test_episodes` 可以设为任意正整数，例如 1、25、100。默认 `null` 保持原来的持续运行和批次平均日志。
`num_envs` 是并行环境数量；episode 上限是所有环境合计完成的数量，成功、失败和提前结束都计入。
同一步多个环境结束时按 `env_id` 升序记录，最后一批只取剩余名额，最终恰好记录指定数量。
物理仿真仍然并行执行，同一步超出名额的结束结果不计入日志、最佳记录或累计指标。

每个结束 episode 独立打印 `episode`、`env_id`、`reward`、`steps`，启用语义评估时追加现有全部语义指标。
如需使用语义指标排序，在双手任务的原命令中同时指定：

```bash
task.env.semanticContactEval.enabled=true
```

原有工具红黑标注等评估前提仍然适用。未开启评估不会自动创建评估器。

最佳记录按以下顺序比较原始数值（不使用打印时的舍入值）：

1. `steps` 越大越好。
2. steps 相同时，`operation_completion_ratio` 越大越好。
3. 前两项相同时，`planned_semantic_contact_coverage` 越大越好。
4. 前三项相同时，`black_violation_frame_ratio` 越小越好。
5. 四项全部相同，保留最早记录的 episode。

缺失或 `NA` 指标在对应比较项中排在有效值之后；未开启语义评估时只比较 steps，同值保留第一条。
`steps` 的现有计算方式不变；随机初始化可能使其包含起始参考帧索引，比较从头执行长度时使用 `randomStateInit=false`。

输出保存在当前测试目录 `dumps/test_<实验名>__<时间>/`：

- `episodes.log`：每个 episode 一行的完整文本日志。
- `episodes.jsonl`：每个 episode 一条结构化数据，包括原始数值、完整日志和可用的语义计数。
- `summary.json`：目标/完成数量、运行状态、平均 reward/steps、最大 steps、最佳记录和语义总统计。

现有 `config.yaml`、`spider_references.json` 保存方式不变。日志逐条刷盘；已有同名 episode 日志不会被覆盖。
结束后先打印 `[test_summary]`，如有语义指标再打印 `[test_semantic_summary]`，最后打印 `[best_steps]` 和最佳记录的完整原始日志。
最佳记录的 `semantic_sr_total` 是记录该 episode 时的累计值；最终累计值位于总统计中。
总统计按原始事件数、帧数加权，不直接平均各个 episode 的比例。
若关闭 player 的 `print_stats`，仍保存三个文件，但不打印 episode 或汇总行。

计数与 `save_rollouts` 独立，正常统计不需要保存轨迹。
同时启用轨迹保存时，目录仍使用原来的 `dump_` 前缀，并保留原有短轨迹过滤、初始化预热跳过和成功轨迹筛选。
`test_episodes` 决定何时结束测试；`num_rollouts_to_save` 仅限制保存数量，旧的 `num_rollouts_to_run` 不再提前结束有限次测试。
最终选中的终止轨迹在退出前按原有对齐方式写入 HDF5（仍受保存数量及筛选条件限制），不额外运行物理步。
同时开启视频录制时，到达成功视频数量上限后停止录制，测试继续到 episode 上限。

普通 Python 异常或 Ctrl+C 会保存已完成部分，并标记 `status=incomplete` 和实际完成数量，然后继续抛出原异常。
进程被强制终止或原生仿真崩溃时，已刷盘的 episode 行仍可读取，最终汇总不保证生成。
