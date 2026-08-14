# OpenClaw Graph 3.0 可选后端

当前适配分支只增加开关，不改变稳定默认行为。

默认仍然是 Cognee：

```bash
scripts/kb_search "自适应放疗" 4 --graph-evidence --json
```

显式开启 Graph 3.0：

```bash
NEUROGRAPH_BACKEND=graph3 \
NEUROGRAPH_GRAPH3_ROOT=/Users/wangcheng/Projects/neurograph-graph-3.0 \
NEUROGRAPH_GRAPH3_STORAGE=/Users/wangcheng/Projects/neurograph-graph-3.0/data/graph3 \
scripts/kb_search "自适应放疗" 4 --graph-evidence --json
```

Graph 3.0 后端返回 JSON EvidencePack，供上层模型回答；回答完成后再把实际引用的
ID 通过 `scripts/graph3_feedback` 回写。需要读取 FSRS 弱先验时增加：

```bash
NEUROGRAPH_GRAPH3_FSRS=1
```

如果调用方需要一个不调用 DeepSeek 的确定性回答草稿，可显式使用：

```bash
NEUROGRAPH_BACKEND=graph3 \
NEUROGRAPH_GRAPH3_ROOT=/Users/wangcheng/Projects/neurograph-graph-3.0 \
NEUROGRAPH_GRAPH3_STORAGE=/Users/wangcheng/Projects/neurograph-graph-3.0/data/graph3 \
scripts/kb_search "GeoDose 的机制和结果" 6 --answer-draft --json
```

该模式返回 `AnswerDraft`，状态为 `answer`、`follow_up` 或 `conflict`，并附带按
机制/量化结果分组的证据、Observation ID 和原始定位。它只组装 EvidencePack，
不生成新的事实；上层模型若继续润色，也只能使用返回的证据和 citations。

回滚只需取消 `NEUROGRAPH_BACKEND=graph3`，无需修改 OpenClaw 稳定配置。

Graph 3.0 仍不接管旧的 `--answer` 模式，避免和上层 DeepSeek 重复生成答案；需要
确定性回答草稿时使用新增的 `--answer-draft`。取消 `NEUROGRAPH_BACKEND=graph3`
即可回滚到 Cognee。

接入前可运行灰度冒烟检查，不修改稳定配置：

```bash
scripts/graph3_adapter_smoke
```

它会实际验证 AnswerDraft、EvidencePack、追问、重复回答保护和环境变量回滚路径。
