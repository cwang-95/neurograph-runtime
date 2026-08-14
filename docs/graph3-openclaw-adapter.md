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

适配器默认优先使用 Graph 3.0 根目录下的 `.venv/bin/python`，这样可直接使用 Graph 3.0 的 ANN 依赖；若该环境不存在，才回退到 `~/.cognee-venv/bin/python`。也可以显式指定：

```bash
NEUROGRAPH_GRAPH3_PYTHON=/path/to/graph3/.venv/bin/python
```

如果 Graph 3.0 的语料已经建立 embedding，可继续通过环境变量启用语义路由；这些变量只透传到底层入口，不改变默认的词法+图检索：

```bash
NEUROGRAPH_GRAPH3_EMBEDDING_ENDPOINT=http://127.0.0.1:8000/v1/embeddings \\
NEUROGRAPH_GRAPH3_EMBEDDING_MODEL=mlx-community/Qwen3-Embedding-0.6B-4bit-DWQ \\
NEUROGRAPH_GRAPH3_ANN_INDEX=/path/to/graph3-ann \\
NEUROGRAPH_GRAPH3_ANN_BACKEND=auto
```

只有设置 `NEUROGRAPH_GRAPH3_EMBEDDING_ENDPOINT` 时才会调用 embedding 服务；未设置或服务不可用时，Graph 3.0 保留词法、数值和图路径结果，不会让整个查询失败。

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
