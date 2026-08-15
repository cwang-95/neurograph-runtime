# OpenClaw Graph 3.0 适配器

当前 `main` 已将 Graph 3.0 接入 OpenClaw，知识域 `knowledge/wiki_full` 默认走
Graph3。Cognee 仍保留为兼容和回滚后端。

默认调用：

```bash
scripts/kb_search "自适应放疗" 4 --graph-evidence --json
```

显式指定 Graph 3.0（用于固定运行目录或调试）：

```bash
NEUROGRAPH_BACKEND=graph3 \
NEUROGRAPH_GRAPH3_ROOT=/path/to/neurograph-runtime \
NEUROGRAPH_GRAPH3_STORAGE=/path/to/neurograph-runtime/data/graph3-openclaw-full \
scripts/kb_search "自适应放疗" 4 --graph-evidence --json
```

Graph 3.0 后端返回 JSON EvidencePack，供上层模型回答；回答完成后再把实际引用的
ID 通过 `scripts/g3fb` 回写。FSRS 默认开启；需要关闭时设置：

```bash
NEUROGRAPH_GRAPH3_FSRS=0
```

适配器默认优先使用 Graph 3.0 根目录下的 `.venv/bin/python`（Windows 为
`.venv/Scripts/python.exe`），这样可直接使用 Graph 3.0 的 ANN 依赖；若该环境不存在，才回退到旧 Cognee 环境。也可以显式指定：

```bash
NEUROGRAPH_GRAPH3_PYTHON=/path/to/graph3/.venv/bin/python
```

当前语料已建立 embedding。适配器默认尝试本机 embedding 服务；端点、模型和 ANN
路径均可通过环境变量覆盖：

```bash
NEUROGRAPH_GRAPH3_EMBEDDING_ENDPOINT=http://127.0.0.1:8003/v1/embeddings \\
NEUROGRAPH_GRAPH3_EMBEDDING_MODEL=mlx-community/Qwen3-Embedding-0.6B-4bit-DWQ \\
NEUROGRAPH_GRAPH3_ANN_INDEX=/path/to/graph3-ann \\
NEUROGRAPH_GRAPH3_ANN_BACKEND=auto
```

默认模型为 `mlx-community/Qwen3-Embedding-0.6B-4bit-DWQ`。如果部署机器没有该服务，
可显式关闭向量路由；Graph 3.0 仍保留词法、数值、实体和图路径结果：

```bash
NEUROGRAPH_GRAPH3_DISABLE_EMBEDDING=1 \\
scripts/kb_search "自适应放疗" 4 --graph-evidence --json
```

未设置 `NEUROGRAPH_GRAPH3_DISABLE_EMBEDDING=1` 时，适配器会按默认端点尝试连接；服务
不可用时也会回退到非向量路线，不会让整个查询失败。

如果调用方需要一个不调用 DeepSeek 的确定性回答草稿，可显式使用：

```bash
NEUROGRAPH_BACKEND=graph3 \
NEUROGRAPH_GRAPH3_ROOT=/path/to/neurograph-runtime \
NEUROGRAPH_GRAPH3_STORAGE=/path/to/neurograph-runtime/data/graph3-openclaw-full \
scripts/kb_search "GeoDose 的机制和结果" 6 --answer-draft --json
```

该模式返回 `AnswerDraft`，状态为 `answer`、`follow_up` 或 `conflict`，并附带按
机制/量化结果分组的证据、Observation ID 和原始定位。它只组装 EvidencePack，
不生成新的事实；上层模型若继续润色，也只能使用返回的证据和 citations。

回滚只需设置 `NEUROGRAPH_BACKEND=cognee`，无需修改 OpenClaw 稳定配置：

```bash
NEUROGRAPH_BACKEND=cognee \\
scripts/kb_search "自适应放疗" 4 --graph-evidence --json
```

Graph 3.0 仍不接管旧的 `--answer` 模式，避免和上层 DeepSeek 重复生成答案；需要
确定性回答草稿时使用新增的 `--answer-draft`。设置 `NEUROGRAPH_BACKEND=cognee`
即可回滚到 Cognee。

接入前可运行灰度冒烟检查，不修改稳定配置：

```bash
scripts/graph3_adapter_smoke
```

它会实际验证 AnswerDraft、EvidencePack、追问、重复回答保护和环境变量回滚路径。
