# Graph 3.0 OpenClaw 部署说明

## 当前状态

OpenClaw 的知识域 `wiki_full` 默认通过 `scripts/kb_search` 使用 Graph 3.0。Graph 3.0 返回可审计的 EvidencePack，由 OpenClaw 当前上层模型完成唯一一次最终回答。

以下域保持兼容路径：

- `--memory`：Cognee memory 域
- `--github`：Cognee GitHub 项目域
- `--all`：Cognee 跨数据集兼容查询
- 显式 `NEUROGRAPH_BACKEND=cognee`：全量回滚

## 运行目录

- OpenClaw 运行脚本：`/Users/wangcheng/.openclaw/workspace/projects/neurograph`
- Graph 3.0 开发 checkout：`/Users/wangcheng/Projects/neurograph-graph-3.0`
- 全量存储：`/Users/wangcheng/Projects/neurograph-graph-3.0/data/graph3-openclaw-full`
- 回滚备份：`/Users/wangcheng/.openclaw/backups/neurograph-graph3-deploy-20260815`

全量库当前包含 249 个来源、2,161 个观测；向量索引使用本地 Qwen3-Embedding-0.6B、1024 维。运行时默认不自动切换本地模型，以避免影响 OpenClaw 常驻视觉模型。若要显式启用语义向量查询，可设置：

```bash
export NEUROGRAPH_GRAPH3_EMBEDDING_ENDPOINT=http://127.0.0.1:8000/v1/embeddings
export NEUROGRAPH_GRAPH3_EMBEDDING_MODEL=mlx-community/Qwen3-Embedding-0.6B-4bit-DWQ
```

## 验证与回滚

```bash
scripts/kb_search "自适应放疗的实时计划流程" 6 --graph-evidence --json
NEUROGRAPH_BACKEND=cognee scripts/kb_search "自适应放疗的实时计划流程" 6 --graph-evidence --json
```

Graph 3.0 不接受旧的 `--answer` 模式，避免后端再次调用回答模型；无上层模型时使用 `--answer-draft`。
