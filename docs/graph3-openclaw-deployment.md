# Graph 3.0 OpenClaw 部署说明

## 当前状态

OpenClaw 的知识域 `wiki_full` 默认通过 `scripts/kb_search` 使用 Graph 3.0。Graph 3.0 返回可审计的 EvidencePack，由 OpenClaw 当前上层模型完成唯一一次最终回答。

以下域保持兼容路径：

- `--memory`：Cognee memory 域
- `--github`：Cognee GitHub 项目域
- `--all`：Cognee 跨数据集兼容查询
- 显式 `NEUROGRAPH_BACKEND=cognee`：全量回滚

## 运行目录与安装

Graph 3.0 与 OpenClaw adapter 位于同一个仓库 checkout。首次安装：

```bash
bash scripts/setup_graph3.sh
```

默认存储为仓库内 `data/graph3-openclaw-full`（该目录被 Git 忽略）。已有外部语料时，通过 `NEUROGRAPH_GRAPH3_STORAGE` 指定，不要把运行数据提交到仓库。

全量库当前包含 249 个来源、2,161 个观测；向量索引使用本地 Qwen3-Embedding-0.6B、1024 维。

`kb_search` 默认尝试本机 8003 端口的 embedding 服务，并默认开启 FSRS。embedding 服务不可用时，Graph3 会保留词法、数值、实体、图路径和 ZenBrain 检索；跨电脑部署时建议按目标机器显式配置：

```bash
export NEUROGRAPH_GRAPH3_EMBEDDING_ENDPOINT=http://127.0.0.1:8003/v1/embeddings
export NEUROGRAPH_GRAPH3_EMBEDDING_MODEL=mlx-community/Qwen3-Embedding-0.6B-4bit-DWQ
```

如果目标机器没有 embedding 服务：

```bash
export NEUROGRAPH_GRAPH3_DISABLE_EMBEDDING=1
```

如果目标机器没有 OpenClaw 的 ZenBrain Node 依赖：

```bash
export NEUROGRAPH_GRAPH3_FSRS=0
```

作答反馈通过以下脚本回写，`N` 表示最多记录前 N 条证据及对应的 claim、图路径和关系：

```bash
scripts/g3fb selected /tmp/pack.json 3
scripts/g3fb cited /tmp/pack.json 3
scripts/g3fb corrected /tmp/pack.json 3
```

## 验证与回滚

```bash
scripts/kb_search "自适应放疗的实时计划流程" 6 --graph-evidence --json
NEUROGRAPH_BACKEND=cognee scripts/kb_search "自适应放疗的实时计划流程" 6 --graph-evidence --json
```

Graph 3.0 不接受旧的 `--answer` 模式，避免后端再次调用回答模型；无上层模型时使用 `--answer-draft`。
