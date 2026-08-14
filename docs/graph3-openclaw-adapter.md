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

回滚只需取消 `NEUROGRAPH_BACKEND=graph3`，无需修改 OpenClaw 稳定配置。

该适配暂不支持 `--answer`，避免 Graph 3.0 后端和上层 DeepSeek 重复生成答案。
