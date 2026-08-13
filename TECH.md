# 记忆图谱 2.0 — 技术文档（方案① cognee 底座 · 可复现版）

> 落实: 2026-08-11 | 配套 README.md（方案介绍）+ 本文件（**工程可复现手册**）
> 目标: **其他 AI / 开发者能照着本文档从零搭建、复现、维护这套"文献+GitHub项目→知识图谱→自动感知检索"系统**。
> 环境基线: Apple Silicon (M1 Pro, 16GB) / macOS / Python 3.13 / cognee 1.4.2（下文所有路径、版本、命令均实测）。

---

## 0. TL;DR（30 秒看懂）

```
文献(wiki 223篇) + GitHub项目(每周) 
   → [config env] cognee(LLM=DeepSeek, Embedding=DashScope)
   → [feed_wiki.py] add(灌入) + cognify(建认知图谱)
   → [query.py / kb_search] cognee.search(CHUNKS 纯向量) + ZenBrain类脑排序
   → 自动感知: 专业问题先查图谱 → 记忆 → 联网兜底(AGENTS.md 第5条)
```

**两条铁律**:
1. **LLM 必须用 DeepSeek**（`deepseek/deepseek-chat`），绝不用 DashScope 免费 key 当 LLM（一灌库就烧光额度，检索跟着全挂）。
2. **Embedding 用 DashScope**（`text-embedding-v4`，1024维，几乎免费；DeepSeek 无 embedding API）。两者独立 env，互不干扰。

---

## 1. 环境搭建（从零复现）

### 1.1 创建 venv + 安装依赖
```bash
# Python 3.13（用户态 pip, 无 brew）
python3 -m venv ~/.cognee-venv
source ~/.cognee-venv/bin/activate
pip install "cognee==1.4.2" litellm networkx
# 画图谱还要 matplotlib（装到 /tmp/viz_pkgs，不污染 venv，见 5.2）
```

**实测锁定版本**（`pip list` 核对）:
| 包 | 版本 |
|---|---|
| cognee | 1.4.2 |
| litellm | 1.96.0 |
| lancedb | 0.37.1 |
| ladybug | 0.18.2 |
| networkx | 3.6.1 |
| pydantic | 2.13.4 |
| SQLAlchemy | 2.0.51 |

### 1.2 需要的文件（本项目的脚本）
```
projects/neurograph/
├── cognee-bridge/
│   ├── feed_wiki.py      # 灌库+建图
│   ├── query.py          # 检索入口（多数据集 + 标题清洗 + 📚来源标注 + ZenBrain排序）
│   ├── zenbrain_llm.py   # ZenBrain 类脑加权（FSRS/Hebbian）
│   ├── visualize.py      # 图谱可视化
│   └── smoke_test.py     # 冒烟测试
├── scripts/kb_search     # 命令行检索封装
```
> 源码就在本项目，其他 AI 复现时直接照抄这几个文件即可；核心逻辑在下面各节逐文件讲清。

---

## 2. 环境变量（这是最容易出错、也是最关键的部分）

### 2.1 脚本外必须 export（连网 + DashScope key）
```bash
# ① 代理（cognee 联网必需；2026-08-14 起本机已切 TUN 全局接管，无需显式 1097）
# export HTTPS_PROXY=http://127.0.0.1:1097 ...  （旧方法已失效）
export SSL_CERT_FILE=/etc/ssl/cert.pem CURL_CA_BUNDLE=/etc/ssl/cert.pem
# ② 排除内部 telemetry（cognee 会向 posthog 发请求，代理下会卡/挂）
export NO_PROXY="us.i.posthog.com,api.posthog.com,127.0.0.1,localhost"
# ③ DashScope embedding key（灌库/检索都要）
export DASHSCOPE_API_KEY=<你的DashScope key>
```

### 2.2 脚本内置的 cognee env（feed_wiki.py / query.py 开头已 setdefault，逻辑见下）
```python
# LLM → DeepSeek（仅建图用）
LLM_PROVIDER      = "openai"
LLM_MODEL         = "deepseek/deepseek-chat"   # ⚠️ 必须带 deepseek/ 前缀！见踩坑1
LLM_API_KEY       = <DeepSeek key>              # 从 openclaw.json 动态读，不硬编码
LLM_ENDPOINT      = "https://api.deepseek.com/v1"

# Embedding → DashScope
EMBEDDING_PROVIDER = "openai_compatible"
EMBEDDING_MODEL    = "text-embedding-v4"
EMBEDDING_API_KEY  = <DashScope key>
EMBEDDING_ENDPOINT = "https://dashscope.aliyuncs.com/compatible-mode/v1"
EMBEDDING_DIMENSIONS = "1024"

# 其他
ENABLE_BACKEND_ACCESS_CONTROL = "false"   # 单用户, ladybug/networkx 不支持访问控制
EMBEDDING_BATCH_SIZE = "10"               # DashScope 每批上限 10
EMBEDDING_MAX_CONCURRENT_DATA_POINTS = "30"
DB_PROVIDER         = "sqlite"
```
> 脚本里必须 `import litellm; litellm.drop_params = True`（防 dimensions 传参坑）。

### 2.3 DeepSeek key 从哪读
```python
# _get_deepseek_key(): 从 ~/.openclaw/openclaw.json 的 models.providers.custom-api-deepseek-com.apiKey 动态读
# 兜底: 环境变量 DEEPSEEK_API_KEY
```

---

## 3. 灌库 + 建图（feed_wiki.py）

### 3.1 用法
```bash
cd ~/.openclaw/workspace/projects/neurograph/cognee-bridge

# ① 灌文献全量/增量 → dataset=wiki_full
~/.cognee-venv/bin/python -u feed_wiki.py wiki_full
# ② 灌 GitHub 项目 → dataset=github_projects（指定目录）
~/.cognee-venv/bin/python -u feed_wiki.py github_projects --dir ~/.openclaw/workspace/ai-agent-learning/graph/
# ③ 只 add 不建图（快）
... feed_wiki.py wiki_full --no-cognify
# ④ 断点续灌 / 只灌前 N 篇
... feed_wiki.py wiki_full --start 30 --papers 50
```

**参数表**:
| 参数 | 类型 | 默认 | 说明 |
|---|---|---|---|
| `dataset` | 位置 | 必填 | 数据集名，如 wiki_full / github_projects |
| `--dir` | str | `~/wiki/raw/papers` | 灌入的 .md 目录（灌 GitHub 项目时改指向 graph/） |
| `--papers` | int | 0(全部) | 只灌前 N 篇 |
| `--start` | int | 0 | 从第 N 篇开始（断点续灌） |
| `--no-cognify` | flag | 关 | 只 add 不建图 |

### 3.2 内部流程（两步）
```python
# 1. add: 把 .md 文件灌进指定 dataset（很快, 223篇只 ~39s）
await cognee.add(file_paths, dataset_name=args.dataset)
# 2. cognify: 建认知图谱（LLM 抽实体/关系; 慢, 223篇 ~271s）
await cognee.cognify(datasets=args.dataset)
```

### 3.3 实测耗时（223篇文献, DeepSeek LLM）
- add: ~39s
- cognify: ~271s（约 4.5 分钟）
- 总: ~310s（约 5 分钟）——不是预估的 1.5-2.5 小时
- 1 篇 GitHub 项目: add 4s + cognify 19s ≈ 23s

---

## 4. 检索（query.py / kb_search）

### 4.0 三种调用模式

```text
evidence       = CHUNKS 原文片段，不调用回答模型
graph-evidence = GRAPH_COMPLETION 检索器 + only_context=True，不执行LLM completion（默认）
answer         = GRAPH_COMPLETION 完整链路，由DeepSeek生成答案（独立问答备用）
```

`graph-evidence` 使用 Cognee 1.4.2 的公开参数 `only_context=True`，默认从向量检索结果选择种子节点并扩展两跳图邻域。输出包含节点正文和 `source --[relationship]--> target` 关系，可直接交给 Codex 或 OpenClaw 的强模型。使用 `--format json` 时返回稳定的机器可读字段：

```json
{
  "mode": "graph-evidence",
  "query": "...",
  "datasets": "wiki_full,github_projects",
  "graph_context": ["Nodes: ... Connections: ..."]
}
```

职责边界：DeepSeek仍负责 `cognify` 建图以及 `answer` 模式；证据模式不让DeepSeek生成最终答案。

### 4.1 用法
```bash
# 直接 python 调用（最灵活）
~/.cognee-venv/bin/python -u query.py "自适应放疗" --top 5
~/.cognee-venv/bin/python -u query.py "多智能体 协作" --datasets wiki_full,github_projects
~/.cognee-venv/bin/python -u query.py "我今天要做什么" --domain memory   # 记忆域

# 三种知识域接口
~/.cognee-venv/bin/python -u query.py "问题" --mode evidence --format json
~/.cognee-venv/bin/python -u query.py "问题" --mode graph-evidence --format json
~/.cognee-venv/bin/python -u query.py "问题" --mode answer --format json

# 封装命令（自动感知第一查, 推荐日常）
bash ~/.openclaw/workspace/projects/neurograph/scripts/kb_search "自适应放疗" [4] [--all|--github|--memory]
```

**参数表**:
| 参数 | 类型 | 默认 | 说明 |
|---|---|---|---|
| `query` | 位置 | 必填 | 检索词 |
| `--top` | int | 5 | 返回几条 |
| `--domain` | choice | knowledge | knowledge=cognee / memory=memory_rag |
| `--datasets` | str | wiki_full | 逗号分隔多数据集（如 `wiki_full,github_projects`） |
| `--mode` | choice | graph-evidence | evidence / graph-evidence / answer |
| `--format` | choice | text | text / json |

### 4.2 cognee.search 调用（关键 API）
```python
import cognee
# ⚠️ datasets 必须是列表！传逗号字符串 "wiki_full,github_projects" 会报
#    DatasetNotFoundError: No datasets found (404) —— 见踩坑2
r = await cognee.search(
    query_text=q,
    query_type=cognee.SearchType.CHUNKS,   # CHUNKS=纯向量; 也有 GRAPH_COMPLETION(图遍历)
    datasets=datasets_list,                # 列表, 如 ["wiki_full","github_projects"]
)
```
> `SearchType` 只有 CHUNKS / GRAPH_COMPLETION 等，**没有 INSIGHTS**（API 变了）。
> `SearchResult` 取内容要剥一层 `.search_result`。

### 4.3 检索结果解析（标题 + 来源）
```python
# 每条 search_result 是 dict, 关键字段:
#   text  = 整段 markdown(开头的 --- frontmatter 含 journal/DOI/authors/date)
# 提取标题(清洗 frontmatter): 见 query.py 的 extract_text()
# 提取来源(📚标注):         见 query.py 的 extract_source() → journal | DOI | authors | date
# ⚠️ 文本以 --- 开头时是 YAML frontmatter, 不能当标题, 要先剥掉(见踩坑3)
```

### 4.4 ZenBrain 类脑加权
```python
from zenbrain_llm import boost
boosted = boost(results)     # 给每条加 _classed_score / _retrievability_pct
```
- **FSRS decay**: 7天 36.8%、recall 后 stability 8.19（与 memory_rag 一致）
- state 落 `data/zenbrain_state.json`（只读更新调度状态，不写回原始文献）

---

## 5. 辅助脚本

### 5.1 smoke_test.py — 冒烟测试
```bash
~/.cognee-venv/bin/python -u smoke_test.py
# 5 个专业查询(自适应放疗/AI、DL勾画OAR、CBCT、宫颈癌ART、剂量预测)全跑, 验召回+排序
# 结果落 data/smoke_test.json
```

### 5.2 visualize.py — 图谱可视化
```bash
~/.cognee-venv/bin/python -u visualize.py --top 40 --min-freq 3
#   --top 40      画最核心 40 个实体
#   --min-freq 3  实体至少出现 3 次才画
# 输出: data/graph_{dataset}.png + .json
```
**关键坑**: 读图必须用 `get_triplets_batch(offset, limit)` 分页取三元组；
`get_model_independent_graph_data` 只返回 node_labels/relationship_types **不返数据**（见踩坑4）。
matplotlib 装 /tmp/viz_pkgs，中文字体用 `/System/Library/Fonts/STHeiti Light.ttc`。

---

## 6. 数据落点（cognee 存哪了）

cognee 数据**落包默认目录**（`COGNEE_DATA_ROOT` 环境变量实测**不生效**，别指望它能改路径）:
```
~/.cognee-venv/lib/python3.13/site-packages/cognee/.cognee_system/databases/
├── cognee.lancedb/        # 向量库(LanceDB): Entity_name/TextDocument_name/DocumentChunk_text/EdgeType... 等 .lance 文件
├── cognee_graph_ladybug/  # 认知图谱(Ladybug 单文件库, ~25MB)
└── cache.db
```
**项目自身产物**在 `projects/neurograph/data/`:
- `graph_wiki_full.png/.json` — 可视化
- `zenbrain_state.json` — ZenBrain 调度状态
- `smoke_test.json` — 冒烟结果
- `logs/` — 灌库日志（feed_wiki_full.log / feed_github_full.log）

---

## 7. Python 环境说明

- **cognee venv**: `~/.cognee-venv`（全部 cognee 依赖, 版本见 1.1）
- **matplotlib**: 单独装 `/tmp/viz_pkgs`（不污染 venv）, visualize.py 自动加入 sys.path
- ⚠️ matplotlib 若缺失, visualize.py 会报 ModuleNotFoundError, 先 `pip install --target /tmp/viz_pkgs matplotlib`

---

## 8. 已知问题 & 规避（踩坑实录）

### 踩坑1: litellm 模型名必须带前缀 ✋
```python
# ❌ 报错: litellm.BadRequestError: LLM Provider NOT provided. You passed model=deepseek-chat
LLM_MODEL = "deepseek-chat"
# ✅ 正确
LLM_MODEL = "deepseek/deepseek-chat"
```
> litellm 原生支持 DeepSeek，但 model 名必须带 `deepseek/` provider 前缀，否则它不知道用哪个 provider。
> （直连测试时手动传 api_key+api_base 才能用无前缀名，cognee 内部不会这样传，所以必须带前缀。）

### 踩坑2: cognee.search 的 datasets 必须是列表 ✋
```python
# ❌ 报错: DatasetNotFoundError: No datasets found (404)
datasets="wiki_full,github_projects"
# ✅
datasets=["wiki_full","github_projects"]
# query.py 已做处理: 接收逗号字符串再拆成列表
```

### 踩坑3: --- frontmatter 会被当标题 ✋
文献 md 开头是 YAML frontmatter（`---\ntitle:...\nsource:...\n---`），cognee 存 text 时把整段含 frontmatter 存了。
提取标题要**先剥 frontmatter**（找 title: 字段或跳过整个 --- 块），否则显示 "---"。query.py extract_text 已处理。

### 踩坑4: 读图谱的 API ✋
```python
# ❌ get_model_independent_graph_data 只返回 node_labels/relationship_types, 不返数据
# ✅ 用分页
triplets = await cognee.get_triplets_batch(offset=0, limit=1000)
```

### 踩坑5: ladybug 单进程锁 ✋
cognee_graph_ladybug 是单文件库，**并发开两个 query 报 `Could not set lock on file... Lock is held by PID xxx`**。
```bash
pkill -9 -f "query.py"   # 清理幽灵进程
# 平时单次跑, 别并发
```

### 踩坑6: OOM（M1 16G）✋
cognee 初始化+检索吃内存，反复跑会 SIGKILL（free 曾低至 ~260MB）。
策略: **单次干净跑、及时 pkill、不并发**。

### 踩坑7: DashScope 免费额度 ✋
**绝不用 DashScope 做 LLM**（qwen-plus 一灌库就烧光，免费额度耗尽后报 `Free quota exhausted`，检索全挂）。LLM 一律 DeepSeek。

### 踩坑8: 来源元数据不全 ✋
部分文献（英文综述/M2M 英文版）frontmatter 没存 journal/DOI，检索时抓不到精确出处。
根源: 灌库时源文献元数据不完整。**对策**: ①阶段不重灌；②阶段统一规范化 frontmatter。

---

## 9. 自动感知链路（AGENTS.md 第5条）

王成 2026-08-11 拍板「档1 自动感知」：
1. **专业/文献/项目问题** → 回答前自动 `kb_search` 查图谱（纯向量, 零 LLM 成本）
2. 图谱不足 → 回想记忆（memory_rag search_memory.py，🧠 记忆域, 私有不串味）
3. 仍缺 → 联网兜底（web_search, 🌐）
4. **实时问题**（新闻/行情/天气）→ 直接联网，不查图谱
5. **来源标注**: 📚=图谱/文献(标题+期刊/DOI/作者), 🧠=记忆, 🌐=联网 —— 回答必须带，缺失要明说"图谱无完整出处"

---

## 10. 每周 GitHub 项目自动灌入（cron a94f9475）

每周五 10:00 自动运行（isolated, 超时 1800s）:
1. 检索 1 个 AI Agent 高赞项目 → 学习笔记 `ai-agent-learning/docs/YYYY-MM-DD_项目.md`
2. **生成灌图谱版** `ai-agent-learning/graph/YYYY-MM-DD_项目.md`（项目简介/基本信息/拆解/原理/应用场景/生态/参考价值）
3. **灌进 cognee github_projects 数据集**: `feed_wiki.py github_projects --dir ai-agent-learning/graph/`
4. 更新 INDEX.md + data/history.json + 汇报王成

> 这样每周 GitHub 项目和文献一样自动进知识图谱，跨数据集检索（如查"多智能体"同时出 CrewAI + 放疗 LLM Agent 文献）。

---

## 11. 扩展（②自研阶段, 周末反馈后）

- 存储 `.npy→LanceDB`、规则共现→LLM 抽实体认知图
- ZenBrain 深内嵌（非外挂后处理）
- 记忆域并入同一套图谱（需王成拍板隐私方案）
- **参照本文档踩坑 1-8 和 README「周日升级版注意事项」避开已知问题**
- 补: 灌库规范化 frontmatter（补齐 journal/DOI/year），提升来源可追溯性（王成强调的规范）

---

## 12. 一键复现检查清单（复现者过一遍）

- [ ] `python3 -m venv ~/.cognee-venv && pip install cognee==1.4.2 litellm networkx`（版本对齐 1.1）
- [ ] `export SSL_CERT_FILE=/etc/ssl/cert.pem NO_PROXY=...`（2.1，TUN 全局接管无需显式 1097）
- [ ] `export DASHSCOPE_API_KEY=...`（2.1）
- [ ] 确认 DeepSeek key 可读（2.3）
- [ ] 拷贝 cognee-bridge/ 4 个脚本 + scripts/kb_search（1.2）
- [ ] `feed_wiki.py wiki_full`（灌文献）→ `feed_wiki.py github_projects --dir .../graph/`（灌项目, 3.1）
- [ ] `kb_search "自适应放疗" --all` 应返回文献+项目, 标题干净、带📚来源（4.1）
- [ ] `smoke_test.py` 5 查询全过（5.1）
- [ ] 若 OOM/锁问题 → 见踩坑5/6
