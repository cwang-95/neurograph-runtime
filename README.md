# NeuroGraph — Graph 3.0 知识图谱与类脑检索

> 当前可用版本已将 Graph 3.0、OpenClaw adapter 和默认部署入口整合在同一仓库。知识域默认返回可审计 EvidencePack，由上层 OpenClaw/Codex 模型完成唯一一次最终回答；Cognee 保留为可回滚兼容后端。

## 快速开始

```bash
bash scripts/setup_graph3.sh
scripts/kb_search "自适应放疗的工作流和时间成本" 8 --graph-evidence --json
```

默认数据目录是 `data/graph3-openclaw-full`，运行数据不会提交到 Git。已有外部语料时设置 `NEUROGRAPH_GRAPH3_STORAGE`。回滚到 Cognee：

```bash
NEUROGRAPH_BACKEND=cognee scripts/kb_search "自适应放疗" 8 --graph-evidence --json
```

如果部署机器提供 OpenAI-compatible embedding 服务，`kb_search` 默认尝试本机
`http://127.0.0.1:8003/v1/embeddings`，也可以显式覆盖；没有该服务时仍会保留词法、数值、实体和图检索：

```bash
export NEUROGRAPH_GRAPH3_EMBEDDING_ENDPOINT=http://127.0.0.1:8003/v1/embeddings
export NEUROGRAPH_GRAPH3_EMBEDDING_MODEL=mlx-community/Qwen3-Embedding-0.6B-4bit-DWQ
# 没有 embedding 服务时显式关闭向量路由，避免等待连接超时
export NEUROGRAPH_GRAPH3_DISABLE_EMBEDDING=1
```

FSRS 默认开启；如果部署机器没有 OpenClaw 的 ZenBrain Node 依赖，可设置
`NEUROGRAPH_GRAPH3_FSRS=0`，不影响基础检索。

作答后的 EvidencePack 可通过反馈脚本回写 ZenBrain，只记录前 N 条证据及其关联的 claims、图路径和关系：

```bash
scripts/g3fb selected /tmp/pack.json 3
```

完整部署说明见 [`docs/graph3-openclaw-deployment.md`](docs/graph3-openclaw-deployment.md)，历史设计记录见下文。

---

# 历史方案记录：记忆图谱 2.0 — 知识库 × 类脑记忆融合

> 本节是 2026-08-10 的方案草稿，仅用于保留决策背景；当前运行状态、默认入口和部署参数以上方说明及 `docs/graph3-openclaw-deployment.md` 为准。文中“计划”“尚未切换”等表述不代表当前 `main`。

> 版本: v1 | 日期: 2026-08-10 | 定调人: 王成
> 一句话: **①用 cognee 铺路先受益,②自研真向量库+认知图谱+类脑算法长成工业级,①验证②、②接手①。**

---

## 一、背景与目标

### 现状痛点（历史记录）
1. 当时的知识库(wiki 198 篇文献 + kb_rag)是"惰性/事后查"——**只有我主动跑 search_kb.py 才用**,答前不带着专业知识,没有"被动底色"。
2. memory_rag(轻量自研)思路先进(类脑算法 ZenBrain:FSRS遗忘/Hebbian强化/情绪加权/睡眠巩固),但**工程是雏形**(.npy + 余弦 + 规则共现图),非工业级。

### 目标
把"知识库"和"类脑记忆"真正合一:一个系统,既有 **Cognee 式工业级工程**(真向量库+认知图谱+增量更新+跨会话),又保留 **ZenBrain 式类脑灵魂**(遗忘调度/巩固/情绪——cognee 没有的深度)。

### 王成拍板
- 倾向方案②(自研深耕、自主可控),认可方案①(现成稳定)。
- 决策:**①②并行,①铺路受益,②按自研走保留类脑灵魂,②成熟后①退居备用。**
- ②底座:王成选**真向量库**(倾向 LanceDB,与 cognee 对齐、文件型轻量、不用常驻服务;Qdrant 为备选)。

---

## 二、整体架构(双轨)

```
┌─────────────── 方案① Cognee 底座(铺路·先受益) ───────────────┐
│  wiki文献 → cognee.add() → cognee.cognify()建认知图          │
│          → cognee.search() 自动召回(被动专业底色)            │
│  现成:真向量库 LanceDB + 认知图谱 + 增量更新 + 跨会话          │
│  验证:8篇文献实测检索命中极高(见实测记录)                     │
└────────────────────────────────────────────────────────────┘
                        ↓ 参照/验证机制,逐步"抄"回自研版
┌─────────────── 方案② 自研升级(深耕·保留灵魂) ────────────────┐
│  底座:真向量库(LanceDB) + 认知图谱(LLM抽实体/关系)           │
│  类脑算法层 ZENBRAIN:FSRS遗忘/Hebbian/情绪/睡眠巩固           │
│          ↑ 原生内嵌(非外挂)——这是①做不到的深度              │
│  编排层:feed/consolidate/search(对齐现有 memory_rag)         │
│  目标:②成熟 → 知识库迁回自研,①退居备用                       │
└────────────────────────────────────────────────────────────┘
                    ↑ 类脑算法(共有的灵魂)
```

---

## 三、方案① Cognee 铺路(马上做)

### 施工步骤
1. **批量灌 wiki 全量文献**(已从 8 篇验证,扩到全部 198 篇)
2. **cognee.cognify()** 建全量认知图谱(LLM 抽实体/关系,DashScope qwen-plus)
3. **可视化**(可选):用 cognee UI/visualize 看图谱,调试+给王成直观感受
4. **接进我日常**:专业话题检索走 cognee → 被动专业底色
5. **每日增量**:新文献进 wiki 后,增量 add+cognify

### 已踩坑经验(别重复)
- DashScope embedding 必须 `EMBEDDING_PROVIDER=openai_compatible`
- `litellm.drop_params=True`(防 dimensions 传参坑)
- `ENABLE_BACKEND_ACCESS_CONTROL=false`(单用户,networkx 不支持访问控制)
- `EMBEDDING_BATCH_SIZE=10`(DashScope 每批最大10)
- API 变了:search 用 `query_text`,SearchType 无 INSIGHTS
- 脚本放 workspace 里 + `python -u`(防 flush 限制 / 输出缓冲)

### ⚠️ 周日升级版项目注意事项(2026-08-11 王成要求记录)
> 这些是在①实测中得出的结论,②自研时必须遵守/参考,避免重蹈覆辙。

1. **LLM 必须用 DeepSeek, 不要用 DashScope 免费 key**:
   - cognee 默认 LLM 是 qwen-plus(DashScope),一灌库(建图)就烧光 DashScope 免费额度(`Free quota exhausted`),检索跟着全卡死。
   - 正确做法: LLM 用 `deepseek/deepseek-chat` + `https://api.deepseek.com/v1`(key 从 openclaw.json `models.providers.custom-api-deepseek-com.apiKey` 动态读, 不硬编码);**Embedding 才用 DashScope**(text-embedding-v4 几乎免费, DeepSeek 无 embedding API)。
   - litellm 坑: model 必须写 `deepseek/deepseek-chat`(带 provider 前缀),写裸 `deepseek-chat` 报 `LLM Provider NOT provided`。
2. **检索应做到零 LLM(纯向量)**: cognee.search CHUNKS 模式检索时**不调 LLM**(实测 stderr LLM 痕迹=0), 低成本免费。只有灌库/建图(低频)才耗 LLM。②的检索也应是纯向量 + ZenBrain,不烧钱。
3. **高频噪声实体需过滤**: 图谱里 person/ai/concept/date/technology/organization 等元数据/泛化实体占中心(1592实体里高频)，与 memory_rag 当年 AI/ART 高频噪声同病,②建认知图时要设噪声黑名单。
4. **`---` frontmatter 提取瑕疵**: 文献 md 开头 `---` 会被 cognee 当成标题显示(部分检索结果标题显示 "---"), 不影响检索,显示层需清洗。
5. **并发/内存**: ladybug 单文件图库单进程锁; 反复跑 query 会留幽灵进程占锁+OOM(本机 M1 16G), 用完即 kill, 单次跑更稳。
6. **成本**: 灌223篇一次性 DeepSeek≈¥16 / qwen-plus≈¥4-5(粗算, token 不精确, 以控制台为准); 检索免费; 插增量/每周插 GitHub 项目是每次几毛到1-2块。

---

## 四、方案② 自研升级(深耕)

### 技术选型(已定)
| 层 | 选型 | 说明 |
|---|---|---|
| 向量库 | **LanceDB**(备选 Qdrant) | 文件型轻量、与 cognee 对齐、从①迁②平滑 |
| 图谱 | 认知图谱(LLM抽实体/关系) | 学 cognee,不再用规则共现 |
| 类脑算法 | **ZenBrain**(@zensation/algorithms) | FSRS/Hebbian/情绪/巩固,**原生内嵌** |
| 编排 | 对齐现有 memory_rag | feed/consolidate/search |
| embedding | DashScope text-embedding-v4(1024维,curl) | 现有资产,复用 |

### 施工步骤(分阶段)
- **阶段2.1 存储升级**:memory_rag 的 `.npy→LanceDB`、`规则共现→LLM抽实体建认知图`
- **阶段2.2 类脑内嵌**:ZenBrain 遗忘/巩固/情绪**直接进检索排序**(不是外挂后处理)
- **阶段2.3 知识库并入**:wiki 文献作为"知识域"并进自研系统(记忆=经历域 + 知识=文献域,一套图谱)
- **阶段2.4 迁移收尾**:①退居备用,②全面接手

---

## 五、分工边界(防混淆)
- **记忆(经历)**:MEMORY.md + memory/*.md → memory_rag(memory 域)
- **知识(文献)**:~/wiki + kb_rag → ①cognee / ②自研知识域
- 领域分开但**一套图谱架构**,类脑调度统一。

---

## 六、红线(延续)
- 记忆/知识只读原文件,绝不写回(①和②都遵守)
- aging 只打标签永不删(王成"不丢记忆"红线)
- 改配置前先备份;删东西前先查引用
- 全程中文;已授权直接执行,不再反复确认

---

## 七、历史状态记录（不代表当前部署）
- [x] ①cognee 8篇验证通过(灌→建图→检索全通,命中极高)
- [x] ①批量灌 wiki 全量 223 篇 + 建认知图谱 + 可视化出图
- [x] ①检索链路打通(LLM 换 DeepSeek,DashScope 免费额度不会被烧光)
- [x] ①图谱可视化(1592 实体/6878 三元组,见 data/graph_wiki_full.png)
- [x] 冒烟测试通过(5 个专业查询全部命中+ZenBrain 排序正常)
- [x] **文献 + GitHub 项目双数据集融合检索**: 新增 github_projects 数据集(每周灌), query.py/kb_search 支持跨数据集检索(--all)
- [x] 每周 cron 已升级: 检索项目→学习笔记→灌图谱版→灌进 github_projects
- [x] ②选型落地：已由 Graph3 自研核心、SQLite 权威存储、可重建 HNSW 和 ZenBrain 反馈机制完成
- [x] ②成熟后迁移收尾：Graph3 已合并 `main` 并成为 OpenClaw knowledge/wiki_full 默认入口

### 已验证能力(2026-08-11)
- 查"自适应放疗" → 命中《AI-driven ART》《AMA ART review》等 3+ 篇
- 查"多智能体协作 角色分工" → **同时带出 GitHub 项目(CrewAI) + 放疗 LLM Agent 文献(GPT-RadPlan/M2M Agent 等)** —— 文献+项目融合的被动专业底色达成
- 跨数据集检索: docs/wiki_full(文献) + github_projects(项目)
- 标题清洗已生效(不再显示 `---` frontmatter)

> 上述“下一步”属于历史计划。当前部署、数据规模和验证结果见 README 顶部及部署文档。

---

## 八、检索接口分层

同一个 NeuroGraph 后端提供三种稳定模式，上层按自身推理能力选择，避免重复生成答案：

| 模式 | 做什么 | 是否调用回答模型 | 推荐调用方 |
|---|---|---:|---|
| `evidence` | 返回语义相关的完整原文片段 | 否 | 精确事实、来源回查 |
| `graph-evidence` | 向量召回种子后扩展两跳图邻域，返回节点和关系 | 否 | Codex、OpenClaw + DeepSeek（默认） |
| `answer` | 基于图上下文生成完整答案 | 是，DeepSeek | 无上层模型的独立问答备用 |

```bash
# OpenClaw/Codex默认：只返回图增强证据，由上层模型回答
bash scripts/kb_search "自适应放疗" 8 --all --graph-evidence --json

# Codex/强模型：只取图关联证据，由上层模型回答
bash scripts/kb_search "自适应放疗" 8 --all --graph-evidence --json

# 精确查数字或原文
bash scripts/kb_search "gamma 98.6%" 8 --evidence --json
```

不传模式时默认 `graph-evidence`，避免 OpenClaw 中的 DeepSeek 与 NeuroGraph 内部 DeepSeek 重复生成答案。`answer` 保留为独立问答备用；旧参数 `--chunks` 继续兼容，等价于 `--evidence`。
