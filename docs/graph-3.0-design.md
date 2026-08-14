# NeuroGraph 3.0 技术设计

状态：设计基线 v0.2（已完成首轮技术 review；Phase 0–5 与证据覆盖基础已在 `feature/graph-3.0` 落地，尚未切换 OpenClaw 默认入口）

当前实现边界：已具备 RawAsset/SourceElement/Observation、Claim/EvidenceLink、保守实体图、显式模式语义关系、多证据关系聚合、DeepSeek 结构化关系候选及严格审核、可控批量构建入口、受 hop/beam/关系白名单约束的多跳图扩展、ZenBrain 追加事件账本与弱先验、多路 lexical/numeric/vector/graph 召回、EvidencePack 槽位覆盖与确定性追问、现有 ZenBrain FSRS 调度器适配、Observation/ClaimVersion/Relation/Path 显式回答反馈接口、ClaimVersion 抑制与冲突投影。向量检索当前是可重建的 SQLite brute-force 基线，DeepSeek 只生成候选，不直接改变权威事实；边/路径/Claim 目前使用事件弱先验，尚未有独立 FSRS 状态，ANN 索引仍未接入。

## 1. 目标、原则与边界

NeuroGraph 3.0 的目标不是在 Cognee 输出后增加一层重排，而是建立可与 ZenBrain 深度融合的自研知识组织与检索核心。系统应从领域问题出发，发现相关材料，跨块、跨页、跨文档跳跃检索，最终返回完整、可核验、可纠正的信息，而不是孤立文本片段。

必须遵守以下原则：

1. 原始材料不可变，任何 OCR、ASR、视觉理解、摘要和 Claim 都是派生数据；
2. 小块用于命中，结构用于扩展，Claim 用于推理，原始证据用于核验；
3. 不存在适合所有问题的唯一 chunk size，也不存在全局唯一的“最小完整知识块”；
4. 事实相关性、证据可信度、记忆可检索性必须分别建模；
5. 图遍历必须由问题、边类型、证据覆盖和预算约束，不能无差别扩散；
6. 纠错采用追加式版本记录，不覆盖历史证据；
7. ZenBrain 只能改变检索优先级，不能改变事实真伪或删除冷知识；
8. 所有核心设计都必须通过真实问题回归集验证，不能只凭主观效果判断。

3.0 自研以下核心环节：

- 统一知识数据模型与稳定 ID 规则；
- 分层解析、多粒度切块和跨模态对齐流程；
- Claim、实体、关系、冲突与纠错逻辑；
- 多路召回、查询规划、图遍历和证据覆盖检查；
- EvidencePack 组装、引用和回答接口；
- ZenBrain 的事件、节点、边、路径和巩固机制。

Docling、python-pptx、OCR/ASR 工具、LanceDB、SQLite、NetworkX 或图数据库可作为解析、存储和计算组件，但不承担 3.0 的核心知识组织逻辑。Cognee 2.0 保留为稳定回退、迁移数据源和效果基准，不作为 3.0 核心。

## 2. 总体架构

```text
RawAsset 原始文件
  → SourceElement 原始结构元素
  → Observation 机器或人工观察结果
  → ClaimVersion 规范化事实版本
  ↔ Entity / Relation 实体与关系

SourceElement 同时组成稳定的 SourceUnit 层级
ClaimVersion 同时组成可版本化的 TopicUnit / KnowledgeUnit

用户问题
  → QueryPlan + EvidenceSlots
  → 多路召回
  → 受约束图遍历
  → EvidencePack
  → Codex / OpenClaw / 独立回答模型

ZenBrain Event Ledger 横向记录检索、采用、引用、确认和纠错事件

3.0 通过 `NodeZenBrainFSRS` 调用 OpenClaw 现有的
`@zensation/algorithms`，将观察节点的 FSRS 状态保存到 SQLite。检索只读
`retrievability`，不会产生强化；`selected`、`cited`、`followed_up` 和
`user_confirmed` 必须由回答层显式回写，才会更新 FSRS。`corrected` 与
`rejected` 只影响事件先验，不触发 recall。ClaimVersion、Relation 和 Path
使用同一追加事件账本，但当前只参与弱先验，不与 Observation 共用 FSRS 状态。
```

权威数据保存在结构化元数据与事件账本中。向量索引、BM25 索引、图投影、摘要和缓存都是可重建的派生视图。

## 3. 数据与证据模型

### 3.1 RawAsset

不可变的原始材料，例如 PPTX、PDF、音频、视频、网页快照、Markdown 或图片。至少记录：

- `asset_id`：基于内容哈希生成的稳定 ID；
- `content_hash`、文件大小、MIME 类型；
- 原始路径或来源 URL；
- 获取时间、来源时间和文档版本；
- 原始文件与规范化副本的位置；
- 数据集、访问域和许可信息。

同一内容重复灌入时必须复用 `asset_id`，不能产生新的知识副本。

### 3.2 SourceElement

从 RawAsset 中确定性定位的原始结构元素，例如幻灯片、文本框、表格单元格、图表、图片区域、论文段落或音频时间段。至少记录：

- `element_id`：由 `asset_id + locator + content_hash` 生成；
- `page` / `slide` / `sheet`；
- `bbox`、`shape_id`、`cell`；
- `audio_time_span` / `video_time_span`；
- `char_span` 或原始 XML/HTML 定位；
- `previous`、`next`、`parent`、`child`；
- 原始文本或原始区域引用。

SourceElement 是引用的最小定位单位。回答中的每个关键事实最终必须能回到一个或多个 SourceElement。

### 3.3 Observation

对 SourceElement 的一次观察或提取结果，例如：

- PPT 原生文本提取；
- OCR 文本；
- ASR 转写；
- 视觉模型对图表、流程图或图片的描述；
- 人工校对结果。

Observation 必须记录：

- `observation_id` 和对应 `element_id`；
- `method`、解析器或模型名称及版本；
- prompt、配置和代码版本；
- 创建时间、语言和置信度；
- 原始输出，不把模型生成描述伪装成原始材料。

同一 SourceElement 可以有多个 Observation。人工校对可以提高优先级，但不会删除机器提取记录。

### 3.4 ClaimVersion

ClaimVersion 是可判断、可比较、可引用的规范化事实版本。至少包含：

```text
claim_id                 同一逻辑事实的稳定身份
claim_version_id         本次具体版本
subject                  主体实体
predicate                关系或属性
object / value           客体或数值
unit                     单位
polarity                 肯定 / 否定
modality                 确定 / 可能 / 建议 / 假设
population               适用人群或样本
condition                实验、设备、场景等条件
method                   测量或计算方法
valid_time               事实在现实中有效的时间
observed_time            系统获取该事实的时间
source_scope             作者自有结果 / 引用结果 / 背景知识
status                   active / disputed / rejected / superseded
extraction_confidence    抽取正确概率
source_quality           来源等级
support_strength         证据对该 Claim 的支持程度
```

Claim 的身份不能只由自然语言或 embedding 相似度决定。数值、单位、极性、时间和条件都是身份与冲突判断的一部分。

### 3.5 EvidenceLink

EvidenceLink 连接 Observation 与 ClaimVersion，并说明证据关系：

- `supports`：直接支持；
- `contradicts`：直接冲突；
- `qualifies`：补充适用条件；
- `derived_from`：经过计算或归纳得到；
- `mentions`：仅提及，不足以支持；
- `quotes`：引用另一来源。

引用准确性在 EvidenceLink 层验证。一个 Claim 可以有多个独立证据；一个 Observation 也可以支持多个 Claim。

### 3.6 Entity 与实体消歧

Entity 至少记录规范名称、类型、别名、缩写、外部标识和来源范围。实体合并分两级：

- `possible_same_as`：候选相同，不参与不可逆合并；
- `confirmed_same_as`：有足够上下文或人工确认后建立。

合并必须保存依据并支持撤销。不同版本的软件、模型、设备和方法默认是不同实体，通过 `version_of` 或 `derived_from` 关联。

### 3.7 SourceUnit、TopicUnit、KnowledgeUnit 与 EvidencePack

- `SourceUnit`：由原始结构确定的稳定单元，例如章节、幻灯片或表格；
- `TopicUnit`：离线聚类或抽取得到的主题单元，是可重建、可版本化的派生视图；
- `KnowledgeUnit`：经过验证、可复用的一组 Claim、关系和证据，服务于一类主题问题；
- `EvidencePack`：针对当前问题动态组装的证据包，是回答层实际消费的单位。

KnowledgeUnit 不再定义为“对所有问题都完整的最小块”。完整性由当前 QueryPlan 的 EvidenceSlots 判断。

## 4. 解析、切块与跨模态对齐

### 4.1 原始层先行

解析流程必须先建立 RawAsset 和 SourceElement，再运行 OCR、ASR、视觉理解和 Claim 抽取。不能只保存合并后的 Markdown，因为合并文本无法完整恢复页面位置、时间范围、提取来源和重复关系。

对于演讲材料：

- 音频先按真实时间段保存唯一 ASR segment；
- ASR segment 可以与多个相邻幻灯片对齐，但文本只存一份；
- 幻灯片原生文字、视觉描述和语音分别作为 Observation；
- 同一句语音出现在多个合并页面时只建立多个 alignment，不复制 Claim；
- 图表数字优先读取 PPT 原生 chart data，其次视觉/OCR，最后才是语音转写。

### 4.2 多粒度索引单元

系统同时保留：

```text
Document / RawAsset
└── Section / SourceUnit
    └── Passage
        ├── Proposition / Claim
        └── SourceElement / Observation
```

- Proposition/Claim 用于精确事实、数字和图路径召回；
- Passage 用于语义和关键词召回；
- SourceUnit 用于恢复局部上下文；
- TopicUnit/KnowledgeUnit 用于主题发现和跨文档组织；
- SourceElement 用于最终核验与引用。

### 4.3 切块边界

切块边界按以下优先级确定：

1. 文档和数据集边界；
2. 标题、章节、主题和幻灯片边界；
3. 表格、图、图注、文本框和语音时间段；
4. 语义主题变化；
5. 句子边界；
6. token 上限强制切分。

固定 token 长度只作为技术保护。块大小、上限和动态扩展窗口由回归集校准，并按文档类型分别配置，不能全库使用一个参数。

### 4.4 动态上下文与重叠控制

存储时少重叠或不重叠，显式保存：

- `parent` / `child`；
- `previous` / `next`；
- `same_page` / `same_figure` / `same_table`；
- `aligned_with`；
- `supports` / `contradicts` / `qualifies`。

命中后根据问题动态扩展父级、相邻块、同页元素和对应证据。若源数据本身已有滑动窗口或重复转写，先进行 span/hash 对齐去重，再进入索引和 ZenBrain。

### 4.5 上下文化索引实验

3.0 将以下方案作为 A/B 实验，不预先认定某一种必然最佳：

1. 原始 Passage embedding；
2. 标题、章节路径和文档元数据前缀；
3. 由完整文档生成的简短块级 contextual prefix；
4. Late Chunking；
5. 向量 + BM25 + reranker。

当前 OpenAI-compatible embedding 接口只返回池化后的向量，无法直接实现需要 token embedding 的 Late Chunking。若不更换本地 embedding 服务接口，3.0 首版优先实现结构前缀和 contextual prefix，Late Chunking 保留为后续实验。

用于检索的 contextual prefix 是派生 Observation，不能伪装成原始 SourceElement；回答时优先返回原始内容而不是索引前缀。

### 4.6 特殊内容

- 表格保存表头、行名、数值、单位、合并单元格和坐标；
- 图表保存标题、坐标轴、图例、序列、数据点和对应图片；
- 图片保存原图、图注、OCR、视觉描述和区域位置；
- 公式保存原式、变量定义、单位和相邻解释；
- 数字同时保存原始写法和规范化值，不忽略误差、范围、比较符号和单位；
- 否定、条件、适用范围和限制与对应 Claim 绑定；
- 页面装饰性年份、页码和步骤编号不能误建为实验结果。

### 4.7 降采样与去重

数据分为三层：

- 原始层：完整保存，不降采样、不覆盖；
- 索引层：允许去重、摘要和压缩，但保留原始证据指针；
- 回答层：按 EvidenceSlots 和上下文预算选择证据。

降采样只能影响候选优先级。数字、单位、否定词、条件限制、异常结果、可信冲突和低频但高相关事实不能因相似度、热度或访问频率而丢失。

近似文本不能仅凭相似度合并。例如 `1.5 ms` 与 `1.5 s` 必须分别保存并触发单位、模块总耗时和来源等级核验。

## 5. 索引与派生视图

3.0 至少维护以下可重建索引：

- Passage/Claim/TopicUnit 向量索引；
- BM25 或等价关键词索引；
- 数字、单位、范围和标识符精确索引；
- 实体规范名和别名索引；
- 类型化图邻接索引；
- 文档父子、相邻、同页和跨模态对齐索引；
- 时间、版本、状态和来源过滤索引。

每条 embedding 必须记录模型、维度、归一化方式、输入模板和版本。更换 embedding 模型时建立新索引版本，验证后原子切换，不混用不同空间的向量。

索引生成必须幂等。相同 RawAsset 重复灌入不增加 Claim、实体或路径数量；解析器或模型升级时创建新的 Observation/派生版本。

## 6. 查询规划与多路检索

### 6.1 QueryPlan

先将问题分为一个或多个查询类型：

- 精确事实或数字；
- 实体说明；
- 方法或机制；
- 比较；
- 多跳因果或流程；
- 主题综述；
- 时间变化；
- 来源核验。

QueryPlan 包含查询类型、候选实体、时间/条件约束、EvidenceSlots、召回路线、预算和停止条件。计划是结构化数据，必须可以记录和回放。

### 6.2 EvidenceSlots

槽位由领域模板和问题动态分解共同生成。可能包括：

- 定义、背景、机制；
- 模块、输入、输出和流程；
- 数据集、样本、设备和实验条件；
- 指标、数值结果和比较对象；
- 限制、适用范围和来源。

每个槽位具有以下状态：

```text
supported
conflicted
missing
not_applicable
low_confidence
```

LLM 生成的槽位只是候选。领域模板、问题类型规则和检索证据共同校正槽位，避免模型凭空要求不存在的信息。

### 6.3 多路召回

每次检索可组合：

1. 语义向量召回；
2. BM25/关键词召回；
3. 数字、单位、公式和标识符精确召回；
4. 实体及别名召回；
5. 文档结构召回；
6. TopicUnit/KnowledgeUnit 召回；
7. 时间和版本约束召回；
8. ZenBrain 提供的弱先验候选。

首版使用可解释的 rank fusion，例如按路线保留配额后执行 RRF，再由轻量 reranker 评估问题相关性和证据质量。不同路线的原始分数不直接相加。

### 6.4 受约束图遍历

图遍历采用类型化 beam search，而不是无差别固定 N 跳：

- 从多路召回得到的 Claim、Entity、SourceUnit 或 TopicUnit 开始；
- 只沿 QueryPlan 允许的边类型扩展；
- 每条路径记录起点、边、终点和支持证据；
- 设置 `max_hops`、`beam_width`、每类边配额和总候选预算；
- 惩罚高 degree 的通用枢纽节点；
- 奖励能够填补缺失 EvidenceSlot 的候选；
- 无 SourceElement 和 EvidenceLink 支持的派生路径不能成为最终关键结论；
- 路径扩展没有带来覆盖增益时提前停止。

精确问题优先 Claim/Statement 路径；宽泛问题先走 Topic/Community，再回落到 Claim 和原始证据。

### 6.5 评分职责

至少分别计算：

- `query_relevance`：与当前问题的相关性；
- `evidence_quality`：来源和提取可信度；
- `support_strength`：证据是否直接支持 Claim；
- `coverage_gain`：是否填补缺失槽位；
- `path_quality`：关系类型、路径长度和枢纽噪声；
- `retrievability_prior`：ZenBrain 的记忆先验；
- `conflict_penalty`：未解决冲突；
- `redundancy_penalty`：与已选证据重复。

这些维度保留原值和解释。最终排序可以组合，但不得只保存一个不可解释的总分。证据质量和当前问题相关性始终高于历史热度。

### 6.6 迭代检索与停止条件

每轮检索后检查 EvidenceSlots：

- 必要槽位均 `supported`，停止；
- 新一轮没有产生覆盖增益，停止并报告缺失项；
- 达到 hop、候选、延迟或 token 预算，停止；
- 关键槽位存在无法裁决的冲突，进入冲突流程；
- 多个解释会实质改变答案且无法从证据裁决，向用户追问。

## 7. EvidencePack 与回答接口

EvidencePack 是 Codex、OpenClaw 或独立回答模型消费的稳定接口，至少包含：

```json
{
  "query": "...",
  "query_plan": {},
  "slot_status": {},
  "claims": [],
  "evidence": [],
  "paths": [],
  "conflicts": [],
  "missing": [],
  "citations": [],
  "retrieval_trace": {},
  "index_version": "..."
}
```

上下文组装规则：

- 数字、单位、条件、限制和否定句尽量保留原文；
- 重复证据折叠，但保留独立来源数量；
- 摘要和原始证据明确分区；
- 按 EvidenceSlot 和结论组织，不简单按相似度堆叠；
- 每个关键 Claim 就近关联引用；
- 控制总长度，避免相关证据在超长上下文中被淹没；
- 回答模型不能引用未进入 EvidencePack 的来源。

`evidence` 和 `graph-evidence` 模式返回 EvidencePack，不调用最终回答模型；`answer` 模式才执行一次回答生成。Codex/OpenClaw 已经承担回答时，不再调用内部 DeepSeek 生成第二份答案。

## 8. 歧义、冲突与纠错

### 8.1 追问策略

- 多个方向可以同时成立时，分类或合并，不追问；
- 宽泛问题先给领域地图和主要分支；
- 缺少证据但可继续检索时，系统自行补检索；
- 只有多个解释会实质改变 QueryPlan 或答案，且证据不足以裁决时才追问；
- 记录追问原因、候选解释及其预计影响，便于评测误问和漏问。

### 8.2 冲突分类

发现差异时先分类：

- 转写、OCR、单位或解析错误；
- 事实随时间变化；
- 实验条件、患者群体或方法不同；
- 作者观点或来源之间存在真实争议；
- 同一来源内部自相矛盾；
- 重复表述但并不冲突。

明显解析错误由单位一致性、上下文、总量约束、原生 PPT 数据和更高等级来源自动裁决，不把所有错误候选展示给用户。只有来源等级接近、条件相同且无法可靠裁决时才请求确认。

### 8.3 追加式纠错

确认纠错后：

1. 追加新的 ClaimVersion；
2. 记录纠错 Activity、依据、操作者和时间；
3. 将旧版本标记为 `rejected` 或 `superseded`，不删除；
4. 更新 EvidenceLink 和当前有效视图；
5. 增量重建受影响的向量、关键词和图投影；
6. 失效相关缓存；
7. 写入 ZenBrain correction 事件并抑制错误路径；
8. 后续回答默认过滤无效版本，同时允许审计历史。

系统同时记录 `valid_time` 和 `observed_time`，避免把旧版本事实误判为错误。

## 9. ZenBrain 深度融合

### 9.1 事件模型

ZenBrain 使用追加式事件账本，不允许检索函数在读取候选时隐式强化。事件至少包括：

```text
retrieved          仅被召回，不强化
selected           进入 EvidencePack，轻度强化
cited              被最终答案引用，中度强化
followed_up         用户基于该知识继续追问，中强强化
user_confirmed      用户明确认可，强强化
corrected           被纠正，抑制错误 Claim/路径
rejected           被判无关或错误，不强化或降权
```

事件记录 query、ClaimVersion、EvidenceLink、path、时间、调用方和反馈来源。重复自动查询不能伪装成用户认可。

### 9.2 作用范围

ZenBrain 分别维护：

- 节点状态：当前实现先维护 Observation 的 FSRS 可检索性，后续扩展到 Claim、Entity、TopicUnit；
- 事实状态：ClaimVersion 已有独立事件先验，纠错默认只作用于选中的版本，不自动修改同一逻辑 Claim 的其他版本；
- 边状态：当前已记录 Relation 的显式回答反馈弱先验，后续扩展为按问题类型的独立调度状态；
- 路径状态：当前已记录稳定 Path ID 的显式回答反馈弱先验，后续扩展为路径级巩固状态；
- 用户上下文：用户近期关注方向，但不改变事实可信度；
- 巩固候选：高频共同激活内容可形成候选 KnowledgeUnit，必须重新核验来源。

### 9.3 安全边界

- `retrievability_prior` 只作为弱先验；
- `query_relevance`、`evidence_quality`、`support_strength` 和 `coverage_gain` 优先；
- 冷知识与当前问题高度相关时必须能够越过记忆衰减；
- 通用热门节点不能因高频占据所有候选；
- 遗忘只降低默认优先级，不删除 RawAsset、SourceElement、Observation 或 ClaimVersion；
- ZenBrain 开启后不得降低冷事实回归集的召回率超过验收阈值。

当前 2.0 的 `boost()` 在“被召回”时立即强化，且默认 graph-evidence 不经过 boost。3.0 不复用该行为，只将现有状态作为可选迁移数据，并默认从新的事件账本重新计分。

回答层反馈接口必须显式调用：

```python
ledger.record_feedback(
    observation_ids,
    ZenBrainEventType.CITED,
    query=query,
    caller="answer-layer",
)
```

事实级反馈使用 `record_claim_feedback(claim_version_ids, event_type)`。
若回答层明确确认来源证据也应同步强化，才传入
`propagate_to_observations=True`；纠错和拒绝不会沿证据链接扩散。

`record_feedback` 不是检索函数的一部分。若只调用 `retrieve()`，事件数和
FSRS 节点状态都不增加；这条约束纳入回归测试。

EvidencePack 的 `conflicts` 只报告同一逻辑 Claim 下仍未裁决的不同版本；
若某一 ClaimVersion 已被明确纠正、拒绝或静态标为 superseded/rejected，
该版本不会进入回答证据，但原始 Observation 仍保留在权威存储中供审计。

## 10. 存储与工程实现

首版建议：

- SQLite：权威元数据、版本、EvidenceLink、事件账本和任务状态；
- LanceDB：可重建的向量索引；
- SQLite FTS5 或独立 BM25：关键词与精确文本检索；
- NetworkX：早期图算法验证；数据量和并发达到瓶颈后再评估 Ladybug 或图数据库；
- 原始文件目录：内容寻址、只读保存；
- JSON Schema/Pydantic：接口和数据校验。

工程要求：

- 所有 ingest、extract、index 和 correct 操作具有 job ID，可重试、可恢复；
- 每一步记录输入版本、输出版本、代码版本和错误；
- 写入采用事务或 staging + 原子切换；
- 派生索引损坏时可从权威库重建；
- 新 embedding、parser 或 schema 使用新版本并行构建，验证后切换；
- 不把数据库放入 Python 包安装目录；
- 运行配置、数据路径和模型版本必须由统一配置读取，文档不得与代码各写一份真相。

## 11. 实施与迁移策略

3.0 在独立分支和 worktree 中开发，避免影响 OpenClaw 当前使用的 2.0：

- 稳定运行目录：`~/.openclaw/workspace/projects/neurograph`
- 计划开发目录：`~/Projects/neurograph-graph-3.0`
- 计划分支：`feature/graph-3.0`

实施阶段：

### Phase 0：基准与数据契约

- 固化 2.0 回归结果；
- 建立真实问题、标准 Claim、EvidenceSlot 和引用定位集；
- 定义 Pydantic/JSON Schema、稳定 ID 和版本规则；
- 建立 RawAsset 内容寻址存储。

### Phase 1：证据底座

- 导入 PPT/PDF/Markdown/音频；
- 建立 SourceElement 和 Observation；
- 解决相邻幻灯片重复语音问题；
- 验证任意 Observation 可回到原始页面、区域或时间段。

### Phase 2：混合检索最小闭环

- Passage、Claim 候选和实体抽取；
- 向量、BM25、数字、实体和结构索引；
- QueryPlan、EvidenceSlots 和 EvidencePack；
- 暂不加入 ZenBrain，先建立可靠基线。

### Phase 3：图与冲突

- 类型化关系和受约束 beam search；
- TopicUnit/KnowledgeUnit；
- ClaimVersion、冲突分类、纠错和增量索引。

### Phase 4：ZenBrain

- 事件账本；
- 节点、边、路径和用户上下文先验；
- 用户反馈与纠错传播；
- 与禁用 ZenBrain 的基线进行消融测试。

### Phase 5：FSRS 与回答反馈

- 接入现有 `@zensation/algorithms` FSRS，保存 Observation 节点状态；
- 将可检索性作为弱先验，不改变事实相关性、证据质量和槽位覆盖判断；
- 提供 `selected/cited/followed_up/user_confirmed/corrected/rejected` 的回答层显式事件接口，覆盖 Observation、Relation 和 Path；
- 让 EvidencePack 返回关联 ClaimVersion，支持事实级确认、纠错和受控证据传播；
- 验证检索不会隐式强化，反馈才会改变 FSRS 状态；
- 后续再扩展边、路径和用户上下文的调度状态。

### Phase 6：迁移与切换

- 从 Cognee 2.0 导入候选实体、关系和文本引用；
- 所有导入内容重新绑定 SourceElement 和 EvidenceLink；
- A/B 比较 2.0 与 3.0；
- 达到切换门槛后再修改 OpenClaw 默认入口；
- 保留 2.0 一键回退。

## 12. 评测集与验收标准

### 12.1 回归问题类型

回归集至少包含：

- 精确数字、单位、误差和范围；
- 术语、定义和缩写；
- 方法模块、输入输出和完整流程；
- 跨页、跨文档、多跳问题；
- 表格行列、图表趋势和公式；
- 否定、限制、适用条件；
- 相似文本中的真实冲突；
- 明显 OCR/ASR 错误；
- 时间变化和版本问题；
- 宽泛领域问题和真正需要追问的歧义问题；
- 冷门但高度相关的事实。

### 12.2 首版切换门槛

以下阈值是首版工程目标；正式实现后可根据人工标注集规模调整，但调整必须记录理由：

| 指标 | 目标 |
|---|---:|
| RawAsset / SourceElement 可追溯率 | 100% |
| 重复灌入新增权威节点数 | 0 |
| 精确事实 Claim Recall@20 | ≥ 95% |
| 多跳问题完整证据集召回率 | ≥ 90% |
| 数字、单位、极性准确率 | ≥ 99% |
| 关键 Claim 引用支持精确率 | ≥ 98% |
| 关键 Claim 引用覆盖率 | ≥ 95% |
| 已知冲突识别率 | ≥ 95% |
| superseded/rejected 事实默认泄漏率 | ≤ 1% |
| 无歧义问题误追问率 | ≤ 5% |
| 有实质歧义问题漏追问率 | ≤ 10% |
| ZenBrain 开启后的冷事实 Recall@20 降幅 | ≤ 2 个百分点 |
| 检索 p95（不含最终回答生成） | ≤ 5 秒 |

### 12.3 对照与消融

每次重要版本至少比较：

1. BM25；
2. Dense retrieval；
3. BM25 + Dense；
4. 2.0 Cognee graph-evidence；
5. 3.0 多路召回但无图；
6. 3.0 图检索；
7. 3.0 图检索 + ZenBrain。

同时对 chunk size、结构前缀、contextual prefix、reranker、hop 数、beam width 和证据预算做消融，分别测召回、完整性、引用质量、延迟和存储开销。

## 13. 当前已知技术债与未决项

### 2.0 技术债

- `zenbrain_llm.py` 在候选被召回时就执行强化，与 3.0 事件原则冲突；
- 默认 graph-evidence 没有经过 ZenBrain；
- `TECH.md` 中的 DeepSeek/Embedding 配置与当前代码存在漂移；
- Cognee 数据实际落在包目录，不能作为 3.0 权威存储；
- 当前合并讲座中存在跨幻灯片重复语音；
- 现有 graph-evidence 是固定种子、固定两跳邻域，不具备 QueryPlan、EvidenceSlots 和覆盖检查。

这些问题在 2.0 保持运行期间只记录，不在 3.0 开发前顺手重构；3.0 通过新数据模型和接口解决。

### 实施时通过实验决定

- Passage 的不同文档类型最佳长度；
- contextual prefix 的生成方式和成本；
- Claim/实体抽取模型与 prompt；
- 实体自动合并阈值；
- rank fusion、reranker 和路径评分权重；
- SQLite + NetworkX 的性能上限和是否需要迁移图数据库；
- 是否扩展本地 embedding 服务以支持 Late Chunking。

## 14. 参考设计依据

- [Microsoft GraphRAG indexing and query documentation](https://microsoft.github.io/graphrag/)
- [RAPTOR: Recursive Abstractive Processing for Tree-Organized Retrieval](https://arxiv.org/abs/2401.18059)
- [Late Chunking: Contextual Chunk Embeddings Using Long-Context Embedding Models](https://arxiv.org/abs/2409.04701)
- [Anthropic Contextual Retrieval](https://www.anthropic.com/engineering/contextual-retrieval)
- [Hierarchical Lexical Graph for Enhanced Multi-Hop Retrieval](https://arxiv.org/abs/2506.08074)
- [RAGChecker](https://arxiv.org/abs/2408.08067)
- [Correctness is not Faithfulness in RAG Attributions](https://arxiv.org/abs/2412.18004)
- [W3C PROV-O](https://www.w3.org/TR/prov-o/)
- [Lost in the Middle](https://aclanthology.org/2024.tacl-1.9/)
- [Docling](https://github.com/docling-project/docling)
