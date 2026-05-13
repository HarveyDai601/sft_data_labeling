# SFT Trace Labeling Pipeline 流程图

## 一、整体流程

```
输入                               处理                              输出
─────                              ─────                             ─────

data/task_001/
├── main.json        ──┐
├── file_retrieval_0.json │         ┌──────────────┐
├── step_retrieval_0.json │         │   Loader     │
├── code_completion_0.json│    ──→  │ 加载所有文件  │  ──→  TaskData
├── consistency_check_0.json        │ 建立对应关系  │
├── reference.py     ──┘           └──────┬───────┘
                                              │
                                              ▼
                                    ┌──────────────────┐
                                    │     Pipeline      │
                                    │ 对 main 和每个    │
                                    │ sub 独立处理      │
                                    └──────┬───────────┘
                                              │
                         ┌────────────────────┼────────────────────┐
                         ▼                    ▼                    ▼
                   ┌───────────┐       ┌───────────┐       ┌───────────┐
                   │ Evaluator │       │  Cleaner  │       │ Dependency│
                   │  评估阶段  │  ──→  │  清洗阶段  │  ──→  │ Resolver  │
                   │           │       │           │       │ 依赖解析  │
                   └───────────┘       └───────────┘       └───────────┘
                                              │
                                              ▼
                                    ┌──────────────────┐
                                    │     Output       │
                                    │ cleaned JSON     │
                                    │ + 处理报告        │
                                    └──────────────────┘
```

## 二、评估阶段详解

评估器是可插拔的，每个评估器独立检查一个维度。核心思路：

```
                ┌─────────────────────────────────────┐
                │         AST 初筛 + LLM 终判          │
                │                                     │
                │  快速标记疑点（确定性） →  LLM 语义判断  │
                │  成本低、无幻觉           成本高、但准确  │
                └─────────────────────────────────────┘
```

### 2.1 代码差异评估 (code_diff)

```
reference.py                    trace 中的代码
─────────────                   ──────────────
def process(data):              def process(data):
    result = transform(data)        result = transform(data)
    return result                   return result
                                +   extra_function()     ← 多余的

         ┌──────────────────────────────────┐
         │  逐行 diff（difflib）              │
         │  → 找出 多余 / 缺少 / 修改 的行     │
         └──────────────┬───────────────────┘
                        ▼
         ┌──────────────────────────────────┐
         │  LLM 语义判断                      │
         │  → 多余的代码是否功能等价？          │
         │  → 缺少的代码是否被替代实现？        │
         └──────────────────────────────────┘
```

### 2.2 函数准召率评估 (function_recall)

```
         reference.py
         ────────────
         def process_data(data):        ← 定义
             result = transform(data)   ← 调用
                                         ──────────────
         ┌──────────────┐               │ AST 解析      │
         │ AST 解析      │               │ 提取所有函数名  │
         │ 提取函数定义   │               └───────┬──────┘
         │ + 外部调用     │                       │
         └──────┬───────┘                       │
                │                               │
                ▼                               ▼
         期望 API 集合:                    实际 API 集合:
         {process_data, transform}       {transform}

                        差集 = 疑点: {process_data}
                                │
                                ▼
                 ┌──────────────────────────────┐
                 │  LLM 终判                      │
                 │  "process_data 是核心函数，     │
                 │   trace 中确实没有 → 确认缺失"   │
                 └──────────────────────────────┘
```

**关键设计**：
- 无 LLM 时 → 只检查 reference 中的**顶层函数定义**（启发式过滤）
- 有 LLM 时 → LLM 判断是"真正缺失"还是"等价实现"
- severity = 缺失比例 × 1.5（不是绝对数量）

### 2.3 参数准召率评估 (param_recall)

```
reference 函数签名:              trace 函数签名:
def process(data, config)       def process(input_data, cfg)
    │                               │
    └─────────┬─────────────────────┘
              ▼
     按函数名配对，比较参数
     process: {data, config} vs {input_data, cfg}
              │
              ▼
     差集 = 疑点: {data, config}
              │
              ▼
     ┌────────────────────────────────────┐
     │  LLM 终判                           │
     │  "input_data ≈ data (语义等价),     │
     │   cfg ≈ config (语义等价)            │
     │   → 无真正缺失"                      │
     └────────────────────────────────────┘
```

### 2.4 其他评估器

```
评估器               检查什么                    判断依据
─────────           ─────────                  ─────────
file_relevance      检索的文件是否相关            reference 的 import 列表
step_relevance      检索的步骤是否相关            reference 的函数/类/关键操作
consistency         一致性检查结论是否正确         LLM 验证
tool_call_decision  调用顺序是否合理              预定义顺序规则
```

## 三、清洗阶段详解

### 3.1 清洗粒度：content item 级别

一条 OpenAI message 的 content 可以是字符串或数组：

```
message.content = [
    {"type": "text", "text": "分析代码..."},          ← item 0
    {"type": "text", "text": "多余的一段"},            ← item 1
    {"type": "tool_use", "name": "write_file"},      ← item 2
]
```

每个 item 独立评估，独立清洗。

### 3.2 清洗动作

```
动作              含义                    触发条件
─────            ─────                   ─────────
KEEP             保留原样                  评估通过
DELETE            删除该 item              评估为 bad，severity 低
REPLACE           替换内容                  评估为 bad，有 suggested_fix
REWRITE           LLM 重写                 评估为 bad，severity 高
INSERT_BEFORE     在前面插入                评估为 missing
INSERT_AFTER      在后面插入                评估为 missing
DELETE_MESSAGE    删除整条 message          所有 item 都被 DELETE
```

### 3.3 冲突解决

多个评估器可能对同一条 message 给出不同判断：

```
评估器 A: "这条 message 有问题" (severity=0.8)
评估器 B: "这条 message 没问题" (severity=0.0)
                    │
                    ▼
            severity 高的优先
            保守策略: DELETE > REPLACE > KEEP
                    │
                    ▼
            最终: REWRITE (severity=0.8)
```

### 3.4 特殊保护：tool_calls 不被误删

```
assistant message:
├── content: "调用 retrieval agent..."     ← 可以清空
└── tool_calls: [{name: "task", ...}]     ← 必须保留

tool response message:
├── content: "retrieval result..."         ← 可以清空
└── tool_call_id: "xxx"                   ← 必须保留
```

DELETE 时：有 tool_calls/tool_call_id 的 message → 只清空 content，保留结构。

## 四、依赖解析

```
消息 #3 被 DELETE
    │
    ▼
检查后续消息是否依赖 #3
    │
    ├── 消息 #5 的 content 引用了 #3 中的变量？
    │       → 是: 消息 #5 标记为 REWRITE
    │
    ├── 消息 #7 的 tool_call 参数依赖 #3 的内容？
    │       → 是: 消息 #7 标记为 REWRITE
    │
    └── 消息 #9 引用了 #5（而 #5 刚被标记 REWRITE）？
            → 是: 消息 #9 也标记为 REWRITE
                    │
                    ▼
            最大追踪深度 = 3（防止连锁反应）
                    │
                    ▼
            LLM 重新生成受影响的消息
```

## 五、输出

```
output/task_001/
├── main_cleaned.json              ← 清洗后的 SFT 数据（OpenAI messages 格式）
├── file_retrieval_0_cleaned.json  ← 每个 sub agent 独立输出
├── step_retrieval_0_cleaned.json
├── code_completion_0_cleaned.json
├── consistency_check_0_cleaned.json
├── main_report.md                 ← 处理报告（为什么这么处理）
├── ...
└── summary.md                     ← 汇总统计
```

报告示例：
```markdown
## 评估结果（按维度）
### function_recall
- 消息 #3: **MISSING** — 核心 API 缺失: process_data (severity: 0.5)

## 清洗操作
- 消息 #3: **REWRITE** — LLM 重写补全缺失逻辑 (来源: function_recall)

## 依赖处理
- 消息 #3 被重写 → 消息 #7 依赖 #3 → LLM 重写消息 #7
```

## 六、配置驱动

```yaml
# 哪些评估器对哪种 trace 生效
evaluators:
  code_diff:        { applies_to: [code_completion] }
  function_recall:  { applies_to: [code_completion, main] }
  param_recall:     { applies_to: [code_completion, main] }
  file_relevance:   { applies_to: [file_retrieval] }
  step_relevance:   { applies_to: [step_retrieval] }
  consistency:      { applies_to: [consistency_check] }
  tool_call_decision: { applies_to: [main] }

# LLM 指向内网，不对外发请求
llm:
  base_url: "http://localhost:8000/v1"
  model: "your-model"
```

新增评估维度：写一个 `XxxEvaluator` 类 + 在 config 注册，不改其他代码。
