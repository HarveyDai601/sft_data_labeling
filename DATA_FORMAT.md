# 输入数据结构说明

## 1. 整体目录结构

```
data/
├── task_001/
│   ├── main.json                    # Main agent trace
│   ├── script_retrieval_0.json      # Sub agent trace
│   ├── script_retrieval_1.json
│   ├── step_retrieval_0.json
│   ├── script_complete_0.json
│   ├── consistency_check_0.json
│   ├── reference.py                 # 参考实现
│   └── context.md                   # （可选）
├── task_002/
│   └── ...
```

## 2. 两层结构

```
Main trace (main.json)
  │
  │  assistant 通过 tool_call(function.name="task") 启动 sub-agent
  │  arguments.subagent_type = "script_retrieval" / "step_retrieval" / ...
  │  content 是对 sub-agent 的任务描述
  │  tool response 是 sub-agent 的最终返回结果
  │
  ├─→ script_retrieval_0.json    ← Sub-agent 独立 trace
  │     内部用 read/grep/glob 等实际工具执行文件检索
  │
  ├─→ step_retrieval_0.json      ← Sub-agent 独立 trace
  │     内部用 search/grep 等工具执行步骤检索
  │
  ├─→ script_complete_0.json     ← Sub-agent 独立 trace
  │     内部用 read 读上下文, write 写代码
  │
  └─→ consistency_check_0.json   ← Sub-agent 独立 trace
        内部检查代码一致性
```

**关键区别：Main trace 只做调度，Sub-agent traces 才做实际文件操作。**

## 3. Main trace — 调度 sub-agent

```json
{
  "messages": [
    {
      "role": "user",
      "content": "根据 math.py 实现 process_data 函数"
    },
    {
      "role": "assistant",
      "content": "我需要先检索 math.py 的内容。",
      "tool_calls": [
        {
          "id": "call_1",
          "type": "function",
          "function": {
            "name": "task",
            "arguments": "{\"subagent_type\": \"script_retrieval\", \"description\": \"检索 math.py 文件\"}"
          }
        }
      ]
    },
    {
      "role": "tool",
      "tool_call_id": "call_1",
      "name": "task",
      "content": "文件检索完成：math.py 包含 add, multiply, transform 三个函数"
    },
    {
      "role": "assistant",
      "content": "检索到了 math.py 的函数列表。现在检索相关步骤。",
      "tool_calls": [
        {
          "id": "call_2",
          "type": "function",
          "function": {
            "name": "task",
            "arguments": "{\"subagent_type\": \"step_retrieval\", \"description\": \"检索 process_data 的实现步骤\"}"
          }
        }
      ]
    },
    {
      "role": "tool",
      "tool_call_id": "call_2",
      "name": "task",
      "content": "步骤：1. 读取数据 2. 调用 transform 3. 返回结果"
    },
    {
      "role": "assistant",
      "content": "现在开始生成代码。",
      "tool_calls": [
        {
          "id": "call_3",
          "type": "function",
          "function": {
            "name": "task",
            "arguments": "{\"subagent_type\": \"script_complete\", \"description\": \"实现 process_data 函数\"}"
          }
        }
      ]
    },
    {
      "role": "tool",
      "tool_call_id": "call_3",
      "name": "task",
      "content": "代码已生成：def process_data(data): return transform(data)"
    }
  ]
}
```

**Main trace 中的 tool_call 模式：**
- `function.name` 始终是 `"task"`
- `arguments` 中包含 `subagent_type`：`script_retrieval` / `step_retrieval` / `script_complete` / `consistency_check`
- `content` 是对 sub-agent 的任务描述（自然语言）

## 4. Sub-agent trace — 实际文件操作

### 4.1 script_retrieval（文件检索）

```json
{
  "messages": [
    {
      "role": "user",
      "content": "检索 math.py 文件"
    },
    {
      "role": "assistant",
      "content": "让我用 glob 找到文件，再用 read 读取内容。",
      "tool_calls": [
        {
          "id": "call_a1",
          "type": "function",
          "function": {
            "name": "glob",
            "arguments": "{\"pattern\": \"**/math.py\"}"
          }
        }
      ]
    },
    {
      "role": "tool",
      "tool_call_id": "call_a1",
      "name": "glob",
      "content": "[\"math.py\"]"
    },
    {
      "role": "assistant",
      "content": "找到 math.py，现在读取内容。",
      "tool_calls": [
        {
          "id": "call_a2",
          "type": "function",
          "function": {
            "name": "read",
            "arguments": "{\"file_path\": \"math.py\"}"
          }
        }
      ]
    },
    {
      "role": "tool",
      "tool_call_id": "call_a2",
      "name": "read",
      "content": "def add(a, b): return a + b\ndef multiply(a, b): return a * b\ndef transform(data): return [x*2 for x in data]"
    },
    {
      "role": "assistant",
      "content": "math.py 包含 add, multiply, transform 三个函数。"
    }
  ]
}
```

### 4.2 script_complete（代码生成）

```json
{
  "messages": [
    {
      "role": "user",
      "content": "实现 process_data 函数"
    },
    {
      "role": "assistant",
      "content": "先读取 math.py 了解可用函数。",
      "tool_calls": [
        {
          "id": "call_b1",
          "type": "function",
          "function": {
            "name": "read",
            "arguments": "{\"file_path\": \"math.py\"}"
          }
        }
      ]
    },
    {
      "role": "tool",
      "tool_call_id": "call_b1",
      "name": "read",
      "content": "def transform(data): return [x*2 for x in data]"
    },
    {
      "role": "assistant",
      "content": "了解了 transform 函数，现在生成代码。",
      "tool_calls": [
        {
          "id": "call_b2",
          "type": "function",
          "function": {
            "name": "write",
            "arguments": "{\"file_path\": \"output.py\", \"content\": \"def process_data(data):\\n    return transform(data)\"}"
          }
        }
      ]
    },
    {
      "role": "tool",
      "tool_call_id": "call_b2",
      "name": "write",
      "content": "写入成功"
    }
  ]
}
```

## 5. 工具类型

### 调度工具（仅 main trace）

| 工具名 | 参数 | 说明 |
|--------|------|------|
| `task` | `subagent_type`, `description` | 启动 sub-agent |

### 文件操作工具（仅 sub-agent trace）

| 工具名 | 参数 | 说明 |
|--------|------|------|
| `read` | `file_path` | 读取文件 |
| `write` | `file_path`, `content` | 写入文件 |
| `edit` | `file_path`, `old_str`, `new_str` | 编辑文件 |
| `glob` | `pattern` | 文件匹配搜索 |
| `grep` / `rg` | `pattern`, `path` | 内容搜索 |

### 其他 sub-agent 工具

| 工具名 | 说明 | 使用的 sub-agent |
|--------|------|-----------------|
| `search` | 语义搜索 | step_retrieval |
| `write` / `edit` | 写/编辑代码 | script_complete |
| `consistency_check` | 一致性检查 | consistency_check |

## 6. 消息角色

| 角色 | 说明 | 关键字段 |
|------|------|----------|
| `user` | 任务输入 | `content` |
| `assistant` | 推理 + 工具调用 | `content`, `tool_calls[]` |
| `tool` | 工具执行结果 | `tool_call_id`, `name`, `content` |
| `system` | 系统指令 | `content` |

## 7. 评判逻辑对应

| 评估器 | 适用的 trace | 评判内容 |
|--------|------------|----------|
| `orchestration` | main | `task` 调用序列、子 agent 类型完整性 |
| `code_correctness` | script_complete | `write` 输出的代码 vs reference 函数级语义比对 |
| `info_usage` | script_complete | 子 agent 的 `read`→`write` 信息流追踪 |
| `consistency` | consistency_check | 一致性检查结论是否正确 |

## 8. 关键约束

1. Main trace 的 tool_call function.name 始终是 `task`
2. Sub-agent trace 的 tool_call function.name 是实际工具名（read/write/grep 等）
3. 每个 `tool_call_id` 必须有对应的 tool 消息
4. content 是 agent 的推理过程，代码在 tool_call arguments 中
5. Main trace 和 Sub-agent traces 是独立的消息序列，不是子集关系
