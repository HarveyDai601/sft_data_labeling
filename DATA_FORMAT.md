# 输入数据结构说明

## 1. 整体目录结构

```
data/
├── task_001/
│   ├── main.json                    # Main agent trace
│   ├── script_retrieval_0.json        # Sub agent trace (type 1)
│   ├── script_retrieval_1.json        # Sub agent trace (type 1)
│   ├── step_retrieval_0.json        # Sub agent trace (type 2)
│   ├── script_completion_0.json       # Sub agent trace (type 3)
│   ├── consistency_check_0.json     # Sub agent trace (type 4)
│   ├── reference.py                 # 参考代码
│   └── context.md                   # （可选）上下文信息
├── task_002/
│   └── ...
└── task_003/
    └── ...
```

## 2. Trace JSON 格式（OpenAI Chat Completion）

每个 `*.json` 文件的结构是标准的 OpenAI 消息格式：

```json
{
  "messages": [
    {
      "role": "user",
      "content": "Write a function that calculates factorial"
    },
    {
      "role": "assistant",
      "content": "I'll help you...",
      "tool_calls": [
        {
          "id": "call_001",
          "type": "function",
          "function": {
            "name": "script_retrieval",
            "arguments": "{\"file_path\": \"math.py\"}"
          }
        }
      ]
    },
    {
      "role": "tool",
      "tool_call_id": "call_001",
      "name": "script_retrieval",
      "content": "def factorial(n):\n    return n * factorial(n-1)"
    },
    {
      "role": "assistant",
      "content": "I found the factorial function. Now I'll..."
    },
    ...
  ]
}
```

## 3. 消息角色类型

| 角色 | 说明 | 字段 |
|------|------|------|
| `user` | 用户或系统输入 | `content` |
| `assistant` | AI agent 的回复 | `content`, `tool_calls` (optional) |
| `tool` | 工具执行结果 | `tool_call_id`, `name`, `content` |
| `system` | 系统级别的说明 | `content` |

## 4. Tool Call 的结构

```json
{
  "id": "call_abc123",           // 唯一 ID，用来关联 tool 消息
  "type": "function",
  "function": {
    "name": "script_retrieval",    // 工具类型：script_retrieval, step_retrieval, 等
    "arguments": "{...}"         // JSON 字符串格式的参数
  }
}
```

## 5. Tool Result 消息

```json
{
  "role": "tool",
  "tool_call_id": "call_abc123",     // 必须匹配对应的 tool_call.id
  "name": "script_retrieval",          // 工具类型
  "content": "file contents..."       // 工具的返回结果
}
```

## 6. Sub Agent 类型

| 类型 | 文件名前缀 | 说明 |
|------|----------|------|
| File Retrieval | `script_retrieval_*.json` | 检索相关文件 |
| Step Retrieval | `step_retrieval_*.json` | 检索解题步骤 |
| Code Completion | `script_completion_*.json` | 代码补全 |
| Consistency Check | `consistency_check_*.json` | 一致性检查 |

## 7. 完整示例

### Main trace（main.json）

```json
{
  "messages": [
    {
      "role": "user",
      "content": "根据 fibonacci.py，写一个计算第 n 项的函数"
    },
    {
      "role": "assistant",
      "content": "我需要先查看 fibonacci.py 的内容",
      "tool_calls": [
        {
          "id": "call_1",
          "type": "function",
          "function": {
            "name": "script_retrieval",
            "arguments": "{\"file_path\": \"fibonacci.py\"}"
          }
        }
      ]
    },
    {
      "role": "tool",
      "tool_call_id": "call_1",
      "name": "script_retrieval",
      "content": "def fib(n):\n    if n <= 1:\n        return n\n    return fib(n-1) + fib(n-2)"
    },
    {
      "role": "assistant",
      "content": "我看到现有的递归实现。现在让我检查性能..."
    },
    {
      "role": "assistant",
      "content": "我需要检查动态规划的方法。让我检索相关步骤。",
      "tool_calls": [
        {
          "id": "call_2",
          "type": "function",
          "function": {
            "name": "step_retrieval",
            "arguments": "{\"topic\": \"dynamic_programming_fibonacci\"}"
          }
        }
      ]
    },
    {
      "role": "tool",
      "tool_call_id": "call_2",
      "name": "step_retrieval",
      "content": "动态规划方案：\n1. 创建数组 dp\n2. dp[0] = 0, dp[1] = 1\n3. 循环填充..."
    },
    {
      "role": "assistant",
      "content": "根据步骤，我实现了 O(n) 的方案。\n\ndef fib_dp(n):\n    if n <= 1:\n        return n\n    dp = [0] * (n + 1)\n    dp[1] = 1\n    for i in range(2, n + 1):\n        dp[i] = dp[i-1] + dp[i-2]\n    return dp[n]"
    }
  ]
}
```

### Sub trace - script_retrieval（script_retrieval_0.json）

```json
{
  "messages": [
    {
      "role": "user",
      "content": "根据 fibonacci.py，写一个计算第 n 项的函数"
    },
    {
      "role": "assistant",
      "content": "我需要先查看 fibonacci.py 的内容",
      "tool_calls": [
        {
          "id": "call_1",
          "type": "function",
          "function": {
            "name": "script_retrieval",
            "arguments": "{\"file_path\": \"fibonacci.py\"}"
          }
        }
      ]
    },
    {
      "role": "tool",
      "tool_call_id": "call_1",
      "name": "script_retrieval",
      "content": "def fib(n):\n    if n <= 1:\n        return n\n    return fib(n-1) + fib(n-2)"
    },
    {
      "role": "assistant",
      "content": "我成功检索了 fibonacci.py 的内容。文件中已经有了基础的递归实现。"
    }
  ]
}
```

## 8. 参考代码文件（reference.py）

```python
# reference.py - 标准答案或参考实现
def fib_optimal(n):
    """最优的 Fibonacci 实现（备忘录法）"""
    memo = {}
    
    def helper(n):
        if n in memo:
            return memo[n]
        if n <= 1:
            return n
        memo[n] = helper(n-1) + helper(n-2)
        return memo[n]
    
    return helper(n)

# 其他可能用到的函数...
```

## 9. 关键约束

1. **消息必须是连贯的**
   - user → assistant → tool → assistant → ...
   - 不能出现 assistant → assistant（中间没有 tool）
   - 不能出现 tool → tool（中间没有 assistant）

2. **tool_call 和 tool 消息必须成对**
   - assistant 中的每个 tool_call 必须在后面有对应的 tool 消息
   - tool 消息的 tool_call_id 必须匹配 tool_call.id

3. **content 不能为空**
   - 每条消息的 content 都必须有值（不能是 null 或空字符串）

4. **Sub traces 应该是 Main trace 的子集**
   - 不要包含 Main trace 中没有的 tool_call
   - 通常 Sub trace 只处理特定类型的工具调用

## 10. 检验清单

使用系统前，检查输入数据：

- [ ] 每个 task 目录包含 main.json 和至少一个 sub*.json
- [ ] 每个 JSON 文件有合法的 messages 数组
- [ ] 所有 tool_call 都有对应的 tool 消息
- [ ] 所有 tool 消息都有对应的 tool_call
- [ ] 消息顺序合理（user → assistant → tool 循环）
- [ ] 没有空的 content 字段
- [ ] reference.py 存在且包含标准答案
