# LangChain 工具链组合设计 - 网络工程师智能助手

使用 LangChain 组合多种工具完成网络工程复杂任务的智能助手项目。

## 项目概述

本项目演示如何使用 LangChain 框架将多种专业工具组合起来，构建一个面向网络工程师的智能助手。通过 Agent 自主选择和调用工具，自动化完成网络故障诊断、配置分析、日志处理等复杂任务。

## 核心功能

### 🔧 工具集

| 工具 | 功能描述 |
|------|---------|
| 文本分析工具 | 文本统计、情感分析、关键词提取 |
| 数据转换工具 | JSON/YAML/CSV 互转、Cisco 配置转 JSON |
| 文本处理工具 | 文本清洗、分割、正则匹配、IP/URL/邮箱提取 |
| 网络诊断工具 | Ping 测试、DNS 解析、端口检测、路由追踪 |
| 配置分析工具 | 设备配置解析、安全检查、接口/路由提取 |
| 日志分析工具 | 日志统计、错误提取、模式识别、防火墙日志分析 |

### 🤖 Agent 能力

- **自动工具选择**：根据用户问题自动选择合适的工具
- **多工具协作**：支持多轮工具调用完成复杂任务
- **对话记忆**：保持上下文，支持多轮对话
- **错误处理**：优雅处理工具调用失败情况

## 项目结构

```
18-CASE-工具链组合设计/
├── code/                      # 核心代码
│   ├── __init__.py
│   ├── main.py               # 主程序入口
│   ├── tools/                # 工具模块
│   │   ├── __init__.py
│   │   ├── text_analysis.py      # 文本分析工具
│   │   ├── data_conversion.py    # 数据转换工具
│   │   ├── text_processing.py    # 文本处理工具
│   │   ├── network_diagnosis.py  # 网络诊断工具
│   │   ├── config_analysis.py    # 配置分析工具
│   │   └── log_analysis.py       # 日志分析工具
│   └── agents/               # Agent 模块
│       ├── __init__.py
│       └── network_engineer_agent.py  # 网络工程师 Agent
├── data/                     # 数据目录
├── docs/                     # 文档目录
├── output/                   # 输出结果
├── tests/                    # 测试文件
│   └── test_tool_chain.py
└── README.md
```

## 快速开始

### 1. 环境准备

```bash
# 进入项目目录
cd practice/18-CASE-工具链组合设计

# 激活虚拟环境（在项目根目录）
source ../../.venv/bin/activate

# 设置 API Key
export DASHSCOPE_API_KEY="your_dashscope_api_key"
# 或
export OPENAI_API_KEY="your_openai_api_key"
```

### 2. 运行模式

```bash
# 交互模式 - 与 Agent 对话
python code/main.py --mode interactive

# 演示模式 - 展示各工具功能
python code/main.py --mode demo

# 测试模式 - 运行单元测试
python code/main.py --mode test
```

## 使用示例

### 网络诊断

```
👤 你: 帮我 ping 一下 www.baidu.com

🤖 助手: [调用网络诊断工具]
📡 Ping 测试：www.baidu.com
--------------------------------------------------
正在 Ping www.baidu.com [180.101.50.188] 具有 64 字节的数据:
  来自 180.101.50.188 的回复: 字节=64 时间=15ms TTL=52
  ...
✅ 状态：主机可达
```

### 配置分析

```
👤 你: 分析这段 Cisco 配置的安全性：
hostname Router1
interface GigabitEthernet0/0
 ip address 192.168.1.1 255.255.255.0
!
line vty 0 4
 transport input ssh
!

🤖 助手: [调用配置分析工具]
🔒 安全配置检查结果：
--------------------------------------------------
✅ 已通过检查：
   • SSH 远程访问 [高]
   • 禁用 Telnet [高]
⚠️ 建议改进：
   • 密码加密服务 [高]
   • 登录横幅设置 [中]
📊 安全评分：50/100
```

### 日志分析

```
👤 你: 提取这段日志中的错误：
2024-01-15 ERROR Connection failed
2024-01-15 INFO Backup completed
2024-01-15 CRITICAL Disk full

🤖 助手: [调用日志分析工具]
❌ 错误日志提取：
--------------------------------------------------
🔴 【CRITICAL】
    2024-01-15 CRITICAL Disk full

🟠 【ERROR】
    2024-01-15 ERROR Connection failed
```

### 文本处理

```
👤 你: 从这段文本中提取所有 IP 地址：
服务器 192.168.1.100 连接到 10.0.0.1 失败

🤖 助手: [调用文本处理工具]
  1. 192.168.1.100
  2. 10.0.0.1
```

## 工具详细说明

### 1. 文本分析工具 (TextAnalysisTool)

```python
from tools import TextAnalysisTool

tool = TextAnalysisTool()
result = tool.run("你的文本内容")
# 返回：字符数、词数、高频关键词、情感倾向等
```

### 2. 数据转换工具 (DataConversionTool)

```python
from tools import DataConversionTool

tool = DataConversionTool()

# JSON 转 YAML
result = tool.run('json2yaml|{"key": "value"}')

# 格式化 JSON
result = tool.run('format|{"key": "value"}')

# Cisco 配置转 JSON
result = tool.run('cisco2json|hostname Router1')
```

### 3. 文本处理工具 (TextProcessingTool)

```python
from tools import TextProcessingTool

tool = TextProcessingTool()

# 提取 IP 地址
result = tool.run('extract_ip|服务器 IP：192.168.1.1')

# 正则匹配
result = tool.run('regex|\d{4}-\d{2}-\d{2}|日期：2024-01-15')

# 文本清洗
result = tool.run('clean|  多余  空白  ')
```

### 4. 网络诊断工具 (NetworkDiagnosisTool)

```python
from tools import NetworkDiagnosisTool

tool = NetworkDiagnosisTool()

# Ping 测试
result = tool.run('ping|www.baidu.com')

# DNS 解析
result = tool.run('dns|www.google.com')

# 端口检测
result = tool.run('port|192.168.1.1|80')

# 综合检查
result = tool.run('check|192.168.1.1')
```

### 5. 配置分析工具 (ConfigAnalysisTool)

```python
from tools import ConfigAnalysisTool

tool = ConfigAnalysisTool()

# 解析配置
result = tool.run('parse|hostname Router1\ninterface GE0/0')

# 安全检查
result = tool.run('security|完整的 Cisco 配置')

# 提取接口
result = tool.run('interfaces|配置内容')

# 识别厂商
result = tool.run('vendor|配置内容')
```

### 6. 日志分析工具 (LogAnalysisTool)

```python
from tools import LogAnalysisTool

tool = LogAnalysisTool()

# 日志摘要
result = tool.run('summary|多行日志内容')

# 错误提取
result = tool.run('errors|多行日志内容')

# IP 统计
result = tool.run('ips|多行日志内容')

# 防火墙日志分析
result = tool.run('firewall|防火墙日志')
```

## 自定义扩展

### 添加新工具

```python
from langchain.tools import Tool

class MyCustomTool:
    def __init__(self):
        self.name = "自定义工具"
        self.description = "工具描述"
    
    def run(self, input_str: str) -> str:
        # 实现你的逻辑
        return "结果"

# 在 Agent 中注册
tools.append(Tool(
    name="自定义工具",
    func=MyCustomTool().run,
    description="工具描述"
))
```

### 切换 LLM

```python
from agents import NetworkEngineerAgent

# 使用通义千问
agent = NetworkEngineerAgent(
    llm_type="tongyi",
    model_name="qwen-turbo"
)

# 使用 OpenAI
agent = NetworkEngineerAgent(
    llm_type="openai",
    model_name="gpt-4"
)
```

## 技术栈

- **LangChain**: LLM 应用开发框架
- **LangChain Community**: 社区组件
- **LangChain OpenAI**: OpenAI 集成
- **通义千问**: 阿里云大语言模型
- **Python 3.11+**: 编程语言

## 参考资料

- [LangChain 官方文档](https://python.langchain.com/docs/)
- [课程 18-LangChain：多任务应用开发](../../courseware/18-LangChain：多任务应用开发/)
- [ReAct Agent 论文](https://arxiv.org/abs/2210.03629)

## 许可证

MIT License
