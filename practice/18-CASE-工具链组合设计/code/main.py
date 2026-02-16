"""
LangChain 工具链组合设计 - 主程序入口

演示如何使用 LangChain 组合多种工具完成网络工程复杂任务。

项目功能：
1. 文本分析工具 - 文本统计、情感分析、关键词提取
2. 数据转换工具 - JSON/YAML/CSV 互转、配置格式转换
3. 文本处理工具 - 文本清洗、分割、正则匹配、IP/URL 提取
4. 网络诊断工具 - Ping、DNS、端口检测、路由追踪
5. 配置分析工具 - 设备配置解析、安全检查、差异对比
6. 日志分析工具 - 日志统计、错误提取、模式识别

使用方法：
    python main.py --mode interactive    # 交互模式
    python main.py --mode demo           # 演示模式
    python main.py --mode test           # 测试模式
"""

import os
import sys
import argparse
from pathlib import Path
from typing import Optional

# 添加项目路径
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# 导入工具和 Agent
from tools import (
    TextAnalysisTool,
    DataConversionTool,
    TextProcessingTool,
    NetworkDiagnosisTool,
    ConfigAnalysisTool,
    LogAnalysisTool,
)
from agents import NetworkEngineerAgent


def print_banner():
    """打印项目横幅"""
    banner = """
╔══════════════════════════════════════════════════════════════╗
║       LangChain 工具链组合设计 - 网络工程师智能助手          ║
╠══════════════════════════════════════════════════════════════╣
║  使用 LangChain 组合多种工具，自动化网络工程复杂任务         ║
║                                                              ║
║  工具列表：                                                  ║
║  • 文本分析工具 - 文本统计、情感分析、关键词提取            ║
║  • 数据转换工具 - JSON/YAML/CSV 互转、配置格式转换          ║
║  • 文本处理工具 - 文本清洗、分割、正则匹配、IP 提取         ║
║  • 网络诊断工具 - Ping、DNS、端口检测、路由追踪             ║
║  • 配置分析工具 - 设备配置解析、安全检查                    ║
║  • 日志分析工具 - 日志统计、错误提取、模式识别              ║
╚══════════════════════════════════════════════════════════════╝
"""
    print(banner)


def interactive_mode():
    """交互模式 - 与 Agent 对话"""
    print_banner()
    
    # 检查 API Key
    api_key = os.environ.get("DASHSCOPE_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("❌ 错误：请设置 DASHSCOPE_API_KEY 或 OPENAI_API_KEY 环境变量")
        print("   你可以创建 .env 文件并添加：DASHSCOPE_API_KEY=your_key")
        return
    
    # 确定 LLM 类型
    llm_type = "tongyi" if os.environ.get("DASHSCOPE_API_KEY") else "openai"
    
    try:
        # 创建 Agent
        print("\n正在初始化智能助手...")
        agent = NetworkEngineerAgent(llm_type=llm_type, verbose=True)
        
        print("\n✅ 智能助手已就绪！")
        print("\n" + "=" * 60)
        print("对话指南：")
        print("  - 描述你的网络问题，助手会自动选择合适的工具")
        print("  - 输入 'quit' 退出程序")
        print("  - 输入 'clear' 清除对话记忆")
        print("  - 输入 'help' 查看使用示例")
        print("=" * 60)
        
        while True:
            try:
                user_input = input("\n👤 你: ").strip()
                
                if not user_input:
                    continue
                
                if user_input.lower() == "quit":
                    print("\n👋 再见！感谢使用网络工程师智能助手")
                    break
                
                if user_input.lower() == "clear":
                    agent.clear_memory()
                    print("✅ 对话记忆已清除")
                    continue
                
                if user_input.lower() == "help":
                    show_help_examples()
                    continue
                
                # 获取响应
                print("\n🤖 助手: ", end="")
                response = agent.chat(user_input)
                print(response)
                
            except KeyboardInterrupt:
                print("\n\n👋 再见！")
                break
            except Exception as e:
                print(f"\n❌ 错误: {e}")
    
    except Exception as e:
        print(f"\n❌ 初始化失败: {e}")


def show_help_examples():
    """显示帮助示例"""
    examples = """
📚 使用示例：

【网络诊断】
  - 帮我 ping 一下 www.baidu.com
  - 检查 192.168.1.1 的 80 端口是否开放
  - 查询 www.google.com 的 DNS 解析

【配置分析】
  - 分析这段 Cisco 配置的安全性：[粘贴配置]
  - 提取配置中的接口信息
  - 识别设备厂商

【日志分析】
  - 分析这段日志中的错误：[粘贴日志]
  - 统计日志中的 IP 地址出现频率
  - 识别日志类型

【文本处理】
  - 从这段文本中提取所有 IP 地址：[文本]
  - 清洗这段文本中的多余空白：[文本]

【数据转换】
  - 将这段 JSON 转换为 YAML 格式：[JSON]
  - 验证这段 JSON 是否有效：[JSON]
"""
    print(examples)


def demo_mode():
    """演示模式 - 展示各工具功能"""
    print_banner()
    print("\n🎬 演示模式 - 展示各工具功能\n")
    
    # 1. 文本分析工具演示
    print("=" * 60)
    print("📝 1. 文本分析工具演示")
    print("=" * 60)
    text_tool = TextAnalysisTool()
    sample_text = """
网络工程师是负责计算机网络设计、实施和维护的专业人员。
他们需要掌握路由器、交换机、防火墙等网络设备的配置和管理。
优秀的网络工程师应该具备故障诊断能力，能够快速定位和解决网络问题。
"""
    print(f"输入文本：{sample_text[:50]}...")
    print("\n" + text_tool.run(sample_text))
    
    # 2. 数据转换工具演示
    print("\n" + "=" * 60)
    print("🔄 2. 数据转换工具演示")
    print("=" * 60)
    data_tool = DataConversionTool()
    json_data = '{"hostname": "Router1", "ip": "192.168.1.1", "status": "active"}'
    print(f"输入 JSON：{json_data}")
    print("\n" + data_tool.run(f"json2yaml|{json_data}"))
    
    # 3. 文本处理工具演示
    print("\n" + "=" * 60)
    print("✂️ 3. 文本处理工具演示")
    print("=" * 60)
    proc_tool = TextProcessingTool()
    log_text = "服务器 192.168.1.100 连接到 10.0.0.1 失败，错误来自 172.16.0.50"
    print(f"输入文本：{log_text}")
    print("\n" + proc_tool.run(f"extract_ip|{log_text}"))
    
    # 4. 网络诊断工具演示
    print("\n" + "=" * 60)
    print("🌐 4. 网络诊断工具演示")
    print("=" * 60)
    net_tool = NetworkDiagnosisTool()
    print("执行：ping www.baidu.com")
    print("\n" + net_tool.run("ping|www.baidu.com"))
    
    # 5. 配置分析工具演示
    print("\n" + "=" * 60)
    print("⚙️ 5. 配置分析工具演示")
    print("=" * 60)
    config_tool = ConfigAnalysisTool()
    cisco_config = """
hostname Router1
!
interface GigabitEthernet0/0
 ip address 192.168.1.1 255.255.255.0
 no shutdown
!
interface GigabitEthernet0/1
 ip address 10.0.0.1 255.255.255.0
 shutdown
!
router ospf 1
 network 192.168.1.0 0.0.0.255 area 0
!
line vty 0 4
 transport input ssh
 login local
!
service password-encryption
!
"""
    print("分析 Cisco 配置...")
    print("\n" + config_tool.run(f"parse|{cisco_config}"))
    print("\n" + config_tool.run(f"security|{cisco_config}"))
    
    # 6. 日志分析工具演示
    print("\n" + "=" * 60)
    print("📊 6. 日志分析工具演示")
    print("=" * 60)
    log_tool = LogAnalysisTool()
    sample_logs = """
2024-01-15 10:23:45 ERROR Connection failed from 192.168.1.100 to 10.0.0.1
2024-01-15 10:24:12 WARNING High CPU usage detected on server 192.168.1.50
2024-01-15 10:25:33 INFO User admin logged in from 192.168.1.10
2024-01-15 10:26:01 ERROR Database connection timeout from 192.168.1.100
2024-01-15 10:27:15 CRITICAL Disk space critical on server 192.168.1.50
2024-01-15 10:28:00 INFO Backup completed successfully
"""
    print("分析日志...")
    print("\n" + log_tool.run(f"summary|{sample_logs}"))
    print("\n" + log_tool.run(f"errors|{sample_logs}"))
    
    print("\n" + "=" * 60)
    print("✅ 演示完成！运行 'python main.py --mode interactive' 开始交互模式")
    print("=" * 60)


def test_mode():
    """测试模式 - 运行单元测试"""
    print_banner()
    print("\n🧪 测试模式 - 运行单元测试\n")
    
    # 运行测试
    test_file = PROJECT_ROOT / "tests" / "test_tool_chain.py"
    
    if test_file.exists():
        import subprocess
        result = subprocess.run(
            [sys.executable, "-m", "pytest", str(test_file), "-v"],
            cwd=str(PROJECT_ROOT)
        )
    else:
        print("❌ 测试文件不存在，正在创建...")
        create_test_file()
        print("✅ 测试文件已创建，请重新运行")


def create_test_file():
    """创建测试文件"""
    test_content = '''"""
工具链测试文件
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools import (
    TextAnalysisTool,
    DataConversionTool,
    TextProcessingTool,
    NetworkDiagnosisTool,
    ConfigAnalysisTool,
    LogAnalysisTool,
)


def test_text_analysis():
    """测试文本分析工具"""
    tool = TextAnalysisTool()
    result = tool.run("这是一个测试文本")
    assert "字符总数" in result
    print("✅ 文本分析工具测试通过")


def test_data_conversion():
    """测试数据转换工具"""
    tool = DataConversionTool()
    result = tool.run(\'format|{"key": "value"}\')
    assert "key" in result
    print("✅ 数据转换工具测试通过")


def test_text_processing():
    """测试文本处理工具"""
    tool = TextProcessingTool()
    result = tool.run("extract_ip|192.168.1.1 和 10.0.0.1")
    assert "192.168.1.1" in result
    print("✅ 文本处理工具测试通过")


def test_network_diagnosis():
    """测试网络诊断工具"""
    tool = NetworkDiagnosisTool()
    result = tool.run("ping|localhost")
    assert "Ping" in result or "ping" in result.lower()
    print("✅ 网络诊断工具测试通过")


def test_config_analysis():
    """测试配置分析工具"""
    tool = ConfigAnalysisTool()
    result = tool.run("parse|hostname TestRouter")
    assert "设备名称" in result or "TestRouter" in result
    print("✅ 配置分析工具测试通过")


def test_log_analysis():
    """测试日志分析工具"""
    tool = LogAnalysisTool()
    result = tool.run("summary|2024-01-15 ERROR Test message")
    assert "日志" in result
    print("✅ 日志分析工具测试通过")


if __name__ == "__main__":
    test_text_analysis()
    test_data_conversion()
    test_text_processing()
    test_network_diagnosis()
    test_config_analysis()
    test_log_analysis()
    print("\\n🎉 所有测试通过！")
'''
    
    test_file = PROJECT_ROOT / "tests" / "test_tool_chain.py"
    test_file.write_text(test_content, encoding="utf-8")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="LangChain 工具链组合设计 - 网络工程师智能助手"
    )
    parser.add_argument(
        "--mode",
        choices=["interactive", "demo", "test"],
        default="interactive",
        help="运行模式：interactive（交互）、demo（演示）、test（测试）"
    )
    
    args = parser.parse_args()
    
    if args.mode == "interactive":
        interactive_mode()
    elif args.mode == "demo":
        demo_mode()
    elif args.mode == "test":
        test_mode()


if __name__ == "__main__":
    main()
