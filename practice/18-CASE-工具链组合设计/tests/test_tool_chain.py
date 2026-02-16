"""
工具链测试文件

测试所有自定义工具的基本功能
"""

import sys
import os
from pathlib import Path

# 添加项目路径
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from code.tools import (
    TextAnalysisTool,
    DataConversionTool,
    TextProcessingTool,
    NetworkDiagnosisTool,
    ConfigAnalysisTool,
    LogAnalysisTool,
)


def test_text_analysis():
    """测试文本分析工具"""
    print("\n📝 测试文本分析工具...")
    tool = TextAnalysisTool()
    
    # 测试基本分析
    result = tool.run("这是一个测试文本，包含一些中文字符。")
    assert "字符总数" in result, "应该包含字符总数"
    assert "中文字符数" in result, "应该包含中文字符数"
    
    # 测试空输入
    result = tool.run("")
    assert "错误" in result or "无效" in result, "应该提示错误"
    
    print("  ✅ 文本分析工具测试通过")


def test_data_conversion():
    """测试数据转换工具"""
    print("\n🔄 测试数据转换工具...")
    tool = DataConversionTool()
    
    # 测试 JSON 格式化
    result = tool.run('format|{"key": "value"}')
    assert "key" in result, "应该包含 key"
    
    # 测试 JSON 转 YAML
    result = tool.run('json2yaml|{"name": "test"}')
    assert "name" in result, "应该包含 name"
    
    # 测试 JSON 验证
    result = tool.run('validate|{"valid": true}')
    assert "正确" in result or "✅" in result, "应该验证通过"
    
    # 测试无效 JSON
    result = tool.run('validate|{invalid}')
    assert "错误" in result or "❌" in result, "应该验证失败"
    
    print("  ✅ 数据转换工具测试通过")


def test_text_processing():
    """测试文本处理工具"""
    print("\n✂️ 测试文本处理工具...")
    tool = TextProcessingTool()
    
    # 测试 IP 提取
    result = tool.run("extract_ip|服务器地址是 192.168.1.1 和 10.0.0.1")
    assert "192.168.1.1" in result, "应该提取到 192.168.1.1"
    assert "10.0.0.1" in result, "应该提取到 10.0.0.1"
    
    # 测试文本清洗
    result = tool.run("clean|  多余   空白  字符  ")
    assert "多余 空白 字符" in result, "应该清洗空白"
    
    # 测试大小写转换
    result = tool.run("uppercase|hello")
    assert "HELLO" in result, "应该转为大写"
    
    result = tool.run("lowercase|HELLO")
    assert "hello" in result, "应该转为小写"
    
    print("  ✅ 文本处理工具测试通过")


def test_network_diagnosis():
    """测试网络诊断工具"""
    print("\n🌐 测试网络诊断工具...")
    tool = NetworkDiagnosisTool()
    
    # 测试 Ping
    result = tool.run("ping|localhost")
    assert "Ping" in result or "ping" in result.lower(), "应该包含 Ping"
    
    # 测试 DNS
    result = tool.run("dns|www.baidu.com")
    assert "DNS" in result or "解析" in result, "应该包含 DNS 或解析"
    
    # 测试端口检测
    result = tool.run("port|192.168.1.1|80")
    assert "端口" in result or "80" in result, "应该包含端口信息"
    
    # 测试综合检查
    result = tool.run("check|localhost")
    assert "检查" in result or "连通" in result, "应该包含检查结果"
    
    print("  ✅ 网络诊断工具测试通过")


def test_config_analysis():
    """测试配置分析工具"""
    print("\n⚙️ 测试配置分析工具...")
    tool = ConfigAnalysisTool()
    
    # 测试配置解析
    config = "hostname TestRouter\ninterface GE0/0\n ip address 192.168.1.1 255.255.255.0"
    result = tool.run(f"parse|{config}")
    assert "TestRouter" in result or "设备名称" in result, "应该包含设备名"
    
    # 测试安全检查
    result = tool.run(f"security|{config}")
    assert "安全" in result or "评分" in result, "应该包含安全检查结果"
    
    # 测试接口提取
    result = tool.run(f"interfaces|{config}")
    assert "接口" in result or "GE0/0" in result, "应该包含接口信息"
    
    # 测试厂商识别
    result = tool.run(f"vendor|{config}")
    assert "厂商" in result or "Cisco" in result, "应该包含厂商信息"
    
    print("  ✅ 配置分析工具测试通过")


def test_log_analysis():
    """测试日志分析工具"""
    print("\n📊 测试日志分析工具...")
    tool = LogAnalysisTool()
    
    sample_logs = """
2024-01-15 10:23:45 ERROR Connection failed from 192.168.1.100
2024-01-15 10:24:12 WARNING High CPU usage
2024-01-15 10:25:33 INFO User logged in
2024-01-15 10:26:01 CRITICAL Disk full
"""
    
    # 测试摘要
    result = tool.run(f"summary|{sample_logs}")
    assert "日志" in result, "应该包含日志统计"
    
    # 测试错误提取
    result = tool.run(f"errors|{sample_logs}")
    assert "ERROR" in result or "错误" in result or "CRITICAL" in result, "应该包含错误信息"
    
    # 测试 IP 统计
    result = tool.run(f"ips|{sample_logs}")
    assert "IP" in result or "192.168.1.100" in result, "应该包含 IP 信息"
    
    # 测试级别统计
    result = tool.run(f"level|{sample_logs}")
    assert "ERROR" in result or "INFO" in result, "应该包含日志级别"
    
    print("  ✅ 日志分析工具测试通过")


def run_all_tests():
    """运行所有测试"""
    print("=" * 60)
    print("🧪 工具链单元测试")
    print("=" * 60)
    
    try:
        test_text_analysis()
        test_data_conversion()
        test_text_processing()
        test_network_diagnosis()
        test_config_analysis()
        test_log_analysis()
        
        print("\n" + "=" * 60)
        print("🎉 所有测试通过！")
        print("=" * 60)
        return True
        
    except AssertionError as e:
        print(f"\n❌ 测试失败: {e}")
        return False
    except Exception as e:
        print(f"\n❌ 测试异常: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
