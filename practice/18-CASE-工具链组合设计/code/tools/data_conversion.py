"""
数据转换工具 - DataConversionTool

功能：
- JSON/YAML/CSV 格式互转
- 数据格式验证
- 编码转换
- 网络数据格式处理（如 Cisco 配置格式转换）
"""

import json
from typing import Dict, Any


class DataConversionTool:
    """数据转换工具类"""
    
    def __init__(self):
        self.name = "数据转换工具"
        self.description = (
            "数据格式转换工具。"
            "支持 JSON/YAML/CSV 格式互转，数据格式验证。"
            "输入：格式为 '转换类型|数据内容'，如 'json2yaml|{\"key\": \"value\"}'。"
            "输出：转换后的数据或错误信息。"
        )
    
    def run(self, input_str: str) -> str:
        """
        运行数据转换
        
        Args:
            input_str: 格式为 "转换类型|数据内容"
            
        Returns:
            转换结果字符串
        """
        if not input_str or "|" not in input_str:
            return self._show_usage()
        
        try:
            parts = input_str.split("|", 1)
            if len(parts) != 2:
                return self._show_usage()
            
            conversion_type = parts[0].strip().lower()
            data = parts[1].strip()
            
            return self._convert(conversion_type, data)
        except Exception as e:
            return f"转换过程中出现错误：{str(e)}"
    
    def _show_usage(self) -> str:
        """显示使用说明"""
        return """📋 数据转换工具使用说明：
格式：转换类型|数据内容

支持的转换类型：
  json2yaml   - JSON 转 YAML
  yaml2json   - YAML 转 JSON
  json2csv    - JSON 数组转 CSV
  validate    - 验证 JSON 格式
  format      - 格式化 JSON
  cisco2json  - Cisco 配置转 JSON

示例：
  json2yaml|{"name": "router1", "ip": "192.168.1.1"}"""
    
    def _convert(self, conversion_type: str, data: str) -> str:
        """执行转换"""
        converters = {
            "json2yaml": self._json_to_yaml,
            "yaml2json": self._yaml_to_json,
            "json2csv": self._json_to_csv,
            "validate": self._validate_json,
            "format": self._format_json,
            "cisco2json": self._cisco_to_json,
        }
        
        if conversion_type not in converters:
            return f"不支持的转换类型：{conversion_type}\n{self._show_usage()}"
        
        return converters[conversion_type](data)
    
    def _json_to_yaml(self, data: str) -> str:
        """JSON 转 YAML"""
        try:
            parsed = json.loads(data)
            return self._dict_to_yaml(parsed)
        except json.JSONDecodeError as e:
            return f"JSON 解析错误：{str(e)}"
    
    def _dict_to_yaml(self, data: Any, indent: int = 0) -> str:
        """字典转 YAML 格式"""
        lines = []
        prefix = "  " * indent
        
        if isinstance(data, dict):
            for key, value in data.items():
                if isinstance(value, (dict, list)) and value:
                    lines.append(f"{prefix}{key}:")
                    lines.append(self._dict_to_yaml(value, indent + 1))
                else:
                    lines.append(f"{prefix}{key}: {value}")
        elif isinstance(data, list):
            for item in data:
                if isinstance(item, dict):
                    lines.append(f"{prefix}-")
                    lines.append(self._dict_to_yaml(item, indent + 1))
                else:
                    lines.append(f"{prefix}- {item}")
        else:
            lines.append(f"{prefix}{data}")
        
        return "\n".join(lines)
    
    def _yaml_to_json(self, data: str) -> str:
        """YAML 转 JSON（简单实现）"""
        try:
            # 简单的 YAML 解析器，处理基本格式
            result = self._parse_simple_yaml(data)
            return json.dumps(result, ensure_ascii=False, indent=2)
        except Exception as e:
            return f"YAML 解析错误：{str(e)}"
    
    def _parse_simple_yaml(self, data: str) -> Dict[str, Any]:
        """简单 YAML 解析"""
        result = {}
        lines = data.strip().split("\n")
        
        for line in lines:
            if ":" in line and not line.strip().startswith("-"):
                key, value = line.split(":", 1)
                result[key.strip()] = value.strip() if value.strip() else {}
        
        return result
    
    def _json_to_csv(self, data: str) -> str:
        """JSON 数组转 CSV"""
        try:
            parsed = json.loads(data)
            if not isinstance(parsed, list):
                return "错误：JSON 数据必须是数组格式"
            if not parsed:
                return "错误：JSON 数组为空"
            
            # 获取所有字段
            if isinstance(parsed[0], dict):
                headers = list(parsed[0].keys())
                lines = [",".join(headers)]
                
                for item in parsed:
                    values = [str(item.get(h, "")) for h in headers]
                    lines.append(",".join(values))
                
                return "\n".join(lines)
            else:
                return "\n".join(str(item) for item in parsed)
        except json.JSONDecodeError as e:
            return f"JSON 解析错误：{str(e)}"
    
    def _validate_json(self, data: str) -> str:
        """验证 JSON 格式"""
        try:
            json.loads(data)
            return "✅ JSON 格式正确"
        except json.JSONDecodeError as e:
            return f"❌ JSON 格式错误：{str(e)}"
    
    def _format_json(self, data: str) -> str:
        """格式化 JSON"""
        try:
            parsed = json.loads(data)
            return json.dumps(parsed, ensure_ascii=False, indent=2)
        except json.JSONDecodeError as e:
            return f"JSON 解析错误：{str(e)}"
    
    def _cisco_to_json(self, data: str) -> str:
        """Cisco 配置转 JSON"""
        result = {
            "hostname": "",
            "interfaces": [],
            "vlans": [],
            "routing": [],
            "acl": [],
        }
        
        lines = data.strip().split("\n")
        current_section = None
        current_interface = {}
        
        for line in lines:
            line = line.strip()
            if not line or line.startswith("!"):
                if current_interface:
                    result["interfaces"].append(current_interface)
                    current_interface = {}
                continue
            
            # 解析 hostname
            if line.startswith("hostname"):
                result["hostname"] = line.split()[1] if len(line.split()) > 1 else ""
            
            # 解析接口配置
            elif line.startswith("interface"):
                if current_interface:
                    result["interfaces"].append(current_interface)
                current_interface = {"name": line.split()[1] if len(line.split()) > 1 else "", "config": []}
            
            elif current_interface and line:
                current_interface["config"].append(line)
            
            # 解析 VLAN
            elif line.startswith("vlan"):
                vlan_num = line.split()[1] if len(line.split()) > 1 else ""
                result["vlans"].append({"id": vlan_num})
        
        if current_interface:
            result["interfaces"].append(current_interface)
        
        return json.dumps(result, ensure_ascii=False, indent=2)


def create_data_conversion_tool():
    """创建 LangChain Tool 实例"""
    from langchain_core.tools import Tool
    
    tool = DataConversionTool()
    return Tool(
        name=tool.name,
        func=tool.run,
        description=tool.description
    )
