"""
配置分析工具 - ConfigAnalysisTool

功能：
- 网络设备配置解析（Cisco/Juniper/Huawei）
- 配置差异对比
- 安全配置检查
- 配置合规性验证
"""

import re


class ConfigAnalysisTool:
    """配置分析工具类"""
    
    def __init__(self):
        self.name = "配置分析工具"
        self.description = (
            "网络设备配置分析工具。"
            "支持 Cisco/Juniper/Huawei 配置解析、差异对比、安全检查。"
            "输入：格式为 '分析类型|配置内容' 或 '分析类型|参数|配置内容'。"
            "输出：分析结果。"
        )
        
        # 安全检查规则
        self.security_rules = {
            "password_encryption": {
                "pattern": r"service password-encryption",
                "description": "密码加密服务",
                "severity": "高"
            },
            "ssh_enabled": {
                "pattern": r"line vty.*\n.*transport input ssh",
                "description": "SSH 远程访问",
                "severity": "高"
            },
            "telnet_disabled": {
                "pattern": r"no.*telnet|transport input ssh",
                "description": "禁用 Telnet",
                "severity": "高"
            },
            "banner_set": {
                "pattern": r"banner (motd|login|exec)",
                "description": "登录横幅设置",
                "severity": "中"
            },
            "aaa_enabled": {
                "pattern": r"aaa new-model",
                "description": "AAA 认证",
                "severity": "高"
            },
            "logging_enabled": {
                "pattern": r"logging (host|buffered|trap)",
                "description": "日志记录",
                "severity": "中"
            },
            "ntp_configured": {
                "pattern": r"ntp (server|master)",
                "description": "NTP 时间同步",
                "severity": "低"
            },
        }
    
    def run(self, input_str: str) -> str:
        """
        运行配置分析
        
        Args:
            input_str: 格式为 "分析类型|配置内容"
            
        Returns:
            分析结果字符串
        """
        if not input_str or "|" not in input_str:
            return self._show_usage()
        
        try:
            parts = input_str.split("|", 1)
            if len(parts) != 2:
                return self._show_usage()
            
            analysis_type = parts[0].strip().lower()
            config = parts[1].strip()
            
            return self._analyze(analysis_type, config)
        except Exception as e:
            return f"分析过程中出现错误：{str(e)}"
    
    def _show_usage(self) -> str:
        """显示使用说明"""
        return """📋 配置分析工具使用说明：
格式：分析类型|配置内容

支持的分析类型：
  parse      - 解析配置结构
  security   - 安全配置检查
  summary    - 配置摘要
  interfaces - 接口配置提取
  routing    - 路由配置提取
  acl        - ACL 配置提取
  vendor     - 识别设备厂商

示例：
  parse|hostname Router1
  interface|interface GigabitEthernet0/0
  security|完整配置内容"""
    
    def _analyze(self, analysis_type: str, config: str) -> str:
        """执行分析"""
        analyzers = {
            "parse": self._parse_config,
            "security": self._security_check,
            "summary": self._config_summary,
            "interfaces": self._extract_interfaces,
            "routing": self._extract_routing,
            "acl": self._extract_acl,
            "vendor": self._identify_vendor,
        }
        
        if analysis_type not in analyzers:
            return f"不支持的分析类型：{analysis_type}\n{self._show_usage()}"
        
        return analyzers[analysis_type](config)
    
    def _parse_config(self, config: str) -> str:
        """解析配置结构"""
        lines = ["📄 配置解析结果："]
        lines.append("-" * 50)
        
        result = {
            "hostname": "",
            "version": "",
            "interfaces": [],
            "vlans": [],
            "routing_protocols": [],
            "acls": [],
            "ntp_servers": [],
            "dns_servers": [],
        }
        
        # 解析 hostname
        hostname_match = re.search(r"hostname\s+(\S+)", config)
        if hostname_match:
            result["hostname"] = hostname_match.group(1)
        
        # 解析 version
        version_match = re.search(r"version\s+(\S+)", config)
        if version_match:
            result["version"] = version_match.group(1)
        
        # 解析接口
        interfaces = re.findall(r"interface\s+(\S+)", config)
        result["interfaces"] = interfaces
        
        # 解析 VLAN
        vlans = re.findall(r"vlan\s+(\d+)", config)
        result["vlans"] = vlans
        
        # 解析路由协议
        if re.search(r"router\s+ospf", config):
            result["routing_protocols"].append("OSPF")
        if re.search(r"router\s+eigrp", config):
            result["routing_protocols"].append("EIGRP")
        if re.search(r"router\s+bgp", config):
            result["routing_protocols"].append("BGP")
        if re.search(r"router\s+rip", config):
            result["routing_protocols"].append("RIP")
        
        # 解析 ACL
        acls = re.findall(r"access-list\s+(\S+)", config)
        result["acls"] = list(set(acls))
        
        # 解析 NTP
        ntp_servers = re.findall(r"ntp\s+server\s+(\S+)", config)
        result["ntp_servers"] = ntp_servers
        
        # 解析 DNS
        dns_servers = re.findall(r"ip\s+name-server\s+(\S+)", config)
        result["dns_servers"] = dns_servers
        
        # 格式化输出
        lines.append(f"  设备名称: {result['hostname'] or '未设置'}")
        lines.append(f"  IOS 版本: {result['version'] or '未知'}")
        lines.append(f"  接口数量: {len(result['interfaces'])}")
        lines.append(f"  VLAN 数量: {len(result['vlans'])}")
        lines.append(f"  路由协议: {', '.join(result['routing_protocols']) or '无'}")
        lines.append(f"  ACL 数量: {len(result['acls'])}")
        lines.append(f"  NTP 服务器: {', '.join(result['ntp_servers']) or '未配置'}")
        lines.append(f"  DNS 服务器: {', '.join(result['dns_servers']) or '未配置'}")
        
        return "\n".join(lines)
    
    def _security_check(self, config: str) -> str:
        """安全配置检查"""
        lines = ["🔒 安全配置检查结果："]
        lines.append("-" * 50)
        
        passed = []
        failed = []
        warnings = []
        
        for rule_name, rule in self.security_rules.items():
            pattern = rule["pattern"]
            description = rule["description"]
            severity = rule["severity"]
            
            if re.search(pattern, config, re.MULTILINE | re.DOTALL):
                passed.append((description, severity))
            else:
                if severity == "高":
                    failed.append((description, severity))
                else:
                    warnings.append((description, severity))
        
        # 输出结果
        lines.append("\n✅ 已通过检查：")
        for desc, sev in passed:
            lines.append(f"   • {desc} [{sev}]")
        
        if failed:
            lines.append("\n❌ 未通过检查：")
            for desc, sev in failed:
                lines.append(f"   • {desc} [{sev}]")
        
        if warnings:
            lines.append("\n⚠️ 建议改进：")
            for desc, sev in warnings:
                lines.append(f"   • {desc} [{sev}]")
        
        # 计算安全评分
        total = len(passed) + len(failed) + len(warnings)
        score = int((len(passed) / total) * 100) if total > 0 else 0
        
        lines.append(f"\n📊 安全评分：{score}/100")
        
        if score >= 80:
            lines.append("   状态：良好 ✨")
        elif score >= 60:
            lines.append("   状态：一般 ⚠️")
        else:
            lines.append("   状态：需要改进 ❌")
        
        return "\n".join(lines)
    
    def _config_summary(self, config: str) -> str:
        """配置摘要"""
        lines = ["📝 配置摘要："]
        lines.append("-" * 50)
        
        # 统计配置行数
        config_lines = [l for l in config.split("\n") if l.strip() and not l.strip().startswith("!")]
        total_lines = len(config_lines)
        
        # 解析主要配置块
        sections = {
            "hostname": len(re.findall(r"^hostname", config, re.MULTILINE)),
            "interface": len(re.findall(r"^interface", config, re.MULTILINE)),
            "vlan": len(re.findall(r"^vlan", config, re.MULTILINE)),
            "router": len(re.findall(r"^router", config, re.MULTILINE)),
            "access-list": len(re.findall(r"^access-list", config, re.MULTILINE)),
            "line": len(re.findall(r"^line", config, re.MULTILINE)),
        }
        
        lines.append(f"  总配置行数: {total_lines}")
        lines.append("\n  配置块统计:")
        for section, count in sections.items():
            if count > 0:
                lines.append(f"    • {section}: {count} 个")
        
        return "\n".join(lines)
    
    def _extract_interfaces(self, config: str) -> str:
        """提取接口配置"""
        lines = ["🔌 接口配置："]
        lines.append("-" * 50)
        
        # 匹配接口配置块
        pattern = r"interface\s+(\S+)\s*\n((?:(?!interface|!)[\s\S])*?)(?=interface|!|$)"
        matches = re.findall(pattern, config, re.MULTILINE)
        
        if not matches:
            lines.append("  未找到接口配置")
            return "\n".join(lines)
        
        for iface_name, iface_config in matches:
            lines.append(f"\n  【{iface_name}】")
            
            # 提取关键配置
            desc_match = re.search(r"description\s+(.+)", iface_config)
            if desc_match:
                lines.append(f"    描述: {desc_match.group(1).strip()}")
            
            ip_match = re.search(r"ip\s+address\s+(\S+)\s+(\S+)", iface_config)
            if ip_match:
                lines.append(f"    IP地址: {ip_match.group(1)}/{ip_match.group(2)}")
            
            status_match = re.search(r"(no\s+)?shutdown", iface_config)
            if status_match:
                status = "关闭" if status_match.group(1) else "开启"
                lines.append(f"    状态: {status}")
            
            vlan_match = re.search(r"switchport\s+access\s+vlan\s+(\d+)", iface_config)
            if vlan_match:
                lines.append(f"    Access VLAN: {vlan_match.group(1)}")
        
        return "\n".join(lines)
    
    def _extract_routing(self, config: str) -> str:
        """提取路由配置"""
        lines = ["🛤️ 路由配置："]
        lines.append("-" * 50)
        
        # 静态路由
        static_routes = re.findall(r"ip\s+route\s+(\S+)\s+(\S+)\s+(\S+)", config)
        if static_routes:
            lines.append("\n  【静态路由】")
            for dest, mask, next_hop in static_routes:
                lines.append(f"    目的: {dest}/{mask} -> 下一跳: {next_hop}")
        
        # OSPF
        ospf_match = re.search(r"router\s+ospf\s+(\d+)", config)
        if ospf_match:
            lines.append(f"\n  【OSPF 进程 {ospf_match.group(1)}】")
            networks = re.findall(r"network\s+(\S+)\s+(\S+)\s+area\s+(\S+)", config)
            for net, mask, area in networks:
                lines.append(f"    网络: {net}/{mask} -> 区域: {area}")
        
        # BGP
        bgp_match = re.search(r"router\s+bgp\s+(\d+)", config)
        if bgp_match:
            lines.append(f"\n  【BGP AS {bgp_match.group(1)}】")
            neighbors = re.findall(r"neighbor\s+(\S+)\s+remote-as\s+(\d+)", config)
            for neighbor, remote_as in neighbors:
                lines.append(f"    邻居: {neighbor} -> AS: {remote_as}")
        
        if not any([static_routes, ospf_match, bgp_match]):
            lines.append("\n  未配置路由协议")
        
        return "\n".join(lines)
    
    def _extract_acl(self, config: str) -> str:
        """提取 ACL 配置"""
        lines = ["🛡️ ACL 配置："]
        lines.append("-" * 50)
        
        # 标准 ACL
        standard_acls = re.findall(r"access-list\s+(\d+)\s+(permit|deny)\s+(\S+)", config)
        if standard_acls:
            lines.append("\n  【标准 ACL】")
            for acl_num, action, source in standard_acls:
                lines.append(f"    ACL {acl_num}: {action} {source}")
        
        # 扩展 ACL
        extended_acls = re.findall(r"access-list\s+(\d+)\s+extended\s+(permit|deny)\s+(\S+)\s+(\S+)\s+(\S+)", config)
        if extended_acls:
            lines.append("\n  【扩展 ACL】")
            for acl_num, action, protocol, source, dest in extended_acls:
                lines.append(f"    ACL {acl_num}: {action} {protocol} {source} -> {dest}")
        
        if not any([standard_acls, extended_acls]):
            lines.append("\n  未配置 ACL")
        
        return "\n".join(lines)
    
    def _identify_vendor(self, config: str) -> str:
        """识别设备厂商"""
        lines = ["🏭 设备厂商识别："]
        lines.append("-" * 50)
        
        vendor_patterns = {
            "Cisco": [
                (r"hostname\s+\S+", "hostname 配置"),
                (r"interface\s+(GigabitEthernet|FastEthernet|Serial)", "Cisco 接口命名"),
                (r"router\s+(ospf|eigrp|bgp)", "Cisco 路由协议"),
                (r"version\s+\d+\.\d+", "IOS 版本"),
            ],
            "Huawei": [
                (r"sysname\s+\S+", "Huawei sysname"),
                (r"interface\s+(GigabitEthernet|Ethernet)\d+/\d+/\d+", "Huawei 接口命名"),
                (r"huawei", "Huawei 关键字"),
            ],
            "Juniper": [
                (r"set\s+system\s+host-name", "Juniper hostname"),
                (r"interfaces\s+\S+\s+unit", "Juniper 接口配置"),
                (r"junos", "Junos 关键字"),
            ],
        }
        
        scores = {}
        for vendor, patterns in vendor_patterns.items():
            score = 0
            matched = []
            for pattern, desc in patterns:
                if re.search(pattern, config, re.IGNORECASE):
                    score += 1
                    matched.append(desc)
            scores[vendor] = (score, matched)
        
        # 找出最可能的厂商
        best_vendor = max(scores, key=lambda x: scores[x][0])
        best_score, matched_items = scores[best_vendor]
        
        if best_score > 0:
            lines.append(f"  识别结果: {best_vendor}")
            lines.append(f"  匹配特征:")
            for item in matched_items:
                lines.append(f"    • {item}")
            lines.append(f"  置信度: {best_score}/{len(vendor_patterns[best_vendor])}")
        else:
            lines.append("  无法识别设备厂商")
        
        return "\n".join(lines)


def create_config_analysis_tool():
    """创建 LangChain Tool 实例"""
    from langchain_core.tools import Tool
    
    tool = ConfigAnalysisTool()
    return Tool(
        name=tool.name,
        func=tool.run,
        description=tool.description
    )
