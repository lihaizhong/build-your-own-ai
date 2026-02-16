"""
日志分析工具 - LogAnalysisTool

功能：
- 日志解析和统计
- 错误日志提取
- 日志模式识别
- 网络日志分析（Syslog、防火墙日志等）
"""

import re
from collections import Counter


class LogAnalysisTool:
    """日志分析工具类"""
    
    def __init__(self):
        self.name = "日志分析工具"
        self.description = (
            "日志分析工具。"
            "支持日志解析、错误提取、模式识别、统计分析。"
            "输入：格式为 '分析类型|日志内容'。"
            "输出：分析结果。"
        )
        
        # 常见日志模式
        self.log_patterns = {
            "syslog": r"(\w+\s+\d+\s+\d+:\d+:\d+)\s+(\S+)\s+(\S+?)(?:\[\d+\])?:\s+(.*)",
            "apache": r'(\S+)\s+(\S+)\s+(\S+)\s+\[([^\]]+)\]\s+"([^"]+)"\s+(\d+)\s+(\d+)',
            "nginx": r'(\S+)\s+-\s+(\S+)\s+\[([^\]]+)\]\s+"([^"]+)"\s+(\d+)\s+(\d+)',
            "firewall": r"(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}).*?(ALLOW|DENY|DROP).*?src=(\S+).*?dst=(\S+)",
        }
        
        # 错误级别关键词
        self.error_keywords = {
            "critical": ["critical", "fatal", "emergency", "panic", "CRITICAL", "FATAL"],
            "error": ["error", "fail", "failed", "exception", "ERROR", "FAIL", "FAILED"],
            "warning": ["warning", "warn", "caution", "WARNING", "WARN"],
            "info": ["info", "information", "notice", "INFO", "NOTICE"],
            "debug": ["debug", "trace", "DEBUG", "TRACE"],
        }
    
    def run(self, input_str: str) -> str:
        """
        运行日志分析
        
        Args:
            input_str: 格式为 "分析类型|日志内容"
            
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
            logs = parts[1].strip()
            
            return self._analyze(analysis_type, logs)
        except Exception as e:
            return f"分析过程中出现错误：{str(e)}"
    
    def _show_usage(self) -> str:
        """显示使用说明"""
        return """📋 日志分析工具使用说明：
格式：分析类型|日志内容

支持的分析类型：
  summary    - 日志统计摘要
  errors     - 提取错误日志
  timeline   - 时间线分析
  ips        - IP 地址统计
  patterns   - 日志模式识别
  firewall   - 防火墙日志分析
  level      - 日志级别统计

示例：
  summary|多行日志内容
  errors|多行日志内容
  ips|多行日志内容"""
    
    def _analyze(self, analysis_type: str, logs: str) -> str:
        """执行分析"""
        analyzers = {
            "summary": self._log_summary,
            "errors": self._extract_errors,
            "timeline": self._timeline_analysis,
            "ips": self._ip_statistics,
            "patterns": self._pattern_recognition,
            "firewall": self._firewall_analysis,
            "level": self._level_statistics,
        }
        
        if analysis_type not in analyzers:
            return f"不支持的分析类型：{analysis_type}\n{self._show_usage()}"
        
        return analyzers[analysis_type](logs)
    
    def _log_summary(self, logs: str) -> str:
        """日志统计摘要"""
        lines = ["📊 日志统计摘要："]
        lines.append("-" * 50)
        
        log_lines = [l for l in logs.split("\n") if l.strip()]
        total_lines = len(log_lines)
        
        # 时间范围
        timestamps = re.findall(r"\d{4}-\d{2}-\d{2}[T\s]\d{2}:\d{2}:\d{2}", logs)
        if not timestamps:
            timestamps = re.findall(r"\w{3}\s+\d+\s+\d{2}:\d{2}:\d{2}", logs)
        
        # 统计各级别日志数量
        level_counts = {}
        for level, keywords in self.error_keywords.items():
            count = sum(logs.lower().count(kw.lower()) for kw in keywords[:2])
            if count > 0:
                level_counts[level] = count
        
        # 统计 IP 地址
        ips = re.findall(r'\b(?:\d{1,3}\.){3}\d{1,3}\b', logs)
        unique_ips = set(ips)
        
        # 统计端口
        ports = re.findall(r'port[:\s]+(\d+)', logs, re.IGNORECASE)
        
        lines.append(f"  总日志行数: {total_lines}")
        if timestamps:
            lines.append(f"  时间戳数量: {len(timestamps)}")
        lines.append(f"  唯一 IP 数: {len(unique_ips)}")
        if ports:
            lines.append(f"  端口引用数: {len(ports)}")
        
        if level_counts:
            lines.append("\n  日志级别分布:")
            for level, count in sorted(level_counts.items(), key=lambda x: -x[1]):
                lines.append(f"    • {level.upper()}: {count}")
        
        return "\n".join(lines)
    
    def _extract_errors(self, logs: str) -> str:
        """提取错误日志"""
        lines = ["❌ 错误日志提取："]
        lines.append("-" * 50)
        
        log_lines = logs.split("\n")
        error_lines = []
        
        for log_line in log_lines:
            if not log_line.strip():
                continue
            
            # 检查是否包含错误关键词
            for level, keywords in self.error_keywords.items():
                if level in ["critical", "error", "warning"]:
                    for kw in keywords:
                        if kw in log_line:
                            error_lines.append((level, log_line.strip()))
                            break
        
        if not error_lines:
            lines.append("  ✅ 未发现错误日志")
            return "\n".join(lines)
        
        # 按级别分组
        by_level = {}
        for level, line in error_lines:
            if level not in by_level:
                by_level[level] = []
            by_level[level].append(line)
        
        for level in ["critical", "error", "warning"]:
            if level in by_level:
                symbol = "🔴" if level == "critical" else "🟠" if level == "error" else "🟡"
                lines.append(f"\n  【{level.upper()}】 {symbol}")
                for line in by_level[level][:5]:  # 只显示前5条
                    lines.append(f"    {line[:100]}{'...' if len(line) > 100 else ''}")
                if len(by_level[level]) > 5:
                    lines.append(f"    ... 还有 {len(by_level[level]) - 5} 条")
        
        lines.append(f"\n  总计发现 {len(error_lines)} 条异常日志")
        
        return "\n".join(lines)
    
    def _timeline_analysis(self, logs: str) -> str:
        """时间线分析"""
        lines = ["📅 时间线分析："]
        lines.append("-" * 50)
        
        # 提取时间戳和事件
        log_lines = logs.split("\n")
        events = []
        
        for line in log_lines:
            if not line.strip():
                continue
            
            # 尝试匹配常见时间格式
            ts_match = re.search(r"(\d{4}-\d{2}-\d{2}[T\s]\d{2}:\d{2}:\d{2})", line)
            if not ts_match:
                ts_match = re.search(r"(\w{3}\s+\d+\s+\d{2}:\d{2}:\d{2})", line)
            
            if ts_match:
                timestamp = ts_match.group(1)
                event = line.strip()[:80]
                events.append((timestamp, event))
        
        if not events:
            lines.append("  未找到时间戳信息")
            return "\n".join(lines)
        
        lines.append(f"  找到 {len(events)} 条带时间戳的日志\n")
        
        # 显示前10条事件
        for ts, event in events[:10]:
            lines.append(f"  [{ts}] {event}{'...' if len(event) >= 80 else ''}")
        
        if len(events) > 10:
            lines.append(f"\n  ... 还有 {len(events) - 10} 条日志")
        
        return "\n".join(lines)
    
    def _ip_statistics(self, logs: str) -> str:
        """IP 地址统计"""
        lines = ["🌐 IP 地址统计："]
        lines.append("-" * 50)
        
        # 提取所有 IP 地址
        ips = re.findall(r'\b(?:\d{1,3}\.){3}\d{1,3}\b', logs)
        
        if not ips:
            lines.append("  未找到 IP 地址")
            return "\n".join(lines)
        
        # 统计频率
        ip_counter = Counter(ips)
        
        lines.append(f"  总 IP 出现次数: {len(ips)}")
        lines.append(f"  唯一 IP 数量: {len(ip_counter)}")
        
        lines.append("\n  高频 IP 地址 TOP 10:")
        for ip, count in ip_counter.most_common(10):
            # 判断 IP 类型
            if ip.startswith("192.168."):
                ip_type = "内网"
            elif ip.startswith("10."):
                ip_type = "内网"
            elif ip.startswith("172."):
                ip_type = "内网"
            elif ip.startswith("127."):
                ip_type = "本地"
            else:
                ip_type = "公网"
            
            lines.append(f"    • {ip}: {count} 次 [{ip_type}]")
        
        return "\n".join(lines)
    
    def _pattern_recognition(self, logs: str) -> str:
        """日志模式识别"""
        lines = ["🔍 日志模式识别："]
        lines.append("-" * 50)
        
        log_lines = [l for l in logs.split("\n") if l.strip()]
        
        # 识别常见模式
        patterns_found = {}
        
        # Syslog 模式
        syslog_matches = re.findall(self.log_patterns["syslog"], logs)
        if syslog_matches:
            patterns_found["Syslog"] = len(syslog_matches)
        
        # Apache 访问日志
        apache_matches = re.findall(self.log_patterns["apache"], logs)
        if apache_matches:
            patterns_found["Apache Access"] = len(apache_matches)
        
        # Nginx 访问日志
        nginx_matches = re.findall(self.log_patterns["nginx"], logs)
        if nginx_matches:
            patterns_found["Nginx Access"] = len(nginx_matches)
        
        # 防火墙日志
        fw_matches = re.findall(self.log_patterns["firewall"], logs, re.IGNORECASE)
        if fw_matches:
            patterns_found["Firewall"] = len(fw_matches)
        
        # SSH 日志
        ssh_matches = re.findall(r"ssh\d?|sshd", logs, re.IGNORECASE)
        if ssh_matches:
            patterns_found["SSH"] = len(ssh_matches)
        
        # DHCP 日志
        dhcp_matches = re.findall(r"dhcp|DHCP", logs)
        if dhcp_matches:
            patterns_found["DHCP"] = len(dhcp_matches)
        
        # DNS 日志
        dns_matches = re.findall(r"dns|named|DNS", logs)
        if dns_matches:
            patterns_found["DNS"] = len(dns_matches)
        
        if patterns_found:
            lines.append("  识别到的日志类型:")
            for pattern, count in sorted(patterns_found.items(), key=lambda x: -x[1]):
                lines.append(f"    • {pattern}: {count} 条")
        else:
            lines.append("  未识别到常见日志模式")
        
        return "\n".join(lines)
    
    def _firewall_analysis(self, logs: str) -> str:
        """防火墙日志分析"""
        lines = ["🔥 防火墙日志分析："]
        lines.append("-" * 50)
        
        # 查找允许/拒绝记录
        allow_pattern = r"(ALLOW|PASS|ACCEPT|permit)"
        deny_pattern = r"(DENY|DROP|BLOCK|reject|deny)"
        
        allows = re.findall(allow_pattern, logs, re.IGNORECASE)
        denies = re.findall(deny_pattern, logs, re.IGNORECASE)
        
        lines.append(f"  允许连接: {len(allows)} 次")
        lines.append(f"  拒绝连接: {len(denies)} 次")
        
        # 提取源/目标 IP
        src_ips = re.findall(r"src[=:\s]+(\S+)", logs, re.IGNORECASE)
        dst_ips = re.findall(r"dst[=:\s]+(\S+)", logs, re.IGNORECASE)
        dst_ports = re.findall(r"dpt[=:\s]+(\d+)|dstport[=:\s]+(\d+)", logs, re.IGNORECASE)
        
        if src_ips:
            src_counter = Counter(src_ips)
            lines.append("\n  源 IP TOP 5:")
            for ip, count in src_counter.most_common(5):
                lines.append(f"    • {ip}: {count} 次")
        
        if dst_ips:
            dst_counter = Counter(dst_ips)
            lines.append("\n  目标 IP TOP 5:")
            for ip, count in dst_counter.most_common(5):
                lines.append(f"    • {ip}: {count} 次")
        
        if dst_ports:
            ports = [p[0] or p[1] for p in dst_ports]
            port_counter = Counter(ports)
            lines.append("\n  目标端口 TOP 5:")
            for port, count in port_counter.most_common(5):
                lines.append(f"    • 端口 {port}: {count} 次")
        
        return "\n".join(lines)
    
    def _level_statistics(self, logs: str) -> str:
        """日志级别统计"""
        lines = ["📈 日志级别统计："]
        lines.append("-" * 50)
        
        level_counts = {}
        for level, keywords in self.error_keywords.items():
            count = sum(logs.count(kw) for kw in keywords)
            if count > 0:
                level_counts[level] = count
        
        if not level_counts:
            lines.append("  未能识别日志级别")
            return "\n".join(lines)
        
        total = sum(level_counts.values())
        
        for level in ["critical", "error", "warning", "info", "debug"]:
            if level in level_counts:
                count = level_counts[level]
                percentage = (count / total) * 100
                bar = "█" * int(percentage / 5)
                
                symbol = {
                    "critical": "🔴",
                    "error": "🟠",
                    "warning": "🟡",
                    "info": "🟢",
                    "debug": "🔵"
                }.get(level, "⚪")
                
                lines.append(f"  {symbol} {level.upper():10} {count:5} ({percentage:5.1f}%) {bar}")
        
        lines.append(f"\n  总计: {total}")
        
        return "\n".join(lines)


def create_log_analysis_tool():
    """创建 LangChain Tool 实例"""
    from langchain_core.tools import Tool
    
    tool = LogAnalysisTool()
    return Tool(
        name=tool.name,
        func=tool.run,
        description=tool.description
    )
