"""
网络诊断工具 - NetworkDiagnosisTool

功能：
- 模拟 Ping 测试
- 模拟 DNS 解析
- 模拟端口检测
- 网络连通性分析
- 路由追踪模拟
"""

import re
import random


class NetworkDiagnosisTool:
    """网络诊断工具类（模拟实现）"""
    
    def __init__(self):
        self.name = "网络诊断工具"
        self.description = (
            "网络诊断工具（模拟）。"
            "支持 Ping 测试、DNS 解析、端口检测、连通性分析等。"
            "输入：格式为 '诊断类型|目标地址' 或 '诊断类型|目标地址|端口'。"
            "输出：诊断结果。"
        )
        
        # 模拟一些已知的主机
        self.known_hosts = {
            "www.baidu.com": {"ip": "180.101.50.188", "status": "up"},
            "www.google.com": {"ip": "142.250.189.68", "status": "up"},
            "www.github.com": {"ip": "140.82.121.3", "status": "up"},
            "localhost": {"ip": "127.0.0.1", "status": "up"},
            "192.168.1.1": {"ip": "192.168.1.1", "status": "up"},
            "192.168.1.100": {"ip": "192.168.1.100", "status": "down"},
            "10.0.0.1": {"ip": "10.0.0.1", "status": "up"},
            "dns.google": {"ip": "8.8.8.8", "status": "up"},
        }
    
    def run(self, input_str: str) -> str:
        """
        运行网络诊断
        
        Args:
            input_str: 格式为 "诊断类型|目标地址" 或 "诊断类型|目标地址|端口"
            
        Returns:
            诊断结果字符串
        """
        if not input_str or "|" not in input_str:
            return self._show_usage()
        
        try:
            parts = input_str.split("|")
            diag_type = parts[0].strip().lower()
            
            if len(parts) == 2:
                target = parts[1].strip()
                return self._diagnose(diag_type, target)
            elif len(parts) >= 3:
                target = parts[1].strip()
                param = parts[2].strip()
                return self._diagnose_with_param(diag_type, target, param)
            else:
                return self._show_usage()
        except Exception as e:
            return f"诊断过程中出现错误：{str(e)}"
    
    def _show_usage(self) -> str:
        """显示使用说明"""
        return """📋 网络诊断工具使用说明：
格式：诊断类型|目标地址 或 诊断类型|目标地址|端口

支持的诊断类型：
  ping     - Ping 测试连通性
  dns      - DNS 解析查询
  port     - 端口检测
  trace    - 路由追踪（模拟）
  check    - 综合连通性检查

示例：
  ping|www.baidu.com
  dns|www.google.com
  port|192.168.1.1|80
  trace|www.github.com"""
    
    def _diagnose(self, diag_type: str, target: str) -> str:
        """执行诊断"""
        diags = {
            "ping": self._ping,
            "dns": self._dns_lookup,
            "trace": self._traceroute,
            "check": self._comprehensive_check,
        }
        
        if diag_type not in diags:
            return f"不支持的诊断类型：{diag_type}\n{self._show_usage()}"
        
        return diags[diag_type](target)
    
    def _diagnose_with_param(self, diag_type: str, target: str, param: str) -> str:
        """执行带参数的诊断"""
        if diag_type == "port":
            return self._port_check(target, param)
        else:
            return f"诊断类型 '{diag_type}' 不支持额外参数"
    
    def _ping(self, target: str) -> str:
        """模拟 Ping 测试"""
        lines = [f"📡 Ping 测试：{target}"]
        lines.append("-" * 50)
        
        # 检查是否是已知主机
        host_info = self.known_hosts.get(target)
        if host_info:
            ip = host_info["ip"]
            status = host_info["status"]
        else:
            # 生成随机 IP 或使用输入作为 IP
            if self._is_valid_ip(target):
                ip = target
            else:
                ip = f"模拟.{random.randint(1, 255)}.{random.randint(1, 255)}.{random.randint(1, 255)}"
            status = random.choice(["up", "down"])
        
        if status == "up":
            # 模拟成功的 ping 结果
            lines.append(f"正在 Ping {target} [{ip}] 具有 64 字节的数据:")
            for i in range(4):
                time_ms = random.randint(1, 50)
                ttl = random.randint(50, 128)
                lines.append(f"  来自 {ip} 的回复: 字节=64 时间={time_ms}ms TTL={ttl}")
            
            lines.append(f"\n{target} 的 Ping 统计信息:")
            lines.append(f"  数据包: 已发送=4，已接收=4，丢失=0 (0% 丢失)")
            avg_time = sum(random.randint(1, 50) for _ in range(4)) // 4
            lines.append(f"  往返行程的估计时间(以毫秒为单位):")
            lines.append(f"    最短=1ms，最长=50ms，平均={avg_time}ms")
            lines.append("\n✅ 状态：主机可达")
        else:
            lines.append(f"正在 Ping {target} [{ip}] 具有 64 字节的数据:")
            lines.append(f"  请求超时。")
            lines.append(f"  请求超时。")
            lines.append(f"  请求超时。")
            lines.append(f"  请求超时。")
            lines.append(f"\n{target} 的 Ping 统计信息:")
            lines.append(f"  数据包: 已发送=4，已接收=0，丢失=4 (100% 丢失)")
            lines.append("\n❌ 状态：主机不可达")
        
        return "\n".join(lines)
    
    def _dns_lookup(self, target: str) -> str:
        """模拟 DNS 解析"""
        lines = [f"🔍 DNS 解析查询：{target}"]
        lines.append("-" * 50)
        
        host_info = self.known_hosts.get(target)
        if host_info:
            ip = host_info["ip"]
            lines.append(f"服务器:  dns.google")
            lines.append(f"Address:  8.8.8.8")
            lines.append(f"\n非权威应答:")
            lines.append(f"  名称:    {target}")
            lines.append(f"  Address:  {ip}")
            lines.append("\n✅ DNS 解析成功")
        else:
            # 模拟解析失败
            if self._is_valid_ip(target):
                lines.append(f"输入的是 IP 地址，进行反向 DNS 查询:")
                lines.append(f"  名称:    unknown-{target.replace('.', '-')}.example.com")
                lines.append("\n✅ 反向解析完成")
            else:
                lines.append(f"服务器:  dns.google")
                lines.append(f"Address:  8.8.8.8")
                lines.append(f"\n*** 未找到 {target} 的主机")
                lines.append("\n❌ DNS 解析失败")
        
        return "\n".join(lines)
    
    def _port_check(self, target: str, port_str: str) -> str:
        """模拟端口检测"""
        lines = [f"🔌 端口检测：{target}:{port_str}"]
        lines.append("-" * 50)
        
        try:
            port = int(port_str)
        except ValueError:
            return "错误：端口号必须是数字"
        
        if port < 1 or port > 65535:
            return "错误：端口号必须在 1-65535 范围内"
        
        # 常见端口状态模拟
        common_ports = {
            22: ("SSH", "up"),
            23: ("Telnet", "down"),
            25: ("SMTP", "up"),
            53: ("DNS", "up"),
            80: ("HTTP", "up"),
            443: ("HTTPS", "up"),
            3306: ("MySQL", "up"),
            3389: ("RDP", "down"),
            5432: ("PostgreSQL", "down"),
            6379: ("Redis", "up"),
            8080: ("HTTP-Alt", "up"),
        }
        
        host_info = self.known_hosts.get(target)
        host_up = host_info["status"] == "up" if host_info else random.choice([True, False])
        
        if not host_up:
            lines.append(f"❌ 主机 {target} 不可达，无法检测端口")
            return "\n".join(lines)
        
        if port in common_ports:
            service, status = common_ports[port]
        else:
            service = "unknown"
            status = random.choice(["up", "down"])
        
        if status == "up":
            lines.append(f"  PORT     STATE    SERVICE")
            lines.append(f"  {port}/tcp   open     {service}")
            lines.append(f"\n✅ 端口 {port} 开放 ({service})")
        else:
            lines.append(f"  PORT     STATE    SERVICE")
            lines.append(f"  {port}/tcp   closed   {service}")
            lines.append(f"\n❌ 端口 {port} 关闭 ({service})")
        
        return "\n".join(lines)
    
    def _traceroute(self, target: str) -> str:
        """模拟路由追踪"""
        lines = [f"🛤️ 路由追踪：{target}"]
        lines.append("-" * 50)
        lines.append(f"traceroute to {target}, 30 hops max, 60 byte packets")
        
        host_info = self.known_hosts.get(target)
        final_ip = host_info["ip"] if host_info else "模拟目标IP"
        
        # 模拟路由跳数
        hops = random.randint(5, 12)
        for i in range(1, hops + 1):
            if i == hops:
                hop_ip = final_ip
                hop_name = target
            else:
                hop_ip = f"{random.randint(1, 255)}.{random.randint(1, 255)}.{random.randint(1, 255)}.{random.randint(1, 255)}"
                hop_name = f"hop-{i}.isp.net"
            
            times = [random.randint(1, 30) for _ in range(3)]
            times_str = "  ".join(f"{t} ms" for t in times)
            lines.append(f" {i:2}  {hop_name} ({hop_ip})  {times_str}")
        
        lines.append(f"\n✅ 路由追踪完成，共 {hops} 跳")
        return "\n".join(lines)
    
    def _comprehensive_check(self, target: str) -> str:
        """综合连通性检查"""
        lines = [f"🔎 综合连通性检查：{target}"]
        lines.append("=" * 50)
        
        # 1. DNS 检查
        lines.append("\n【DNS 解析】")
        host_info = self.known_hosts.get(target)
        if host_info:
            lines.append(f"  ✅ 解析成功: {host_info['ip']}")
            ip = host_info['ip']
        elif self._is_valid_ip(target):
            ip = target
            lines.append(f"  ℹ️ 输入为 IP 地址，无需解析")
        else:
            ip = "未知"
            lines.append(f"  ❌ 解析失败")
        
        # 2. Ping 检查
        lines.append("\n【Ping 测试】")
        if host_info:
            if host_info["status"] == "up":
                lines.append(f"  ✅ 主机可达")
            else:
                lines.append(f"  ❌ 主机不可达")
        else:
            lines.append(f"  ⚠️ 状态未知（模拟环境）")
        
        # 3. 常见端口检查
        lines.append("\n【常见端口状态】")
        if host_info and host_info["status"] == "up":
            common = [(80, "HTTP"), (443, "HTTPS"), (22, "SSH")]
            for port, name in common:
                status = random.choice(["open", "closed"])
                symbol = "✅" if status == "open" else "❌"
                lines.append(f"  {symbol} {port}/{name}: {status}")
        else:
            lines.append(f"  ⚠️ 主机不可达，跳过端口检测")
        
        lines.append("\n" + "=" * 50)
        lines.append("检查完成")
        
        return "\n".join(lines)
    
    def _is_valid_ip(self, ip: str) -> bool:
        """验证 IP 地址格式"""
        pattern = r'^(\d{1,3}\.){3}\d{1,3}$'
        if re.match(pattern, ip):
            parts = ip.split('.')
            return all(0 <= int(part) <= 255 for part in parts)
        return False


def create_network_diagnosis_tool():
    """创建 LangChain Tool 实例"""
    from langchain_core.tools import Tool
    
    tool = NetworkDiagnosisTool()
    return Tool(
        name=tool.name,
        func=tool.run,
        description=tool.description
    )
