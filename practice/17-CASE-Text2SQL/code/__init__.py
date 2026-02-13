"""
Text2SQL Vanna 实现示例
演示完整的使用流程
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from text2sql_vanna import create_vanna
from rich.console import Console
from rich.table import Table
from rich.syntax import Syntax

console = Console()


def demo():
    """演示 Text2SQL 功能"""
    console.print("[bold cyan]Text2SQL 功能演示[/bold cyan]\n")
    
    # 1. 创建 Vanna 实例
    console.print("[1] 初始化 Vanna...")
    vanna = create_vanna(llm_provider="dashscope")
    
    # 2. 演示问题列表
    questions = [
        "查询所有战士类英雄的名称和生命值",
        "查询生命值最高的前5个英雄",
        "统计每个定位有多少个英雄",
        "查询周免英雄有哪些",
        "查询击杀数最高的3场比赛记录"
    ]
    
    for i, question in enumerate(questions, 1):
        console.print(f"\n[bold]问题 {i}: {question}[/bold]")
        
        try:
            # 生成 SQL
            sql = vanna.generate_sql(question)
            console.print("[green]生成的 SQL:[/green]")
            console.print(Syntax(sql, "sql", theme="monokai"))
            
            # 执行查询
            results = vanna.run_sql(sql)
            
            # 显示结果
            if results:
                table = Table()
                for key in results[0].keys():
                    table.add_column(key)
                
                for row in results[:5]:
                    table.add_row(*[str(v) if v is not None else "" for v in row.values()])
                
                console.print(table)
                console.print(f"[dim]共 {len(results)} 条记录[/dim]")
            
        except Exception as e:
            console.print(f"[red]错误: {e}[/red]")
    
    # 关闭连接
    vanna.close()
    console.print("\n[bold green]演示完成！[/bold green]")


if __name__ == "__main__":
    demo()
