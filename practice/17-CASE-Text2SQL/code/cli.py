"""
Text2SQL 交互式命令行界面
"""

import sys
from pathlib import Path
from typing import Optional
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.syntax import Syntax
from rich.prompt import Prompt
from rich.markdown import Markdown

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent))

from text2sql_vanna import create_vanna, SimpleVanna
from loguru import logger

console = Console()


class Text2SQLCLI:
    """Text2SQL 交互式命令行界面"""
    
    def __init__(self, llm_provider: str = "dashscope"):
        self.vanna: Optional[SimpleVanna] = None
        self.llm_provider = llm_provider
        self.history: list = []
    
    def start(self):
        """启动交互式界面"""
        self._print_welcome()
        self._init_vanna()
        self._run_repl()
    
    def _print_welcome(self):
        """打印欢迎信息"""
        console.print(Panel.fit(
            "[bold cyan]Text2SQL 智能查询系统[/bold cyan]\n"
            "[dim]基于 Vanna + 大语言模型实现[/dim]\n\n"
            "[yellow]命令说明：[/yellow]\n"
            "  • 输入自然语言问题，自动生成 SQL 并执行\n"
            "  • [green]history[/green] - 查看历史查询\n"
            "  • [green]train[/green] - 添加训练数据\n"
            "  • [green]schema[/green] - 查看表结构\n"
            "  • [green]help[/green] - 显示帮助\n"
            "  • [green]quit/exit[/green] - 退出程序",
            title="🎮 欢迎使用",
            border_style="cyan"
        ))
    
    def _init_vanna(self):
        """初始化 Vanna"""
        console.print("\n[bold]正在初始化...[/bold]")
        
        try:
            self.vanna = create_vanna(llm_provider=self.llm_provider)
            console.print("[green]✓[/green] 系统初始化成功\n")
        except Exception as e:
            console.print(f"[red]✗ 初始化失败: {e}[/red]")
            sys.exit(1)
    
    def _run_repl(self):
        """运行交互式循环"""
        while True:
            try:
                question = Prompt.ask("\n[bold cyan]请输入问题[/bold cyan]")
                
                if not question.strip():
                    continue
                
                # 处理命令
                if question.lower() in ['quit', 'exit', 'q']:
                    self._handle_quit()
                    break
                elif question.lower() == 'help':
                    self._handle_help()
                elif question.lower() == 'history':
                    self._handle_history()
                elif question.lower() == 'schema':
                    self._handle_schema()
                elif question.lower() == 'train':
                    self._handle_train()
                else:
                    self._handle_question(question)
                    
            except KeyboardInterrupt:
                console.print("\n[yellow]使用 quit 或 exit 退出[/yellow]")
            except Exception as e:
                console.print(f"[red]错误: {e}[/red]")
    
    def _handle_question(self, question: str):
        """处理用户问题"""
        console.print(f"\n[bold]🔍 正在处理问题...[/bold]")
        
        try:
            # 生成 SQL
            sql = self.vanna.generate_sql(question)
            
            # 显示生成的 SQL
            console.print("\n[bold green]生成的 SQL:[/bold green]")
            syntax = Syntax(sql, "sql", theme="monokai", line_numbers=False)
            console.print(syntax)
            
            # 执行 SQL
            console.print("\n[bold]📊 执行查询...[/bold]")
            results = self.vanna.run_sql(sql)
            
            # 显示结果
            self._display_results(results)
            
            # 保存历史
            self.history.append({
                "question": question,
                "sql": sql,
                "row_count": len(results)
            })
            
        except Exception as e:
            console.print(f"[red]错误: {e}[/red]")
    
    def _display_results(self, results: list):
        """显示查询结果"""
        if not results:
            console.print("[yellow]查询结果为空[/yellow]")
            return
        
        # 创建表格
        table = Table(show_header=True, header_style="bold magenta")
        
        # 添加列
        for key in results[0].keys():
            table.add_column(str(key))
        
        # 添加行
        for row in results[:20]:  # 限制显示20行
            table.add_row(*[str(v) if v is not None else "" for v in row.values()])
        
        console.print(table)
        
        if len(results) > 20:
            console.print(f"[dim]显示前 20 条，共 {len(results)} 条记录[/dim]")
        else:
            console.print(f"[dim]共 {len(results)} 条记录[/dim]")
    
    def _handle_help(self):
        """显示帮助"""
        help_text = """
# Text2SQL 帮助

## 使用方式

直接输入自然语言问题，系统会自动：
1. 理解问题意图
2. 生成对应的 SQL 语句
3. 执行查询并展示结果

## 示例问题

- 查询所有战士类英雄
- 查询生命值最高的5个英雄
- 统计每个定位的英雄数量
- 查询周免英雄有哪些
- 查询比赛记录中击杀数最高的3场比赛

## 可用表

- `heros` - 英雄基本信息
- `hero_skills` - 英雄技能
- `match_records` - 比赛记录
"""
        console.print(Markdown(help_text))
    
    def _handle_history(self):
        """显示历史"""
        if not self.history:
            console.print("[yellow]暂无历史查询[/yellow]")
            return
        
        table = Table(title="历史查询")
        table.add_column("#", style="dim")
        table.add_column("问题")
        table.add_column("结果数")
        
        for i, item in enumerate(self.history[-10:], 1):
            table.add_row(
                str(i),
                item["question"][:50] + "..." if len(item["question"]) > 50 else item["question"],
                str(item["row_count"])
            )
        
        console.print(table)
    
    def _handle_schema(self):
        """显示表结构"""
        if not self.vanna:
            return
        
        console.print(Panel(
            Syntax(self.vanna.schema_info, "sql", theme="monokai"),
            title="数据库表结构",
            border_style="blue"
        ))
    
    def _handle_train(self):
        """添加训练数据"""
        console.print("\n[bold]添加训练数据[/bold]")
        
        question = Prompt.ask("请输入问题")
        sql = Prompt.ask("请输入对应的 SQL")
        
        if question and sql:
            self.vanna.train(question=question, sql=sql)
            console.print("[green]训练数据添加成功！[/green]")
    
    def _handle_quit(self):
        """退出程序"""
        if self.vanna:
            self.vanna.close()
        console.print("\n[bold cyan]感谢使用，再见！[/bold cyan]")


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Text2SQL 交互式查询")
    parser.add_argument(
        "--provider",
        choices=["dashscope", "openai", "ollama"],
        default="dashscope",
        help="LLM 提供商"
    )
    args = parser.parse_args()
    
    cli = Text2SQLCLI(llm_provider=args.provider)
    cli.start()


if __name__ == "__main__":
    main()
