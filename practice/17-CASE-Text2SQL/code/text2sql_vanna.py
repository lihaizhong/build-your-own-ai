"""
Vanna Text2SQL 核心模块
支持多种LLM提供商：OpenAI、通义千问(DashScope)、Ollama
"""

import os
import sqlite3
from pathlib import Path
from typing import Optional, List, Dict, Any
from abc import ABC, abstractmethod

from loguru import logger
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()


def get_project_path(*paths: str) -> Path:
    """获取项目路径"""
    current_dir = Path(__file__).parent
    project_dir = current_dir.parent.parent.parent  # 回到根目录
    return project_dir.joinpath(*paths)


class BaseVanna(ABC):
    """Vanna 基类抽象"""
    
    @abstractmethod
    def train(self, *args, **kwargs):
        """训练模型"""
        pass
    
    @abstractmethod
    def generate_sql(self, question: str) -> str:
        """生成 SQL"""
        pass
    
    @abstractmethod
    def run_sql(self, sql: str) -> Any:
        """执行 SQL"""
        pass


class SimpleVanna(BaseVanna):
    """
    简化版 Vanna 实现
    不依赖 vanna 库，直接使用 LLM API 实现 Text2SQL
    """
    
    def __init__(
        self,
        llm_provider: str = "dashscope",
        db_path: Optional[str] = None,
        chroma_persist_dir: Optional[str] = None
    ):
        self.llm_provider = llm_provider
        self.db_path = db_path or str(get_project_path("practice/17-CASE-Text2SQL/data/heros.db"))
        self.training_data: List[Dict[str, str]] = []
        self.schema_info: str = ""
        
        # 初始化向量存储路径
        self.chroma_persist_dir = chroma_persist_dir or str(
            get_project_path("practice/17-CASE-Text2SQL/data/chroma")
        )
        
        # 初始化 LLM 客户端
        self._init_llm_client()
        
        # 初始化数据库连接
        self._init_db_connection()
        
        # 加载训练数据
        self._load_training_data()
    
    def _init_llm_client(self):
        """初始化 LLM 客户端"""
        if self.llm_provider == "dashscope":
            self._init_dashscope()
        elif self.llm_provider == "openai":
            self._init_openai()
        elif self.llm_provider == "ollama":
            self._init_ollama()
        else:
            raise ValueError(f"不支持的 LLM 提供商: {self.llm_provider}")
    
    def _init_dashscope(self):
        """初始化通义千问"""
        api_key = os.getenv("DASHSCOPE_API_KEY")
        if not api_key:
            raise ValueError("请设置 DASHSCOPE_API_KEY 环境变量")
        
        from openai import OpenAI
        self.client = OpenAI(
            api_key=api_key,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
        )
        self.model = os.getenv("DASHSCOPE_MODEL", "qwen-turbo")
        logger.success(f"DashScope 客户端初始化成功，模型: {self.model}")
    
    def _init_openai(self):
        """初始化 OpenAI"""
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("请设置 OPENAI_API_KEY 环境变量")
        
        from openai import OpenAI
        base_url = os.getenv("OPENAI_API_BASE")
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        logger.success(f"OpenAI 客户端初始化成功，模型: {self.model}")
    
    def _init_ollama(self):
        """初始化 Ollama"""
        from openai import OpenAI
        base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
        self.client = OpenAI(api_key="ollama", base_url=base_url)
        self.model = os.getenv("OLLAMA_MODEL", "deepseek-r1:7b")
        logger.success(f"Ollama 客户端初始化成功，模型: {self.model}")
    
    def _init_db_connection(self):
        """初始化数据库连接"""
        db_path = Path(self.db_path)
        if not db_path.exists():
            logger.warning(f"数据库文件不存在: {db_path}，请先运行 prepare_data.py")
            self.conn = None
            return
        
        self.conn = sqlite3.connect(str(db_path))
        self.conn.row_factory = sqlite3.Row
        logger.success(f"数据库连接成功: {db_path}")
        
        # 获取表结构信息
        self._load_schema_info()
    
    def _load_schema_info(self):
        """加载表结构信息"""
        if not self.conn:
            return
        
        cursor = self.conn.cursor()
        
        # 获取所有表
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [row[0] for row in cursor.fetchall()]
        
        schema_parts = []
        for table in tables:
            # 获取表结构
            cursor.execute(f"PRAGMA table_info({table})")
            columns = cursor.fetchall()
            
            col_defs = []
            for col in columns:
                col_name = col[1]
                col_type = col[2]
                col_defs.append(f"{col_name} {col_type}")
            
            schema_parts.append(f"CREATE TABLE {table} (\n  " + ",\n  ".join(col_defs) + "\n);")
        
        self.schema_info = "\n\n".join(schema_parts)
        logger.info(f"已加载 {len(tables)} 个表的结构信息")
    
    def _load_training_data(self):
        """加载训练数据（SQL问答对）"""
        # 预置的示例训练数据
        self.training_data = [
            {
                "question": "查询所有英雄的名称和定位",
                "sql": "SELECT hero_name, role FROM heros"
            },
            {
                "question": "查询所有战士类英雄",
                "sql": "SELECT * FROM heros WHERE role = '战士'"
            },
            {
                "question": "查询生命值最高的前5个英雄",
                "sql": "SELECT hero_name, health FROM heros ORDER BY health DESC LIMIT 5"
            },
            {
                "question": "查询每个定位有多少个英雄",
                "sql": "SELECT role, COUNT(*) as count FROM heros GROUP BY role"
            },
            {
                "question": "查询难度为3的英雄数量",
                "sql": "SELECT COUNT(*) FROM heros WHERE difficulty = 3"
            },
            {
                "question": "查询周免英雄",
                "sql": "SELECT hero_name, hero_title FROM heros WHERE is_free = 1"
            },
            {
                "question": "查询价格低于10000金币的英雄",
                "sql": "SELECT hero_name, price FROM heros WHERE price > 0 AND price < 10000"
            },
            {
                "question": "查询每个阵营的英雄数量",
                "sql": "SELECT region, COUNT(*) as count FROM heros GROUP BY region ORDER BY count DESC"
            },
            {
                "question": "查询近战英雄的平均生命值",
                "sql": "SELECT AVG(health) as avg_health FROM heros WHERE attack_type = '近战'"
            },
            {
                "question": "查询比赛记录中获胜次数最多的英雄",
                "sql": """
                    SELECT h.hero_name, COUNT(*) as win_count 
                    FROM match_records m 
                    JOIN heros h ON m.hero_id = h.hero_id 
                    WHERE m.win = 1 
                    GROUP BY h.hero_name 
                    ORDER BY win_count DESC 
                    LIMIT 1
                """
            },
            {
                "question": "查询击杀数最高的比赛记录",
                "sql": """
                    SELECT h.hero_name, m.player_name, m.kill_count, m.match_date 
                    FROM match_records m 
                    JOIN heros h ON m.hero_id = h.hero_id 
                    ORDER BY m.kill_count DESC 
                    LIMIT 5
                """
            },
            {
                "question": "查询每个英雄的平均击杀数",
                "sql": """
                    SELECT h.hero_name, AVG(m.kill_count) as avg_kills 
                    FROM match_records m 
                    JOIN heros h ON m.hero_id = h.hero_id 
                    GROUP BY h.hero_name 
                    ORDER BY avg_kills DESC
                """
            }
        ]
        logger.info(f"已加载 {len(self.training_data)} 条训练数据")
    
    def train(self, question: str = None, sql: str = None, ddl: str = None, documentation: str = None):
        """
        添加训练数据
        
        Args:
            question: 自然语言问题
            sql: 对应的 SQL 语句
            ddl: 表结构定义
            documentation: 文档说明
        """
        if question and sql:
            self.training_data.append({"question": question, "sql": sql})
            logger.success(f"添加训练数据: {question}")
        
        if ddl:
            self.schema_info += f"\n\n{ddl}"
            logger.success("添加 DDL 训练数据")
        
        if documentation:
            self.training_data.append({"question": documentation, "sql": ""})
            logger.success("添加文档训练数据")
    
    def _build_prompt(self, question: str) -> str:
        """构建提示词"""
        # 查找相似的问题作为 few-shot 示例
        similar_examples = self._find_similar_questions(question)
        
        prompt = """你是一个专业的 SQL 专家，请根据用户的问题生成对应的 SQL 查询语句。

## 数据库表结构

{schema}

## 示例问答

{examples}

## 用户问题

{question}

## 要求

1. 只输出 SQL 语句，不要有其他解释
2. SQL 语句要符合 SQLite 语法
3. 使用表名和字段名要准确
4. 注意 SQL 注入防护

## SQL 语句

""".format(
            schema=self.schema_info,
            examples="\n\n".join([
                f"问题：{ex['question']}\nSQL：{ex['sql']}"
                for ex in similar_examples
            ]),
            question=question
        )
        
        return prompt
    
    def _find_similar_questions(self, question: str, top_k: int = 3) -> List[Dict]:
        """查找相似问题（简单的关键词匹配）"""
        # 简单实现：基于关键词匹配
        question_lower = question.lower()
        scored_examples = []
        
        for example in self.training_data:
            score = 0
            ex_question = example['question'].lower()
            
            # 关键词匹配
            keywords = ['查询', '统计', '最高', '最低', '平均', '数量', '排序', '分组', '英雄', '比赛']
            for kw in keywords:
                if kw in question_lower and kw in ex_question:
                    score += 1
            
            scored_examples.append((score, example))
        
        # 按分数排序，取前 top_k
        scored_examples.sort(key=lambda x: x[0], reverse=True)
        return [ex for _, ex in scored_examples[:top_k]]
    
    def generate_sql(self, question: str, allow_llm_to_see_data: bool = True) -> str:
        """
        生成 SQL 语句
        
        Args:
            question: 自然语言问题
            allow_llm_to_see_data: 是否允许 LLM 查看数据
            
        Returns:
            生成的 SQL 语句
        """
        prompt = self._build_prompt(question)
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "你是一个专业的 SQL 专家，只输出 SQL 语句。"},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=500
            )
            
            sql = response.choices[0].message.content.strip()
            
            # 清理 SQL 输出
            sql = self._clean_sql(sql)
            
            logger.info(f"生成 SQL: {sql}")
            return sql
            
        except Exception as e:
            logger.error(f"生成 SQL 失败: {e}")
            raise
    
    def _clean_sql(self, sql: str) -> str:
        """清理 SQL 输出"""
        # 移除 markdown 代码块标记
        sql = sql.replace("```sql", "").replace("```", "").strip()
        
        # 移除多余空白
        sql = " ".join(sql.split())
        
        return sql
    
    def run_sql(self, sql: str) -> List[Dict]:
        """
        执行 SQL 查询
        
        Args:
            sql: SQL 语句
            
        Returns:
            查询结果列表
        """
        if not self.conn:
            raise ValueError("数据库未连接")
        
        try:
            cursor = self.conn.cursor()
            cursor.execute(sql)
            
            # 转换为字典列表
            columns = [description[0] for description in cursor.description]
            rows = cursor.fetchall()
            
            results = []
            for row in rows:
                results.append(dict(zip(columns, row)))
            
            logger.success(f"查询成功，返回 {len(results)} 条记录")
            return results
            
        except Exception as e:
            logger.error(f"SQL 执行失败: {e}")
            raise
    
    def ask(self, question: str, visualize: bool = False) -> Dict[str, Any]:
        """
        完整的问答流程
        
        Args:
            question: 自然语言问题
            visualize: 是否可视化
            
        Returns:
            包含 SQL 和结果的字典
        """
        logger.info(f"处理问题: {question}")
        
        # 生成 SQL
        sql = self.generate_sql(question)
        
        # 执行 SQL
        try:
            results = self.run_sql(sql)
        except Exception as e:
            return {
                "question": question,
                "sql": sql,
                "error": str(e),
                "results": None
            }
        
        return {
            "question": question,
            "sql": sql,
            "results": results,
            "row_count": len(results)
        }
    
    def close(self):
        """关闭数据库连接"""
        if self.conn:
            self.conn.close()
            logger.info("数据库连接已关闭")


def create_vanna(
    llm_provider: str = None,
    db_path: str = None
) -> SimpleVanna:
    """
    创建 Vanna 实例
    
    Args:
        llm_provider: LLM 提供商 (dashscope/openai/ollama)
        db_path: 数据库路径
        
    Returns:
        Vanna 实例
    """
    provider = llm_provider or os.getenv("LLM_PROVIDER", "dashscope")
    
    return SimpleVanna(
        llm_provider=provider,
        db_path=db_path
    )


if __name__ == "__main__":
    # 测试
    vanna = create_vanna()
    
    # 测试问题
    questions = [
        "查询所有战士类英雄",
        "查询生命值最高的3个英雄",
        "每个定位有多少英雄"
    ]
    
    for q in questions:
        print(f"\n问题: {q}")
        result = vanna.ask(q)
        print(f"SQL: {result['sql']}")
        if result['results']:
            print(f"结果: {result['row_count']} 条记录")
        print("-" * 50)
    
    vanna.close()
