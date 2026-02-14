"""
数据准备脚本
创建heros数据表和示例数据
"""

import sqlite3
from pathlib import Path
from loguru import logger


def get_project_path(*paths: str) -> Path:
    """获取项目路径"""
    current_dir = Path(__file__).parent
    project_dir = current_dir.parent
    return project_dir.joinpath(*paths)


def create_heros_database():
    """创建heros数据库和表结构"""
    db_path = get_project_path("data", "heros.db")
    db_path.parent.mkdir(parents=True, exist_ok=True)
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # 删除已存在的表
    cursor.execute("DROP TABLE IF EXISTS heros")
    cursor.execute("DROP TABLE IF EXISTS hero_skills")
    cursor.execute("DROP TABLE IF EXISTS hero_items")
    cursor.execute("DROP TABLE IF EXISTS match_records")
    
    # 创建英雄基本信息表
    cursor.execute("""
    CREATE TABLE heros (
        hero_id INTEGER PRIMARY KEY AUTOINCREMENT,
        hero_name VARCHAR(50) NOT NULL,
        hero_title VARCHAR(100),
        role VARCHAR(20),
        attack_type VARCHAR(20),
        difficulty INTEGER,
        health INTEGER,
        mana INTEGER,
        attack_damage INTEGER,
        magic_damage INTEGER,
        armor INTEGER,
        magic_resist INTEGER,
        attack_speed DECIMAL(3,2),
        move_speed INTEGER,
        release_date DATE,
        is_free BOOLEAN DEFAULT FALSE,
        price INTEGER,
        region VARCHAR(50),
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """)
    
    # 创建英雄技能表
    cursor.execute("""
    CREATE TABLE hero_skills (
        skill_id INTEGER PRIMARY KEY AUTOINCREMENT,
        hero_id INTEGER NOT NULL,
        skill_name VARCHAR(50) NOT NULL,
        skill_type VARCHAR(20),
        description TEXT,
        cooldown DECIMAL(4,1),
        mana_cost INTEGER,
        damage_type VARCHAR(20),
        FOREIGN KEY (hero_id) REFERENCES heros(hero_id)
    )
    """)
    
    # 创建英雄装备推荐表
    cursor.execute("""
    CREATE TABLE hero_items (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        hero_id INTEGER NOT NULL,
        item_name VARCHAR(50) NOT NULL,
        item_type VARCHAR(30),
        price INTEGER,
        priority INTEGER,
        FOREIGN KEY (hero_id) REFERENCES heros(hero_id)
    )
    """)
    
    # 创建比赛记录表
    cursor.execute("""
    CREATE TABLE match_records (
        match_id INTEGER PRIMARY KEY AUTOINCREMENT,
        hero_id INTEGER NOT NULL,
        player_name VARCHAR(50),
        kill_count INTEGER,
        death_count INTEGER,
        assist_count INTEGER,
        damage_dealt INTEGER,
        damage_taken INTEGER,
        gold_earned INTEGER,
        win BOOLEAN,
        match_date DATE,
        match_duration INTEGER,
        FOREIGN KEY (hero_id) REFERENCES heros(hero_id)
    )
    """)
    
    conn.commit()
    logger.success(f"数据库创建成功: {db_path}")
    return conn


def insert_hero_data(conn: sqlite3.Connection):
    """插入英雄示例数据"""
    cursor = conn.cursor()
    
    # 英雄数据
    heros_data = [
        # 战士类英雄
        ('亚瑟', '圣光骑士', '战士', '近战', 1, 3500, 450, 180, 0, 120, 50, 0.85, 370, '2015-11-26', True, 0, '圣光教廷'),
        ('吕布', '无双战神', '战士', '近战', 3, 3800, 0, 200, 0, 100, 40, 0.75, 375, '2016-02-16', False, 13888, '群雄'),
        ('典韦', '狂战士', '战士', '近战', 2, 3600, 500, 190, 0, 130, 55, 0.82, 370, '2016-07-15', True, 5888, '魏国'),
        ('关羽', '武圣', '战士', '近战', 4, 3400, 400, 175, 0, 95, 45, 0.80, 390, '2016-06-28', False, 18888, '蜀国'),
        ('老夫子', '万世师表', '战士', '近战', 2, 3400, 480, 170, 0, 100, 45, 0.88, 365, '2016-01-14', True, 5888, '稷下学院'),
        
        # 法师类英雄
        ('妲己', '魅惑之狐', '法师', '远程', 1, 2800, 600, 150, 280, 60, 80, 0.70, 360, '2015-11-26', True, 0, '商朝'),
        ('安琪拉', '暗夜萝莉', '法师', '远程', 2, 2700, 580, 140, 300, 55, 75, 0.68, 360, '2015-11-26', True, 0, '魔法学院'),
        ('王昭君', '冰雪之华', '法师', '远程', 2, 2600, 620, 130, 310, 50, 90, 0.65, 355, '2016-01-20', True, 6888, '北夷'),
        ('貂蝉', '绝世舞姬', '法师', '远程', 4, 2800, 550, 145, 260, 65, 70, 0.72, 365, '2016-05-24', False, 13888, '群雄'),
        ('小乔', '恋之微风', '法师', '远程', 2, 2700, 590, 135, 290, 55, 75, 0.66, 358, '2016-04-15', True, 5888, '吴国'),
        
        # 射手类英雄
        ('后羿', '半神之弓', '射手', '远程', 1, 2600, 420, 200, 0, 70, 40, 0.90, 350, '2015-11-26', True, 0, '神话'),
        ('鲁班七号', '机关造物', '射手', '远程', 1, 2500, 400, 210, 0, 60, 35, 0.95, 350, '2016-02-02', True, 0, '稷下学院'),
        ('孙尚香', '千金重弩', '射手', '远程', 3, 2700, 450, 195, 0, 75, 45, 0.88, 355, '2016-03-29', False, 5888, '吴国'),
        ('马可波罗', '远游之枪', '射手', '远程', 3, 2800, 480, 180, 0, 80, 50, 0.85, 360, '2016-09-27', False, 13888, '西域'),
        ('狄仁杰', '断案大师', '射手', '远程', 2, 2600, 430, 185, 0, 70, 45, 0.88, 352, '2016-08-23', True, 0, '唐朝'),
        
        # 辅助类英雄
        ('蔡文姬', '天籁弦音', '辅助', '远程', 1, 2800, 550, 130, 150, 65, 95, 0.65, 355, '2016-07-19', True, 0, '汉末'),
        ('明世隐', '灵魂劫卜', '辅助', '远程', 3, 3000, 480, 140, 180, 70, 90, 0.68, 360, '2018-01-16', False, 13888, '长安'),
        ('瑶', '鹿灵守心', '辅助', '远程', 2, 2700, 560, 125, 160, 60, 100, 0.62, 350, '2019-04-16', False, 13888, '神话'),
        ('大乔', '沧海之曜', '辅助', '远程', 3, 2600, 540, 120, 170, 55, 85, 0.64, 355, '2017-02-28', False, 13888, '吴国'),
        ('庄周', '逍遥幻梦', '辅助', '近战', 2, 3200, 520, 145, 140, 90, 80, 0.70, 365, '2015-11-26', True, 2888, '道家'),
        
        # 坦克类英雄
        ('张飞', '燕人张翼德', '坦克', '近战', 2, 4200, 0, 160, 0, 150, 70, 0.72, 375, '2016-06-28', False, 13888, '蜀国'),
        ('程咬金', '热烈之斧', '坦克', '近战', 2, 4000, 400, 175, 0, 140, 60, 0.80, 370, '2016-08-02', True, 5888, '隋唐'),
        ('廉颇', '正义爆轰', '坦克', '近战', 1, 4300, 380, 155, 0, 160, 75, 0.70, 380, '2015-11-26', True, 5888, '赵国'),
        ('白起', '最终兵器', '坦克', '近战', 2, 4100, 0, 165, 0, 155, 65, 0.74, 375, '2016-09-13', True, 6888, '秦国'),
        ('项羽', '霸王', '坦克', '近战', 1, 4000, 420, 170, 0, 145, 65, 0.75, 372, '2015-11-26', True, 0, '楚汉'),
        
        # 刺客类英雄
        ('阿轲', '信念之刃', '刺客', '近战', 3, 2900, 400, 220, 0, 75, 35, 0.92, 380, '2016-04-26', False, 13888, '刺客联盟'),
        ('李白', '青莲剑仙', '刺客', '近战', 4, 3000, 450, 195, 0, 80, 40, 0.85, 385, '2016-03-01', False, 18888, '唐朝'),
        ('韩信', '国士无双', '刺客', '近战', 3, 2900, 420, 200, 0, 70, 35, 0.90, 385, '2016-04-19', False, 13888, '汉初'),
        ('娜可露露', '鹰之守护', '刺客', '近战', 3, 2800, 380, 210, 0, 65, 30, 0.95, 382, '2016-05-24', False, 13888, 'SNK'),
        ('孙悟空', '齐天大圣', '刺客', '近战', 2, 3000, 450, 190, 0, 85, 45, 0.82, 378, '2015-11-26', False, 18888, '神话'),
    ]
    
    cursor.executemany("""
        INSERT INTO heros (hero_name, hero_title, role, attack_type, difficulty,
                          health, mana, attack_damage, magic_damage, armor, magic_resist,
                          attack_speed, move_speed, release_date, is_free, price, region)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, heros_data)
    
    conn.commit()
    logger.success(f"已插入 {len(heros_data)} 个英雄数据")


def insert_skill_data(conn: sqlite3.Connection):
    """插入技能数据"""
    cursor = conn.cursor()
    
    # 获取所有英雄ID
    cursor.execute("SELECT hero_id, hero_name FROM heros")
    hero_ids = {name: hid for hid, name in cursor.fetchall()}
    
    # 技能数据 (简化示例)
    skills_data = [
        # 亚瑟技能
        (hero_ids.get('亚瑟'), '圣光守护', '被动', '被动：亚瑟获得圣光护盾，每2秒恢复2%最大生命值', 0, 0, None),
        (hero_ids.get('亚瑟'), '誓约之盾', '主动', '举起盾牌冲锋，对敌人造成伤害并沉默', 10, 50, '物理'),
        (hero_ids.get('亚瑟'), '回旋打击', '主动', '旋转剑刃，对周围敌人造成持续伤害', 12, 60, '物理'),
        (hero_ids.get('亚瑟'), '圣剑裁决', '大招', '跃向目标，造成大额物理伤害', 40, 100, '物理'),
        
        # 李白技能
        (hero_ids.get('李白'), '侠客行', '被动', '被动：每次普攻积累剑气，四层后进入侠客行状态', 0, 0, None),
        (hero_ids.get('李白'), '将进酒', '主动', '向指定方向突进，造成物理伤害', 12, 50, '物理'),
        (hero_ids.get('李白'), '神来之笔', '主动', '在地上画剑阵，对范围内敌人造成伤害', 14, 60, '物理'),
        (hero_ids.get('李白'), '青莲剑歌', '大招', '化身为剑气，对范围内敌人造成多段伤害', 30, 80, '物理'),
        
        # 更多技能...
    ]
    
    # 过滤掉None值
    skills_data = [(hid, *rest) for hid, *rest in skills_data if hid is not None]
    
    cursor.executemany("""
        INSERT INTO hero_skills (hero_id, skill_name, skill_type, description, cooldown, mana_cost, damage_type)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, skills_data)
    
    conn.commit()
    logger.success(f"已插入 {len(skills_data)} 个技能数据")


def insert_match_data(conn: sqlite3.Connection):
    """插入比赛记录数据"""
    import random
    from datetime import datetime, timedelta
    
    cursor = conn.cursor()
    
    # 获取所有英雄ID
    cursor.execute("SELECT hero_id FROM heros")
    hero_ids = [row[0] for row in cursor.fetchall()]
    
    # 生成随机比赛数据
    players = ['玩家A', '玩家B', '玩家C', '玩家D', '玩家E', 
               '玩家F', '玩家G', '玩家H', '玩家I', '玩家J']
    
    match_records = []
    base_date = datetime(2024, 1, 1)
    
    for i in range(100):
        hero_id = random.choice(hero_ids)
        player = random.choice(players)
        k = random.randint(0, 20)
        d = random.randint(0, 10)
        a = random.randint(0, 25)
        damage_dealt = random.randint(50000, 200000)
        damage_taken = random.randint(20000, 80000)
        gold = random.randint(8000, 18000)
        win = random.choice([True, False])
        match_date = base_date + timedelta(days=random.randint(0, 180))
        duration = random.randint(600, 1800)
        
        match_records.append((hero_id, player, k, d, a, damage_dealt, damage_taken, 
                            gold, win, match_date.date(), duration))
    
    cursor.executemany("""
        INSERT INTO match_records (hero_id, player_name, kill_count, death_count, assist_count,
                                   damage_dealt, damage_taken, gold_earned, win, match_date, match_duration)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, match_records)
    
    conn.commit()
    logger.success(f"已插入 {len(match_records)} 条比赛记录")


def generate_ddl_documentation(conn: sqlite3.Connection) -> str:
    """生成DDL文档"""
    cursor = conn.cursor()
    
    doc = "# 数据库表结构\n\n"
    
    # 获取所有表
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = [row[0] for row in cursor.fetchall()]
    
    for table in tables:
        doc += f"## {table} 表\n\n"
        
        # 获取表结构
        cursor.execute(f"PRAGMA table_info({table})")
        columns = cursor.fetchall()
        
        doc += "| 字段名 | 类型 | 是否为空 | 默认值 | 说明 |\n"
        doc += "|--------|------|----------|--------|------|\n"
        
        for col in columns:
            name, type_, notnull, default, pk = col[1], col[2], col[3], col[4], col[5]
            doc += f"| {name} | {type_} | {'否' if notnull else '是'} | {default or '-'} | {'主键' if pk else ''} |\n"
        
        doc += "\n"
    
    # 保存文档
    doc_path = get_project_path("docs", "database_schema.md")
    doc_path.parent.mkdir(parents=True, exist_ok=True)
    doc_path.write_text(doc, encoding='utf-8')
    logger.success(f"DDL文档已生成: {doc_path}")
    
    return doc


def main():
    """主函数"""
    print("=" * 60)
    print("Text2SQL 数据准备脚本")
    print("=" * 60)
    
    # 创建数据库
    conn = create_heros_database()
    
    # 插入数据
    insert_hero_data(conn)
    insert_skill_data(conn)
    insert_match_data(conn)
    
    # 生成文档
    generate_ddl_documentation(conn)
    
    # 关闭连接
    conn.close()
    
    # 统计数据
    db_path = get_project_path("data", "heros.db")
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    print("\n数据库统计:")
    for table in ['heros', 'hero_skills', 'hero_items', 'match_records']:
        cursor.execute(f"SELECT COUNT(*) FROM {table}")
        count = cursor.fetchone()[0]
        print(f"  - {table}: {count} 条记录")
    
    conn.close()
    
    print("\n" + "=" * 60)
    print("数据准备完成!")
    print("=" * 60)


if __name__ == "__main__":
    main()
