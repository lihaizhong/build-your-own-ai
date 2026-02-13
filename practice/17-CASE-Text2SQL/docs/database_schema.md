# 数据库表结构

## heros 表

| 字段名 | 类型 | 是否为空 | 默认值 | 说明 |
|--------|------|----------|--------|------|
| hero_id | INTEGER | 是 | - | 主键 |
| hero_name | VARCHAR(50) | 否 | - |  |
| hero_title | VARCHAR(100) | 是 | - |  |
| role | VARCHAR(20) | 是 | - |  |
| attack_type | VARCHAR(20) | 是 | - |  |
| difficulty | INTEGER | 是 | - |  |
| health | INTEGER | 是 | - |  |
| mana | INTEGER | 是 | - |  |
| attack_damage | INTEGER | 是 | - |  |
| magic_damage | INTEGER | 是 | - |  |
| armor | INTEGER | 是 | - |  |
| magic_resist | INTEGER | 是 | - |  |
| attack_speed | DECIMAL(3,2) | 是 | - |  |
| move_speed | INTEGER | 是 | - |  |
| release_date | DATE | 是 | - |  |
| is_free | BOOLEAN | 是 | FALSE |  |
| price | INTEGER | 是 | - |  |
| region | VARCHAR(50) | 是 | - |  |
| created_at | TIMESTAMP | 是 | CURRENT_TIMESTAMP |  |

## sqlite_sequence 表

| 字段名 | 类型 | 是否为空 | 默认值 | 说明 |
|--------|------|----------|--------|------|
| name |  | 是 | - |  |
| seq |  | 是 | - |  |

## hero_skills 表

| 字段名 | 类型 | 是否为空 | 默认值 | 说明 |
|--------|------|----------|--------|------|
| skill_id | INTEGER | 是 | - | 主键 |
| hero_id | INTEGER | 否 | - |  |
| skill_name | VARCHAR(50) | 否 | - |  |
| skill_type | VARCHAR(20) | 是 | - |  |
| description | TEXT | 是 | - |  |
| cooldown | DECIMAL(4,1) | 是 | - |  |
| mana_cost | INTEGER | 是 | - |  |
| damage_type | VARCHAR(20) | 是 | - |  |

## hero_items 表

| 字段名 | 类型 | 是否为空 | 默认值 | 说明 |
|--------|------|----------|--------|------|
| id | INTEGER | 是 | - | 主键 |
| hero_id | INTEGER | 否 | - |  |
| item_name | VARCHAR(50) | 否 | - |  |
| item_type | VARCHAR(30) | 是 | - |  |
| price | INTEGER | 是 | - |  |
| priority | INTEGER | 是 | - |  |

## match_records 表

| 字段名 | 类型 | 是否为空 | 默认值 | 说明 |
|--------|------|----------|--------|------|
| match_id | INTEGER | 是 | - | 主键 |
| hero_id | INTEGER | 否 | - |  |
| player_name | VARCHAR(50) | 是 | - |  |
| kill_count | INTEGER | 是 | - |  |
| death_count | INTEGER | 是 | - |  |
| assist_count | INTEGER | 是 | - |  |
| damage_dealt | INTEGER | 是 | - |  |
| damage_taken | INTEGER | 是 | - |  |
| gold_earned | INTEGER | 是 | - |  |
| win | BOOLEAN | 是 | - |  |
| match_date | DATE | 是 | - |  |
| match_duration | INTEGER | 是 | - |  |

