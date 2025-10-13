#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
香港疫情数据可视化大屏 - Flask应用
使用ECharts实现数据可视化
"""

from flask import Flask, render_template, jsonify
import pandas as pd
from pathlib import Path
import json
from datetime import datetime, timedelta
import random

app = Flask(__name__)

class CovidDataProcessor:
    """疫情数据处理器"""
    
    def __init__(self):
        self.data_file = Path(__file__).parent / "香港各区疫情数据_20250322.xlsx"
        self.df = None
        self.load_data()
    
    def load_data(self):
        """加载疫情数据"""
        try:
            if self.data_file.exists():
                self.df = pd.read_excel(self.data_file)
                print(f"成功加载数据：{len(self.df)} 条记录")
            else:
                print("数据文件不存在，请先运行数据生成脚本")
        except Exception as e:
            print(f"数据加载失败：{e}")
    
    def get_daily_summary(self):
        """获取每日汇总数据"""
        if self.df is None:
            return {"dates": [], "new_cases": [], "total_cases": []}
        
        # 按日期汇总
        daily_data = self.df.groupby('日期').agg({
            '新增确诊': 'sum',
            '累计确诊': 'sum'
        }).reset_index()
        
        daily_data = daily_data.sort_values('日期')
        
        return {
            "dates": daily_data['日期'].tolist(),
            "new_cases": daily_data['新增确诊'].tolist(),
            "total_cases": daily_data['累计确诊'].tolist()
        }
    
    def get_district_distribution(self):
        """获取各区域疫情分布"""
        if self.df is None:
            return []
        
        try:
            # 按地区汇总最新数据
            latest_date = self.df['日期'].max()
            latest_data = self.df[self.df['日期'] == latest_date]
            
            district_data = latest_data.groupby('地区').agg({
                '新增确诊': 'sum',
                '累计确诊': 'sum',
                '现有确诊': 'sum'
            }).reset_index()
            
            # 香港18个行政区名称映射（确保与地图数据一致）
            district_name_mapping = {
                '中西区': '中西区',
                '湾仔区': '湾仔区', 
                '东区': '东区',
                '南区': '南区',
                '深水埗区': '深水埗区',
                '油尖旺区': '油尖旺区',
                '九龙城区': '九龙城区',
                '黄大仙区': '黄大仙区',
                '观塘区': '观塘区',
                '荃湾区': '荃湾区',
                '屯门区': '屯门区',
                '元朗区': '元朗区',
                '北区': '北区',
                '大埔区': '大埔区',
                '沙田区': '沙田区',
                '西贡区': '西贡区',
                '离岛区': '离岛区',
                '葵青区': '葵青区'
            }
            
            # 转换为地图格式
            result = []
            for _, row in district_data.iterrows():
                # 确保数值为正数，如果为负数或NaN则设为0
                cumulative = max(0, int(row['累计确诊']) if not pd.isna(row['累计确诊']) else 0)
                new_cases = max(0, int(row['新增确诊']) if not pd.isna(row['新增确诊']) else 0)
                active_cases = max(0, int(row['现有确诊']) if not pd.isna(row['现有确诊']) else 0)
                
                district_name = str(row['地区'])
                # 使用映射后的名称，确保与地图一致
                mapped_name = district_name_mapping.get(district_name, district_name)
                
                result.append({
                    "name": mapped_name,
                    "value": cumulative,
                    "new_cases": new_cases,
                    "active_cases": active_cases
                })
            
            # 按累计确诊数排序
            result = sorted(result, key=lambda x: x['value'], reverse=True)
            
            print(f"地图数据: {len(result)}个地区")
            # 输出前几个地区的数据供调试
            for item in result[:5]:
                print(f"  {item['name']}: 累计{item['value']}例")
                
            return result
            
        except Exception as e:
            print(f"获取区域分布数据错误: {e}")
            return []
    
    def get_trend_analysis(self):
        """获取趋势分析数据"""
        if self.df is None:
            return {"dates": [], "growth_rate": [], "new_cases_trend": []}
        
        daily_data = self.df.groupby('日期')['新增确诊'].sum().reset_index()
        daily_data = daily_data.sort_values('日期')
        
        # 计算增长率
        daily_data['growth_rate'] = daily_data['新增确诊'].pct_change() * 100
        daily_data['growth_rate'] = daily_data['growth_rate'].fillna(0)
        
        # 计算7日移动平均
        daily_data['ma7'] = daily_data['新增确诊'].rolling(window=3, min_periods=1).mean()
        
        return {
            "dates": daily_data['日期'].tolist(),
            "new_cases_trend": daily_data['新增确诊'].tolist(),
            "growth_rate": [round(x, 2) for x in daily_data['growth_rate'].tolist()],
            "ma7": [round(x, 1) for x in daily_data['ma7'].tolist()]
        }
    
    def get_district_ranking(self):
        """获取各区域排名数据"""
        if self.df is None:
            return {"districts": [], "values": []}
        
        # 获取最新数据
        latest_date = self.df['日期'].max()
        latest_data = self.df[self.df['日期'] == latest_date]
        
        ranking_data = latest_data.groupby('地区')['累计确诊'].sum().sort_values(ascending=True)
        
        return {
            "districts": ranking_data.index.tolist()[-10:],  # 取前10名
            "values": ranking_data.values.tolist()[-10:]
        }
    
    def get_key_indicators(self):
        """获取关键指标数据"""
        if self.df is None:
            return {}
        
        latest_date = self.df['日期'].max()
        latest_data = self.df[self.df['日期'] == latest_date]
        
        total_new = latest_data['新增确诊'].sum()
        total_cases = latest_data['累计确诊'].sum()
        total_recovered = latest_data['累计康复'].sum()
        total_active = latest_data['现有确诊'].sum()
        avg_vaccination = latest_data['疫苗接种率'].mean()
        
        # 计算恢复率
        recovery_rate = (total_recovered / total_cases * 100) if total_cases > 0 else 0
        
        return {
            "total_new": int(total_new),
            "total_cases": int(total_cases),
            "total_active": int(total_active),
            "recovery_rate": round(recovery_rate, 1),
            "vaccination_rate": round(avg_vaccination, 1)
        }

# 创建数据处理器实例
data_processor = CovidDataProcessor()

@app.route('/')
def dashboard():
    """主页路由 - 显示可视化大屏"""
    return render_template('dashboard.html')

@app.route('/hongkong.json')
def hongkong_map():
    """提供香港地图数据"""
    map_file = Path(__file__).parent / "templates" / "hongkong.json"
    try:
        with open(map_file, 'r', encoding='utf-8') as f:
            import json
            map_data = json.load(f)
        return jsonify(map_data)
    except Exception as e:
        print(f"加载地图文件失败: {e}")
        return jsonify({"error": "地图数据加载失败"}), 404

@app.route('/api/daily_summary')
def api_daily_summary():
    """每日汇总数据API"""
    data = data_processor.get_daily_summary()
    return jsonify(data)

@app.route('/api/district_distribution')
def api_district_distribution():
    """区域分布数据API"""
    data = data_processor.get_district_distribution()
    return jsonify(data)

@app.route('/api/trend_analysis')
def api_trend_analysis():
    """趋势分析数据API"""
    data = data_processor.get_trend_analysis()
    return jsonify(data)

@app.route('/api/district_ranking')
def api_district_ranking():
    """区域排名数据API"""
    data = data_processor.get_district_ranking()
    return jsonify(data)

@app.route('/api/key_indicators')
def api_key_indicators():
    """关键指标数据API"""
    data = data_processor.get_key_indicators()
    return jsonify(data)

if __name__ == '__main__':
    print("=" * 80)
    print("🚀 启动香港疫情数据可视化大屏")
    print("=" * 80)
    print("📊 访问地址：http://localhost:5001")
    print("📈 包含5个核心图表的疫情数据可视化")
    print("=" * 80)
    
    app.run(debug=True, host='0.0.0.0', port=5001)