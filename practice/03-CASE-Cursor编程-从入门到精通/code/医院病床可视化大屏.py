#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
医院病床使用情况可视化大屏 - Flask应用
使用ECharts实现数据可视化
"""

from flask import Flask, render_template, jsonify
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
import random

app = Flask(__name__)

class HospitalBedDataProcessor:
    """医院病床数据处理器"""
    
    def __init__(self):
        self.data_file = Path(__file__).parent / "医院病床数据.xlsx"
        self.df = None
        self.load_data()
    
    def load_data(self):
        """加载病床数据"""
        try:
            if self.data_file.exists():
                self.df = pd.read_excel(self.data_file)
                print(f"成功加载数据：{len(self.df)} 条记录")
            else:
                # 生成模拟数据
                self.generate_mock_data()
        except Exception as e:
            print(f"数据加载失败：{e}")
            self.generate_mock_data()
    
    def generate_mock_data(self):
        """生成模拟病床数据"""
        print("生成模拟病床数据...")
        
        # 科室信息
        departments = [
            {'name': '内科', 'total_beds': 120, 'floor': 2},
            {'name': '外科', 'total_beds': 100, 'floor': 3},
            {'name': '儿科', 'total_beds': 80, 'floor': 4},
            {'name': '妇产科', 'total_beds': 60, 'floor': 5},
            {'name': 'ICU', 'total_beds': 40, 'floor': 6},
            {'name': '急诊科', 'total_beds': 50, 'floor': 1},
            {'name': '心内科', 'total_beds': 70, 'floor': 7},
            {'name': '神经科', 'total_beds': 55, 'floor': 8}
        ]
        
        # 生成24小时趋势数据
        hours = [(datetime.now() - timedelta(hours=i)).strftime('%H:%M') for i in range(23, -1, -1)]
        
        # 创建数据
        data = []
        for dept in departments:
            # 模拟占用率在70-95%之间波动
            base_occupancy = random.uniform(0.7, 0.95)
            occupied_beds = int(dept['total_beds'] * base_occupancy)
            available_beds = dept['total_beds'] - occupied_beds
            
            data.append({
                'department': dept['name'],
                'total_beds': dept['total_beds'],
                'occupied_beds': occupied_beds,
                'available_beds': available_beds,
                'occupancy_rate': round(occupied_beds / dept['total_beds'] * 100, 1),
                'floor': dept['floor']
            })
        
        self.df = pd.DataFrame(data)
        print(f"生成模拟数据：{len(self.df)} 个科室")
    
    def get_key_indicators(self):
        """获取关键指标数据"""
        if self.df is None:
            return {}
        
        total_beds = self.df['total_beds'].sum()
        total_occupied = self.df['occupied_beds'].sum()
        total_available = self.df['available_beds'].sum()
        overall_occupancy = round(total_occupied / total_beds * 100, 1)
        
        return {
            "total_beds": int(total_beds),
            "occupied_beds": int(total_occupied),
            "available_beds": int(total_available),
            "occupancy_rate": overall_occupancy
        }
    
    def get_department_occupancy(self):
        """获取各科室占用率数据"""
        if self.df is None:
            return []
        
        result = []
        for _, row in self.df.iterrows():
            result.append({
                "name": row['department'],
                "value": row['occupancy_rate'],
                "occupied": int(row['occupied_beds']),
                "total": int(row['total_beds'])
            })
        
        return sorted(result, key=lambda x: x['value'], reverse=True)
    
    def get_occupancy_trend(self):
        """获取占用率趋势数据（模拟24小时数据）"""
        hours = [(datetime.now() - timedelta(hours=i)).strftime('%H:%M') for i in range(23, -1, -1)]
        
        # 模拟一天的占用率变化
        base_rate = 78  # 基础占用率
        trend_data = []
        occupied_data = []
        
        total_beds = self.df['total_beds'].sum() if self.df is not None else 575
        
        for i, hour in enumerate(hours):
            # 模拟占用率变化：夜间较低，白天较高
            hour_num = int(hour.split(':')[0])
            if 6 <= hour_num <= 18:  # 白天
                rate_modifier = random.uniform(-5, 10)
            else:  # 夜间
                rate_modifier = random.uniform(-15, 5)
            
            current_rate = max(50, min(95, base_rate + rate_modifier))
            current_occupied = int(total_beds * current_rate / 100)
            
            trend_data.append(round(current_rate, 1))
            occupied_data.append(current_occupied)
        
        return {
            "times": hours,
            "occupancy_rates": trend_data,
            "occupied_counts": occupied_data
        }
    
    def get_available_beds_by_department(self):
        """获取各科室可用床位数据"""
        if self.df is None:
            return {"departments": [], "available_counts": [], "colors": []}
        
        # 按可用床位数排序
        sorted_df = self.df.sort_values('available_beds', ascending=True)
        
        departments = sorted_df['department'].tolist()
        available_counts = sorted_df['available_beds'].tolist()
        
        # 根据可用床位数设置颜色
        colors = []
        for count in available_counts:
            if count < 5:
                colors.append('#ff4757')  # 红色：紧急
            elif count < 10:
                colors.append('#ffa502')  # 橙色：警告
            else:
                colors.append('#2ed573')  # 绿色：充足
        
        return {
            "departments": departments,
            "available_counts": available_counts,
            "colors": colors
        }
    
    def get_bed_distribution_heatmap(self):
        """获取病床分布热力图数据"""
        if self.df is None:
            return {"data": [], "departments": [], "floors": []}
        
        # 创建楼层和科室的热力图数据
        floors = sorted(self.df['floor'].unique())
        departments = self.df['department'].tolist()
        
        heatmap_data = []
        for i, dept in enumerate(departments):
            row_data = self.df[self.df['department'] == dept].iloc[0]
            floor_idx = floors.index(row_data['floor'])
            
            # [x轴索引, y轴索引, 占用率]
            heatmap_data.append([i, floor_idx, row_data['occupancy_rate']])
        
        return {
            "data": heatmap_data,
            "departments": departments,
            "floors": [f"{f}楼" for f in floors]
        }

# 创建数据处理器实例
data_processor = HospitalBedDataProcessor()

@app.route('/')
def dashboard():
    """主页路由 - 显示可视化大屏"""
    return render_template('hospital_dashboard.html')

@app.route('/api/key_indicators')
def api_key_indicators():
    """关键指标数据API"""
    data = data_processor.get_key_indicators()
    return jsonify(data)

@app.route('/api/department_occupancy')
def api_department_occupancy():
    """各科室占用率数据API"""
    data = data_processor.get_department_occupancy()
    return jsonify(data)

@app.route('/api/occupancy_trend')
def api_occupancy_trend():
    """占用率趋势数据API"""
    data = data_processor.get_occupancy_trend()
    return jsonify(data)

@app.route('/api/available_beds')
def api_available_beds():
    """可用床位数据API"""
    data = data_processor.get_available_beds_by_department()
    return jsonify(data)

@app.route('/api/bed_distribution')
def api_bed_distribution():
    """病床分布热力图数据API"""
    data = data_processor.get_bed_distribution_heatmap()
    return jsonify(data)

if __name__ == '__main__':
    print("=" * 80)
    print("🏥 启动医院病床使用情况可视化大屏")
    print("=" * 80)
    print("📊 访问地址：http://localhost:5002")
    print("📈 包含5个核心图表的病床数据可视化")
    print("=" * 80)
    
    app.run(debug=True, host='0.0.0.0', port=5002)