#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试疫情可视化大屏的API数据
"""

import requests
import json
from pathlib import Path

def test_api_endpoints():
    """测试所有API端点"""
    base_url = "http://localhost:5001"
    
    endpoints = [
        "/api/daily_summary",
        "/api/district_distribution", 
        "/api/trend_analysis",
        "/api/district_ranking",
        "/api/key_indicators"
    ]
    
    print("=" * 80)
    print("🧪 测试疫情可视化大屏API数据")
    print("=" * 80)
    
    for endpoint in endpoints:
        url = f"{base_url}{endpoint}"
        try:
            print(f"\n📊 测试 {endpoint}...")
            response = requests.get(url, timeout=5)
            
            if response.status_code == 200:
                data = response.json()
                print(f"✅ 成功获取数据")
                
                if endpoint == "/api/district_distribution":
                    print(f"   数据数量: {len(data)} 个地区")
                    if data:
                        print(f"   样例数据: {data[0]}")
                        for item in data[:5]:  # 显示前5个地区
                            print(f"   - {item['name']}: 累计{item['value']}例")
                elif endpoint == "/api/daily_summary":
                    print(f"   日期数量: {len(data.get('dates', []))}")
                    print(f"   新增确诊: {data.get('new_cases', [])}")
                elif endpoint == "/api/key_indicators":
                    print(f"   关键指标: {data}")
                else:
                    print(f"   数据类型: {type(data)}")
                    if isinstance(data, dict):
                        print(f"   数据键: {list(data.keys())}")
            else:
                print(f"❌ HTTP错误: {response.status_code}")
                
        except requests.exceptions.ConnectionError:
            print(f"❌ 连接失败: Flask应用可能未启动")
        except Exception as e:
            print(f"❌ 请求失败: {e}")
    
    print(f"\n{'='*80}")
    print("测试完成！")

if __name__ == "__main__":
    test_api_endpoints()