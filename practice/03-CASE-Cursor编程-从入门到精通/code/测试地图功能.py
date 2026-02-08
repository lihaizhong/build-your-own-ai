#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试地图功能是否正常工作
"""

import requests

def test_map_functionality():
    """测试地图功能"""
    base_url = "http://localhost:5001"
    
    print("=" * 80)
    print("🗺️ 测试香港疫情地图功能")
    print("=" * 80)
    
    try:
        # 测试主页是否正常
        print("📊 测试主页访问...")
        response = requests.get(f"{base_url}/", timeout=5)
        if response.status_code == 200:
            print("✅ 主页访问正常")
        else:
            print(f"❌ 主页访问失败: {response.status_code}")
            return
            
        # 测试地图数据API
        print("\n📊 测试地图数据API...")
        response = requests.get(f"{base_url}/api/district_distribution", timeout=5)
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ 地图数据获取成功")
            print(f"   数据数量: {len(data)} 个地区")
            
            if data:
                print(f"\n📋 地图数据样例:")
                for i, item in enumerate(data[:5]):
                    print(f"   {i+1}. {item['name']}: 累计{item['value']}例, 新增{item['new_cases']}例")
                
                # 检查数据完整性
                required_fields = ['name', 'value', 'new_cases', 'active_cases']
                all_fields_present = all(
                    all(field in item for field in required_fields) 
                    for item in data
                )
                
                if all_fields_present:
                    print("✅ 数据字段完整")
                else:
                    print("❌ 数据字段不完整")
                
                # 检查地区名称
                hk_districts = [
                    '中西区', '湾仔区', '东区', '南区', 
                    '深水埗区', '油尖旺区', '九龙城区', '黄大仙区', '观塘区',
                    '荃湾区', '屯门区', '元朗区', '北区', '大埔区', 
                    '沙田区', '西贡区', '离岛区', '葵青区'
                ]
                
                data_districts = [item['name'] for item in data]
                print(f"\n🏙️ 地区名称检查:")
                print(f"   预期地区数: {len(hk_districts)}")
                print(f"   实际地区数: {len(data_districts)}")
                
                missing_districts = set(hk_districts) - set(data_districts)
                if missing_districts:
                    print(f"   缺失地区: {missing_districts}")
                else:
                    print("   ✅ 所有地区都有数据")
                    
            else:
                print("❌ 地图数据为空")
        else:
            print(f"❌ 地图数据获取失败: {response.status_code}")
            
    except requests.exceptions.ConnectionError:
        print("❌ 连接失败: Flask应用可能未启动")
        print("   请先运行: uv run 疫情可视化大屏.py")
    except Exception as e:
        print(f"❌ 测试失败: {e}")
    
    print(f"\n{'='*80}")
    print("测试完成！")
    print("如果地图仍然不显示，可能是前端ECharts地图注册问题")
    print("建议打开浏览器开发者工具查看控制台错误信息")

if __name__ == "__main__":
    test_map_functionality()