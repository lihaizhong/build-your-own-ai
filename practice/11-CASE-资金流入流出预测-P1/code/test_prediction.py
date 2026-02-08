#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from ...shared import get_project_path


def generate_prediction_template():
    """
    生成考试预测文件模板
    基于 data/comp_predict_table.csv 的格式
    """
    print("=== 生成考试预测文件模板 ===")
    
    # 读取格式参考文件
    format_ref_file = get_project_path('..', 'data', 'comp_predict_table.csv')
    output_file = get_project_path('..', 'prediction_result', 'tc_comp_predict_table.csv')
    
    try:
        # 读取参考文件的日期
        with open(format_ref_file, 'r') as f:
            lines = f.readlines()
        
        if not lines:
            print("错误：格式参考文件为空")
            return False
        
        # 提取日期并使用示例数据生成完整预测文件
        prediction_dates = []
        for line in lines:
            date_str = line.strip().split(',')[0]  # 获取日期部分
            prediction_dates.append(date_str)
        
        print(f"检测到需要预测的日期: {prediction_dates}")
        
        # 生成预测文件（使用简单的示例逻辑，实际应该用训练好的模型）
        # 这里先用示例值，实际项目中应该用机器学习模型预测
        with open(output_file, 'w') as f:
            for date in prediction_dates:
                # 示例：使用固定的预测值，实际应该根据训练好的模型计算
                # 这里使用参考文件中第一行的值作为示例
                if date == "20140901":
                    purchase_pred = 50000000  # 5亿
                    redeem_pred = 35000000   # 3.5亿
                elif date == "20140902":
                    purchase_pred = 52000000  # 5.2亿
                    redeem_pred = 37000000   # 3.7亿
                elif date == "20140903":
                    purchase_pred = 51000000  # 5.1亿
                    redeem_pred = 36000000   # 3.6亿
                else:
                    # 默认值，实际应该预测
                    purchase_pred = 50000000
                    redeem_pred = 35000000
                
                f.write(f"{date},{purchase_pred},{redeem_pred}\n")
        
        print(f"✅ 预测文件已生成: {output_file}")
        print("📋 文件内容预览:")
        
        with open(output_file, 'r') as f:
            for i, line in enumerate(f, 1):
                print(f"  {i}. {line.strip()}")
        
        return True
        
    except FileNotFoundError:
        print(f"错误：找不到格式参考文件 {format_ref_file}")
        return False
    except Exception as e:
        print(f"生成预测文件时发生错误: {e}")
        return False


if __name__ == "__main__":
    generate_prediction_template()
