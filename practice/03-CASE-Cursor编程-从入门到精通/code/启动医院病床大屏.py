#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
启动医院病床可视化大屏
"""

import subprocess
import sys
from pathlib import Path

def main():
    current_dir = Path(__file__).parent
    app_file = current_dir / "医院病床可视化大屏.py"
    
    print("🏥 正在启动医院病床使用情况可视化大屏...")
    print("📊 访问地址：http://localhost:5002")
    print("=" * 60)
    
    try:
        # 启动Flask应用
        subprocess.run([sys.executable, str(app_file)], check=True)
    except KeyboardInterrupt:
        print("\n👋 应用已停止")
    except Exception as e:
        print(f"❌ 启动失败：{e}")

if __name__ == "__main__":
    main()