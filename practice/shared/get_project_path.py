import inspect
from pathlib import Path

def get_project_path(*paths: str) -> Path:
    """获取相对于调用脚本的路径
    
    使用示例：
    在 practice/11-CASE-资金流入流出预测-P1/code/prophet_v7_prediction.py 中：
    
    from practice.shared.get_project_path import get_project_path
    data_dir = get_project_path("..", "data")  # 返回 practice/11-CASE-资金流入流出预测-P1/data
    """
    try:
        # 获取调用者的文件路径
        current_frame = inspect.currentframe()
        if current_frame is None:
            raise RuntimeError("Cannot get current frame")
        
        frame = current_frame.f_back
        if frame is None:
            raise RuntimeError("Cannot get caller frame")
        
        caller_file = Path(frame.f_globals['__file__']).resolve()
        caller_dir = caller_file.parent
        
        if paths:
            return (caller_dir / Path(*paths)).resolve()
        else:
            return caller_dir
    except (NameError, AttributeError):
        # 交互式环境（Jupyter等）
        return Path.cwd().joinpath(*paths) if paths else Path.cwd()
