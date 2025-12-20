async def main(args: Args) -> Output:
    params = args.params
    # 获取输入数据
    data = params['input']  # 这里data就是一个list
    # 过滤掉空字符串和只包含空白字符的项
    filtered_data = [item for item in data if item.strip() != ""]
    ret: Output = {
        "key0": filtered_data
    }
    return ret
