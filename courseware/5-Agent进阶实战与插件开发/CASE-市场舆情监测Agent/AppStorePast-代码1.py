from typing import Any
async def main(args: Args) -> Output:
    params = args.params
    input_list = params['input_list']
    key0 = []
    key1 = []
    for obj in input_list:
        for item in obj['output']:
            if item['estimate'] == '好评':
                key0.append(item)
            elif item['estimate'] == '差评':
                key1.append(item)

    ret: Output = {
        "key0_good": key0,
        "key1_bad": key1,
    }
    return ret
