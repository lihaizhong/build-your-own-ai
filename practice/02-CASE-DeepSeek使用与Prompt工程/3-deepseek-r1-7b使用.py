import os
from modelscope import AutoModelForCausalLM, AutoTokenizer

# 1. 加载模型：从本地路径加载 DeepSeek-R1 7B 模型
model_name = "./models/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
print(f"模型路径: {model_name}")
print(f"路径是否存在: {os.path.exists(model_name)}")
print("开始加载模型...")
model = AutoModelForCausalLM.from_pretrained(
  model_name,
  torch_dtype="auto",
  device_map="auto",
)
tokenizer = AutoTokenizer.from_pretrained(model_name)
prompt = "帮我用JavaScript写一个二分查找法"
messages = [
  {"role": "system", "content": "you are a helpful assistant"},
  {"role": "user", "content": prompt}
]

# 2. 构建对话：创建包含系统提示和用户请求的对话
text = tokenizer.apply_chat_template(
  messages,
  tokenize=False,
  add_generation_prompt=True
)

# 3. 文本处理：将对话转换为模型输入格式
model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

# 4. 生成文本：模型根据输入生成回复（最多2000个token）
generated_ids = model.generate(
  **model_inputs,
  max_new_tokens=2000
)

# 5. 提取回复：从生成结果中提取只属于模型回复的部分
generated_ids = [
  # 模型生成的 generated_ids 包含了原始输入 + 新生成的内容
  # 通过切片 [len(input_ids):] 只保留新生成的部分
  # 去掉原始输入，只获取模型新产生的回复
  output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
]

# 等价的传统for循环写法
# new_generated_ids = []
# for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids):
  # new_tokens = output_ids[len(input_ids):]
  # new_generated_ids.append(new_tokens)
# generated_ids = new_generated_ids

# 6. 解码输出：将token转换为可读文本并打印
response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
print(response)
