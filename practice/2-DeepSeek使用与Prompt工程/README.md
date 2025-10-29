# 二、DeepSeek使用和提示词工程

## DeepSeek的创新

- MLA（Multi-Head Latent Attention）
  - 一种通过低秩键值联合压缩的注意力机制
- MoE（Mix of Expert）
  - 原来都是Dense架构  
- 混合精度框架

---

## DeepSeek私有化部署选择

- ollama  
- vLLM  
- SGLang

---

## Ollama部署DeepSeek-R1

![Pasted image 20250831235203.png](../../public/Pasted_Image_20250831235203.png)

[ollama](https://ollama.com/)

---

## vLLM部署DeepSeek-R1

![Pasted image 20250831235339.png](../../public/Pasted_Image_20250831235339.png)

[vLLM](https://docs.vllm.ai/en/stable/)

---

## SGLang部署Deepseek-R1

![Pasted image 20250831235454.png](../../public/Pasted_Image_20250831235454.png)

[SGLang](https://docs.sglang.ai/)

---

### 不同尺寸大模型选择

大模型大小乘以2 就是对显存的需求 例如：7B的模型，使用14GB显存

---

## 提示词工程

- **具体指导**：给予模型明确的指导和约束  
- **简洁明了**：使用简练、清晰的语言表达Prompt
- **适当引导**：通过示例或问题边界引导模型  
- **优化迭代**：根据输出结果，持续调整和优化

---
测试：

> 请从伊利集团财务报表中提取以下信息，包括：公司名称，股票代码，营收，净利润，毛利，总资产，总负债。并以JSON格式返回。

## Prompt工程：设计与优化

### Prompt原理

---

### 提示词关键原则

---

### 提示词编写框架

重要性排序

---

### 提示词编写技巧

限制格式、区分、小样本学习、CoT、角色扮演等

---

### Prompt工程实战
