import os
import requests
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

load_dotenv(verbose=True)
ollama_base_url = os.getenv("OLLAMA_BASE_URL")

app = FastAPI()

# 定义请求模型
class ChatRequest(BaseModel):
  prompt: str
  model: str = "deepseek-r1:8b"

# 允许跨域请求（根据需要配置）
app.add_middleware(
  CORSMiddleware,
  allow_origins=["*"],
  allow_methods=["*"],
  allow_headers=["*"],
)

@app.post("/api/chat")
async def chat(request: ChatRequest):
  url = f"{ollama_base_url}/api/generate"
  print(f"ollama url: {url}")
  data = {
    "model": request.model,
    "prompt": request.prompt,
    "stream": False
  }
  response = requests.post(url, json=data)
  if response.status_code == 200:
    return {"response": response.json()["response"]}
  else:
    return {"error": "Failed to get response from Ollama"}, 500

if __name__ == "__main__":
  uvicorn.run(app, host="0.0.0.0", port=8000)
