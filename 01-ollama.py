# from langchain_ollama import ChatOllama

# model=ChatOllama(
#     model='deepseek-r1:1.5b',
#     base_url="http://localhost:11434",
#     temperature=0.1
#     )


# langchain1.0
from langchain.chat_models import init_chat_model

model = init_chat_model(
    model="deepseek-r1:1.5b",
    model_provider="ollama",  # 必须
    # model="ollama:deepseek-r1:1.5b",
    base_url="http://localhost:11434",
    temperature=0.1,
    timeout=30,
    max_token=2000,
)
for chunk in model.stream("来一段唐诗"):
    print(chunk.content, end="", flush=True)
"""
init_chat_model 看到模型名称包含 "deepseek"，
自动将其识别为 DeepSeek 官方 API，
因此要求设置 DEEPSEEK_API_KEY 环境变量。
通过显式指定 model_provider="ollama"，
可以强制使用本地 Ollama 服务。
"""
