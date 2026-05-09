import os
import sys

# 【关键】清除所有代理环境变量，强制直连
for key in ['http_proxy', 'https_proxy', 'HTTP_PROXY', 'HTTPS_PROXY', 'all_proxy', 'ALL_PROXY']:
    if key in os.environ:
        del os.environ[key]

from langchain_ollama import OllamaEmbeddings
import requests

# 使用 localhost，因为现在我们在 WSL 内部，且没有代理干扰
# 如果 localhost 依然不行，再换回 WSL IP
BASE_URL = 'http://127.0.0.1:11434' 

print(f"正在测试连接: {BASE_URL}")

# 1. 先用 requests 裸测
try:
    resp = requests.post(
        f"{BASE_URL}/api/embeddings", 
        json={"model": "nomic-embed-text", "prompt": "test"},
        timeout=10
    )
    print(f"Requests 状态码: {resp.status_code}")
    if resp.status_code == 200:
        print("✅ Requests 测试成功!")
    else:
        print(f"❌ Requests 失败: {resp.text}")
except Exception as e:
    print(f"❌ Requests 异常: {e}")

# 2. 再用 LangChain 测试
try:
    embedding = OllamaEmbeddings(
        model='nomic-embed-text',
        base_url=BASE_URL
    )
    test_vector = embedding.embed_query("test text")
    print(f"✅ LangChain 成功! 维度: {len(test_vector)}")
except Exception as e:
    print(f"❌ LangChain 失败: {e}")