# 语义搜索
## 索引
# 1. 读取PDF， 按照页来管理，每页封装成一个Document对象, List[Document]
# 2. 分割文本, 文本段（chunk），每段封装成一个Document, List[Document]
# 3. 向量化：  文本段 <=> 向量， 需要嵌入模型来辅助
# 4. 向量库：  把多个 文本段:向量 存到向量库，OK了。
# uv add pypdf
import os
import sys

# 【关键】清除所有代理环境变量，强制直连
for key in [
    "http_proxy",
    "https_proxy",
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "all_proxy",
    "ALL_PROXY",
]:
    if key in os.environ:
        del os.environ[key]
# 1. 读取PDF，按照页来管理，Document, List[Document]
from langchain_community.document_loaders import PyPDFLoader

# from pathlib import Path
# file_path = Path(__file__).with_name("Git.pdf")
file_path = "Git.pdf"
loader = PyPDFLoader(file_path)
docs = loader.load()
print(len(docs))  # 117
print(type(docs[0]))  # <class 'langchain_core.documents.base.Document'>
print(docs[0])
# page_content='Git教程 ...https://liaoxuefeng.com/books/git/'
# metadata={
#   'producer': 'pdf-lib (https://github.com/Hopding/pdf-lib)',
#   'creator': 'Chromium',
#   'creationdate': '2025-06-16T11:49:04+00:00',
#   'title': 'Git教程',
#   'author': '廖雪峰',
#   'subject': '浅显易懂，适合小白用户入门的Git教程！',
#   'moddate': '2025-06-16T11:49:12+00:00',
#   'source': 'Git.pdf',
#   'total_pages': 117,
#   'page': 0, # 当前页
#   'page_label': '1'  # 当前页码
# }
# 2. 分割文本，文本段(chunk), Document, List[Document]
from langchain_text_splitters import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=200, add_start_index=True
)

all_splits = text_splitter.split_documents(docs)  # List[Document]
print("-" * 80)
print(len(all_splits))  # 134
print(all_splits[0])

# page_content='Git教程...'
# metadata={
#   'producer': 'pdf-lib (https://github.com/Hopding/pdf-lib)',
#   'creator': 'Chromium',
#   'creationdate': '2025-06-16T11:49:04+00:00',
#   'title': 'Git教程',
#   'author': '廖雪峰',
#   'subject': '浅显易懂，适合小白用户入门的Git教程！',
#   'moddate': '2025-06-16T11:49:12+00:00',
#   'source': 'Git.pdf',
#   'total_pages': 117,
#   'page': 0,
#   'page_label': '1',
#   'start_index': 0
# }

# 3. 向量化：文本段 <=> 向量，需要嵌入模型来辅助
# from langchain_ollama import OllamaEmbeddings

# embedding=OllamaEmbeddings(model='nomic-embed-text')
# vector_0=embedding.embed_query(all_splits[0].page_content)
# print(len(vector_0))
# print(vector_0)

# 4. 文本块:向量 的存储, 包含第3步
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

embedding = OllamaEmbeddings(model="nomic-embed-text")
vector_store = Chroma(
    collection_name="example_collection",
    embedding_function=embedding,
    persist_directory="./chroma_langchain_db",
)

ids = vector_store.add_documents(documents=all_splits)
print("*" * 80)
print(len(ids))
print(ids)
