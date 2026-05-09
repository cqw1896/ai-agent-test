'''阿里云百炼（DashScope）的嵌入模型可以通过 langchain_community 中的 DashScopeEmbeddings 类来使用。虽然 init_chat_model 主要用于聊天模型，但嵌入模型可以使用 init_embeddings（如果支持）或直接使用 DashScopeEmbeddings。
以下是使用阿里云百炼嵌入模型（如 text-embedding-v3）的代码：
'''
# 语义搜索 - 使用阿里云百炼（DashScope）嵌入模型
## 索引
# 1. 读取PDF，按照页来管理，每页封装成一个Document对象, List[Document]
# 2. 分割文本, 文本段（chunk），每段封装成一个Document, List[Document]
# 3. 向量化：  文本段 <=> 向量， 需要嵌入模型来辅助
# 4. 向量库：  把多个 文本段:向量 存到向量库，OK了。
# uv add pypdf langchain-community dashscope

import os
import sys

# 【关键】清除所有代理环境变量，强制直连
for key in ['http_proxy', 'https_proxy', 'HTTP_PROXY', 'HTTPS_PROXY', 'all_proxy', 'ALL_PROXY']:
    if key in os.environ:
        del os.environ[key]

# 设置阿里云百炼 DashScope API Key（请替换为你的实际密钥）
# 也可以在创建 DashScopeEmbeddings 时直接传入 api_key 参数
os.environ['DASHSCOPE_API_KEY'] = 'your-dashscope-api-key-here'

# 1. 读取PDF，按照页来管理，Document, List[Document]
from langchain_community.document_loaders import PyPDFLoader

file_path = 'Git.pdf'
loader = PyPDFLoader(file_path)
docs = loader.load()
print(len(docs))  # 117
print(type(docs[0]))  # <class 'langchain_core.documents.base.Document'>
print(docs[0])  

# 2. 分割文本，文本段(chunk), Document, List[Document]
from langchain_text_splitters import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    add_start_index=True
)

all_splits = text_splitter.split_documents(docs)  # List[Document]
print('-' * 80)
print(len(all_splits))  # 134
print(all_splits[0])

# 3. 向量化：文本段 <=> 向量，需要嵌入模型来辅助
# 4. 文本块:向量 的存储, 包含第3步
from langchain_chroma import Chroma
from langchain_community.embeddings import DashScopeEmbeddings

# 使用 DashScopeEmbeddings 初始化阿里云百炼嵌入模型
# 常用模型: text-embedding-v3, text-embedding-v2, text-embedding-v1
embedding = DashScopeEmbeddings(
    model='text-embedding-v3',
    dashscope_api_key=os.environ.get('DASHSCOPE_API_KEY')
)

vector_store = Chroma(
    collection_name='example_collection',
    embedding_function=embedding,
    persist_directory='./chroma_langchain_db_dashscope'
)

ids = vector_store.add_documents(documents=all_splits)
print('*' * 80)
print(len(ids))
print(ids)