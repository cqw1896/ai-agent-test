from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
load_dotenv()
# model=ChatOpenAI(
#     model='MiniMax-M2.5-highspeed',  # 兼容openai，.env中的变量名必须是OPENAI_API_KEY or OPENAI_ADMIN_KEY
#     base_url='https://blazeai.boxu.dev/api/', # 兼容openai的第三方模型 
#     temperature=0.1,
#     max_tokens=2000,
#     timeout=None,
#     max_retries=2
# )

# langchain0.1
from langchain.chat_models import init_chat_model
model=init_chat_model(
    # model='qwen3.5-omni-flash-thinking-search',
    # model_provider='openai',
    model='openai:qwen3.5-omni-flash-thinking-search',
    base_url='https://blazeai.boxu.dev/api/',
    temperature=0.1
    
)
for chunk in model.stream('来段毛泽东诗词'):
    print(chunk.content,end='',flush=True)