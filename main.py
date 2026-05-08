from langchain_ollama import ChatOllama
def main():
    print("Hello from ai-agent-test!")
    llm=ChatOllama(
    model="qwen3:4b",
    temperature=0,
    # other params...
)
    resop=llm.invoke("你是谁")
    print(resop)

if __name__ == "__main__":
    main()
