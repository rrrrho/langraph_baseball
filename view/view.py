from langchain_ollama import ChatOllama
from controller.controller import Controller

app = Controller(model = ChatOllama(
        model="llama3.2",
        temperature=0,
        max_retries=2
    ))

def ask(question: str):
    print("\n" + "="*70)
    print("❓ Question:", question)
    print("="*70)

    result = app.ask(question)

    print("\n" + "="*70)
    print("💡 ANSWER")
    print("="*70)
    print(result.get("answer", ""))
    print("="*70 + "\n")

    return result

result = ask("which pitcher is the best?")