import ollama

response = ollama.chat(
    model="qwen2.5:7b",
    messages=[{"role": "user", "content": "Hello, how are you?"}]
)

print(response)