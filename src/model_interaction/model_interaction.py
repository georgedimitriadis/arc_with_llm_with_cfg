
from openai import OpenAI

class ModelClient:
    def __init__(self, system_content:str = "You are a helpful coding assistant."):
        self.client = OpenAI(
            base_url="http://127.0.0.1:8080/v1",
            api_key="not-needed"  # llama.cpp doesn't check this, but the client requires it
        )

        self.model = 'qwen3.6'
        self.messages = [{'role': 'system', 'content': system_content}]
        self.context = ""
        self.response = ""

    def ask(self, question):
        self.messages.append({"role": "user", "content": question})
        response = self.client.chat.completions.create(model=self.model, messages=self.messages)
        reply = response.choices[0].message.content
        self.messages.append({"role": "assistant", "content": reply})
        return reply


