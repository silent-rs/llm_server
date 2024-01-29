from openai import OpenAI

if __name__ == '__main__':
    client = OpenAI(base_url="http://localhost:8000/v1", api_key="123456")

    completion = client.chat.completions.create(
        model="yi-chat-6b.Q5_K_M.gguf",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello!"}
        ]
    )

    print(completion.choices[0].message)
