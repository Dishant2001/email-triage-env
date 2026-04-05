import os
from openai import OpenAI

client = OpenAI(
    base_url="https://router.huggingface.co/v1",
    api_key="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",
)

completion = client.chat.completions.create(
    model="openai/gpt-oss-120b:groq",
    messages=[
        {
            "role": "user",
            "content": "What is the capital of France?"
        }
    ],
)

print(completion.choices[0].message)