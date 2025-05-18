import asyncio
from openai import AsyncOpenAI

with open('prompt.txt', 'r', encoding='utf-8') as file:
    prompt1 = file.read()

async def deep_seek(data,prompt=prompt1):
    client = AsyncOpenAI(
        api_key="",
        base_url="https://api.deepseek.com"
    )

    response = await client.chat.completions.create(
        model="deepseek-reasoner",
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": str(data)},
        ],
        stream=False
    )

    print(response.choices[0].message.content)
