import json

from openai import OpenAI

with open('config/keys.json', 'r') as f:
    keys = json.load(f)
    KEY = keys['openai']['key']

client = OpenAI(api_key=KEY)
completion = client.chat.completions.create(
  model="gpt-4o-mini",
  store=True,
  messages=[{"role": "user", "content": "write a haiku about ai"}])
print(completion.choices[0].message);


