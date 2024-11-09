import os

from anthropic import Anthropic
from dotenv import load_dotenv


def main():
    load_dotenv()
    key = os.getenv('ANTHROPIC_API_KEY')
    client = Anthropic(api_key=key)
    msg = client.messages.create(
        model='claude-3-haiku-20240307',
        max_tokens=1000,
        messages=[{
            'role': 'user',
            'content': 'Hi there! Tell me a joke about little brothers.'
        }])
    print(msg.content[0].text)


if __name__ == '__main__':
    main()

    

    
