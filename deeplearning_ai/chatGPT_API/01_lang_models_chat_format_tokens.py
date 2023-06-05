import os

from dotenv import load_dotenv, find_dotenv
import openai
import tiktoken


_ = load_dotenv(find_dotenv())  # read local .env file
openai.api_key = os.environ['OPENAI_API_KEY']


def main():
    # prompt mod and get a completion
    resp = get_completion('What is the capital of Namibia?')
    print(resp)
    # Tokens
    reps = get_completion('Take the letters in "lollipop" and reverse them')
    print(resp)
    reps = get_completion('Take the letters in l-o-l-l-i-p-o-p and reverse them')
    print(resp)
    # Chat format
    msgs = [
        {'role': 'system',
         'content': 'You are an assistant who responds in the style of Dr. Seuss'},
        {'role': 'user', 'content': 'Sing the praises of the humble carrot'}]
    resp = get_completion_from_messages(msgs, temp=1)
    print(resp)
    msgs = [
        {'role': 'system',
         'content': 'All of your responses must be one sentence long'},
        {'role': 'user', 'content': 'Sing the praises of the humble carrot'}]
    resp = get_completion_from_messages(msgs, temp=1)
    print(resp)
    resp, token_dict = get_completion_from_messages(
        msgs, temp=1, return_token_count=True)
    print(resp)
    print(token_dict)    


def get_completion(prompt, mod='gpt-3.5-turbo', temp=0):
    msgs = [{'role': 'user', 'content': prompt}]
    resp = openai.ChatCompletion.create(model=mod, messages=msgs, temperature=temp)
    return resp.choices[0].message['content']


def get_completion_from_messages(
        msgs, mod='gpt-3.5-turbo', temp=0, max_tokens=500, return_token_count=False):
    resp = openai.ChatCompletion.create(
        model=mod, messages=msgs, temperature=temp, max_tokens=max_tokens)
    content = resp.choices[0].message['content']
    if return_token_count:
        token_dict = {
            'prompt_tokens': reps['usage']['prompt_tokens'],
            'completion_tokens': resp['usage']['completion_tokens'],
            'total_tokens': resp['usage']['total_tokens']}
        return content, token_dict
    return content
