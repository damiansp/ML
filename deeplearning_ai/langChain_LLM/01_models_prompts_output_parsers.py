import os

from dotenv import find_dotenv, load_dotenv
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
import openai


_ = load_dotenv(find_dotenv())
openai.api_key = os.environ['OPENAI_API_KEY']

MOD = 'gpt-3.5-turbo'


def main():
    customer_email = (
        '''Arrr, I be fuming that me blender lid flew off and splattered me kitchen
        walls with smoothie! And to make matters worse, the warranty don't cover the
        cost of cleaning up me kitchen. I need yer help right now, matey!''')
    style = 'American English in a calm and respectful tone'
    prompt = (
        f'''Translate the text that is delimited by triple backticks into a style
        that is {style}. Text: ```{customer_email}```''')
    # LangChain
    chat = ChatOpenAI(temperature=0.)
    template_str = (
        '''Translate the text that is delimited by triple backticks into a style
        that is {style}. Text: ```{text}```''')
    prompt_template = ChatPromptTemplate.from_template(template_str)
    print(prompt_template.messages[0].prompt)
    print(prompt_template.messages[0].prompt.input_variables)
    customer_msgs = prompt_template.format_messages(style=style, text=customer_email)
    print(type(customer_msgs[0]))
    # output parsers
    customer_review = (
        '''This leaf blower is pretty amazing.  It has four settings: candle blower,
        gentle breeze, windy city, and tornado. It arrived in two days, just in time
        for my wife's anniversary present. I think my wife liked it so much she was
        speechless. So far I've been the only one using it, and I've been using it
        every other morning to clear the leaves on our lawn. It's slightly more
        expensive than the other leaf blowers out there, but I think it's worth it
        for the extra features.''')
    review_template = (
        '''For the following text, extract the following information:
        gift: Was the item purchased as a gift for someone else?
        Answer True if yes, False if not or unknown.
        delivery_days: How many days did it take for the product to arrive? If this
        information is not found, output -1.
        price_value: Extract any sentences about the value or price, and output them
        as a comma separated Python list.
        Format the output as JSON with the following keys:
        gift
        delivery_days
        price_value
        text: {text}''')
    prompt_template = ChatPromptTemplate.from_template(review_template)
    print(prompt_template)
    

    
def get_completion(prompt, temp=0):
    msgs = [{'role': 'user', 'content': prompt}]
    resp = openai.ChatCompletion.create(model=MOD, messages=msgs, temperature=temp)
    return resp.choices[0].message['content']
