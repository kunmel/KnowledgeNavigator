import requests
import openai
import config

openai.api_key = 'openai.api_key'
openai.base_url  = 'openai.base_url'

def bert_predict(query, task):
    payload = {"text": query}
    response = requests.post(config.bert_url_list[task], json=payload).json()
    return response["answer"]

def chat_llama2(query):
    payload = {"texts": query}
    response = requests.post(config.llama2_url, json=payload).json()['answer'][0]
    return response

def chat_llama2_multi(query):
    payload = {"texts": query}
    response = requests.post(config.llama2_url, json=payload).json()['answer']
    return response

def chat_gpt_4(query):
    completion = openai.chat.completions.create(
    model="gpt-4-1106-preview",
    messages=[
        {"role": "user", "content": query}
    ]
    )
    return completion.choices[0].message.content
