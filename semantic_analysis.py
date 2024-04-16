import os
import functools
import requests
import json
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from CustomChatZhipuAI import ChatZhipuAI
from dotenv import load_dotenv
load_dotenv()

word_format = {'评价维度': [{'观点词': ['情感倾向[正向，负向，不清晰]']}]}

user_input = '这个牙刷价格有点小贵，不过贵有贵的道理，清洁效果还是很不错的，物美价廉，下次还会买的。牙膏的包装也很好看，就是如果量再大点就好了。'

global_classify_question_prompt = None

def load_prompt():
    global global_classify_question_prompt  
    if global_classify_question_prompt is None:
        template_path = "asset\\template.md"
        with open(template_path, 'r', encoding='utf-8') as file:
            global_classify_question_prompt = file.read()
    return global_classify_question_prompt



def get_chain():
    prompt = ChatPromptTemplate.from_template(load_prompt())
    llm = ChatZhipuAI(
        temperature=0.1,
        api_key=os.getenv("ZHIPUAI_API_KEY"),
        model_name="glm-3-turbo",
    )

    # llm_with_stop = llm.bind(stop=["</label>"])
    chain = prompt | llm | StrOutputParser()
    return chain

if __name__ == '__main__':
    chain = get_chain()
    print("================>chain",chain)
    print("\n")
    res = chain.invoke({
        "word_format": word_format,
        "user_input": user_input,
    })
    print(f"============>LLM label:{res}")



