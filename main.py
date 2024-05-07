import os
import functools
import requests
import json
from utils.parse import parse_output
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from CustomChatZhipuAI import ChatZhipuAI
from dotenv import load_dotenv
load_dotenv()

# word_format = {'评价维度': [{'观点词': ['情感倾向[正向，负向，不清晰]']}]}

# user_input = '这个牙刷价格有点小贵，不过贵有贵的道理，清洁效果还是很不错的，物美价廉，下次还会买的。牙膏的包装也很好看，就是如果量再大点就好了。'
# user_input = '618囤货，送了点小礼品。对比线下价格划算。每年618囤一次，节约点开支。一直都在用云南白药，虽然价格贵了些,但是用起来感觉确实有功效，毕竟还有药物成分,\
# 防止牙龈出血还是很有效果的,618促销很划算，希望平时也会经常有这么大的促销力度,考虑回购，给个好评'
# user_input = '感觉选了很温和的一个味道了，味道还是有点冲'
# user_input = '每次收货都是非常愉快的,可是只要一想到还要给评价, 头就大了。\
#     幸好万能的网友出来一套通用的网购模板,如果你是 想看评论决定买不买这个宝贝,你可以打住了。\
#         因为我说的你不 一定信,但是我自己却坚定不移的要给好评,为啥呢'
# user_input = '东西很好，希望价格便宜点！'
user_input = '还可以吧，这种牙膏价格方面偏贵些，希望以后更优惠一些吧，质量也更好一些，\
没什么异味，还可以吧，希望以后更优惠些，还可以吧。'
global_classify_question_prompt = None

def load_prompt():
    global global_classify_question_prompt  
    if global_classify_question_prompt is None:
        template_path = "asset\\template2.md"
        with open(template_path, 'r', encoding='utf-8') as file:
            global_classify_question_prompt = file.read()
    return global_classify_question_prompt



def get_chain():
    prompt = ChatPromptTemplate.from_template(load_prompt())
    llm = ChatZhipuAI(
        temperature=0.01,
        api_key=os.getenv("ZHIPUAI_API_KEY"),
        model_name="glm-3-turbo",
    )

    # llm_with_stop = llm.bind(stop=["</label>"])
    chain = prompt | llm | StrOutputParser()
    return chain


    
if __name__ == '__main__':
    chain = get_chain()
    # print("================>chain",chain)
    res = chain.invoke({
        # "word_format": word_format,
        "user_input": user_input,
    })
    print(f"============>LLM label:{res}")

    #后处理
    res = parse_output(res, ['answer'])
    print("============>最终结果",res)
    print("============>answer",res['answer'][0])




