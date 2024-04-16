from CustomChatZhipuAI import ChatZhipuAI
import os
from dotenv import load_dotenv
load_dotenv()


llm = ChatZhipuAI(
    temperature=0.1,
    api_key=os.getenv("ZHIPUAI_API_KEY"),
    model_name="glm-4",
)
res = llm.invoke("langsmith如何帮助测试?")
print(res)