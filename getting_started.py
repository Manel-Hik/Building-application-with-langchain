from dotenv import load_dotenv
from langchain.llms import OpenAI


load_dotenv()


llm = OpenAI(model_name='text-davinci-003', 
             temperature=0.9, #degree of randomness
             max_tokens = 256)



print(llm("tell me a funny joke please."))
