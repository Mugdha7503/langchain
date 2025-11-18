from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv() 

model = ChatGoogleGenerativeAI(model='gemini-2.5-pro')

template1= PromptTemplate(
    template="Write a detailed report on the {topic}",
    input_variables=["topic"]
)

template2= PromptTemplate(
    template="Write a 5 line summary on the following text. /n {text}",
    input_variables=["text"]
)

parser = StrOutputParser()

chain = template1 | model |parser | template2 | model | parser

result = chain.invoke({"topic":"Impact of AI on Society"})
print(result)