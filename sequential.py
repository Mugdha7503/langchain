from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

prompt1= PromptTemplate(
    template='Give me a brief summary about {subject} in less than 50 words.',
    input_variables=['subject']
)

prompt2= PromptTemplate(
    template='Now, provide three interesting facts about {text}',
    input_variables=['text']
)

model = ChatGoogleGenerativeAI(model='gemini-2.5-pro')

parser= StrOutputParser()


chain= prompt1 | model | parser | prompt2 | model | parser

result= chain.invoke({'subject':'Unemployment in India'})
print(result)


