from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

prompt = PromptTemplate(
    template= "Tell me five facts about {topic}",
    input_variables=["topic"]
)

model= ChatGoogleGenerativeAI(model='gemini-2.5-pro')

parser= StrOutputParser()

chain = prompt | model | parser
result= chain.invoke({"topic":"Cricket"})

print(result)
chain.get_graph().draw_ascii()