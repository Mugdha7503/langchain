from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence
from dotenv import load_dotenv

load_dotenv()

model= ChatGoogleGenerativeAI(model="gemini-2.5-pro")
parser= StrOutputParser()

prompt= PromptTemplate (
    template= "Give me the joke about {topic}",
    input_variables=["topic"]
)

chain= RunnableSequence(prompt, model, parser)

result = chain.invoke({"topic":"AI"})
print(result)