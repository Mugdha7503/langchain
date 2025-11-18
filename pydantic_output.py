from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel,Field
from dotenv import load_dotenv

load_dotenv()

model = ChatGoogleGenerativeAI(model='gemini-2.5-pro')
class Person(BaseModel):
    name: str= Field(description="The full name of the person")
    age: int= Field(description="The age of the person")
    city: str = Field(description="The city of the person")

parser= PydanticOutputParser(pydantic_object=Person)

template= PromptTemplate(
    template='Give me the name, age and city of a fictional {place} person \n {format_instruction}',
    input_variables=["place"],
    partial_variables={"format_instruction": parser.get_format_instructions()}
)

chain= template | model | parser
result=chain.invoke({"place":"American"})

print(result)


