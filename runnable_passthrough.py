from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence,RunnablePassthrough, RunnableParallel
from dotenv import load_dotenv

load_dotenv()


model= ChatGoogleGenerativeAI(model='gemini-2.5-pro')

parser= StrOutputParser()

prompt1 = PromptTemplate(

    template="Tell me the joke about {topic}",
    input_variables=["topic"]
)

prompt2= PromptTemplate(
    template= "Tell me the explaination too{topic}",
    input_variables=["topic"]
)

joke_gen_chain=RunnableSequence(prompt1, model, parser)

parallel_chain= RunnableParallel({
    'joke': RunnablePassthrough(),
    'explanation': RunnableSequence(prompt2,model,parser)
}
)

final_chain = joke_gen_chain | parallel_chain





result = final_chain.invoke({"topic":"cricket"})
print(result)




