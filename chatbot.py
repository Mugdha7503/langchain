from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI 
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

model=ChatGoogleGenerativeAI(model='gemini-2.5-pro')


#Indexing
video_id = "LPZh9BOjkQs" 
ytt_api = YouTubeTranscriptApi()
try:
    transcript_list = ytt_api.fetch(video_id,languages=["en"])
    raw = transcript_list.to_raw_data()
   
    transcript = " ".join(chunk["text"] for chunk in raw)
   
    

except TranscriptsDisabled:
    print("No captions available for this video.")


splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = splitter.create_documents([transcript])

vector_store = Chroma(
    embedding_function=GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001"),
    persist_directory='chatbot_db',
    collection_name='chatbot'
)


vector_store.add_documents(chunks)
vector_store.persist()


def ask_chatbot(query: str):
    docs = vector_store.similarity_search(query, k=3)
    context = "\n\n".join([doc.page_content for doc in docs])

    prompt = f"""
    Use ONLY the following transcript to answer the question.
    If answer is unknown, say "I don't know".

    Transcript:
    {context}

    Question: {query}
    """

    response = model.invoke(prompt)
    return response.content


print(ask_chatbot("What is the video mainly about?"))


