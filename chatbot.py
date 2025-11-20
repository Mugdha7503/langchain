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
print(transcript)

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
    You are a helpful AI chatbot answering questions ONLY using the following transcript content.

    - If information is not available, reply: "I don't know based on the video."
    - Do not hallucinate or add outside infowhat is the meaning og mind-boogling from the textrmation.
    - Keep answers clear and concise.

    Transcript:
    {context}

    Question: {query}
    """

    response = model.invoke(prompt)
    return response.content.strip()

print("\n Chatbot is ready! Ask anything about the video. Type 'exit' to quit.\n")
while True:
    user_input = input("You: ").strip()

    if user_input.lower() in ["exit", "quit", "bye"]:
        print("Bot: Goodbye!")
        break

    answer = ask_chatbot(user_input)
    print("Bot:", answer)


