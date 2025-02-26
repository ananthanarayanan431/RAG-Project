
import os 
from dotenv import load_dotenv
from langchain_openai.chat_models import ChatOpenAI
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain.chains.hyde.base import HypotheticalDocumentEmbedder
from langchain.schema.runnable import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_chroma.vectorstores import Chroma
from langchain import hub
from langchain.schema import Document

load_dotenv()


STORAGE = "./LINKEDINCONTENT"

base_embeddings = OpenAIEmbeddings()
llm = ChatOpenAI(model="gpt-4o-mini")

docs = [
    Document(
        page_content="The Indian Premier League (IPL) 2024 features ten teams, including franchises like Mumbai Indians, Chennai Super Kings, and Royal Challengers Bangalore. The tournament is scheduled to take place across various cities in India from March to May 2024.",
        metadata={"source": "tournament_info.txt"},
    ),
    Document(
        page_content="The defending champions of IPL 2024 are Chennai Super Kings, who won the previous season by defeating Gujarat Titans in a thrilling final. With key players retained and new signings, they aim to secure another title this season.",
        metadata={"source": "teams.txt"},
    ),
    Document(
        page_content="Ticket prices for IPL 2024 vary depending on the venue and match importance. General admission tickets start at ₹800, while premium seats and hospitality packages range from ₹3,000 to ₹20,000.",
        metadata={"source": "ticket_info.txt"},
    ),
    Document(
        page_content="IPL 2024 matches will be broadcast live on major sports channels and streaming platforms. Fans can watch the games on Star Sports and Disney+ Hotstar, with regional language commentary available for an enhanced experience.",
        metadata={"source": "broadcast_info.txt"},
    )
]


embedding = HypotheticalDocumentEmbedder.from_llm(
    llm=llm,
    base_embeddings=base_embeddings,
    prompt_key="web_search"
)


if not os.path.exists(STORAGE):
    vectorstore = Chroma.from_documents(documents=docs,embedding=embedding,persist_directory=STORAGE)
else:
    vectorstore = Chroma(persist_directory=STORAGE,embedding_function=embedding)

retriever = vectorstore.as_retriever()
prompt = hub.pull("rlm/rag-prompt") 

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

chain = (
     {'context': retriever | format_docs, "question": RunnablePassthrough()}
     | prompt
     | llm
     | StrOutputParser()
)

query = "Tell about IPL"
print(chain.invoke(query))
