
from dotenv import load_dotenv
load_dotenv()

from langchain_ollama.chat_models import ChatOllama
from langchain_openai.embeddings.base import OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_chroma.vectorstores import Chroma
from langchain.schema.runnable import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain.schema import Document
from langchain import hub
import os 

PATH = r"File"
STORAGE = "./storedb"

llm = ChatOllama(model="deepseek-r1:1.5b",temperature=0.7)

documents = []
pdf_content = PyPDFLoader(PATH)
loader = pdf_content.load()

for i in range(len(loader)):
    documents.append(Document(metadata=loader[i].metadata, page_content=loader[i].page_content))

if not os.path.exists(STORAGE):
    vectorstore = Chroma.from_documents(documents=documents,embedding=OpenAIEmbeddings(),persist_directory=STORAGE)
else:
    vectorstore = Chroma(persist_directory=STORAGE,embedding_function=OpenAIEmbeddings())
    
retriever = vectorstore.as_retriever()
prompt = hub.pull("rlm/rag-prompt")
        
chain = (
     {'context': retriever, "question": RunnablePassthrough()}
     | prompt
     | llm
     | StrOutputParser()
)

query = "How RAG Works?"
print(chain.invoke(query))
