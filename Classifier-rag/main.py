
import os 
from dotenv import load_dotenv
load_dotenv()

from langchain_openai.chat_models import ChatOpenAI
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.schema import Document
from langchain_chroma.vectorstores import Chroma
from langchain import hub

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from typing import TypedDict, Literal
from langgraph.graph import StateGraph, START, END

class AgentState(TypedDict):
    messages: list[BaseMessage]
    documents: list[Document]
    on_topic: str 

class GradeQuestion(BaseModel):
    """Boolean value to check whether a question is releated to the restaurant Bella Vista"""
    score: Literal['Yes','No'] = Field(
        description="Question is about <Condition> ? If yes -> 'Yes' if not -> 'No'"
    )

llm = ChatOpenAI(model="gpt-4o-mini",temperature=0.7)
embedding = OpenAIEmbeddings()

PATH = r"<FILE>"
STORAGE = "./storedb"

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

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = prompt | llm 

def question_classifier(state: AgentState):
    question = state["messages"][-1].content
    
    system = """
    You are a classifier that determines whether a user's question is about one of the following topics:
    
    1. <Question 1>
    2. <Question 2>
    
    If the question IS about any of these topics, respond with 'Yes'. Otherwise, respond with 'No'. Remember, ONLY YES or NO, nothing else in the response!
    """
    grade_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            ("human", "User question: {question}"),
        ]
    )
    structured_llm = llm.with_structured_output(GradeQuestion)
    grader_llm = grade_prompt | structured_llm
    result = grader_llm.invoke({"question": question})
    print("RESULT", result)
    state["on_topic"] = result.score
    return state

def on_topic_router(state:AgentState):
    if state['on_topic'].lower()=="yes":
        return "on_topic"
    return "off_topic"

def retrieve(state: AgentState):
    question = state["messages"][-1].content
    documents = retriever.invoke(question)
    state["documents"] = documents
    return state

def generate_answer(state: AgentState):
    question = state["messages"][-1].content
    documents = format_docs(state["documents"])
    generation = rag_chain.invoke({"context": documents, "question": question})
    state["messages"].append(generation)
    return state

def off_topic_response(state: AgentState):
    state["messages"].append(AIMessage(content="I cant respond to that!"))
    return state

builder = StateGraph(AgentState)
builder.add_node("topic_decision",question_classifier)
builder.add_node("off_topic_response",off_topic_response)
builder.add_node("retrieve",retrieve)
builder.add_node('generate_answer',generate_answer)

builder.add_conditional_edges(
    "topic_decision",
    on_topic_router,{
        'on_topic': 'retrieve',
        'off_topic': 'off_topic_response'
    },
)

builder.add_edge('retrieve','generate_answer')
builder.add_edge('generate_answer',END)
builder.add_edge('off_topic_response',END)
builder.add_edge(START,'topic_decision')

graph = builder.compile()

val = graph.invoke(input={"messages": [HumanMessage(content="<question> ?")]})
print(val)
