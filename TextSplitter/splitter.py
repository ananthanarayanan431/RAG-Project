
import re 
from typing import List, Optional, Any
from langchain_text_splitters import TextSplitter
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from rag_prompt import SPLITTER_PROMPT


from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
import os

loader = PyPDFLoader("/Users/ananthanarayanan/Desktop/project/data/Robbins - chapter 3 - Inflamation.pdf")
docs = loader.load()

class CustomTextSplitter(TextSplitter):

    def __init__(self,model_name: str="gpt-4o-mini",**kwargs: Any)-> None:
        super().__init__(**kwargs)
        self.model = ChatOpenAI(model=model_name)
        self.prompts = ChatPromptTemplate.from_template(SPLITTER_PROMPT)
        self.output_parser = StrOutputParser()
        self.chain = (
            {'text': RunnablePassthrough()}
            | self.prompts
            | self.model
            | self.output_parser
        )

    def split_text(self, text:str)-> List[str]:
        response = self.chain.invoke({"text": text})
        chunks = re.findall(r'<<<(.*?)>>>', response, re.DOTALL)
        return [chunk.strip() for chunk in chunks]

    
custom_splitter = CustomTextSplitter(model_name="gpt-4o-mini")
chunks = custom_splitter.split_text(docs)

print(f"Number of chunks: {len(chunks)}")
