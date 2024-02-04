#!/bin/env python

from langchain_community.llms import Ollama 
from datetime import datetime

from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import GPT4AllEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA

loader = WebBaseLoader('https://en.wikipedia.org/wiki/Muhammad')
data = loader.load()

start_time = datetime.now()
llm = Ollama(base_url='http://localhost:11434',model='dolphin-phi')

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500,chunk_overlap=0)
all_splits = text_splitter.split_documents(data)

vectorstore = Chroma.from_documents(documents=all_splits, embedding=GPT4AllEmbeddings())

qachain = RetrievalQA.from_chain_type(llm,retriever=vectorstore.as_retriever())
question = "What did Muhammad teach?"

print(qachain.invoke({"query":question}))

# print(llm.invoke('why is the sky blue?'))
end_time = datetime.now()
print("Time:",end_time-start_time)
