#!/bin/env python

import os
import textract

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import GPT4AllEmbeddings
from langchain_community.vectorstores import Chroma
# from langchain_community.document_loaders import PyPDFDirectoryLoader

# loader = PyPDFDirectoryLoader("docs/")
# docs = loader.load()
# vectorstore = Chroma.from_documents(documents=docs, embedding=GPT4AllEmbeddings(),persist_directory='./chroma_db/')

for root, dirs, files in os.walk("docs", topdown=False):
    for name in files:
        print('File:',os.path.join(root, name))
        filetxt = textract.process(os.path.join(root, name)).decode('utf-8')
        print("Content:",filetxt)
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500,chunk_overlap=50)
        all_splits = text_splitter.create_documents([filetxt])
        if all_splits:
            vectorstore = Chroma.from_documents(documents=all_splits, embedding=GPT4AllEmbeddings(),persist_directory='./chroma_db/')
    for name in dirs:
        print('Dir:',os.path.join(root, name))
