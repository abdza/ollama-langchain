#!/bin/env python

import cherrypy
import os
from langchain_community.llms import Ollama 
from datetime import datetime

from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import GPT4AllEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA

from jinja2 import Environment, PackageLoader, select_autoescape

env = Environment(
    loader=PackageLoader("ollama_docs"),
    autoescape=select_autoescape()
)


# print(qachain.invoke({"query":question}))

# print(llm.invoke('why is the sky blue?'))
# end_time = datetime.now()
# print("Time:",end_time-start_time)

# llm = Ollama(base_url='http://localhost:11434',model='dolphin-phi')
llm = Ollama(base_url='http://localhost:11434',model='llama2')
vectorstore = Chroma(embedding_function=GPT4AllEmbeddings(),persist_directory="./chroma_db/")
qachain = RetrievalQA.from_chain_type(llm,retriever=vectorstore.as_retriever())

class DocTalk(object):
    @cherrypy.expose 
    def index(self):
        template = env.get_template("query_index.html")
        return template.render()

    @cherrypy.expose 
    def query(self,querystr):
        answer = qachain.invoke({"query":querystr})
        print("Answer:",answer)
        return "<dl class='row text-start'><dt class='col-sm-1 text-end'>Query</dt><dd class='col-sm-11 text-start'>" + querystr + "</dd></dl><dl class='row text-start'><dt class='col-sm-1 text-end'>Answer</dt><dd class='col-sm-11 text-start'>" + answer['result'] + "</dd></dl>"

if __name__ == '__main__':
    conf = {
        '/': {
            'tools.sessions.on': True,
            'tools.staticdir.root': os.path.abspath(os.getcwd())
        },
        '/static': {
            'tools.staticdir.on': True,
            'tools.staticdir.dir': './public'
        }
    }
    cherrypy.quickstart(DocTalk(),'/',conf)
