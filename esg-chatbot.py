import sys
import array
import time
import os
from dotenv import load_dotenv

import oracledb
from langchain_community.vectorstores import oraclevs
from langchain_community.vectorstores.oraclevs import OracleVS

from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_core.documents import BaseDocumentTransformer, Document

from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

from langchain_upstage import ChatUpstage
from langchain_upstage import UpstageEmbeddings
from langchain_core.messages import HumanMessage, SystemMessage

import streamlit as st
from openai import OpenAI

import warnings
warnings.filterwarnings("ignore")

load_dotenv()

username=os.environ["DB_USER"]
password=os.environ["DB_PASSWORD"]
dsn=os.environ["DSN"]

con = oracledb.connect(user=username, password=password, dsn=dsn)

try: 
    conn23c = oracledb.connect(user=username, password=password, dsn=dsn)
    print("Connection successful!", conn23c.version)
except Exception as e:
    print("Connection failed!")

upstage_embeddings = UpstageEmbeddings(model="solar-embedding-1-large")
vector_store = OracleVS(client=conn23c, 
                        embedding_function=upstage_embeddings, 
                        table_name="text_embeddings2", 
                        distance_strategy=DistanceStrategy.DOT_PRODUCT)

retriever = vector_store.as_retriever()

llm = ChatUpstage()

template = """Answer the question based only on the following context:
              {context} 
              Question: {question} 
              """
prompt = PromptTemplate.from_template(template)

# with st.sidebar:
#     openai_api_key = st.text_input("OpenAI API Key", key="chatbot_api_key", type="password")
#     "[Get an OpenAI API key](https://platform.openai.com/account/api-keys)"
#     "[View the source code](https://github.com/streamlit/llm-examples/blob/main/Chatbot.py)"
#     "[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new/streamlit/llm-examples?quickstart=1)"

st.title("💬 ESG Chatbot")

if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if question := st.chat_input():
    # client = OpenAI(
    #     api_key=os.environ["UPSTAGE_API_KEY"], base_url="https://api.upstage.ai/v1/solar"
    # )
    st.session_state.messages.append({"role": "user", "content": question})
    st.chat_message("user").write(question)

    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
            )
    print(prompt)
    response = chain.invoke(question)

    st.chat_message("assistant").write(response)
    st.session_state.messages.append({"role": "assistant", "content": response})

    