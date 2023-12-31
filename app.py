__import__('pysqlite3')
import sys

sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
import streamlit as st
import pickle
import os
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import PyPDFLoader
from langchain.vectorstores import Chroma
from chromadb.config import Settings
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains import ConversationalRetrievalChain

# 本番環境の場合はStreamlit Community Cloudの「Secrets」からOpenAI API keyを取得
if st.secrets.env.env == 'prod':
    os.environ["OPENAI_API_KEY"] = st.secrets.OpenAIAPI.openai_api_key
else:
    os.environ["OPENAI_API_KEY"] = openai_api_key_dev

# embeddingとllmの初期設定
embeddings = OpenAIEmbeddings(model = "text-embedding-ada-002")
llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo-16k")

# pdfの処理を行う
loader = PyPDFLoader("https://www.mhlw.go.jp/content/001018385.pdf")
pages = loader.load_and_split()

# embeddingしてchroma-dbに登録する
db = Chroma.from_documents(pages, OpenAIEmbeddings(), persist_directory = "db")
db.persist()
# QAの設定
pdf_qa = ConversationalRetrievalChain.from_llm(llm, db.as_retriever(), return_source_documents=True)

# st.session_stateを使いメッセージのやりとりを保存
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# チャットボットとやりとりする関数
def communicate():
    messages = st.session_state["messages"]
    chat_history = []
    
    result = pdf_qa({"question": st.session_state["user_input"], "chat_history": chat_history})
    messages.append({"answer": result["answer"], "question": result["question"]})
    # 入力欄を消去
    st.session_state["user_input"] = ""  


# ユーザーインターフェイスの構築
st.title("My AI Assistant")
st.write("ChatGPT APIを使ったチャットボットです。総務・人事担当に聞きたい事を入力してください。")

user_input = st.text_input("[厚生労働省のモデル就業規則（PDF）](https://www.mhlw.go.jp/file/06-Seisakujouhou-11200000-Roudoukijunkyoku/0000118951.pdf)を元に回答します。", key="user_input", on_change=communicate)

if st.session_state["messages"]:
    messages = st.session_state["messages"]

    for message in reversed(messages):  # 直近のメッセージを上に
        st.write("🤖" + ": " + message["answer"])
        st.write("🙂" + ": " + message["question"])
