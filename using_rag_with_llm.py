from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter ##
from typing import List
from langchain_core.prompts import ChatPromptTemplate ##
from langchain_core.output_parsers import StrOutputParser ##
from langchain_core.runnables import RunnableParallel, RunnablePassthrough ##
from langchain_core.documents import Document ##
from langchain_ollama import ChatOllama
import os
os.environ["USER_AGENT"] = "MyLangChainApp/1.0 (colimop977@mustaer.com)"

llm = ChatOllama(
    base_url="http://localhost:11434",
    model="deepseek-r1:8b",
    temperature=0.5
)

# load data from web
loader = WebBaseLoader("https://roarofganjam.blogspot.com/2019/05/food-of-ganjam-ganjam-is-well-known-for.html")
data = loader.load()

# split data into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
all_splits = text_splitter.split_documents(data)

# Add chunks to vector db
embedding = OllamaEmbeddings(model="nomic-embed-text:latest")
vectordb = Chroma.from_documents(documents=all_splits, embedding=embedding)

# Create a retriever
retriever = vectordb.as_retriever()

def format_docs(docs:List[Document]) -> str:
    return "\n\n".join([d.page_content for d in docs])

template = """Answer the question based only on the following context:
{context}
Give a summary not the full detail
Question: {question}
"""

prompt = ChatPromptTemplate.from_template(template)
# We can also use the below method inplace of context_retriever
def retrieve_and_format(question):
    docs = retriever.invoke(question)
    formatted_docs = format_docs(docs)

# Create a runnable that retrieves and formats context
context_retriever = retriever | format_docs   # This is a Runnable that takes a str question and returns formatted str

chain = (
        RunnableParallel({"context": context_retriever, "question": RunnablePassthrough()})
        | prompt
        | llm
        | StrOutputParser()
)

# response = chain.invoke("Where is the best place to eat Puri Upma in Berhampur?")
# print(response)



