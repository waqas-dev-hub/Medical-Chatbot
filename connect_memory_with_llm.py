import os

from langchain_groq import ChatGroq
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate

from langchain_classic import hub
from langchain_classic.chains.combine_documents import (
   create_stuff_documents_chain,
)
from langchain_classic.chains import create_retrieval_chain


from dotenv import load_dotenv
load_dotenv()


GROQ_API_KEY=os.environ.get("GROQ_API_KEY")
GROQ_MODEL_NAME="llama-3.1-8b-instant"

llm=ChatGroq(
    model_name=GROQ_MODEL_NAME,
    api_key=GROQ_API_KEY,
    temperature=0.5,
    max_tokens=512
)


DB_FAISS_PATH="vectorstore/db_faiss"
embedding_model=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db=FAISS.load_local(DB_FAISS_PATH,embedding_model,allow_dangerous_deserialization=True)




retrieval_qa_chat_prompt = ChatPromptTemplate.from_messages([
    ("system", "Answer the user's questions based on the below context:\n\n{context}"),
    ("human", "{input}"),
])
combine_docs_chain=create_stuff_documents_chain(llm,retrieval_qa_chat_prompt)
rag_chain=create_retrieval_chain(db.as_retriever(search_kwargs={"k":3}),combine_docs_chain)



user_query=input("write query here: ")
response=rag_chain.invoke({"input":user_query})
print("Result: ",response['answer'])
print("\nSource Documents: ")
for doc in response['context']:
    print(f"-{doc.metadata}-> {doc.page_content[:200]}...")