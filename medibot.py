import os
import streamlit as st

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

DB_FAISS_PATH="vectorstore/db_faiss"
@st.cache_resource
def get_vectorstore():
    embedding_model=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db=FAISS.load_local(DB_FAISS_PATH,embedding_model,allow_dangerous_deserialization=True)
    return db

def set_custom_prompt(custom_prompt_template):
    prompt=ChatPromptTemplate.from_messages([
        ("system", custom_prompt_template),
        ("human", "{input}")
    ])
    return prompt


def main():
    st.title("Ask chatbot")

    if 'messages' not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        st.chat_message(message['role']).markdown(message['content'])

    prompt=st.chat_input("Enter your query here:")

    if prompt:
        st.chat_message('user').markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})


       
        try:
            vectorstore=get_vectorstore()
            if vectorstore is None:
                st.error("Failed to load the vector store")

            GROQ_MODEL_NAME="llama-3.1-8b-instant"
            GROQ_API_KEY=os.environ.get("GROQ_API_KEY")    

            llm=ChatGroq(
                    model=GROQ_MODEL_NAME,
                    api_key=GROQ_API_KEY,
                    temperature=0.5,
                    max_tokens=512
            )
            
            retrieval_qa_chat_prompt = ChatPromptTemplate.from_messages([
                ("system", "Answer the user's questions based on the below context:\n\n{context}"),
                ("human", "{input}"),
            ])
            combine_docs_chain=create_stuff_documents_chain(llm,retrieval_qa_chat_prompt)
            rag_chain=create_retrieval_chain(vectorstore.as_retriever(search_kwargs={"k":3}),combine_docs_chain)

            response=rag_chain.invoke({"input":prompt})
            
            result=response['answer']
            st.chat_message('assistant').markdown(result)
            st.session_state.messages.append({"role": "assistant", "content": result})

        except Exception as e:
            st.error(f"An error occurred: {str(e)}")    

if __name__=="__main__":
    main()        