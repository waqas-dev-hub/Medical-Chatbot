import streamlit as st
import requests

FASTAPI_URL = "http://localhost:8000"


def main():
    st.title("🩺 MediBot — Ask a Medical Question")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        st.chat_message(message["role"]).markdown(message["content"])

    prompt = st.chat_input("Enter your query here:")

    if prompt:
        st.chat_message("user").markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        try:
            response = requests.post(
                f"{FASTAPI_URL}/chat",
                json={"query": prompt},
                timeout=60,
            )
            response.raise_for_status()
            data = response.json()

            answer = data["answer"]
            sources = data.get("source_documents", [])

            st.chat_message("assistant").markdown(answer)
            st.session_state.messages.append(
                {"role": "assistant", "content": answer}
            )

           
            if sources:
                with st.expander("📄 Source Documents"):
                    for i, doc in enumerate(sources, 1):
                        st.markdown(f"**Source {i}** — {doc.get('metadata', {})}")
                        st.caption(doc.get("page_content", "")[:300] + "...")
                        st.divider()

        except requests.exceptions.ConnectionError:
            st.error(
                "⚠️ Cannot connect to the MediBot API. "
                "Make sure the FastAPI backend is running: "
                "`uvicorn app:app --reload`"
            )
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")


if __name__ == "__main__":
    main()