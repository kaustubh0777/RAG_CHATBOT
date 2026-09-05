import streamlit as st
from connectmemorywithhugface import build_qa_chain


@st.cache_resource
def get_qa_chain():
    return build_qa_chain()

def main():
    st.title("Ask Chatbot")
    qa_chain=get_qa_chain()
    prompt=st.chat_input("Ask your prompt")

    if prompt:
        st.chat_message('user').markdown(prompt)
        response=qa_chain.invoke(prompt)
        st.chat_message('assistant').markdown(response)


if __name__=="__main__":
    main()