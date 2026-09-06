import streamlit as st
from connectmemorywithhugface import build_qa_chain

@st.cache_resource
def get_qa_chain():
    return build_qa_chain()

def main():
    st.title("Ask Chatbot")
    if 'messages' not in st.session_state:
        st.session_state.messages=[]
        
    for message in st.session_state.messages:
        st.chat_message(message['role']).markdown(message['content'])
    qa_chain=get_qa_chain()
    prompt=st.chat_input("Ask your prompt")

    if prompt:
        st.chat_message('user').markdown(prompt)
        st.session_state.messages.append({'role':'user','content':prompt})
        response=qa_chain.invoke(prompt)
        st.chat_message('assistant').markdown(response)
        st.session_state.messages.append({'role':'assistant', 'content':response})


if __name__=="__main__":
    main()