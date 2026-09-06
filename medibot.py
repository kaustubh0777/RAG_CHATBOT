"""Streamlit interface for the medical-document RAG chatbot."""

import streamlit as st

from connectmemorywithhugface import build_qa_chain


SUGGESTIONS = {
    ":material/medication: What is hypertension?": "What is hypertension?",
    ":material/monitor_heart: What are common diabetes symptoms?": (
        "What are common diabetes symptoms?"
    ),
    ":material/vaccines: How do vaccines work?": "How do vaccines work?",
}


@st.cache_resource
def get_qa_chain():
    """Create the RAG chain once per Streamlit server."""
    return build_qa_chain()


def initialize_session_state() -> None:
    if "messages" not in st.session_state:
        st.session_state.messages = []


def show_sidebar() -> None:
    with st.sidebar:
        st.title("MediBot", icon=":material/medical_services:")
        st.caption("Medical reference assistant")

        if st.button(
            "Start a new chat",
            icon=":material/add_comment:",
            width="stretch",
        ):
            st.session_state.messages = []
            st.rerun()

        st.subheader("How it works", icon=":material/menu_book:")
        st.caption(
            "Answers are grounded in the medical reference material included "
            "with this application."
        )

        st.subheader("Important", icon=":material/warning:")
        st.caption(
            "MediBot provides general educational information, not a diagnosis "
            "or treatment plan. For urgent concerns, contact a qualified clinician "
            "or local emergency service."
        )


def render_message(role: str, content: str) -> None:
    avatar = ":material/person:" if role == "user" else ":material/medical_services:"
    with st.chat_message(role, avatar=avatar):
        st.markdown(content)


def main() -> None:
    st.set_page_config(
        page_title="MediBot | Medical reference assistant",
        page_icon=":material/medical_services:",
        layout="centered",
        initial_sidebar_state="expanded",
    )
    initialize_session_state()
    show_sidebar()

    st.title("MediBot", icon=":material/medical_services:")
    st.caption("Ask questions about the medical reference guide")

    if not st.session_state.messages:
        with st.container(border=True):
            st.subheader("Your medical reference assistant")
            st.write(
                "Ask a health-information question and I’ll search the provided "
                "medical reference to answer it."
            )
            st.caption("For educational use only — not medical advice.")

    for message in st.session_state.messages:
        render_message(message["role"], message["content"])

    suggested_prompt = None
    if not st.session_state.messages:
        suggested_prompt = st.pills(
            "Try a question",
            list(SUGGESTIONS),
            selection_mode="single",
        )

    typed_prompt = st.chat_input(
        "Ask a question about a medical topic",
        submit_mode="disable",
    )
    prompt = typed_prompt or SUGGESTIONS.get(suggested_prompt)

    if not prompt:
        return

    st.session_state.messages.append({"role": "user", "content": prompt})
    render_message("user", prompt)

    with st.chat_message("assistant", avatar=":material/medical_services:"):
        with st.status("Searching the medical reference", expanded=False) as status:
            try:
                response = get_qa_chain().invoke(prompt)
                status.update(
                    label="Reference search complete",
                    state="complete",
                )
            except Exception:
                status.update(label="Unable to answer right now", state="error")
                st.error(
                    "I couldn't reach the medical reference service. Please try again."
                )
                st.caption("If the issue continues, check the Hugging Face token and service status.")
                return
        st.markdown(response)

    st.session_state.messages.append({"role": "assistant", "content": response})


if __name__ == "__main__":
    main()
