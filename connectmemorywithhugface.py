"""Ask questions about the documents stored in the local FAISS index."""

import os
from pathlib import Path

from dotenv import load_dotenv
from huggingface_hub import InferenceClient
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_huggingface import HuggingFaceEmbeddings


PROJECT_DIR = Path(__file__).resolve().parent
DB_FAISS_PATH = PROJECT_DIR / "vectorstore" / "db_faiss"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_MODEL = "meta-llama/Llama-3.1-8B-Instruct"

CUSTOM_PROMPT_TEMPLATE = """
Use the pieces of information provided in the context to answer the user's question.

If you don't know the answer, just say that you don't know.
Don't try to make up an answer.
Don't provide anything outside of the given context.

Context:
{context}

Question:
{question}

Start the answer directly. No small talk please.
"""


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def create_chat_model(model_id, hf_token):
    """Return a runnable for Hugging Face's current chat-completion API."""
    client = InferenceClient(
        model=model_id,
        api_key=hf_token,
        provider=os.getenv("HF_PROVIDER", "auto"),
        timeout=120,
    )

    def invoke_chat(prompt_value):
        completion = client.chat_completion(
            messages=[{"role": "user", "content": prompt_value.to_string()}],
            max_tokens=512,
            temperature=0.5,
        )
        return completion.choices[0].message.content or ""

    return RunnableLambda(invoke_chat)


def build_qa_chain():
    """Load the local index and return a runnable RAG chain."""
    load_dotenv(PROJECT_DIR / ".env")
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        raise RuntimeError("HF_TOKEN is missing. Add it to .env or set it in the environment.")
    if not (DB_FAISS_PATH / "index.faiss").exists():
        raise FileNotFoundError(
            f"FAISS index not found at {DB_FAISS_PATH}. Run memory.py first."
        )

    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    database = FAISS.load_local(
        str(DB_FAISS_PATH), embeddings, allow_dangerous_deserialization=True
    )
    retriever = database.as_retriever(search_kwargs={"k": 3})
    prompt = PromptTemplate(
        template=CUSTOM_PROMPT_TEMPLATE, input_variables=["context", "question"]
    )
    llm = create_chat_model(os.getenv("HF_REPO_ID", DEFAULT_MODEL), hf_token)

    return (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )


def main():
    user_query = input("Write query here: ").strip()
    if not user_query:
        raise ValueError("Please enter a non-empty question.")

    response = build_qa_chain().invoke(user_query)
    print("\nRESULT:\n")
    print(response)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"Error: {error}")
        raise SystemExit(1) from error
