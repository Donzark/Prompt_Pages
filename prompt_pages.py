import streamlit as st
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
import os
import html
from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OllamaEmbeddings
from langchain_cohere import CohereEmbeddings
from langchain_core.documents import Document

# Load environment variables
# load_dotenv()
# groq_api_key = os.getenv("GROQ_API_KEY")
# os.environ["GROQ_API_KEY"] = groq_api_key

# cohere_api_key = os.getenv("COHERE_API_KEY")
# os.environ["COHERE_API_KEY"] = cohere_api_key

groq_api_key = st.secrets["groq_api_key"]
cohere_api_key = st.secrets["cohere_api_key"]

# Streamlit page config
st.set_page_config(page_title="📄Prompt Pages", layout="centered")

# --- Hero Section ---
st.markdown("""
    <div style="text-align:center; padding: 2rem 0;">
        <h1 style="font-size:3rem;">📄 Prompt Pages</h1>
        <p style="font-size:1.2rem; color:gray;">Your AI-powered documentation companion</p>
    </div>
""", unsafe_allow_html=True)

# --- About Section ---
with st.expander("ℹ️ About this app"):
    st.markdown("""
        **Prompt Pages** helps you:
        - 🔍 Scrape and process technical documentation from the web  
        - 🧠 Generate summaries of scraped content  
        - 💬 Lets you ask questions and get accurate answers using AI  

        Built with [Streamlit](https://streamlit.io), [LangChain](https://www.langchain.com/), and [Cohere](https://cohere.com).
    """)

# Initialize session state
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "summary" not in st.session_state:
    st.session_state.summary = None
if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None
if "query_history" not in st.session_state:
    st.session_state.query_history = []
if "submitted_query" not in st.session_state:
    st.session_state.submitted_query = ""

# --- Input Section ---
with st.container():
    st.markdown("#### 🧾 Enter a documentation URL to begin")
    doc_url = st.text_input(
        label="Paste documentation URL",
        placeholder="https://example.com/docs/...",
        help="Make sure it's a public and structured web page.",
        label_visibility="collapsed"
    )
    scrape = st.button("🔍 Scrape and Analyze")

# --- Scrape and Embed ---
if scrape and doc_url:
    with st.spinner("Scraping and embedding content..."):
        try:
            response = requests.get(doc_url, timeout=10)
            soup = BeautifulSoup(response.text, 'html.parser')
            for tag in soup(['script', 'style', 'header', 'footer', 'nav']):
                tag.decompose()
            text = soup.get_text(separator='\n')
            cleaned_text = '\n'.join([line.strip() for line in text.splitlines() if line.strip()])
            docs = [Document(page_content=cleaned_text)]

            # Create vector store
            embeddings = CohereEmbeddings(cohere_api_key=cohere_api_key, model="embed-english-v3.0")
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            chunks = text_splitter.split_documents(docs)
            vector_store = FAISS.from_documents(chunks, embeddings)

            st.session_state.vector_store = vector_store

            # Summarize
            llm = ChatGroq(groq_api_key=groq_api_key, model_name="llama-3.3-70b-versatile")
            summary_prompt = ChatPromptTemplate.from_template("""
                Summarize the content of the following documentation in 3-5 sentences:
                <context>
                {context}
                </context>
            """)
            summary_chain = create_stuff_documents_chain(llm=llm, prompt=summary_prompt)
            summary = summary_chain.invoke({'context': chunks})
            st.session_state.summary = summary
            st.success("✅ Web page scraped, summarized, and embedded!")

        except Exception as e:
            st.error(f"❌ Full error: {type(e).__name__}: {e}")

# --- Show Summary ---
if st.session_state.vector_store and st.session_state.summary:
    st.markdown("### 📘 What this page is about")
    st.write(st.session_state.summary)

    st.markdown("### 💬 Ask a question from the documentation")

    # --- Handle user input ---
    def handle_submit():
        st.session_state.submitted_query = st.session_state.user_query
        st.session_state.user_query = ""

    st.text_input(
        "Your question",
        placeholder="What does this API do?",
        key="user_query",
        on_change=handle_submit
    )

    # Process submitted question
    if st.session_state.submitted_query:
        user_prompt = st.session_state.submitted_query

        if st.session_state.qa_chain is None:
            qa_prompt = ChatPromptTemplate.from_template("""
                Give correct answers to the questions taking from provided context and general knowledge:
                <context>
                {context}
                </context>
                Question: {input}
            """)
            qa_llm = ChatGroq(groq_api_key=groq_api_key, model_name="llama-3.3-70b-versatile")
            qa_chain = create_stuff_documents_chain(qa_llm, prompt=qa_prompt)
            retriever = st.session_state.vector_store.as_retriever()
            st.session_state.qa_chain = create_retrieval_chain(retriever, qa_chain)

        response = st.session_state.qa_chain.invoke({'input': user_prompt})
        answer = response['answer']
        context_chunks = response.get('context', [])

        st.session_state.query_history.append({
            "question": user_prompt,
            "answer": answer,
            "context": context_chunks
        })

        # Clear submitted_query
        st.session_state.submitted_query = ""

# --- Show Q&A History ---
if st.session_state.query_history:
    st.markdown("### 🧠 Answers")
    for q in reversed(st.session_state.query_history):
        escaped_question = html.escape(q['question'])
        escaped_answer = html.escape(q['answer'])

        with st.container():
            st.markdown(
                f"""
                <div style='
                    background-color:#1f77b4;
                    padding: 12px;
                    border-radius: 10px;
                    margin-bottom: 8px;
                    color: white;
                    font-size: 16px;
                    font-weight: bold;
                '>
                    Question: {escaped_question}
                </div>
                """,
                unsafe_allow_html=True
            )
            st.markdown("**Answer:**")
            st.markdown(q['answer']) 
            st.markdown("<br>", unsafe_allow_html=True)

# --- Footer ---
st.markdown("""
    <hr>
    <div style='text-align: center; font-size: 0.9rem; color: gray;'>
        Powered by LangChain, Cohere, and Streamlit
    </div>
""", unsafe_allow_html=True)



