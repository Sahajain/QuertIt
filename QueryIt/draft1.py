import os
import streamlit as st
from dotenv import load_dotenv
from langchain_openai import AzureOpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai.chat_models import AzureChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from document_processing11 import get_text_chunks_with_metadata
import pandas as pd
from io import StringIO
import base64
import sqlite3
import sqlparse
from langchain_core.prompts import PromptTemplate

def get_vectorstore(chunks_with_metadata):
    embeddings = AzureOpenAIEmbeddings(
        model="text-embedding-ada-002",
        openai_api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        azure_endpoint=os.getenv("AZURE_API_ENDPOINT")
    )
    texts = [chunk["page_content"] for chunk in chunks_with_metadata]
    metadatas = [chunk["metadata"] for chunk in chunks_with_metadata]
    if not texts:
        raise ValueError("No texts found in chunks_with_metadata to create vectorstore.")
    return FAISS.from_texts(texts=texts, embedding=embeddings, metadatas=metadatas)

def get_conversation_chain(vectorstore):
    llm = AzureChatOpenAI(
        deployment_name="gpt-35-turbo",
        openai_api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        azure_endpoint=os.getenv("AZURE_API_ENDPOINT"),
        api_version=os.getenv("OPENAI_API_VERSION", "2023-05-15")
    )
    memory = ConversationBufferMemory(
        memory_key='chat_history',
        return_messages=True,
        output_key='answer'
    )
    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory,
        return_source_documents=True
    )

def generate_sql_query(question, metadata, llm):
    """
    Generate a specific SQL query based on user question and metadata using LLM
    """
    tables = metadata.get('tables', [])
    schema = metadata.get('schema', {})
    
    if not tables:
        return None, None
    
    # Create prompt for LLM to generate SQL query
    prompt_template = PromptTemplate(
        input_variables=["question", "schema"],
        template="""
        Given the following database schema and a user question, generate an appropriate SQL query to answer the question.
        Schema: {schema}
        Question: {question}
        Return only the SQL query without any explanation.
        """
    )
    
    schema_str = "\n".join([f"Table {table}: {', '.join(columns)}" for table, columns in schema.items()])
    query = llm.invoke(prompt_template.format(question=question, schema=schema_str)).content.strip()
    
    # Validate and clean query
    if not query.lower().startswith("select"):
        return None, None
    
    # Execute query for .db files
    if metadata.get('file_type') == 'db':
        try:
            with open("temp.db", "wb") as f:
                f.write(st.session_state.uploaded_db_file)
            conn = sqlite3.connect("temp.db")
            df = pd.read_sql_query(query, conn)
            conn.close()
            os.remove("temp.db")
            return query, df
        except Exception as e:
            st.error(f"Error executing SQL query: {str(e)}")
            return query, None
    else:
        # For .sql files, use sample data
        table_name = tables[0]
        df = pd.DataFrame(metadata['sample_data'][table_name], columns=metadata['schema'][table_name])
        return query, df  # Simplified; in production, filter sample data based on query

def download_table_button(df, filename):
    """
    Create a download button for a DataFrame as CSV
    """
    csv_buffer = StringIO()
    df.to_csv(csv_buffer, index=False)
    csv_data = csv_buffer.getvalue().encode('utf-8')
    b64 = base64.b64encode(csv_data).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}.csv" class="download-button">💾 Download Table as CSV</a>'
    return href

def handle_userinput(user_question):
    if not st.session_state.conversation:
        st.error("Please process documents first.")
        return
    
    response = st.session_state.conversation.invoke({'question': user_question})

    if "display_messages" not in st.session_state:
        st.session_state.display_messages = []

    st.session_state.display_messages.append({"content": user_question, "role": "user"})
    answer = response['chat_history'][-1].content
    source_documents = response.get('source_documents', [])
    
    sql_metadata = None
    for doc in source_documents:
        if doc.metadata.get('file_type') in ['sql', 'db']:
            sql_metadata = doc.metadata
            break
    
    if sql_metadata:
        llm = AzureChatOpenAI(
            deployment_name="gpt-35-turbo",
            openai_api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            azure_endpoint=os.getenv("AZURE_API_ENDPOINT"),
            api_version=os.getenv("OPENAI_API_VERSION", "2023-05-15")
        )
        sql_query, df = generate_sql_query(user_question, sql_metadata, llm)
        if sql_query:
            answer += "\n\n**Table Results:**"
            message = {"content": answer, "role": "assistant"}
            if df is not None and not df.empty:
                message["table"] = df
                message["table_download"] = download_table_button(df, f"{sql_metadata['tables'][0]}_results")
            st.session_state.display_messages.append(message)
            
            if sql_query:
                st.session_state.display_messages.append({
                    "content": f"\n{sql_query}\n",
                    "role": "assistant"
                })
    else:
        st.session_state.display_messages.append({"content": answer, "role": "assistant"})

    sources = response.get('source_documents', [])
    if sources:
        if "sources" not in st.session_state:
            st.session_state.sources = {}
        source_list = []
        for doc in sources:
            source = doc.metadata.get("source", "Unknown source")
            page = doc.metadata.get("page", "")
            if page:
                source_info = f"{source} (page {page})"
            else:
                source_info = source
            if source_info not in source_list:
                source_list.append(source_info)
        st.session_state.sources[len(st.session_state.display_messages)-1] = source_list

    st.session_state.last_question = ""

def on_question_change():
    user_question = st.session_state.last_question
    if user_question:
        handle_userinput(user_question)

def download_button(chat_history, file_format):
    if not chat_history:
        st.warning("No chat history to save.")
        return

    if file_format == "txt":
        text_data = ""
        for msg in chat_history:
            if isinstance(msg["content"], str):
                text_data += f"{msg['role'].upper()}: {msg['content']}\n"
        b64 = base64.b64encode(text_data.encode()).decode()
        href = f'<a href="data:file/txt;base64,{b64}" download="chat_history.txt" class="download-button">Save as TXT</a>'
        st.markdown(href, unsafe_allow_html=True)
    elif file_format == "csv":
        data = []
        for msg in chat_history:
            if isinstance(msg["content"], str):
                data.append({"role": msg["role"], "content": msg["content"]})
        df = pd.DataFrame(data)
        csv_buffer = StringIO()
        df.to_csv(csv_buffer, index=False)
        csv_data = csv_buffer.getvalue().encode('utf-8')
        b64 = base64.b64encode(csv_data).decode()
        href = f'<a href="data:file/csv;base64,{b64}" download="chat_history.csv" class="download-button">Save as CSV</a>'
        st.markdown(href, unsafe_allow_html=True)

def main():
    load_dotenv()
    st.set_page_config(
        page_title="QueryIt",
        page_icon="📚",
        layout="wide",
        initial_sidebar_state="collapsed"
    )

    with open("styles1.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

    st.markdown("""
    <div class="app-header">
        <div>
            <h1><span class="app-logo">📚</span> QueryIt</h1>
            <p class="app-tagline">Intelligent Document & Database Query</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    hide_menu = """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
    """
    st.markdown(hide_menu, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None
    if "last_question" not in st.session_state:
        st.session_state.last_question = ""
    if "display_messages" not in st.session_state:
        st.session_state.display_messages = []
    if "processing" not in st.session_state:
        st.session_state.processing = False
    if "sources" not in st.session_state:
        st.session_state.sources = {}
    if "uploaded_db_file" not in st.session_state:
        st.session_state.uploaded_db_file = None

    with st.sidebar:
        st.markdown('<div class="section-header">📂 Document Library</div>', unsafe_allow_html=True)
        with st.container():
            st.markdown('<div class="section-title">Upload Documents</div>', unsafe_allow_html=True)
            st.markdown("""
            <div class="file-types">
                <span>PDF</span><span>DOCX</span><span>PPTX</span><span>TXT</span>
                <span>CSV</span><span>XLSX</span><span>Images</span><span>MD</span>
                <span>SQL</span><span>DB</span>
            </div>
            """, unsafe_allow_html=True)

            doc_files = st.file_uploader(
                "Drop your files here",
                type=["pdf", "docx", "pptx", "txt", "csv", "xlsx", "jpg", "jpeg", "png", "md", "sql", "db"],
                accept_multiple_files=True,
                label_visibility="collapsed"
            )

            if doc_files:
                file_count = len(doc_files)
                st.markdown(f"<div class='file-count'>{file_count} file{'s' if file_count > 1 else ''} uploaded:</div>", unsafe_allow_html=True)
                for file in doc_files:
                    file_type = file.name.split('.')[-1].upper()
                    file_icon = {
                        'PDF': '📄', 'DOCX': '📝', 'PPTX': '📊',
                        'TXT': '📃', 'CSV': '📈', 'XLSX': '📉',
                        'JPG': '🖼️', 'JPEG': '🖼️', 'PNG': '🖼️',
                        'MD': '📋', 'SQL': '🗄️', 'DB': '🗄️'
                    }.get(file_type, '📁')
                    st.markdown(f"<div class='file-item'>{file_icon} {file.name}</div>", unsafe_allow_html=True)
                
                # Store .db file content for query execution
                for file in doc_files:
                    if file.name.endswith('.db'):
                        st.session_state.uploaded_db_file = file.getvalue()

                st.markdown("<div class='spacer'></div>", unsafe_allow_html=True)
                process_col1, process_col2 = st.columns([3, 2])
                with process_col1:
                    process_button = st.button(
                        "Analyze Document",
                        key="process_button",
                        help="Create an AI-searchable knowledge base from your documents",
                        use_container_width=True
                    )
                with process_col2:
                    clear_button = st.button(
                        "🗑️ Clear",
                        key="clear_docs",
                        use_container_width=True
                    )
                    if clear_button:
                        st.session_state.conversation = None
                        st.session_state.chat_history = None
                        st.session_state.display_messages = []
                        st.session_state.processing = False
                        st.session_state.last_question = ""
                        st.session_state.sources = {}
                        st.session_state.uploaded_db_file = None
                        st.experimental_rerun()

                if process_button:
                    st.session_state.processing = True
                    with st.spinner("Processing documents..."):
                        try:
                            chunks_with_metadata = get_text_chunks_with_metadata(doc_files)
                            vectorstore = get_vectorstore(chunks_with_metadata)
                            st.session_state.conversation = get_conversation_chain(vectorstore)
                            st.session_state.processing = False
                            st.success("✅ Knowledge base created successfully! Ask any question about your documents.")
                        except Exception as e:
                            st.session_state.processing = False
                            st.error(f"❌ Error processing documents: {str(e)}")
            else:
                st.info("👆 Upload one or more documents to start")

        with st.expander("ℹ️ User Guide"):
            st.markdown("""
            <div class="user-guide">
                <div class="section-title">Getting Started</div>
                <ol>
                    <li>Upload documents or databases (SQL, SQLite DB)</li>
                    <li>Click "Analyze Documents" to process content</li>
                    <li>Ask questions in the query area</li>
                    <li>View tailored results, SQL queries, and download tables</li>
                </ol>
            </div>
            """, unsafe_allow_html=True)

    st.markdown('<div class="section-header">💬 Query Area</div>', unsafe_allow_html=True)
    chat_container = st.container(height=500)
    with chat_container:
        if not st.session_state.display_messages:
            st.markdown("""
            <div class="welcome-card">
                <div class="welcome-icon">📚</div>
                <div class="welcome-title">QueryIt</div>
                <div class="welcome-text">
                    Upload your documents or databases to start querying.
                    Ask questions to extract insights and get precise answers.
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            for i, msg in enumerate(st.session_state.display_messages):
                if msg["role"] == "user":
                    st.markdown(f'<div class="user-bubble">{msg["content"]}</div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="assistant-bubble">{msg["content"]}</div>', unsafe_allow_html=True)
                    if "table" in msg:
                        st.markdown("<div class='table-container'>", unsafe_allow_html=True)
                        st.table(msg["table"])
                        if "table_download" in msg:
                            st.markdown(msg["table_download"], unsafe_allow_html=True)
                        st.markdown("</div>", unsafe_allow_html=True)

                if msg["role"] == "assistant" and i in st.session_state.sources:
                    with st.expander("📚 Sources for this answer"):
                        st.markdown("<div class='source-title'>Referenced Documents</div>", unsafe_allow_html=True)
                        for source in st.session_state.sources[i]:
                            st.markdown(f"- {source}")

    st.divider()
    is_ready = st.session_state.conversation is not None and not st.session_state.processing
    input_col1, input_col2, input_col3 = st.columns([5, 1.5, 2])
    with input_col1:
        st.text_input(
            label="Ask a question about your documents",
            key="last_question",
            on_change=on_question_change,
            placeholder="Type your question here..." if is_ready else "Process documents first to enable chat",
            disabled=not is_ready,
            label_visibility="collapsed"
        )
    with input_col2:
        st.button(
            "Send 📤",
            on_click=on_question_change,
            disabled=not is_ready,
            use_container_width=True
        )
    with input_col3:
        if st.session_state.display_messages:
            
            with st.expander("💾 Save Chat"):
                download_button(st.session_state.display_messages, "txt")
                download_button(st.session_state.display_messages, "csv")
                
        else:
            st.markdown("<div style='visibility: hidden;'>💾 Save Chat</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()