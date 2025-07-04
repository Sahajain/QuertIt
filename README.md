# QueryIt: Intelligent Document Query System

QueryIt is an advanced document intelligence system designed to process and query diverse file formats, including PDF, DOCX, PPTX, TXT, CSV, XLSX, images, Markdown, SQL, and SQLite database files. Built with a retrieval-augmented generation (RAG) framework, QueryIt enables semantic search and conversational querying, providing accurate and context-aware responses. The system features a professional Streamlit-based user interface, custom styling, and specialized support for structured data queries via SQL query generation. QueryIt is ideal for enterprise applications requiring unified access to heterogeneous document collections.

## Features

- **Multi-Format Document Processing**: Supports a wide range of file types:
  - Text-based: PDF, DOCX, PPTX, TXT, Markdown
  - Tabular: CSV, XLSX
  - Visual: Images (via OCR)
  - Structured: SQL files and SQLite databases
- **Semantic Search**: Uses Azure OpenAI’s `text-embedding-ada-002` for dense vector embeddings and FAISS for efficient similarity search.
- **Conversational Interface**: Streamlit-based UI with chat history, source attribution, and downloadable responses (TXT/CSV).
- **SQL Query Generation**: Generates and executes SQL queries for SQL and SQLite files, with table results downloadable as CSV.
- **Privacy and Robustness**: Implements secure temporary file handling (e.g., SQLite databases) and comprehensive error handling.
- **Custom Styling**: Professional UI with a Deep Navy, Warm Gold, and Soft Teal color palette, animations, and responsive design.

## Installation

### Prerequisites
- Python 3.9+
- Git
- Azure OpenAI API key (for embeddings and LLM)
- Optional: Docker for containerized deployment

### Dependencies
Install the required Python packages using the provided `requirements.txt` or manually:

```bash
pip install streamlit==1.27.0 langchain==0.0.267 faiss-cpu==1.7.4 azure-ai-textanalytics==5.3.0 python-docx==0.8.11 pypdf2==3.0.1 python-pptx==0.6.21 pandas==2.0.3 sqlparse==0.4.4 markdown==3.4.4 tesseract==5.3.0 openpyxl==3.0.10
```

### Setup
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/your-username/queryit.git
   cd queryit
   ```

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure Environment Variables**:
   Create a `.env` file in the project root and add your Azure OpenAI credentials:
   ```plaintext
   AZURE_OPENAI_API_KEY=your-api-key
   AZURE_OPENAI_ENDPOINT=your-endpoint
   AZURE_OPENAI_DEPLOYMENT_NAME=text-embedding-ada-002
   ```

4. **Run the Application**:
   ```bash
   streamlit run draft1.py
   ```
   The app will be available at `http://localhost:8501`.


## Usage

1. **Launch the Application**:
   Run `streamlit run draft1.py` and access the UI in your browser.

2. **Upload Documents**:
   - Use the document upload interface to add files (PDF, DOCX, PPTX, TXT, CSV, XLSX, images, Markdown, SQL, or SQLite DB).
   - Files are processed into chunks, embedded, and indexed in a FAISS vector store.

3. **Query Documents**:
   - Enter queries in the conversational interface.
   - For SQL/SQLite files, ask structured data questions (e.g., “What are the top 5 customers by sales?”).
   - Responses include source attribution and, for SQL queries, downloadable table results.

4. **Download Results**:
   - Export chat history as TXT or CSV.
   - Download SQL query results as CSV.

## File Structure

```plaintext
queryit/
├── document_processing11.py  # Document processing pipeline (text extraction, chunking, metadata)
├── draft1.py                # Main Streamlit app (UI, RAG, SQL query generation)
├── styles1.css             # Custom CSS for UI styling
├── requirements.txt        # Python dependencies
├── .env                   # Environment variables (not tracked)
└── README.md              # Project documentation
```

## Dependencies

- **Streamlit (1.27.0)**: Web application framework for the UI.
- **LangChain (0.0.267)**: Framework for RAG and LLM integration.
- **FAISS (1.7.4)**: Vector store for similarity search.
- **Azure OpenAI SDK**: For embeddings and LLM queries.
- **PyPDF2, python-docx, python-pptx**: Document parsing for PDF, DOCX, PPTX.
- **pandas, openpyxl**: Tabular data processing for CSV/XLSX.
- **sqlparse, sqlite3**: SQL and SQLite database parsing.
- **Tesseract**: OCR for image processing.
- **markdown**: Markdown parsing.



