import streamlit as st
import requests
import base64
import uuid
import json
import os
import re
import pdfkit
import markdown
from PyPDF2 import PdfReader
from google.generativeai import GenerativeModel, configure
import sqlite3
import hashlib
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import pandas as pd
import tempfile

# Configure API keys (best practice would be to use environment variables)
MISTRAL_API_KEY = "JZslDyH2l2tQHDksRbDfHYVKgO50LfkN"  # Your Mistral API key
GEMINI_API_KEY = "AIzaSyAwY29cyESToWBGM3Rg2mEghTJUGyMaoJw"  # Your Gemini API key

# Configure Gemini
configure(api_key=GEMINI_API_KEY)

# Database setup
DB_NAME = "database/financial_analyzer.db"
DATABASE_DIR = "database"
if not os.path.exists(DATABASE_DIR):
    os.makedirs(DATABASE_DIR)

# Initialize SentenceTransformer for embeddings
embedder = SentenceTransformer('all-MiniLM-L6-v2')  # A lightweight model for embeddings

def init_db():
    """Initialize the SQLite database and create the table if it doesn't exist."""
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS financial_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            file_hash TEXT UNIQUE,
            file_name TEXT,
            extracted_text TEXT,
            analysis_result TEXT,
            extracted_tables TEXT,
            faiss_index_path TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.commit()
    conn.close()

def get_file_hash(uploaded_file):
    """Generate a hash of the uploaded file to check for duplicates."""
    uploaded_file.seek(0)
    file_content = uploaded_file.read()
    file_hash = hashlib.sha256(file_content).hexdigest()
    uploaded_file.seek(0)
    return file_hash

def save_to_db(file_hash, file_name, extracted_text, analysis_result, extracted_tables, faiss_index_path):
    """Save extracted text, tables, analysis result, and FAISS index path to the database."""
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute('''
        INSERT OR REPLACE INTO financial_data 
        (file_hash, file_name, extracted_text, analysis_result, extracted_tables, faiss_index_path)
        VALUES (?, ?, ?, ?, ?, ?)
    ''', (file_hash, file_name, extracted_text, analysis_result, extracted_tables, faiss_index_path))
    conn.commit()
    conn.close()

def get_existing_data(file_hash):
    """Retrieve existing data from the database based on file hash."""
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute('SELECT file_name, extracted_text, analysis_result, extracted_tables, faiss_index_path FROM financial_data WHERE file_hash = ?', (file_hash,))
    result = c.fetchone()
    conn.close()
    return result

def get_all_faiss_indexes():
    """Retrieve all FAISS index paths and file names from the database."""
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute('SELECT file_name, faiss_index_path FROM financial_data')
    results = c.fetchall()
    conn.close()
    return results

def build_faiss_index(text_chunks, file_hash):
    """Build and save a FAISS index from text chunks with a dynamic name."""
    if not text_chunks:
        return None, None, None
    embeddings = embedder.encode(text_chunks, convert_to_numpy=True)
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)  # L2 distance index
    index.add(embeddings)
    faiss_index_path = os.path.join(DATABASE_DIR, f"faiss_{file_hash}.index")
    faiss.write_index(index, faiss_index_path)
    return index, embeddings, faiss_index_path

def load_faiss_index(faiss_index_path):
    """Load the FAISS index from file if it exists."""
    if os.path.exists(faiss_index_path):
        return faiss.read_index(faiss_index_path)
    return None

def extract_text_with_mistral(uploaded_file, pages_to_process):
    """Extract text from PDF using Mistral AI OCR API"""
    api_url = "https://api.mistral.ai/v1/ocr"
    unique_id = str(uuid.uuid4())
    pdf_base64 = base64.b64encode(uploaded_file.getvalue()).decode('utf-8')
    
    payload = {
        "model": "mistral-ocr-latest",
        "id": unique_id,
        "document": {
            "document_url": f"data:application/pdf;base64,{pdf_base64}",
            "document_name": uploaded_file.name,
            "type": "document_url"
        },
        "pages": pages_to_process,
        "include_image_base64": True,
        "image_limit": 0,
        "image_min_size": 0
    }
    
    headers = {
        "Authorization": f"Bearer {MISTRAL_API_KEY}",
        "Content-Type": "application/json"
    }
    
    try:
        response = requests.post(api_url, headers=headers, json=payload)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Error calling Mistral OCR API: {str(e)}")
        if 'response' in locals() and hasattr(response, 'text'):
            st.error(f"Response: {response.text}")
        return None

def extract_tables_from_text(text_content):
    """Extract tables from the markdown text content"""
    table_pattern = r'(\|[^\n]+\|\n\|[-:| ]+\|\n(?:\|[^\n]+\|\n)+)'
    tables = re.findall(table_pattern, text_content)
    return tables

def create_summary_tables(analysis_text):
    """Ask Gemini to create summary tables based on the analysis"""
    try:
        model = GenerativeModel('gemini-2.0-flash')
        prompt = f"""
        Based on the following financial analysis, create 3-5 summary tables in Markdown format. 
        These tables should highlight key financial metrics, trends, and insights from the analysis.
        
        For example, you might create tables for:
        1. Key Financial Metrics Summary
        2. Income Statement Highlights
        3. Balance Sheet Overview
        4. EBITDA Adjustments
        5. Working Capital Summary
        
        Each table should have a clear title and organized columns with meaningful data.
        IMPORTANT: Format all tables in proper Markdown format using pipe (|) syntax.
        
        For each table created, provide specific citations that justify the data, including:
        1. Exact information source from the analysis (quote the specific text)
        2. Page numbers where this information appears in the original document
        3. A brief explanation of how you interpreted this data for the table
        
        After each table, include a "**Table Justification:**" section that explains where each 
        data point came from and how it relates to the original document.
        
        Analysis:
        {analysis_text}
        """
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        st.error(f"Error creating summary tables with Gemini: {str(e)}")
        return "Error creating summary tables."

def analyze_with_gemini(text_content, pages_to_process):
    """Analyze the extracted text with Google's Gemini API and include detailed citations."""
    try:
        # First, let's create page-specific segments to be able to track data sources
        page_text_dict = {}
        current_text = ""
        page_markers = re.findall(r'\n\*\*Page (\d+)\*\*\n', text_content)
        if page_markers:
            # If we have explicit page markers
            segments = re.split(r'\n\*\*Page \d+\*\*\n', text_content)
            for i, page_num in enumerate(page_markers):
                if i + 1 < len(segments):
                    page_text_dict[page_num] = segments[i + 1]
        else:
            # If no explicit markers, divide text by estimated page size
            words = text_content.split()
            words_per_page = 500  # Estimated words per page
            for i, page in enumerate(pages_to_process):
                start_idx = i * words_per_page
                end_idx = (i + 1) * words_per_page
                if start_idx < len(words):
                    page_text = " ".join(words[start_idx:min(end_idx, len(words))])
                    page_text_dict[str(page)] = page_text
        
        model = GenerativeModel('gemini-2.0-flash')
        prompt = f"""
        Analyze the following financial document and provide a detailed analysis with these sections:
        1. BUSINESS OVERVIEW
        2. KEY FINDINGS, FINANCIAL DUE DILIGENCE
        3. INCOME STATEMENT OVERVIEW
        4. BALANCE SHEET OVERVIEW
        5. ADJ EBITDA (IF DETAILED INFORMATION IS PROVIDED)
        6. ADJ WORKING CAPITAL (IF DETAILED INFORMATION IS PROVIDED)
        
        IMPORTANT REQUIREMENTS FOR CITATIONS AND JUSTIFICATIONS:
        
        For each section:
        1. After every paragraph, provide a detailed citation in this format:
           [**Source: Page X, Paragraph Y**] where X is the page number and Y is an approximate paragraph number.
        
        2. After each section, include a detailed "**Justification:**" subsection that:
           - Quotes specific text from the original document that supports your analysis (use direct quotes in "quotation marks")
           - Explains how you interpreted this information
           - Lists ALL pages where supporting evidence was found
           - Explains any assumptions or inferences made when information was implicit
        
        3. For any tables or financial data, provide the exact source, including:
           - The exact numbers as they appear in the original document
           - The page numbers where each data point was found
           - Any calculations or transformations you performed
        
        4. If information seems inconsistent or contradictory, note this explicitly with a "**Data Inconsistency Note:**" 
           explaining the discrepancy and which source you relied on more heavily.
        
        5. If certain information was inferred rather than explicitly stated, mark it clearly with 
           "**Inference:**" and explain your reasoning.
        
        IMPORTANT: Include relevant data in table format where appropriate. Use proper Markdown format
        for all tables using | syntax. For each key financial metric or comparison, present the data
        in a clear, structured table.
        
        Document content:
        {text_content}
        
        Page-specific content:
        {json.dumps(page_text_dict)}
        """
        response = model.generate_content(prompt)
        
        # Post-process to ensure citations are properly formatted
        analysis_text = response.text
        
        # Add an overview of pages processed at the beginning
        page_overview = f"""
## DOCUMENT INFORMATION
- **Pages Analyzed:** {', '.join(map(str, pages_to_process))}
- **Total Pages Processed:** {len(pages_to_process)}

"""
        analysis_text = page_overview + analysis_text
        
        return analysis_text
    except Exception as e:
        st.error(f"Error analyzing with Gemini: {str(e)}")
        return "Error analyzing the content with Gemini API."

def chat_with_gemini_simple(context, user_query):
    """Simple chat with Gemini without FAISS."""
    try:
        model = GenerativeModel('gemini-2.0-flash')
        prompt = f"""
        Based on the following context, answer the user's query:
        
        Context:
        {context}
        
        User Query:
        {user_query}
        
        IMPORTANT FOR CITATIONS:
        1. Cite specific parts of the document that support your answer using [Page X] format
        2. If you make any inference not directly stated in the document, mark it as [Inference]
        3. When providing facts or figures, always include where they came from in the document
        4. If the document contains contradictory information, acknowledge this and explain which source you relied on
        
        If your response should include data, present it in a well-formatted table using Markdown syntax.
        """
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        st.error(f"Error chatting with Gemini: {str(e)}")
        return "Error processing your query."

def chat_with_gemini_faiss(context_chunks, index, full_context, user_query, k=3):
    """Chat with Gemini using FAISS to retrieve relevant context, combined with full context from DB."""
    if not full_context or not full_context.strip():
        return "No full context available to process the query."

    if not context_chunks:
        # Fallback to full context if no chunks are available
        return chat_with_gemini_simple(full_context, user_query)
    
    # Adjust k to be at most the number of chunks
    k = min(k, len(context_chunks))
    if k == 0:
        # Fallback to full context if k becomes 0
        return chat_with_gemini_simple(full_context, user_query)
    
    query_embedding = embedder.encode([user_query], convert_to_numpy=True)
    distances, indices = index.search(query_embedding, k)  # Retrieve top-k similar chunks
    
    # Filter valid indices
    valid_indices = [i for i in indices[0] if i >= 0 and i < len(context_chunks)]
    if not valid_indices:
        # Fallback to full context if no relevant chunks are found
        return chat_with_gemini_simple(full_context, user_query)
    
    relevant_chunks = [context_chunks[i] for i in valid_indices]
    relevant_context = "\n\n".join(relevant_chunks)
    
    try:
        model = GenerativeModel('gemini-2.0-flash')
        prompt = f"""
        Based on the following full context and the most relevant sections, answer the user's query:
        
        Full Context:
        {full_context}
        
        Most Relevant Sections:
        {relevant_context}
        
        User Query:
        {user_query}
        
        IMPORTANT FOR CITATIONS:
        1. Always support your answer with specific citations from the document in [Page X] format
        2. Quote relevant text that supports your answer
        3. List the evidence that led to your conclusion for each major point
        4. If you make any inference not directly stated in the document, mark it as [Inference]
        5. If the information seems incomplete or uncertain, acknowledge this and explain what additional information would help
        
        If your response should include data, present it in a well-formatted table using Markdown syntax.
        """
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        st.error(f"Error chatting with Gemini: {str(e)}")
        return "Error processing your query."

def extract_page_numbers(text_content):
    """Extract page numbers from text content to improve citation accuracy."""
    page_numbers = {}
    lines = text_content.split('\n')
    current_page = 1
    
    for i, line in enumerate(lines):
        # Look for page markers like "Page X" or "Page X of Y"
        page_match = re.search(r'(?i)page\s+(\d+)(?:\s+of\s+\d+)?', line)
        if page_match:
            current_page = int(page_match.group(1))
            start_line = i
            page_numbers[current_page] = {
                'start_line': start_line,
                'content': line
            }
    
    # Add end lines for each page
    sorted_pages = sorted(page_numbers.keys())
    for i, page in enumerate(sorted_pages):
        if i < len(sorted_pages) - 1:
            page_numbers[page]['end_line'] = page_numbers[sorted_pages[i+1]]['start_line'] - 1
        else:
            page_numbers[page]['end_line'] = len(lines) - 1
    
    return page_numbers

def split_into_chunks(text, chunk_size=200):  # Reduced chunk size for better granularity
    """Split text into smaller chunks for FAISS indexing."""
    if not text or not text.strip():
        return []
    words = text.split()
    
    # Add page number information to chunks when possible
    chunks = []
    page_info = extract_page_numbers(text)
    
    for i in range(0, len(words), chunk_size):
        chunk = ' '.join(words[i:i + chunk_size])
        # Try to find page number for this chunk
        for page_num, page_data in page_info.items():
            chunk_position = text.find(chunk)
            if chunk_position >= 0:
                chunk_line = text[:chunk_position].count('\n')
                if page_data['start_line'] <= chunk_line <= page_data['end_line']:
                    chunk = f"[Page {page_num}] {chunk}"
                    break
        chunks.append(chunk)
    
    return chunks

def convert_markdown_to_pdf(markdown_content, output_path):
    """Convert markdown content to PDF using pdfkit/wkhtmltopdf."""
    try:
        # Convert markdown to HTML
        html_content = markdown.markdown(markdown_content, extensions=['tables', 'fenced_code'])
        
        # Add basic styling for better appearance
        styled_html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; line-height: 1.6; }}
                h1, h2, h3 {{ color: #333366; }}
                table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                tr:nth-child(even) {{ background-color: #f9f9f9; }}
                pre {{ background-color: #f5f5f5; padding: 10px; border-radius: 5px; overflow-x: auto; }}
                code {{ font-family: Consolas, monospace; }}
                .citation {{ background-color: #f0f7ff; padding: 5px; border-left: 3px solid #3498db; margin: 10px 0; }}
                .justification {{ background-color: #f0fff0; padding: 10px; border-left: 3px solid #2ecc71; margin: 15px 0; }}
                .inference {{ background-color: #fff9e6; padding: 5px; border-left: 3px solid #f39c12; margin: 10px 0; }}
            </style>
        </head>
        <body>
            {html_content}
        </body>
        </html>
        """
        
        # Create temporary HTML file
        with tempfile.NamedTemporaryFile(suffix='.html', delete=False, mode='w', encoding='utf-8') as temp_html:
            temp_html_path = temp_html.name
            temp_html.write(styled_html)
        
        # Configure pdfkit with path to wkhtmltopdf if needed
        config = None
        try:
            # Try to find wkhtmltopdf in common locations
            wkhtmltopdf_paths = [
                '/usr/local/bin/wkhtmltopdf',
                '/usr/bin/wkhtmltopdf',
                'C:\\Program Files\\wkhtmltopdf\\bin\\wkhtmltopdf.exe',
                'C:\\Program Files (x86)\\wkhtmltopdf\\bin\\wkhtmltopdf.exe'
            ]
            
            for path in wkhtmltopdf_paths:
                if os.path.exists(path):
                    config = pdfkit.configuration(wkhtmltopdf=path)
                    break
            
            # Use the config if found, otherwise try without it
            if config:
                pdfkit.from_file(temp_html_path, output_path, configuration=config)
            else:
                pdfkit.from_file(temp_html_path, output_path)
                
            os.remove(temp_html_path)  # Clean up temp file
            return True, None
            
        except Exception as e:
            os.remove(temp_html_path)  # Clean up temp file
            return False, str(e)
            
    except Exception as e:
        return False, str(e)

def main():
    # Initialize database
    init_db()

    # Streamlit UI
    st.title("Financial Document Analyzer with Enhanced Citations")
    st.write("Upload a financial PDF document for analysis with detailed citations and justifications")

    # Tabs for Upload and Chat with Existing FAISS
    tab1, tab2 = st.tabs(["Upload New PDF", "Chat with Existing FAISS"])

    with tab1:
        # File uploader
        uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])
        
        if uploaded_file is not None:
            # Calculate file hash to check for duplicates
            file_hash = get_file_hash(uploaded_file)
            existing_data = get_existing_data(file_hash)

            if existing_data:
                # If data exists in the database
                file_name, extracted_text, analysis_result, extracted_tables, faiss_index_path = existing_data
                st.success(f"Found existing analysis for '{file_name}' in the database!")
                
                with st.expander("View Extracted Text"):
                    st.markdown(extracted_text)
                
                # Show tables if available
                if extracted_tables:
                    with st.expander("View Extracted Tables"):
                        tables = json.loads(extracted_tables)
                        for i, table in enumerate(tables):
                            st.markdown(f"**Table {i+1}**")
                            st.markdown(table)
                
                st.subheader("Financial Analysis Results:")
                st.markdown(analysis_result)
                
                # Add download buttons
                col1, col2 = st.columns(2)
                with col1:
                    st.download_button(
                        label="Download Analysis (Markdown)",
                        data=analysis_result.encode(),
                        file_name="financial_analysis.md",
                        mime="text/markdown"
                    )
                
                with col2:
                    if st.button("Generate PDF"):
                        with st.spinner("Generating PDF..."):
                            # Create a temporary directory to store the PDF
                            with tempfile.TemporaryDirectory() as temp_dir:
                                pdf_path = os.path.join(temp_dir, "financial_analysis.pdf")
                                success, error = convert_markdown_to_pdf(analysis_result, pdf_path)
                                
                                if success:
                                    with open(pdf_path, "rb") as pdf_file:
                                        pdf_bytes = pdf_file.read()
                                    
                                    st.download_button(
                                        label="Download Analysis (PDF)",
                                        data=pdf_bytes,
                                        file_name="financial_analysis.pdf", 
                                        mime="application/pdf"
                                    )
                                else:
                                    st.error(f"Error generating PDF: {error}")
                                    st.info("Installing wkhtmltopdf:")
                                    st.code("""
# Ubuntu/Debian:
sudo apt-get install wkhtmltopdf

# macOS:
brew install wkhtmltopdf

# Windows:
# Download from https://wkhtmltopdf.org/downloads.html
                                    """)

                # Chat with existing data
                st.subheader("Chat with Existing Data")
                chat_mode = st.radio("Chat mode:", ("Simple (Full Context)", "FAISS (Relevant Chunks)"))
                chat_option = st.radio("Chat based on:", ("Extracted Text", "Analysis Result"))
                context_text = extracted_text if chat_option == "Extracted Text" else analysis_result
                
                if chat_mode == "FAISS (Relevant Chunks)":
                    # Load all available FAISS indexes
                    faiss_indexes = get_all_faiss_indexes()
                    faiss_options = {f"{name} ({path})": path for name, path in faiss_indexes}
                    selected_faiss = st.selectbox("Select FAISS index to use:", list(faiss_options.keys()))
                    faiss_index_path = faiss_options[selected_faiss] if selected_faiss else faiss_index_path
                    
                    context_chunks = split_into_chunks(context_text)
                    index = load_faiss_index(faiss_index_path)
                    if index is None or st.button("Rebuild FAISS Index"):
                        index, _, new_faiss_index_path = build_faiss_index(context_chunks, file_hash)
                        save_to_db(file_hash, file_name, extracted_text, analysis_result, extracted_tables, new_faiss_index_path)
                        st.success("FAISS index rebuilt!")
                
                # Initialize chat history in session state
                if "chat_history_upload" not in st.session_state:
                    st.session_state.chat_history_upload = []
                
                # Display chat history
                for message in st.session_state.chat_history_upload:
                    with st.chat_message(message["role"]):
                        st.markdown(message["content"])
                
                # Chat input
                user_query = st.chat_input("Ask a question about the document (Upload Tab):")
                if user_query:
                    # Add user message to chat history
                    st.session_state.chat_history_upload.append({"role": "user", "content": user_query})
                    with st.chat_message("user"):
                        st.markdown(user_query)
                    
                    # Get response from Gemini
                    with st.spinner("Thinking..."):
                        if chat_mode == "Simple (Full Context)":
                            response = chat_with_gemini_simple(context_text, user_query)
                        else:  # FAISS mode
                            response = chat_with_gemini_faiss(context_chunks, index, context_text, user_query)
                        st.session_state.chat_history_upload.append({"role": "assistant", "content": response})
                        with st.chat_message("assistant"):
                            st.markdown(response)

            else:
                # Process the new file
                try:
                    pdf_reader = PdfReader(uploaded_file)
                    num_pages = len(pdf_reader.pages)
                    
                    st.write(f"Total Pages: {num_pages}")
                    
                    pages_to_process = st.multiselect(
                        "Select pages to process (default: all)", 
                        options=list(range(num_pages)), 
                        default=list(range(num_pages))
                    )
                    
                    citation_level = st.select_slider(
                        "Citation Detail Level",
                        options=["Basic", "Detailed", "Comprehensive"],
                        value="Detailed",
                        help="Controls how detailed the citations and justifications will be in the analysis"
                    )
                    
                    if st.button("Extract and Analyze"):
                        with st.spinner("Extracting text from PDF..."):
                            ocr_result = extract_text_with_mistral(uploaded_file, pages_to_process)
                            
                            if ocr_result:
                                all_text = ""
                                for page in ocr_result.get("pages", []):
                                    # Add explicit page markers to help with citation
                                    page_num = page.get("page_num", 0)
                                    all_text += f"\n**Page {page_num}**\n" + page.get("markdown", "") + "\n\n"
                                
                                with st.expander("View Extracted Text"):
                                    st.markdown(all_text)
                                
                                # Extract tables from the OCR result
                                tables = extract_tables_from_text(all_text)
                                if tables:
                                    with st.expander("View Extracted Tables"):
                                        for i, table in enumerate(tables):
                                            st.markdown(f"**Table {i+1}**")
                                            st.markdown(table)
                                
                                with st.spinner("Analyzing financial information with detailed citations..."):
                                    # First analyze the document
                                    analysis = analyze_with_gemini(all_text, pages_to_process)
                                    
                                    # Then create additional summary tables
                                    with st.spinner("Creating summary tables with justifications..."):
                                        summary_tables = create_summary_tables(analysis)
                                        # Combine analysis with summary tables
                                        combined_analysis = f"{analysis}\n\n## SUMMARY TABLES\n\n{summary_tables}"
                                    
                                    st.subheader("Financial Analysis Results with Citations:")
                                    st.markdown(combined_analysis)
                                    
                                    # Add download buttons
                                    col1, col2 = st.columns(2)
                                    with col1:
                                        st.download_button(
                                            label="Download Analysis (Markdown)",
                                            data=combined_analysis.encode(),
                                            file_name="financial_analysis.md",
                                            mime="text/markdown"
                                        )
                                    
                                    with col2:
                                        if st.button("Generate PDF"):
                                            with st.spinner("Generating PDF..."):
                                                # Create a temporary directory to store the PDF
                                                with tempfile.TemporaryDirectory() as temp_dir:
                                                    pdf_path = os.path.join(temp_dir, "financial_analysis.pdf")
                                                    success, error = convert_markdown_to_pdf(combined_analysis, pdf_path)
                                                    
                                                    if success:
                                                        with open(pdf_path, "rb") as pdf_file:
                                                            pdf_bytes = pdf_file.read()
                                                        
                                                        st.download_button(
                                                            label="Download Analysis (PDF)",
                                                            data=pdf_bytes,
                                                            file_name="financial_analysis.pdf", 
                                                            mime="application/pdf"
                                                        )
                                                    else:
                                                        st.error(f"Error generating PDF: {error}")
                                                        st.info("Installing wkhtmltopdf:")
                                                        st.code("""
# Ubuntu/Debian:
sudo apt-get install wkhtmltopdf

# macOS:
brew install wkhtmltopdf

# Windows:
# Download from https://wkhtmltopdf.org/downloads.html
                                                        """)
                                    
                                    # Build FAISS index for new data
                                    context_chunks = split_into_chunks(all_text)
                                    index, embeddings, faiss_index_path = build_faiss_index(context_chunks, file_hash)
                                    
                                    # Save to database
                                    save_to_db(
                                        file_hash, 
                                        uploaded_file.name, 
                                        all_text, 
                                        combined_analysis, 
                                        json.dumps(tables), 
                                        faiss_index_path
                                    )
                                    
                                    st.success("Analysis completed and saved to database.")
                                    
                                    # Initialize chat with the newly processed data
                                    st.subheader("Chat with This Document")
                                    if "chat_history_new" not in st.session_state:
                                        st.session_state.chat_history_new = []
                                    
                                    chat_query = st.text_input("Ask a question about this document:")
                                    if chat_query:
                                        response = chat_with_gemini_faiss(context_chunks, index, combined_analysis, chat_query)
                                        st.session_state.chat_history_new.append({"role": "user", "content": chat_query})
                                        st.session_state.chat_history_new.append({"role": "assistant", "content": response})
                                        
                                        # Display chat history
                                        for message in st.session_state.chat_history_new:
                                            with st.chat_message(message["role"]):
                                                st.markdown(message["content"])
                            else:
                                st.error("Failed to extract text from the PDF.")
                
                except Exception as e:
                    st.error(f"Error processing PDF: {str(e)}")

    with tab2:
        # Chat with existing FAISS indexes
        st.subheader("Chat with Existing Documents")
        
        # Load all available FAISS indexes
        faiss_indexes = get_all_faiss_indexes()
        
        if not faiss_indexes:
            st.info("No documents have been processed yet. Please upload a document in the Upload tab first.")
        else:
            # Create a dictionary mapping display names to actual file paths
            faiss_options = {f"{name}": path for name, path in faiss_indexes}
            selected_doc = st.selectbox("Select document to chat with:", list(faiss_options.keys()))
            selected_path = faiss_options[selected_doc]
            
            # Get the file hash from the path
            file_hash = os.path.basename(selected_path).replace("faiss_", "").replace(".index", "")
            
            # Get the document data from the database
            conn = sqlite3.connect(DB_NAME)
            c = conn.cursor()
            c.execute('SELECT extracted_text, analysis_result FROM financial_data WHERE file_hash = ?', (file_hash,))
            result = c.fetchone()
            conn.close()
            
            if result:
                extracted_text, analysis_result = result
                
                # Choose context source
                context_source = st.radio("Choose context source:", ["Analysis Result", "Extracted Text"])
                context_text = analysis_result if context_source == "Analysis Result" else extracted_text
                
                # Initialize chat history in session state for tab2
                if "chat_history_tab2" not in st.session_state:
                    st.session_state.chat_history_tab2 = []
                
                # Display chat history
                for message in st.session_state.chat_history_tab2:
                    with st.chat_message(message["role"]):
                        st.markdown(message["content"])
                
                # Load FAISS index
                index = load_faiss_index(selected_path)
                if index is None:
                    st.error(f"Could not load FAISS index from {selected_path}.")
                else:
                    # Split context into chunks
                    context_chunks = split_into_chunks(context_text)
                    
                    # Chat input
                    user_query = st.chat_input("Ask a question about the document (Tab2):")
                    if user_query:
                        # Add user message to chat history
                        st.session_state.chat_history_tab2.append({"role": "user", "content": user_query})
                        with st.chat_message("user"):
                            st.markdown(user_query)
                        
                        # Get response from Gemini
                        with st.spinner("Thinking..."):
                            response = chat_with_gemini_faiss(context_chunks, index, context_text, user_query)
                            st.session_state.chat_history_tab2.append({"role": "assistant", "content": response})
                            with st.chat_message("assistant"):
                                st.markdown(response)
            else:
                st.error(f"Could not find document data for the selected index.")

    # Footer with information
    st.markdown("---")
    st.markdown("""
    ### About This Tool
    This Financial Document Analyzer extracts text from financial PDFs using Mistral OCR, analyzes the content using Google's Gemini model, and provides detailed citations and justifications.
    
    Features:
    - OCR text extraction with Mistral AI
    - Financial analysis with detailed citations using Google Gemini
    - FAISS semantic search for efficient document querying
    - Storage and retrieval of previously analyzed documents
    - Export to Markdown and PDF formats
    
    For more information, contact support.
    """)
    
if __name__ == "__main__":
    main()