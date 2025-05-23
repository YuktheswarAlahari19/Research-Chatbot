import sys
import os
import glob
import uuid

# Ensure project root is on path
def add_project_root_to_path():
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

add_project_root_to_path()

from chatbot_theme_identifier.backend.document_processor import DocumentProcessor
from chatbot_theme_identifier.backend.vector_store import VectorStore
from chatbot_theme_identifier.backend.query_processor import QueryProcessor
from chatbot_theme_identifier.backend.theme_identifier import ThemeIdentifier

# Initialize components
processor = DocumentProcessor()
vector_store = VectorStore()
query_processor = QueryProcessor(vector_store)
theme_identifier = ThemeIdentifier(query_processor)

# Helper to load and index a PDF
def index_pdf(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"PDF not found: {path}")
    pages = processor.process(path)
    if not pages:
        raise RuntimeError(f"No pages extracted from PDF: {path}")
    doc_id = str(uuid.uuid4())
    vector_store.adding_documents(doc_id, pages)
    return doc_id

# Index all PDFs in a directory
def index_pdf_directory(directory):
    if not os.path.exists(directory):
        raise FileNotFoundError(f"Directory not found: {directory}")
    pdf_files = glob.glob(os.path.join(directory, "*.pdf"))
    if not pdf_files:
        raise FileNotFoundError(f"No PDFs found in directory: {directory}")
    
    doc_ids = []
    for pdf_path in pdf_files:
        print(f"Indexing {pdf_path}...")
        doc_id = index_pdf(pdf_path)
        doc_ids.append(doc_id)
    return doc_ids

# Test scenarios
def run_tests(pdf_directory):
    # Index all PDFs
    try:
        doc_ids = index_pdf_directory(pdf_directory)
        print(f"Indexed {len(doc_ids)} PDFs.")
    except Exception as e:
        sys.exit(f"Error indexing PDFs: {str(e)}")

    # Define test queries
    queries = [
        "What is Linear Regression?",
       
    ]

    # Process each query
    for qry in queries:
        print(f"\n=== Query: {qry} ===")
        result = theme_identifier.identify_themes(qry)
        
        # Display results
        print(result.get('summary', 'No summary'))
        print("\nThemes:")
        for i, theme in enumerate(result.get('themes', []), 1):
            print(f"{i}. {theme}")
        print("\nCitations:")
        for citation in result.get('citations', []):
            print(f"- {citation}")
        
        # Display individual responses in tabular format
        theme_identifier.print_tabular_responses(result.get('individual_responses', []))

if __name__ == '__main__':
    # Specify the directory containing PDFs
    pdf_directory = "/home/yuk/Documents/Task/pdfs"
    run_tests(pdf_directory)