import os
import glob
import logging
import uuid
import sys
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


# Configure logging for production use
logging.basicConfig(filename='test.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize all components of the chatbot pipeline
# Creates instances of DocumentProcessor, VectorStore, QueryProcessor, and ThemeIdentifier
# Prepares the system to process documents and queries
def initialize_components():
    """Initialize the chatbot pipeline components."""
    processor = DocumentProcessor(use_gpu=False)
    vector_store = VectorStore(collection_name="documents")
    query_processor = QueryProcessor(vector_store)
    theme_identifier = ThemeIdentifier(query_processor)
    return processor, vector_store, query_processor, theme_identifier

# Index all PDFs in a directory for storage
# Processes each PDF using DocumentProcessor and stores pages in VectorStore
# Returns a list of document IDs for tracking
def index_pdf_directory(directory: str, processor: DocumentProcessor, vector_store: VectorStore) -> list[str]:
    """Index all PDFs in the specified directory."""
    if not os.path.exists(directory):
        logging.error(f"Directory not found: {directory}")
        return []
    
    pdf_files = glob.glob(os.path.join(directory, "*.pdf"))
    if not pdf_files:
        logging.warning(f"No PDFs found in directory: {directory}")
        return []
    
    doc_ids = []
    for pdf_path in pdf_files:
        try:
            logging.info(f"Indexing {pdf_path}")
            pages = processor.process_file(pdf_path)
            if not pages:
                logging.warning(f"No pages extracted from {pdf_path}")
                continue
            doc_id = str(uuid.uuid4())
            vector_store.add_documents(doc_id, pages)
            doc_ids.append(doc_id)
        except Exception as e:
            logging.error(f"Error indexing {pdf_path}: {e}")
            continue
    return doc_ids

# Run test queries and generate results
# Processes each query using ThemeIdentifier to get themes, summary, and responses
# Returns a list of formatted result strings for output
def run_tests(queries: list[str], theme_identifier: ThemeIdentifier) -> list[str]:
    """Run test queries and collect formatted results."""
    results = []
    for query in queries:
        try:
            logging.info(f"Processing query: {query}")
            result = theme_identifier.identify_themes(query)
            formatted_result = f"\n=== Query: {query} ===\n"
            formatted_result += result.get('summary', 'No summary available') + "\n"
            formatted_result += "\nThemes:\n"
            for i, theme in enumerate(result.get('themes', []), 1):
                formatted_result += f"{i}. {theme}\n"
            formatted_result += "\nCitations:\n"
            for citation in result.get('citations', []):
                formatted_result += f"- {citation}\n"
            formatted_result += theme_identifier.format_tabular_responses(result.get('individual_responses', []))
            results.append(formatted_result)
        except Exception as e:
            logging.error(f"Error processing query '{query}': {e}")
            results.append(f"\n=== Query: {query} ===\nError: {str(e)}\n")
    return results

# Main function to execute the test pipeline
# Initializes components, indexes PDFs, runs test queries, and logs results
# Outputs results to console and log file
def main():
    """Execute the full chatbot test pipeline."""
    # Define test queries
    queries = [
        "What is Linear Regression?",
        "Explain SGD",
        "Describe Neural Networks",
    ]
    
    # Define PDF directory
    pdf_directory = "/home/yuk/Documents/Task/pdfs"
    
    # Initialize components
    try:
        processor, vector_store, query_processor, theme_identifier = initialize_components()
    except Exception as e:
        logging.error(f"Failed to initialize components: {e}")
        return
    
    # Index PDFs
    doc_ids = index_pdf_directory(pdf_directory, processor, vector_store)
    if not doc_ids:
        logging.error("No documents indexed. Exiting.")
        return
    logging.info(f"Indexed {len(doc_ids)} PDFs")
    
    # Run test queries
    results = run_tests(queries, theme_identifier)
    
    # Output results
    for result in results:
        print(result)
        logging.info(result)

if __name__ == '__main__':
    main()