import sys
import os

# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

from chatbot_theme_identifier.backend.document_processor import DocumentProcessor
from chatbot_theme_identifier.backend.vector_store import VectorStore
import uuid

# Create processor and vector store
processor = DocumentProcessor()
vector_store = VectorStore()

# Test document
file_path = "/home/yuk/Documents/Task/pdfs/ML BOOk.pdf"

# Process and store document
if os.path.exists(file_path):
    print(f"Processing file: {file_path}")
    pages = processor.process(file_path)
    if pages:
        doc_id = str(uuid.uuid4())
        vector_store.adding_documents(doc_id, pages)
        print(f"Saved document {doc_id} with {len(pages)} pages")
    else:
        print("No pages processed.")
        exit()
else:
    print(f"File not found: {file_path}")
    exit()

# Test search
query = "What is ROC curve?"
print(f"\nSearching: {query}")
results = vector_store.search(query)
if results:
    for result in results:
        print(f"Document {result['doc_id']}, Page {result['page']}: {result['text'][:50]}... (Distance: {result['distance']})")
else:
    print("No search results found.")