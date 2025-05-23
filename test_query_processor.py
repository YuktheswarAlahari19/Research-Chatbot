import sys
import os

# Add the project folder to Python’s path so it can find our code files
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

# Import the tools we need for the project
from chatbot_theme_identifier.backend.document_processor import DocumentProcessor  # Reads PDFs
from chatbot_theme_identifier.backend.vector_store import VectorStore  # Saves document text
from chatbot_theme_identifier.backend.query_processor import QueryProcessor  # Answers questions
import uuid  # Makes unique names for documents

# Create the tools we’ll use
processor = DocumentProcessor()  # Tool to read and pull text from PDFs
vector_store = VectorStore()    # Tool to store the text in a searchable way
query_processor = QueryProcessor(vector_store)  # Tool to find and answer questions

# Set the path to your PDF file
file_path = "/home/yuk/Documents/Task/pdfs/ML BOOk.pdf"

# Check if the PDF file is there
if os.path.exists(file_path):
    print(f"Processing file: {file_path}")
    # Read the PDF and get its pages with text
    pages = processor.process(file_path)
    # If we got some pages, save them
    if pages:
        # Create a unique name for the document
        doc_id = str(uuid.uuid4())
        # Save the pages to our storage (ChromaDB)
        vector_store.adding_documents(doc_id, pages)
        print(f"Saved document {doc_id} with {len(pages)} pages")
    else:
        # If no pages were read, stop the program
        print("No pages processed. Check the PDF.")
        exit()
else:
    # If the PDF file isn’t found, stop the program
    print(f"File not found: {file_path}")
    exit()

# Show what documents we saved in ChromaDB
print("\nStored Documents:")
documents = vector_store.list_documents()
if documents:
    # Print details for each saved page
    for doc in documents:
        print(f"ID: {doc['id']}, Document: {doc['doc_id']}, Page: {doc['page']}, Text: {doc['text']}")
else:
    print("No documents found in ChromaDB.")

# Ask the question we want to answer
query = "Explain about Multioutput Classification"
print(f"\nAsking: {query}")
# Get the answers from the model
answers = query_processor.process_query(query)
# Show the answers
query_processor.display_answers(answers)