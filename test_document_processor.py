import sys
import os

# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

from chatbot_theme_identifier.backend.document_processor import DocumentProcessor

# Create processor
processor = DocumentProcessor()

# Test one file
file_path = "/home/yuk/Documents/Task/pdfs/ML BOOk.pdf"  # Update to your file

print(f"Testing file: {file_path}")
if os.path.exists(file_path):
    results = processor.process(file_path)
    print("Results:")
    if results:  # Check if results is not empty
        for page in results:
            print(f"Page {page['page']}: {page['text'][:50]}...")
    else:
        print("No results returned (empty list).")
else:
    print(f"File not found: {file_path}")