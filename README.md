# Research Chatbot

**Research Chatbot** is a powerful system designed to process documents, enable semantic search, answer user queries, and identify key themes in responses. By integrating technologies like OCR, vector databases, and advanced language models, it provides an efficient solution for analyzing and understanding large document collections.

**Note**: Use Python 3.10. The files which have the names "test.py" contains scripts to test each backend module.

## System Components

- **DocumentProcessor**: Manages text extraction from PDFs and images, ensuring compatibility with diverse inputs.  
- **VectorStore**: Handles storage and retrieval of document embeddings in ChromaDB for efficient search functionality.  
- **QueryProcessor**: Searches the vector store and uses the Phi-3-mini model to deliver accurate query responses.  
- **ThemeIdentifier**: Extracts and summarizes key themes from responses, adding deeper analytical value.  

## Technologies Used

- **PyPDF2**: Library for extracting text from PDF files.  
- **PaddleOCR**: Advanced OCR tool for text extraction from images.  
- **ChromaDB**: Vector database for storing and searching text embeddings.  
- **SentenceTransformer**: Generates embeddings for semantic search capabilities.  
- **Phi-3-mini Language Model**: Powers query answering and theme identification with natural language understanding.  

## How It Works

1. Upload PDFs or images to the system.  
2. The DocumentProcessor extracts text using PyPDF2 and PaddleOCR.  
3. Text is converted into embeddings and stored in ChromaDB via the VectorStore.  
4. Users submit queries, which the QueryProcessor answers using the Phi-3-mini model.  
5. The ThemeIdentifier analyzes responses to highlight recurring themes and insights.  
