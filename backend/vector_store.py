# Importing libraries needed for managing a vector store and creating embeddings
# chromadb: Library for creating and managing a vector database to store and search text embeddings
# SentenceTransformer: Library for generating embeddings (numerical representations) of text
# uuid: Library for generating unique identifiers for documents


import chromadb
from sentence_transformers import SentenceTransformer
import uuid


# Defining a class called VectorStore to manage a collection of documents and search them

class VectorStore:
    
    # Constructor method that runs when a new VectorStore object is created
    # It initializes a vector database and an embedding model
    # collection_name: Name of the collection in the vector database (default is "documents")
    
    
    def __init__(self, collection_name="documents"):
        
        
        self.client = chromadb.Client()  # Creating a new client to interact with the Chroma vector database
        # Creating or retrieving a collection (like a table) in the database with the given name
        self.collection = self.client.get_or_create_collection(collection_name)
        # Loading a pre-trained model to generate embeddings for text
        self.embedder = SentenceTransformer("all-MiniLM-L6-v2")


    # Method to add documents to the vector store
    # It takes a document ID (string) and a list of pages as input
    
    
    def adding_documents(self, doc_id: str, pages: list):
        try:
            # Looping through each page in the list of pages
            
            for page in pages:
                page_num = page['page']  # Getting the page number from the page data
                text = page['text']  # Getting the text content from the page data
                
                
                # Checking if the text is not empty after removing spaces
                if text.strip():
                    # Printing a message to show that the page is being saved
                    print(f"Saving page {page_num} of document {doc_id}")
                    
                    # Converting the text into an embedding (numerical representation) using the model
                    embedding = self.embedder.encode(text).tolist()
                    
                    # Adding the text, embedding, and metadata to the vector store collection
                    self.collection.add(
                        ids=[f"{doc_id}_page_{page_num}"],  # Unique ID for the page (e.g., "doc1_page_1")
                        documents=[text],  # The text content of the page
                        embeddings=[embedding],  # The numerical embedding of the text
                        metadatas=[{"doc_id": doc_id, "page": page_num}]  # Metadata including document ID and page number
                    )
        except Exception as e:
            # Printing any errors that occur while adding documents
            print(f"Error adding documents to vector store: {e}")


    # Method to search the vector store for documents matching a query
    # It takes a query (string) and a maximum number of results (default is 5) as input
    # Returns a list of matching documents
    
    
    def search(self, query: str, max_results: int = 5) -> list:
        
        
        try:
            # Ensuring the query is a single string
            if isinstance(query, list):  # Checking if the query is a list
                # If the query is a list and its first element is a string, use that string
                if query and isinstance(query[0], str):
                    query = query[0]
                else:
                    # Raising an error if the query list is empty or doesn't contain a string
                    raise ValueError("Query must be a string or a list containing a single string")
            # Raising an error if the query is not a string
            if not isinstance(query, str):
                raise ValueError("Query must be a string")

            # Converting the query into an embedding (numerical representation) using the model
            query_embedding = self.embedder.encode(query).tolist()
            
            # Searching the vector store for documents matching the query embedding
            results = self.collection.query(
                query_embeddings=[query_embedding],  # The embedding of the query
                n_results=max_results  # Maximum number of results to return
            )
            
            
            # Creating a list to store the search matches
            matches = []
            # Looping through each result to format it into a dictionary
            for i in range(len(results['ids'][0])):
                match = {
                    "doc_id": results['metadatas'][0][i]['doc_id'],  # Document ID from metadata
                    "page": results['metadatas'][0][i]['page'],  # Page number from metadata
                    "text": results['documents'][0][i],  # The text of the matching document
                    "distance": results['distances'][0][i]  # Distance score indicating how close the match is
                }
                matches.append(match)  # Adding the match to the list
            return matches  # Returning the list of matches
        except Exception as e:
            # Printing any errors that occur during the search
            print(f"Error searching vector store: {e}")
            return []  # Returning an empty list if the search fails


    # Method to list all documents stored in the vector store
    # It returns a list of documents with their details
    
    
    def list_documents(self) -> list:
        try:
            # Getting all documents from the vector store collection
            results = self.collection.get()
            # Creating a list to store the document details
            documents = []
            # Looping through each document to format its details
            for i in range(len(results["ids"])):
                document = {
                    "id": results["ids"][i],  # Unique ID of the document
                    "doc_id": results["metadatas"][i]["doc_id"],  # Document ID from metadata
                    "page": results["metadatas"][i]["page"],  # Page number from metadata
                    # Taking the first 50 characters of the text, adding "..." if longer
                    "text": results["documents"][i][:50] + "..." if results["documents"][i] else ""
                }
                documents.append(document)  # Adding the document to the list
            return documents  # Returning the list of documents
        
        
        except Exception as e:
            # Printing any errors that occur while listing documents
            print(f"Error listing documents: {e}")
            return []  # Returning an empty list if listing fails
