import chromadb
from sentence_transformers import SentenceTransformer
import uuid

class VectorStore:
    
    def __init__(self, collection_name="documents"):
        self.client = chromadb.Client()
        self.collection = self.client.get_or_create_collection(collection_name)
        self.embedder = SentenceTransformer("all-MiniLM-L6-v2")
        
    def adding_documents(self, doc_id: str, pages: list):
        try:
            for page in pages:
                page_num = page['page']
                text = page['text']
                if text.strip():
                    print(f"Saving page {page_num} of document {doc_id}")
                    embedding = self.embedder.encode(text).tolist()
                    self.collection.add(
                        ids=[f"{doc_id}_page_{page_num}"],
                        documents=[text],
                        embeddings=[embedding],
                        metadatas=[{"doc_id": doc_id, "page": page_num}]
                    )
        except Exception as e:
            print(f"Error adding documents to vector store: {e}")
            
    def search(self, query: str, max_results: int = 5) -> list:
        try:
            # Ensure query is a single string
            if isinstance(query, list):
                # If query is a list, take the first element if it’s a string, otherwise raise an error
                if query and isinstance(query[0], str):
                    query = query[0]
                else:
                    raise ValueError("Query must be a string or a list containing a single string")
            if not isinstance(query, str):
                raise ValueError("Query must be a string")

            # Encode the query
            query_embedding = self.embedder.encode(query).tolist()
            
            # Search the vector store
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=max_results
            )
            
            matches = []
            for i in range(len(results['ids'][0])):
                match = {
                    "doc_id": results['metadatas'][0][i]['doc_id'],
                    "page": results['metadatas'][0][i]['page'],
                    "text": results['documents'][0][i],
                    "distance": results['distances'][0][i]
                }
                matches.append(match)
            return matches
        except Exception as e:
            print(f"Error searching vector store: {e}")
            return []
        
    def list_documents(self) -> list:
        try:
            results = self.collection.get()
            documents = []
            for i in range(len(results["ids"])):
                document = {
                    "id": results["ids"][i],
                    "doc_id": results["metadatas"][i]["doc_id"],
                    "page": results["metadatas"][i]["page"],
                    "text": results["documents"][i][:50] + "..." if results["documents"][i] else ""
                }
                documents.append(document)
            return documents
        except Exception as e:
            print(f"Error listing documents: {e}")
            return []