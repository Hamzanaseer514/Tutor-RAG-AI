import faiss
import numpy as np
import pickle
import os
from sentence_transformers import SentenceTransformer

class VectorStoreFAISS:
    def __init__(self, persist_directory: str = "./faiss_db"):
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.persist_directory = persist_directory
        self.index = None
        self.documents = []
        
        # Create directory if not exists
        os.makedirs(persist_directory, exist_ok=True)
        
        # Try to load existing index
        self._load_index()
    
    def _load_index(self):
        index_path = os.path.join(self.persist_directory, "index.faiss")
        docs_path = os.path.join(self.persist_directory, "documents.pkl")
        
        if os.path.exists(index_path) and os.path.exists(docs_path):
            self.index = faiss.read_index(index_path)
            with open(docs_path, 'rb') as f:
                self.documents = pickle.load(f)
        else:
            # Create new index
            self.index = faiss.IndexFlatL2(384)  # 384 is the dimension of all-MiniLM-L6-v2
            self.documents = []
    
    def _save_index(self):
        index_path = os.path.join(self.persist_directory, "index.faiss")
        docs_path = os.path.join(self.persist_directory, "documents.pkl")
        
        faiss.write_index(self.index, index_path)
        with open(docs_path, 'wb') as f:
            pickle.dump(self.documents, f)
    
    def add_document(self, doc_id: str, chunks: list):
        embeddings = self.embedding_model.encode(chunks)
        
        # Add to index
        if self.index.ntotal == 0:
            self.index.add(embeddings.astype('float32'))
        else:
            self.index.add(embeddings.astype('float32'))
        
        # Store documents with their IDs
        for i, chunk in enumerate(chunks):
            self.documents.append({
                'id': f"{doc_id}_{i}",
                'content': chunk,
                'doc_id': doc_id
            })
        
        self._save_index()
    
    def search(self, query: str, top_k: int = 5):
        query_embedding = self.embedding_model.encode([query]).astype('float32')
        
        # Search in index
        distances, indices = self.index.search(query_embedding, top_k)
        
        # Get relevant documents
        results = []
        for i, idx in enumerate(indices[0]):
            if idx < len(self.documents):
                results.append((
                    self.documents[idx]['content'],
                    float(distances[0][i])
                ))
        
        return results