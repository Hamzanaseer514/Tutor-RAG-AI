import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import numpy as np
import logging
from typing import List, Tuple, Dict
import gc
import hashlib

logger = logging.getLogger(__name__)

class VectorStore:
    def __init__(self, persist_directory: str = "./chroma_db"):
        logger.info("Initializing VectorStore")
        try:
            self.client = chromadb.PersistentClient(path=persist_directory)
            # Use a better embedding model for improved search quality
            self.embedding_model = SentenceTransformer('all-mpnet-base-v2')
            self.collection = self.client.get_or_create_collection(
                name="documents",
                metadata={"hnsw:space": "cosine"}
            )
            logger.info("VectorStore initialized with all-mpnet-base-v2 model")
        except Exception as e:
            logger.error(f"Failed to initialize VectorStore: {str(e)}")
            raise
    
    def add_document(self, doc_id: str, chunks: list):
        if not chunks:
            logger.error("No chunks to add to vector store")
            raise ValueError("No text chunks were extracted from the PDF")
        
        logger.info(f"Adding document {doc_id} with {len(chunks)} chunks")
        
        try:
            # Process chunks in larger batches for speed
            batch_size = 10  # Increased from 5 to 10 for faster processing
            total_batches = (len(chunks) + batch_size - 1) // batch_size
            logger.info(f"Processing {total_batches} batches with batch size {batch_size}")
            
            for i in range(0, len(chunks), batch_size):
                batch = chunks[i:i + batch_size]
                batch_num = i // batch_size + 1
                logger.info(f"Processing batch {batch_num}/{total_batches} ({len(batch)} chunks)")
                
                # Create metadata for each chunk
                metadatas = []
                for j, chunk in enumerate(batch):
                    metadata = {
                        "doc_id": doc_id,
                        "chunk_index": i + j,
                        "chunk_length": len(chunk),
                        "chunk_hash": hashlib.md5(chunk.encode()).hexdigest()[:8]
                    }
                    metadatas.append(metadata)
                
                logger.info(f"Generated metadata for batch {batch_num}")
                
                # Generate embeddings
                logger.info(f"Generating embeddings for batch {batch_num}...")
                try:
                    embeddings = self.embedding_model.encode(batch, show_progress_bar=False).tolist()
                    logger.info(f"Generated embeddings for batch {batch_num} successfully")
                except Exception as e:
                    logger.error(f"Failed to generate embeddings for batch {batch_num}: {str(e)}")
                    raise
                
                # Create unique IDs for each chunk
                chunk_ids = [f"{doc_id}_{i + j}" for j in range(len(batch))]
                logger.info(f"Created chunk IDs for batch {batch_num}")
                
                # Add to ChromaDB
                logger.info(f"Adding batch {batch_num} to ChromaDB...")
                try:
                    self.collection.add(
                        embeddings=embeddings,
                        documents=batch,
                        metadatas=metadatas,
                        ids=chunk_ids
                    )
                    logger.info(f"Successfully added batch {batch_num} to ChromaDB")
                except Exception as e:
                    logger.error(f"Failed to add batch {batch_num} to ChromaDB: {str(e)}")
                    raise
                
                # Clean up batch data
                del embeddings, batch, metadatas, chunk_ids
                logger.info(f"Cleaned up batch {batch_num} data")
            
            logger.info("Document added to vector store successfully")
        except Exception as e:
            logger.error(f"Vector store error: {str(e)}", exc_info=True)
            raise ValueError(f"Failed to process vector store: {str(e)}")

    def search(self, query: str, top_k: int = 5, doc_id: str = None) -> List[Tuple[str, float, Dict]]:
        logger.info(f"Searching for: '{query}' with top_k={top_k}")
        
        if self.collection.count() == 0:
            logger.warning("No documents in vector store to search from")
            return [("No documents have been processed yet. Please upload a PDF first.", 1.0, {})]
        
        try:
            # Generate query embedding
            query_embedding = self.embedding_model.encode([query]).tolist()
            logger.info("Query embedding generated")
            
            # Prepare search parameters
            search_params = {
                "query_embeddings": query_embedding,
                "n_results": min(top_k, self.collection.count())
            }
            
            # Add document filter if specified
            if doc_id:
                search_params["where"] = {"doc_id": doc_id}
                logger.info(f"Filtering search to document: {doc_id}")
            
            # Perform search
            results = self.collection.query(**search_params)
            logger.info(f"Search returned {len(results['documents'][0])} results")
            
            documents = results['documents'][0]
            distances = results['distances'][0] if 'distances' in results else [0] * len(documents)
            metadatas = results['metadatas'][0] if 'metadatas' in results else [{}] * len(documents)
            
            # Convert distances to similarity scores (cosine similarity)
            similarities = [1 - dist for dist in distances]
            
            # Log results
            for i, (doc, sim, meta) in enumerate(zip(documents, similarities, metadatas)):
                logger.info(f"Result {i+1} (similarity: {sim:.4f}): {doc[:100]}...")
                if meta:
                    logger.info(f"  Metadata: {meta}")
            
            # Return results with metadata
            return list(zip(documents, similarities, metadatas))
            
        except Exception as e:
            logger.error(f"ChromaDB search failed: {str(e)}")
            raise ValueError(f"Failed to search vector store: {str(e)}")
    
    def get_document_chunks(self, doc_id: str) -> dict:
        """Get information about chunks for a specific document"""
        try:
            # Get all chunks for this document
            results = self.collection.get(
                where={"doc_id": doc_id},
                include=["metadatas", "documents"]
            )
            
            if not results['ids']:
                return {"chunk_count": 0, "chunks": []}
            
            chunks_info = []
            for i, (chunk_id, metadata, document) in enumerate(zip(results['ids'], results['metadatas'], results['documents'])):
                chunks_info.append({
                    "chunk_id": chunk_id,
                    "chunk_index": metadata.get("chunk_index", i),
                    "chunk_length": metadata.get("chunk_length", len(document)),
                    "preview": document[:100] + "..." if len(document) > 100 else document
                })
            
            return {
                "chunk_count": len(chunks_info),
                "chunks": chunks_info,
                "total_characters": sum(chunk["chunk_length"] for chunk in chunks_info)
            }
            
        except Exception as e:
            logger.error(f"Failed to get document chunks: {str(e)}")
            return {"error": str(e)}

    def get_document_stats(self) -> dict:
        """Get overall statistics about the vector store"""
        try:
            # Get all documents
            results = self.collection.get(include=["metadatas"])
            
            if not results['ids']:
                return {"total_documents": 0, "total_chunks": 0, "total_characters": 0}
            
            # Group by document
            doc_stats = {}
            for metadata in results['metadatas']:
                doc_id = metadata.get("doc_id", "unknown")
                if doc_id not in doc_stats:
                    doc_stats[doc_id] = {
                        "chunk_count": 0,
                        "total_chars": 0
                    }
                doc_stats[doc_id]["chunk_count"] += 1
                doc_stats[doc_id]["total_chars"] += metadata.get("chunk_length", 0)
            
            total_chunks = sum(stats["chunk_count"] for stats in doc_stats.values())
            total_chars = sum(stats["total_chars"] for stats in doc_stats.values())
            
            return {
                "total_documents": len(doc_stats),
                "total_chunks": total_chunks,
                "total_characters": total_chars,
                "documents": [
                    {
                        "doc_id": doc_id,
                        "chunk_count": stats["chunk_count"],
                        "total_chars": stats["total_chars"]
                    }
                    for doc_id, stats in doc_stats.items()
                ]
            }
            
        except Exception as e:
            logger.error(f"Failed to get document stats: {str(e)}")
            return {"error": str(e)}
    
    def delete_document(self, doc_id: str) -> bool:
        """Delete all chunks for a specific document"""
        try:
            # Get all chunk IDs for the document
            results = self.collection.get(
                where={"doc_id": doc_id},
                include=["ids"]
            )
            
            if results['ids']:
                self.collection.delete(ids=results['ids'])
                logger.info(f"Deleted {len(results['ids'])} chunks for document {doc_id}")
                return True
            else:
                logger.warning(f"No chunks found for document {doc_id}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to delete document: {str(e)}")
            return False
    
    def get_stats(self) -> Dict:
        """Get vector store statistics"""
        try:
            total_chunks = self.collection.count()
            
            # Get unique document IDs
            if total_chunks > 0:
                results = self.collection.get(include=["metadatas"])
                doc_ids = set(meta.get('doc_id') for meta in results['metadatas'] if meta)
                unique_docs = len(doc_ids)
            else:
                unique_docs = 0
            
            return {
                "total_chunks": total_chunks,
                "unique_documents": unique_docs,
                "embedding_model": "all-mpnet-base-v2"
            }
            
        except Exception as e:
            logger.error(f"Failed to get stats: {str(e)}")
            return {"error": str(e)}