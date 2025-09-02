import os
import uuid
import logging
import asyncio
from datetime import datetime
from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, Request, Form
from fastapi.responses import JSONResponse, HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session
from typing import List, Optional
from dotenv import load_dotenv
import PyPDF2
import psutil
import gc
import json

# Configure real-time logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s.%(msecs)03d %(levelname)s:%(name)s:%(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
from app import models, schemas, database
from app.pdf_processor import PDFProcessor  # Use absolute import
from app.vector_store import VectorStore
from app.grok_integration import DeepSeekIntegration

app = FastAPI(
    title="Tuterby RAG System",
    version="1.0.0",
    docs_url=None,
    redoc_url=None
)

@app.on_event("startup")
def startup_event():
    logger.info("Initializing database tables...")
    models.Base.metadata.create_all(bind=database.engine)
    logger.info("Database tables initialized!")

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Initialize with smaller chunk sizes for better memory management
pdf_processor = PDFProcessor(chunk_size=300, chunk_overlap=50)
vector_store = VectorStore()
deepseek_integration = DeepSeekIntegration()

def get_db():
    db = database.SessionLocal()
    try:
        yield db
    finally:
        db.close()

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/upload-pdf/")
async def upload_pdf(
    file: UploadFile = File(...),
    password: Optional[str] = Form(None),
    db: Session = Depends(get_db)
):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed")
    
    # Store filename before reading file content
    original_filename = file.filename
    doc_id = str(uuid.uuid4())
    file_path = f"temp_{doc_id}.pdf"
    
    try:
        # Log memory usage
        process = psutil.Process()
        mem_before = process.memory_info().rss / 1024 / 1024
        logger.info(f"Memory usage before processing: {mem_before:.2f} MB")
        
        logger.info(f"Starting file save for {original_filename}")
        file_size = 0
        with open(file_path, "wb") as f:
            while True:
                chunk = await file.read(8192)
                if not chunk:
                    break
                f.write(chunk)
                file_size += len(chunk)
        
        # Reduce file size limit to prevent memory issues
        if file_size > 25 * 1024 * 1024:  # 25MB limit instead of 50MB
            raise HTTPException(status_code=413, detail="File too large (max 25MB)")
        
        logger.info(f"File saved: {file_path}, size: {file_size} bytes")
        
        # Validate PDF
        logger.info("Validating PDF...")
        with open(file_path, 'rb') as pdf_file:
            try:
                pdf_reader = PyPDF2.PdfReader(pdf_file, strict=True)
                # Check page count to prevent memory issues
                if len(pdf_reader.pages) > 100:
                    raise HTTPException(
                        status_code=400, 
                        detail="PDF too large (max 100 pages). Please use a smaller document."
                    )
                logger.info("PDF validation successful")
            except Exception as e:
                logger.error(f"Invalid PDF: {str(e)}")
                raise HTTPException(status_code=400, detail=f"Invalid or corrupted PDF file: {str(e)}")
        
        # Process PDF with memory management
        logger.info("Starting PDF text extraction...")
        try:
            logger.info("Calling PDF processor...")
            text_chunks = pdf_processor.process_pdf(file_path, password)
            logger.info(f"PDF processor returned {len(text_chunks)} chunks")
            
            if text_chunks:
                logger.info("Text extraction completed successfully")
            else:
                logger.warning("PDF processor returned empty chunks list")
                
        except MemoryError:
            logger.error("Memory error during PDF processing")
            raise HTTPException(
                status_code=500, 
                detail="PDF too large to process. Please try a smaller document or contact support."
            )
        except Exception as e:
            logger.error(f"PDF processing error: {str(e)}", exc_info=True)
            raise HTTPException(
                status_code=500, 
                detail=f"PDF processing failed: {str(e)}"
            )
        
        if not text_chunks:
            raise HTTPException(
                status_code=400,
                detail="Could not extract text from PDF. The PDF might be scanned, protected, or corrupted."
            )
        
        logger.info(f"Successfully extracted {len(text_chunks)} text chunks")
        
        # Log memory usage
        mem_after_extraction = process.memory_info().rss / 1024 / 1024
        logger.info(f"Memory usage after text extraction: {mem_after_extraction:.2f} MB")
        
        # Add to vector store
        logger.info("Starting vector store processing...")
        try:
            logger.info("Calling vector store add_document...")
            vector_store.add_document(doc_id, text_chunks)
            logger.info("Vector store processing completed successfully")
        except Exception as e:
            logger.error(f"Vector store error: {str(e)}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Failed to store document in vector database: {str(e)}")
        
        # Log memory usage
        mem_after_vector = process.memory_info().rss / 1024 / 1024
        logger.info(f"Memory usage after vector store: {mem_after_vector:.2f} MB")
        
        # Store metadata in SQL database
        logger.info("Saving metadata to database...")
        try:
            db_document = models.Document(
                id=doc_id,
                filename=original_filename,  # Use stored filename
                upload_date=datetime.utcnow(),
                chunk_count=len(text_chunks)
            )
            db.add(db_document)
            db.commit()
            logger.info("Database metadata saved successfully")
        except Exception as e:
            logger.error(f"Database error: {str(e)}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Failed to save document metadata: {str(e)}")
        
        # Clean up
        logger.info(f"Cleaning up temporary file: {file_path}")
        os.remove(file_path)
        
        # Force garbage collection
        gc.collect()
        
        # Log final memory usage
        mem_final = process.memory_info().rss / 1024 / 1024
        logger.info(f"Memory usage after processing: {mem_final:.2f} MB")
        
        logger.info("PDF processing completed successfully")
        return JSONResponse(
            status_code=200,
            content={"message": "PDF processed successfully", "document_id": doc_id}
        )
    except ValueError as ve:
        if os.path.exists(file_path):
            os.remove(file_path)
        logger.error(f"ValueError processing PDF: {str(ve)}")
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        if os.path.exists(file_path):
            os.remove(file_path)
        logger.error(f"Unexpected error processing PDF: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Unexpected error processing PDF: {str(e)}")

@app.get("/conversations/{conversation_id}")
async def get_conversation(
    conversation_id: str,
    db: Session = Depends(get_db)
):
    messages = db.query(models.Conversation).filter(
        models.Conversation.conversation_id == conversation_id
    ).order_by(models.Conversation.timestamp).all()
    
    return {
        "conversation_id": conversation_id,
        "messages": [
            {
                "message": msg.message,
                "is_user": msg.is_user,
                "timestamp": msg.timestamp.isoformat()
            }
            for msg in messages
        ]
    }

@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.utcnow().isoformat()}

@app.post("/query/")
async def query_document(
    request: Request,
    db: Session = Depends(get_db)
):
    try:
        body = await request.json()
        question = body.get("question", "").strip()
        conversation_id = body.get("conversation_id")
        
        if not question:
            raise HTTPException(status_code=400, detail="Question is required")
        
        logger.info(f"Processing question: {question}")
        
        # Search for relevant context in vector store
        try:
            search_results = vector_store.search(question, top_k=3)
            logger.info(f"Found {len(search_results)} relevant chunks")
        except Exception as e:
            logger.error(f"Vector search failed: {str(e)}")
            raise HTTPException(status_code=500, detail="Failed to search document content")
        
        # Prepare context from search results
        if search_results and search_results[0][0] != "No documents have been processed yet. Please upload a PDF first.":
            # Extract chunks from the new tuple format (chunk, similarity, metadata)
            context = "\n\n".join([chunk for chunk, _, _ in search_results])
            logger.info(f"Context length: {len(context)} characters")
        else:
            context = "No relevant documents found. Please upload a PDF first."
            logger.info("No documents available for context")
        
        # Generate response using Grok AI
        try:
            # Prepare conversation history
            history = []
            if conversation_id:
                db_messages = db.query(models.Conversation).filter(
                    models.Conversation.conversation_id == conversation_id
                ).order_by(models.Conversation.timestamp).limit(10).all()
                
                for msg in db_messages:
                    role = "user" if msg.is_user else "assistant"
                    history.append({"role": role, "content": msg.message})
            
            response = deepseek_integration.generate_response(question, context, history)
            logger.info(f"Generated response: {len(response)} characters")
            
        except Exception as e:
            logger.error(f"Grok AI generation failed: {str(e)}")
            response = "I apologize, but I'm having trouble generating a response right now. Please try again later."
        
        # Store conversation in database
        if not conversation_id:
            conversation_id = str(uuid.uuid4())
        
        try:
            # Store user question
            user_msg = models.Conversation(
                conversation_id=conversation_id,
                message=question,
                is_user=True,
                timestamp=datetime.utcnow()
            )
            db.add(user_msg)
            
            # Store AI response
            ai_msg = models.Conversation(
                conversation_id=conversation_id,
                message=response,
                is_user=False,
                timestamp=datetime.utcnow()
            )
            db.add(ai_msg)
            
            db.commit()
            logger.info(f"Conversation stored with ID: {conversation_id}")
            
        except Exception as e:
            logger.error(f"Failed to store conversation: {str(e)}")
            # Don't fail the request if conversation storage fails
        
        return JSONResponse(
            status_code=200,
            content={
                "response": response,
                "conversation_id": conversation_id,
                "context_used": len(context) if context != "No relevant documents found. Please upload a PDF first." else 0
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in query: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")

@app.post("/query-stream/")
async def query_document_stream(
    request: Request,
    db: Session = Depends(get_db)
):
    """Streaming endpoint for real-time chat responses with typing effect"""
    try:
        body = await request.json()
        question = body.get("question", "").strip()
        conversation_id = body.get("conversation_id")
        
        if not question:
            raise HTTPException(status_code=400, detail="Question is required")
        
        logger.info(f"Processing streaming question: {question}")
        
        # Search for relevant context in vector store
        try:
            search_results = vector_store.search(question, top_k=3)
            logger.info(f"Found {len(search_results)} relevant chunks")
        except Exception as e:
            logger.error(f"Vector search failed: {str(e)}")
            raise HTTPException(status_code=500, detail="Failed to search document content")
        
        # Prepare context from search results
        if search_results and search_results[0][0] != "No documents have been processed yet. Please upload a PDF first.":
            context = "\n\n".join([chunk for chunk, _, _ in search_results])
            logger.info(f"Context length: {len(context)} characters")
        else:
            context = "No relevant documents found. Please upload a PDF first."
            logger.info("No documents available for context")
        
        # Store conversation in database
        if not conversation_id:
            conversation_id = str(uuid.uuid4())
        
        try:
            # Store user question
            user_msg = models.Conversation(
                conversation_id=conversation_id,
                message=question,
                is_user=True,
                timestamp=datetime.utcnow()
            )
            db.add(user_msg)
            db.commit()
            logger.info(f"User question stored with ID: {conversation_id}")
            
        except Exception as e:
            logger.error(f"Failed to store user question: {str(e)}")
        
        async def generate_stream():
            """Generate streaming response with real-time typing effect"""
            # Send conversation ID first
            yield f"data: {json.dumps({'type': 'conversation_id', 'conversation_id': conversation_id})}\n\n"
            
            # Send typing indicator
            yield f"data: {json.dumps({'type': 'typing', 'status': 'start'})}\n\n"
            
            # Simulate thinking delay
            await asyncio.sleep(0.3)
            
            try:
                # Generate response using Grok AI
                history = []
                if conversation_id:
                    db_messages = db.query(models.Conversation).filter(
                        models.Conversation.conversation_id == conversation_id
                    ).order_by(models.Conversation.timestamp).limit(10).all()
                    
                    for msg in db_messages:
                        role = "user" if msg.is_user else "assistant"
                        history.append({"role": role, "content": msg.message})
                
                response = await deepseek_integration.generate_response_async(question, context, history)
                logger.info(f"Generated streaming response: {len(response)} characters")
                
                # Split response into words for smooth typing
                words = response.split()
                
                # Send words one by one for realistic typing effect
                for i, word in enumerate(words):
                    # Add space after each word except the last one
                    if i < len(words) - 1:
                        chunk = word + " "
                    else:
                        chunk = word
                    
                    yield f"data: {json.dumps({'type': 'content', 'chunk': chunk})}\n\n"
                    
                    # Smooth typing speed - faster for better UX
                    await asyncio.sleep(0.05)  # 50ms between words
                
                # Send typing end
                yield f"data: {json.dumps({'type': 'typing', 'status': 'end'})}\n\n"
                
                # Send complete response
                yield f"data: {json.dumps({'type': 'complete', 'response': response})}\n\n"
                
                # Store AI response in database
                try:
                    ai_msg = models.Conversation(
                        conversation_id=conversation_id,
                        message=response,
                        is_user=False,
                        timestamp=datetime.utcnow()
                    )
                    db.add(ai_msg)
                    db.commit()
                    logger.info(f"AI response stored for conversation: {conversation_id}")
                except Exception as e:
                    logger.error(f"Failed to store AI response: {str(e)}")
                
            except Exception as e:
                logger.error(f"Grok AI generation failed: {str(e)}")
                error_response = "I apologize, but I'm having trouble generating a response right now. Please try again later."
                
                # Send error response
                yield f"data: {json.dumps({'type': 'content', 'chunk': error_response})}\n\n"
                yield f"data: {json.dumps({'type': 'typing', 'status': 'end'})}\n\n"
                yield f"data: {json.dumps({'type': 'complete', 'response': error_response})}\n\n"
                
                # Store error response in database
                try:
                    ai_msg = models.Conversation(
                        conversation_id=conversation_id,
                        message=error_response,
                        is_user=False,
                        timestamp=datetime.utcnow()
                    )
                    db.add(ai_msg)
                    db.commit()
                except Exception as db_error:
                    logger.error(f"Failed to store error response: {str(db_error)}")
        
        return StreamingResponse(
            generate_stream(),
            media_type="text/plain",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "Content-Type": "text/event-stream"
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in streaming query: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")

@app.get("/documents/")
async def list_documents(db: Session = Depends(get_db)):
    """Get list of all uploaded documents"""
    try:
        documents = db.query(models.Document).order_by(models.Document.upload_date.desc()).all()
        return {
            "documents": [
                {
                    "id": doc.id,
                    "filename": doc.filename,
                    "upload_date": doc.upload_date.isoformat(),
                    "chunk_count": doc.chunk_count,
                    "status": "processed"
                }
                for doc in documents
            ],
            "total_count": len(documents)
        }
    except Exception as e:
        logger.error(f"Failed to fetch documents: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to fetch documents")

@app.delete("/documents/{document_id}")
async def delete_document(document_id: str, db: Session = Depends(get_db)):
    """Delete a document and its data"""
    try:
        # Delete from database
        document = db.query(models.Document).filter(models.Document.id == document_id).first()
        if not document:
            raise HTTPException(status_code=404, detail="Document not found")
        
        # Delete from vector store
        try:
            vector_store.delete_document(document_id)
            logger.info(f"Deleted document {document_id} from vector store")
        except Exception as e:
            logger.warning(f"Failed to delete from vector store: {str(e)}")
        
        # Delete from database
        db.delete(document)
        db.commit()
        
        return {"message": "Document deleted successfully"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete document: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to delete document")

@app.get("/documents/{document_id}/info")
async def get_document_info(document_id: str, db: Session = Depends(get_db)):
    """Get detailed information about a specific document"""
    try:
        document = db.query(models.Document).filter(models.Document.id == document_id).first()
        if not document:
            raise HTTPException(status_code=404, detail="Document not found")
        
        # Get vector store stats for this document
        try:
            stats = vector_store.get_document_chunks(document_id)
        except Exception as e:
            stats = {"error": str(e)}
        
        return {
            "id": document.id,
            "filename": document.filename,
            "upload_date": document.upload_date.isoformat(),
            "chunk_count": document.chunk_count,
            "vector_store_info": stats
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get document info: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to get document info")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)