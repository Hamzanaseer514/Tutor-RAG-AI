from sqlalchemy import Column, String, DateTime, Boolean, Integer
from .database import Base

class Document(Base):
    __tablename__ = "documents"
    
    id = Column(String, primary_key=True, index=True)
    filename = Column(String)
    upload_date = Column(DateTime)
    chunk_count = Column(Integer)

class Conversation(Base):
    __tablename__ = "conversations"
    
    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    conversation_id = Column(String, index=True)
    message = Column(String)
    is_user = Column(Boolean)
    timestamp = Column(DateTime)