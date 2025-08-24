# 🚀 Tuterby AI - Professional Company Knowledge Base

**Transform your company documents into an intelligent, searchable knowledge base powered by AI.**

## ✨ **What is Tuterby AI?**

Tuterby AI is a professional-grade **Retrieval-Augmented Generation (RAG)** system that transforms your company's PDF documents into an intelligent knowledge base. It allows employees to ask questions in natural language and get instant, accurate answers based on your company's documentation.

## 🎯 **Perfect For:**

- **HR Departments** - Company policies, employee handbooks, procedures
- **Technical Teams** - API documentation, technical manuals, troubleshooting guides
- **Training Departments** - Training materials, onboarding documents, SOPs
- **Legal Teams** - Contracts, compliance documents, legal procedures
- **Sales Teams** - Product catalogs, pricing sheets, sales materials
- **Any Business** - Standard operating procedures, company knowledge

## 🏗️ **System Architecture**

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Frontend UI   │    │   FastAPI       │    │   Vector Store  │
│   (React/HTML)  │◄──►│   Backend       │◄──►│   (ChromaDB)    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                │
                                ▼
                       ┌─────────────────┐
                       │   Grok AI       │
                       │   (OpenRouter)  │
                       └─────────────────┘
```

## 🚀 **Key Features**

### 📚 **Document Management**
- **Persistent Storage** - Documents stay available after upload
- **Multiple Formats** - Supports text-based and scanned PDFs
- **Password Protection** - Handles encrypted PDFs
- **Batch Processing** - Efficient handling of large documents
- **Document Analytics** - Track document usage and performance

### 🤖 **AI-Powered Chat**
- **Natural Language Queries** - Ask questions in plain English
- **Context-Aware Responses** - AI understands document context
- **Conversation Memory** - Maintains chat history
- **Multilingual Support** - Works with English and Urdu text
- **Real-time Processing** - Instant responses

### 🎨 **Professional Interface**
- **Modern UI/UX** - Clean, professional design
- **Responsive Design** - Works on all devices
- **Drag & Drop** - Easy file uploads
- **Real-time Updates** - Live progress indicators
- **Professional Branding** - Ready for company use

### 🔒 **Enterprise Features**
- **Secure Processing** - Local document processing
- **Memory Management** - Optimized for large files
- **Error Handling** - Robust error management
- **Logging & Monitoring** - Comprehensive system logs
- **Scalable Architecture** - Ready for production use

## 🛠️ **Technology Stack**

- **Backend**: FastAPI (Python)
- **AI Model**: Grok AI via OpenRouter
- **Vector Database**: ChromaDB
- **Embeddings**: Sentence Transformers (all-mpnet-base-v2)
- **Frontend**: HTML5, CSS3, JavaScript
- **Database**: SQLite with SQLAlchemy ORM
- **PDF Processing**: PyMuPDF, pdfplumber, PyPDF2, OCR

## 📋 **Requirements**

### System Requirements
- **OS**: Windows 10/11, macOS, Linux
- **RAM**: Minimum 4GB, Recommended 8GB+
- **Storage**: 2GB free space
- **Python**: 3.8 or higher

### Software Dependencies
- Python 3.8+
- pip package manager
- Modern web browser

## 🚀 **Quick Start**

### 1. **Clone & Setup**
```bash
git clone <your-repo-url>
cd tuterby_rag
```

### 2. **Install Dependencies**
```bash
pip install -r requirements.txt
```

### 3. **Configure Environment**
```bash
# Copy environment template
cp env_template.txt .env

# Edit .env file with your API keys
GROK_API_KEY=your_openrouter_api_key_here
DATABASE_URL=sqlite:///tuterby.db
CHROMA_DB_PATH=./chroma_db
LOG_LEVEL=INFO
```

### 4. **Start the System**
```bash
python start.py
```

### 5. **Access the Interface**
Open your browser and go to: `http://localhost:8000`

## 📖 **Usage Guide**

### **Uploading Documents**
1. Click "Upload Document" button
2. Drag & drop your PDF or click to browse
3. Enter password if PDF is protected
4. Wait for processing (usually 10-30 seconds)
5. Document is now available for queries

### **Asking Questions**
1. Type your question in the chat box
2. Press Enter or click Send
3. AI searches through your documents
4. Get instant, accurate answers
5. Continue the conversation naturally

### **Managing Documents**
- **View All Documents**: Click "Documents" button
- **Document Info**: Click "Info" on any document
- **Delete Documents**: Click "Delete" (with confirmation)
- **System Stats**: View total documents, chunks, and characters

## 🔧 **Configuration Options**

### **Memory Management**
```python
# In config.py
MAX_FILE_SIZE_MB = 25          # Maximum PDF file size
MAX_PDF_PAGES = 100           # Maximum pages per PDF
MAX_TEXT_LENGTH = 1000000     # Maximum text length (1MB)
MAX_CHUNKS = 50               # Maximum chunks per document
CHUNK_SIZE = 300              # Characters per chunk
CHUNK_OVERLAP = 50            # Overlap between chunks
```

### **AI Model Settings**
```python
# In grok_integration.py
MODEL_NAME = "grok-beta"      # AI model to use
TEMPERATURE = 0.7             # Response creativity (0.0-1.0)
MAX_TOKENS = 1500             # Maximum response length
```

## 📊 **Performance & Optimization**

### **Speed Optimizations**
- **Fast Chunking**: Simple, efficient text splitting
- **Batch Processing**: Vector operations in batches
- **Memory Management**: Optimized for large documents
- **Caching**: Intelligent response caching

### **Memory Optimizations**
- **Garbage Collection**: Automatic memory cleanup
- **Chunk Limits**: Prevents memory overflow
- **File Size Limits**: Configurable upload limits
- **Progress Monitoring**: Real-time memory usage tracking

## 🔒 **Security Features**

- **Local Processing**: Documents processed on your server
- **No External Storage**: All data stays within your system
- **Secure API Keys**: Environment variable protection
- **Input Validation**: Robust file and input validation
- **Error Handling**: Secure error messages

## 📈 **Monitoring & Analytics**

### **System Health**
- **Health Check Endpoint**: `/health`
- **Memory Usage**: Real-time monitoring
- **Processing Times**: Performance tracking
- **Error Logging**: Comprehensive error tracking

### **Document Analytics**
- **Upload Statistics**: Document count and sizes
- **Usage Patterns**: Query frequency and types
- **Performance Metrics**: Response times and accuracy
- **System Resources**: Memory and CPU usage

## 🚀 **Deployment Options**

### **Development**
```bash
python start.py
```

### **Production**
```bash
# Using Gunicorn
gunicorn app.main:app -w 4 -k uvicorn.workers.UvicornWorker

# Using Docker
docker build -t tuterby-ai .
docker run -p 8000:8000 tuterby-ai
```

### **Cloud Deployment**
- **AWS**: EC2 with RDS and S3
- **Azure**: App Service with Cosmos DB
- **Google Cloud**: Compute Engine with Cloud SQL
- **Heroku**: Simple deployment with add-ons

## 🧪 **Testing**

### **System Health Check**
```bash
python test_system.py
```

### **API Testing**
```bash
# Health check
curl http://localhost:8000/health

# List documents
curl http://localhost:8000/documents/

# Upload test
curl -X POST -F "file=@test.pdf" http://localhost:8000/upload-pdf/
```

## 🐛 **Troubleshooting**

### **Common Issues**

#### **PDF Upload Fails**
- Check file size (max 25MB)
- Verify PDF is not corrupted
- Check if password is required

#### **Slow Processing**
- Reduce chunk size in config
- Check available memory
- Monitor system resources

#### **AI Responses Poor**
- Verify Grok API key
- Check internet connection
- Review document quality

### **Performance Tips**
- **Small Documents**: Use smaller chunk sizes
- **Large Documents**: Increase memory limits
- **Frequent Queries**: Enable response caching
- **Multiple Users**: Scale horizontally

## 🔮 **Future Enhancements**

### **Planned Features**
- **Multi-User Support**: User authentication and roles
- **Document Versioning**: Track document changes
- **Advanced Analytics**: Usage insights and reports
- **API Integration**: Connect with other business systems
- **Mobile App**: Native mobile applications

### **AI Improvements**
- **Custom Training**: Train on company-specific data
- **Multi-Modal**: Support images and diagrams
- **Voice Interface**: Speech-to-text capabilities
- **Smart Summaries**: Automatic document summarization

## 📞 **Support & Community**

### **Getting Help**
- **Documentation**: Check this README first
- **Issues**: Report bugs via GitHub issues
- **Discussions**: Join community discussions
- **Email**: Contact support team

### **Contributing**
1. Fork the repository
2. Create feature branch
3. Make your changes
4. Submit pull request
5. Join the team!

## 📄 **License**

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 **Acknowledgments**

- **Grok AI**: Advanced language model capabilities
- **OpenRouter**: API access and management
- **ChromaDB**: Vector database technology
- **FastAPI**: Modern Python web framework
- **Open Source Community**: All contributors and maintainers

---

**Built with ❤️ for modern businesses**

*Transform your company knowledge into intelligent insights with Tuterby AI*
