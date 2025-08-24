# Tuterby AI Troubleshooting Guide

## 🚨 Common Issues and Solutions

### 1. MemoryError During PDF Processing

**Problem:** `MemoryError` when uploading large PDFs

**Causes:**
- PDF file is too large (>25MB)
- PDF has too many pages (>100)
- System doesn't have enough RAM
- PDF contains very long text

**Solutions:**
- Use smaller PDF files (under 25MB)
- Split large PDFs into smaller documents
- Close other applications to free up RAM
- Use PDFs with fewer pages

**Technical Details:**
- Maximum file size: 25MB
- Maximum pages: 100
- Maximum text length: 1MB
- Maximum chunks: 1000

### 2. ChromaDB Initialization Error

**Problem:** `'SentenceTransformer' object has no attribute 'name'`

**Solution:** ✅ **FIXED** - This was a compatibility issue between ChromaDB and SentenceTransformers

### 3. PDF Text Extraction Fails

**Problem:** "Could not extract text from PDF"

**Causes:**
- Scanned PDF (image-based)
- Corrupted PDF file
- Password-protected PDF
- PDF contains only images

**Solutions:**
- Install Tesseract OCR for scanned PDFs
- Check if PDF is corrupted
- Provide password if encrypted
- Use text-based PDFs when possible

### 4. Grok AI API Errors

**Problem:** "Error calling Grok API"

**Causes:**
- Missing or invalid API key
- Network connectivity issues
- API rate limits exceeded

**Solutions:**
- Check your `.env` file has `GROK_API_KEY`
- Verify API key from OpenRouter
- Check internet connection
- Wait if rate limit exceeded

### 5. Slow Performance

**Problem:** PDF processing takes too long

**Causes:**
- Large PDF files
- Complex document structure
- OCR processing for scanned PDFs
- Limited system resources

**Solutions:**
- Use smaller PDFs
- Ensure PDFs are text-based
- Close unnecessary applications
- Increase system RAM if possible

## 🔧 System Requirements

### Minimum Requirements
- **RAM:** 4GB (8GB recommended)
- **Storage:** 2GB free space
- **Python:** 3.8+
- **OS:** Windows 10+, macOS 10.14+, Ubuntu 18.04+

### Recommended Requirements
- **RAM:** 8GB or more
- **Storage:** 5GB free space
- **CPU:** Multi-core processor
- **Internet:** Stable connection for AI API calls

## 📱 Performance Optimization

### For Large PDFs
1. **Split documents** into smaller sections
2. **Use text-based PDFs** instead of scanned
3. **Process during off-peak hours**
4. **Monitor system resources**

### Memory Management
- System automatically limits:
  - File size to 25MB
  - Page count to 100
  - Text length to 1MB
  - Chunks to 1000

## 🚀 Getting Help

### Check Logs
Look for error messages in the console output when running the application.

### Common Error Messages
- `MemoryError`: PDF too large
- `File too large`: Exceeds 25MB limit
- `PDF too large`: Exceeds 100 page limit
- `GROK_API_KEY required`: Missing API key

### Support Steps
1. Check this troubleshooting guide
2. Verify system requirements
3. Check configuration files
4. Review error logs
5. Try with a smaller PDF first

## 🔍 Debug Mode

To enable detailed logging, set in your `.env` file:
```env
LOG_LEVEL=DEBUG
```

This will show more detailed information about what's happening during PDF processing.

## 📊 Memory Usage Monitoring

The system automatically logs memory usage:
- Before processing
- After text extraction
- After vector storage
- After cleanup

Monitor these logs to identify memory bottlenecks.

---

**Need more help?** Check the main README.md for installation and usage instructions.
