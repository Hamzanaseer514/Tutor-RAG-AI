// Global variables
let conversationId = null;
let currentDocumentId = null;
let documents = [];

// Initialize the application
document.addEventListener('DOMContentLoaded', function() {
    console.log('🚀 Tuterby AI Company Knowledge Base initialized');
    setupEventListeners();
    loadDocuments();
});

// Setup event listeners
function setupEventListeners() {
    // File input change
    const pdfFileInput = document.getElementById('pdfFile');
    if (pdfFileInput) {
        pdfFileInput.addEventListener('change', handleFileSelect);
    }
    
    // Drag and drop
    const uploadArea = document.getElementById('uploadArea');
    if (uploadArea) {
        uploadArea.addEventListener('dragover', handleDragOver);
        uploadArea.addEventListener('drop', handleDrop);
        uploadArea.addEventListener('dragleave', handleDragLeave);
    }
    
    // Auto-resize textarea
    const chatInput = document.getElementById('chatInput');
    if (chatInput) {
        chatInput.addEventListener('input', autoResizeTextarea);
    }
    
    // Close modals when clicking outside
    window.addEventListener('click', function(event) {
        if (event.target.classList.contains('modal')) {
            event.target.style.display = 'none';
        }
    });
}

// Handle file selection
function handleFileSelect(event) {
    const file = event.target.files[0];
    if (file) {
        console.log('File selected:', file.name);
        uploadFile(file);
    }
}

// Handle drag and drop
function handleDragOver(event) {
    event.preventDefault();
    event.currentTarget.style.borderColor = '#764ba2';
    event.currentTarget.style.background = 'linear-gradient(135deg, #f0f2ff 0%, #e8ebff 100%)';
}

function handleDragLeave(event) {
    event.preventDefault();
    event.currentTarget.style.borderColor = '#667eea';
    event.currentTarget.style.background = 'linear-gradient(135deg, #f8f9ff 0%, #f0f2ff 100%)';
}

function handleDrop(event) {
    event.preventDefault();
    const file = event.dataTransfer.files[0];
    if (file && file.type === 'application/pdf') {
        console.log('File dropped:', file.name);
        uploadFile(file);
    } else {
        showNotification('Please select a valid PDF file', 'error');
    }
    
    // Reset styling
    event.currentTarget.style.borderColor = '#667eea';
    event.currentTarget.style.background = 'linear-gradient(135deg, #f8f9ff 0%, #f0f2ff 100%)';
}

// Upload file
async function uploadFile(file) {
    console.log('Starting upload for:', file.name);
    
    const password = document.getElementById('password') ? document.getElementById('password').value : '';
    
    // Show upload status
    showUploadStatus('Starting upload...', 10);
    
    const formData = new FormData();
    formData.append('file', file);
    if (password) {
        formData.append('password', password);
    }
    
    try {
        showUploadStatus('Processing PDF...', 30);
        
        const response = await fetch('/upload-pdf/', {
            method: 'POST',
            body: formData
        });
        
        if (response.ok) {
            const result = await response.json();
            showUploadStatus('Upload successful!', 100);
            
            setTimeout(() => {
                closeModal('uploadModal');
                showNotification('Document uploaded successfully!', 'success');
                loadDocuments();
                startNewChat();
            }, 1000);
            
        } else {
            const error = await response.json();
            showUploadStatus('Upload failed', 0);
            showNotification(error.detail || 'Upload failed', 'error');
        }
        
    } catch (error) {
        console.error('Upload error:', error);
        showUploadStatus('Upload failed', 0);
        showNotification('Upload failed. Please try again.', 'error');
    }
}

// Show upload status
function showUploadStatus(message, progress) {
    const status = document.getElementById('uploadStatus');
    const statusText = document.getElementById('statusText');
    const progressFill = document.getElementById('progressFill');
    
    if (status && statusText && progressFill) {
        status.style.display = 'block';
        statusText.textContent = message;
        progressFill.style.width = progress + '%';
    }
}

// Load documents
async function loadDocuments() {
    try {
        console.log('Loading documents...');
        const response = await fetch('/documents/');
        if (response.ok) {
            const data = await response.json();
            documents = data.documents;
            console.log('Documents loaded:', documents.length);
            updateDocumentsUI(data);
        } else {
            console.error('Failed to load documents:', response.status);
        }
    } catch (error) {
        console.error('Failed to load documents:', error);
    }
}

// Update documents UI
function updateDocumentsUI(data) {
    const stats = document.getElementById('documentsStats');
    const list = document.getElementById('documentsList');
    
    if (stats) {
        const totalDocs = document.getElementById('totalDocs');
        const totalChunks = document.getElementById('totalChunks');
        const totalChars = document.getElementById('totalChars');
        
        if (totalDocs) totalDocs.textContent = data.total_count;
        if (totalChunks) totalChunks.textContent = data.documents.reduce((sum, doc) => sum + doc.chunk_count, 0);
        if (totalChars) totalChars.textContent = 'N/A'; // Will be updated with vector store stats
    }
    
    if (list) {
        list.innerHTML = '';
        
        if (data.documents.length === 0) {
            list.innerHTML = `
                <div class="empty-state">
                    <i class="fas fa-folder-open"></i>
                    <h3>No documents uploaded yet</h3>
                    <p>Upload your first company document to get started!</p>
                </div>
            `;
            return;
        }
        
        data.documents.forEach(doc => {
            const docElement = createDocumentElement(doc);
            list.appendChild(docElement);
        });
    }
}

// Create document element
function createDocumentElement(doc) {
    const div = document.createElement('div');
    div.className = 'document-item';
    div.innerHTML = `
        <div class="document-info">
            <div class="document-name">${doc.filename}</div>
            <div class="document-meta">
                Uploaded: ${new Date(doc.upload_date).toLocaleDateString()} | 
                Chunks: ${doc.chunk_count}
            </div>
        </div>
        <div class="document-actions">
            <button class="btn-info" onclick="viewDocumentInfo('${doc.id}')">
                <i class="fas fa-info-circle"></i> Info
            </button>
            <button class="btn-danger" onclick="deleteDocument('${doc.id}')">
                <i class="fas fa-trash"></i> Delete
            </button>
        </div>
    `;
    return div;
}

// View document info
async function viewDocumentInfo(docId) {
    try {
        console.log('Viewing document info for:', docId);
        const response = await fetch(`/documents/${docId}/info`);
        if (response.ok) {
            const doc = await response.json();
            showDocumentInfoModal(doc);
        } else {
            showNotification('Failed to get document info', 'error');
        }
    } catch (error) {
        console.error('Failed to get document info:', error);
        showNotification('Failed to get document info', 'error');
    }
}

// Show document info modal
function showDocumentInfoModal(doc) {
    // Create a simple modal for document info
    const modal = document.createElement('div');
    modal.className = 'modal';
    modal.style.display = 'block';
    
    modal.innerHTML = `
        <div class="modal-content">
            <div class="modal-header">
                <h3><i class="fas fa-file-pdf"></i> Document Information</h3>
                <button class="close-btn" onclick="this.closest('.modal').remove()">
                    <i class="fas fa-times"></i>
                </button>
            </div>
            <div class="modal-body">
                <div class="document-details">
                    <p><strong>Filename:</strong> ${doc.filename}</p>
                    <p><strong>Upload Date:</strong> ${new Date(doc.upload_date).toLocaleString()}</p>
                    <p><strong>Chunk Count:</strong> ${doc.chunk_count}</p>
                    <p><strong>Document ID:</strong> ${doc.id}</p>
                </div>
            </div>
        </div>
    `;
    
    document.body.appendChild(modal);
}

// Delete document
async function deleteDocument(docId) {
    if (!confirm('Are you sure you want to delete this document? This action cannot be undone.')) {
        return;
    }
    
    try {
        console.log('Deleting document:', docId);
        const response = await fetch(`/documents/${docId}`, {
            method: 'DELETE'
        });
        
        if (response.ok) {
            showNotification('Document deleted successfully', 'success');
            loadDocuments();
        } else {
            const error = await response.json();
            showNotification(error.detail || 'Failed to delete document', 'error');
        }
    } catch (error) {
        console.error('Failed to delete document:', error);
        showNotification('Failed to delete document', 'error');
    }
}

// Send message
async function sendMessage() {
    const input = document.getElementById('chatInput');
    if (!input) return;
    
    const message = input.value.trim();
    if (!message) return;
    
    // Add user message to chat
    addMessage(message, true);
    input.value = '';
    autoResizeTextarea();
    
    // Show loading message
    const loadingId = addLoadingMessage();
    
    try {
        const response = await fetch('/query/', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                question: message,
                conversation_id: conversationId
            })
        });
        
        if (response.ok) {
            const result = await response.json();
            
            // Remove loading message
            removeLoadingMessage(loadingId);
            
            // Add AI response
            addMessage(result.response, false);
            
            // Update conversation ID
            if (result.conversation_id) {
                conversationId = result.conversation_id;
            }
            
        } else {
            const error = await response.json();
            removeLoadingMessage(loadingId);
            addMessage(`Error: ${error.detail}`, false);
        }
        
    } catch (error) {
        console.error('Query error:', error);
        removeLoadingMessage(loadingId);
        addMessage('Sorry, I encountered an error. Please try again.', false);
    }
}

// Add message to chat
function addMessage(text, isUser) {
    const chatMessages = document.getElementById('chatMessages');
    if (!chatMessages) return;
    
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${isUser ? 'user-message' : 'ai-message'}`;
    
    const timestamp = new Date().toLocaleTimeString();
    
    messageDiv.innerHTML = `
        <div class="message-avatar">
            <i class="fas fa-${isUser ? 'user' : 'robot'}"></i>
        </div>
        <div class="message-content">
            <div class="message-header">
                <strong>${isUser ? 'You' : 'Tuterby AI'}</strong>
                <span class="message-time">${timestamp}</span>
            </div>
            <div class="message-text">${text}</div>
        </div>
    `;
    
    chatMessages.appendChild(messageDiv);
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

// Add loading message
function addLoadingMessage() {
    const chatMessages = document.getElementById('chatMessages');
    if (!chatMessages) return null;
    
    const loadingDiv = document.createElement('div');
    const loadingId = 'loading-' + Date.now();
    loadingDiv.id = loadingId;
    loadingDiv.className = 'message ai-message';
    
    loadingDiv.innerHTML = `
        <div class="message-avatar">
            <i class="fas fa-robot"></i>
        </div>
        <div class="message-content">
            <div class="message-header">
                <strong>Tuterby AI</strong>
                <span class="message-time">${new Date().toLocaleTimeString()}</span>
            </div>
            <div class="message-text">
                <i class="fas fa-spinner fa-spin"></i> Thinking...
            </div>
        </div>
    `;
    
    chatMessages.appendChild(loadingDiv);
    chatMessages.scrollTop = chatMessages.scrollHeight;
    return loadingId;
}

// Remove loading message
function removeLoadingMessage(loadingId) {
    const loadingDiv = document.getElementById(loadingId);
    if (loadingDiv) {
        loadingDiv.remove();
    }
}

// Start new chat
function startNewChat() {
    conversationId = null;
    currentDocumentId = null;
    
    const chatMessages = document.getElementById('chatMessages');
    if (!chatMessages) return;
    
    chatMessages.innerHTML = `
        <div class="welcome-message">
            <div class="welcome-icon">
                <i class="fas fa-robot"></i>
            </div>
            <h3>Welcome to Tuterby AI!</h3>
            <p>I'm your company knowledge assistant. I can help you with questions about:</p>
            <ul>
                <li>Company policies and procedures</li>
                <li>Technical documentation</li>
                <li>Training materials</li>
                <li>Any uploaded company documents</li>
            </ul>
            <p><strong>Start by asking me a question or upload a document to get started!</strong></p>
        </div>
    `;
    
    showNotification('New chat started', 'info');
}

// Handle key press
function handleKeyDown(event) {
    if (event.key === 'Enter' && !event.shiftKey) {
        event.preventDefault();
        sendMessage();
    }
}

// Auto-resize textarea
function autoResizeTextarea() {
    const textarea = document.getElementById('chatInput');
    if (!textarea) return;
    
    textarea.style.height = 'auto';
    textarea.style.height = Math.min(textarea.scrollHeight, 120) + 'px';
}

// Modal functions
function showUploadModal() {
    const modal = document.getElementById('uploadModal');
    if (modal) {
        modal.style.display = 'block';
        const status = document.getElementById('uploadStatus');
        if (status) status.style.display = 'none';
        
        const fileInput = document.getElementById('pdfFile');
        if (fileInput) fileInput.value = '';
        
        const password = document.getElementById('password');
        if (password) password.value = '';
    }
}

function showDocumentsModal() {
    const modal = document.getElementById('documentsModal');
    if (modal) {
        modal.style.display = 'block';
        loadDocuments();
    }
}

function showSystemInfo() {
    const modal = document.getElementById('systemInfoModal');
    if (modal) {
        modal.style.display = 'block';
    }
}

function closeModal(modalId) {
    const modal = document.getElementById(modalId);
    if (modal) {
        modal.style.display = 'none';
    }
}

// Show notification
function showNotification(message, type = 'info') {
    // Create notification element
    const notification = document.createElement('div');
    notification.className = `notification notification-${type}`;
    notification.innerHTML = `
        <i class="fas fa-${type === 'success' ? 'check-circle' : type === 'error' ? 'exclamation-circle' : 'info-circle'}"></i>
        <span>${message}</span>
    `;
    
    // Add to page
    document.body.appendChild(notification);
    
    // Show notification
    setTimeout(() => {
        notification.classList.add('show');
    }, 100);
    
    // Hide and remove
    setTimeout(() => {
        notification.classList.remove('show');
        setTimeout(() => {
            if (notification.parentNode) {
                notification.remove();
            }
        }, 300);
    }, 3000);
}

// Add notification styles
const notificationStyles = `
    .notification {
        position: fixed;
        top: 20px;
        right: 20px;
        background: white;
        padding: 1rem 1.5rem;
        border-radius: 10px;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.2);
        display: flex;
        align-items: center;
        gap: 0.75rem;
        z-index: 3000;
        transform: translateX(400px);
        transition: transform 0.3s ease;
    }
    
    .notification.show {
        transform: translateX(0);
    }
    
    .notification-success {
        border-left: 4px solid #28a745;
        color: #155724;
    }
    
    .notification-error {
        border-left: 4px solid #dc3545;
        color: #721c24;
    }
    
    .notification-info {
        border-left: 4px solid #17a2b8;
        color: #0c5460;
    }
    
    .notification i {
        font-size: 1.2rem;
    }
    
    .empty-state {
        text-align: center;
        padding: 3rem 2rem;
        color: #666;
    }
    
    .empty-state i {
        font-size: 4rem;
        color: #ddd;
        margin-bottom: 1rem;
    }
    
    .document-details p {
        margin-bottom: 0.5rem;
        padding: 0.5rem 0;
        border-bottom: 1px solid #eee;
    }
`;

// Inject notification styles
const styleSheet = document.createElement('style');
styleSheet.textContent = notificationStyles;
document.head.appendChild(styleSheet);