import React, { useState, useEffect, useCallback } from 'react';
import { api } from './services/api';

function App() {
  const [backendStatus, setBackendStatus] = useState('checking');
  const [fileId, setFileId] = useState(localStorage.getItem('policyai_fileId'));
  const [uploadedFile, setUploadedFile] = useState(() => {
    const saved = localStorage.getItem('policyai_fileInfo');
    return saved ? JSON.parse(saved) : null;
  });
  const [messages, setMessages] = useState(() => {
    const saved = localStorage.getItem('policyai_messages');
    return saved ? JSON.parse(saved) : [];
  });
  const [isProcessing, setIsProcessing] = useState(false);
  const [inputMessage, setInputMessage] = useState('');

  // Check backend connection
  const checkBackend = useCallback(async () => {
    try {
      await api.healthCheck();
      setBackendStatus('connected');
    } catch (error) {
      setBackendStatus('disconnected');
      console.error('Backend connection failed:', error);
    }
  }, []);

  useEffect(() => {
    checkBackend();
    const interval = setInterval(checkBackend, 10000);
    return () => clearInterval(interval);
  }, [checkBackend]);

  // Save state to localStorage
  const saveState = useCallback(() => {
    if (fileId) localStorage.setItem('policyai_fileId', fileId);
    if (uploadedFile) localStorage.setItem('policyai_fileInfo', JSON.stringify(uploadedFile));
    localStorage.setItem('policyai_messages', JSON.stringify(messages));
  }, [fileId, uploadedFile, messages]);

  useEffect(() => {
    saveState();
  }, [saveState]);

  const addMessage = useCallback((type, content, meta = null) => {
    const newMessage = { type, content, meta, timestamp: new Date().toLocaleTimeString() };
    setMessages(prev => [...prev, newMessage]);
  }, []);

  const handleFileUpload = async (file) => {
    if (backendStatus !== 'connected') {
      addMessage('error', 'Backend server is offline. Please start the server first.');
      return;
    }

    setIsProcessing(true);
    addMessage('system', `📤 Uploading "${file.name}"...`);

    try {
      const result = await api.uploadFile(file);
      
      const fileInfo = {
        name: file.name,
        size: (file.size / 1024 / 1024).toFixed(2) + ' MB',
        uploadTime: new Date().toLocaleTimeString(),
        fileId: result.file_id
      };

      setFileId(result.file_id);
      setUploadedFile(fileInfo);
      addMessage('system', `✅ Document uploaded successfully! You can now ask questions.`);
      
    } catch (error) {
      addMessage('error', `Upload failed: ${error.message}`);
    } finally {
      setIsProcessing(false);
    }
  };

  const handleSendMessage = async (message) => {
    if (!fileId || backendStatus !== 'connected') return;

    setIsProcessing(true);
    addMessage('user', message);
    setInputMessage('');

    try {
      const startTime = Date.now();
      const result = await api.queryDocument(fileId, message);
      const processingTime = ((Date.now() - startTime) / 1000).toFixed(1);

      if (result.answers && result.answers.length > 0) {
        addMessage('assistant', result.answers[0], { processingTime });
      } else {
        throw new Error('No answer received');
      }
    } catch (error) {
      addMessage('error', `Query failed: ${error.message}`);
    } finally {
      setIsProcessing(false);
    }
  };

  const handleSampleQuestion = (question) => {
    if (fileId && backendStatus === 'connected' && !isProcessing) {
      handleSendMessage(question);
    }
  };

  const clearChat = () => {
    setMessages([]);
  };

  const sampleQuestions = [
    "What's covered under this policy?",
    "What are the exclusions?",
    "What is the claims process?",
    "Are pre-existing conditions covered?",
    "What is the premium amount?",
    "What's the policy duration?"
  ];

  return (
    <div className="insurance-app">
      {/* Header */}
      <header className="insurance-header">
        <div className="header-content">
          <div className="brand">
            <div className="brand-icon">
              <i className="fas fa-shield-alt"></i>
            </div>
            <div className="brand-text">
              <h1>PolicyAI</h1>
              <div className="tagline">Insurance Document Intelligence</div>
            </div>
          </div>
          <div className="status-indicator">
            <div className={`status-dot ${backendStatus === 'connected' ? 'ready' : backendStatus === 'disconnected' ? 'error' : 'processing'}`}></div>
            <span>
              {backendStatus === 'connected' ? 'Connected' : 
               backendStatus === 'disconnected' ? 'Offline' : 'Checking...'}
            </span>
          </div>
        </div>
      </header>

      {/* Main Layout */}
      <main className="insurance-layout">
        {/* Document Sidebar */}
        <aside className="document-sidebar">
          {/* Upload Card */}
          <div className="upload-card" onClick={() => document.getElementById('fileInput')?.click()}>
            <div className="upload-icon">
              <i className="fas fa-cloud-upload-alt"></i>
            </div>
            <div className="upload-text">
              <h3>Upload Policy Document</h3>
              <p>Drag & drop your insurance PDF or click to browse</p>
            </div>
            <input
              id="fileInput"
              type="file"
              accept=".pdf"
              onChange={(e) => {
                const file = e.target.files[0];
                if (file && file.type === 'application/pdf') {
                  handleFileUpload(file);
                }
                e.target.value = '';
              }}
              disabled={backendStatus !== 'connected' || isProcessing}
              style={{ display: 'none' }}
            />
          </div>

          {/* Current Document */}
          {uploadedFile && (
            <div className="document-list">
              <div className="section-title">
                <i className="fas fa-file-pdf"></i>
                <span>Current Document</span>
              </div>
              <div className="document-item active">
                <div className="doc-icon">
                  <i className="fas fa-file-pdf"></i>
                </div>
                <div className="doc-info">
                  <div className="doc-name">{uploadedFile.name}</div>
                  <div className="doc-meta">
                    <span>{uploadedFile.size}</span>
                    <span>Uploaded</span>
                  </div>
                </div>
              </div>
            </div>
          )}

          {/* Quick Questions */}
          <div className="quick-questions">
            <div className="section-title">
              <i className="fas fa-bolt"></i>
              <span>Quick Questions</span>
            </div>
            <div className="question-grid">
              {sampleQuestions.map((question, index) => (
                <button
                  key={index}
                  className="question-btn"
                  onClick={() => handleSampleQuestion(question)}
                  disabled={!fileId || backendStatus !== 'connected' || isProcessing}
                >
                  <i className="fas fa-play"></i>
                  {question}
                </button>
              ))}
            </div>
          </div>
        </aside>

        {/* Chat Interface */}
        <section className="chat-interface">
          <div className="chat-header">
            <h2>Policy Analysis</h2>
            <div className="status-indicator">
              <div className={`status-dot ${!fileId ? 'error' : isProcessing ? 'processing' : 'ready'}`}></div>
              <span>
                {!fileId ? 'Upload a document to start' : 
                 isProcessing ? 'Processing...' : 'Ready for questions'}
              </span>
            </div>
          </div>

          <div className="chat-messages">
            {messages.length === 0 ? (
              <div className="welcome-message">
                <div className="message assistant">
                  <div className="message-avatar">
                    <i className="fas fa-robot"></i>
                  </div>
                  <div className="message-content">
                    <div className="message-text">
                      Welcome! Upload an insurance policy PDF and I'll help you understand the coverage, 
                      exclusions, claims process, and answer any specific questions you have about the document.
                    </div>
                  </div>
                </div>
              </div>
            ) : (
              messages.map((message, index) => (
                <div key={index} className={`message ${message.type}`}>
                  <div className="message-avatar">
                    {message.type === 'user' ? (
                      <i className="fas fa-user"></i>
                    ) : message.type === 'error' ? (
                      <i className="fas fa-exclamation-triangle"></i>
                    ) : (
                      <i className="fas fa-robot"></i>
                    )}
                  </div>
                  <div className="message-content">
                    <div className="message-text">{message.content}</div>
                    <div className="message-time">{message.timestamp}</div>
                    {message.meta?.processingTime && (
                      <div className="processing-time">
                        ⏱️ {message.meta.processingTime}s
                      </div>
                    )}
                  </div>
                </div>
              ))
            )}
          </div>

          <div className="chat-input-area">
            <div className="input-group">
              <textarea
                className="chat-input"
                placeholder={!fileId ? "Upload a document to ask questions..." : "Ask about coverage, claims, exclusions..."}
                value={inputMessage}
                onChange={(e) => setInputMessage(e.target.value)}
                onKeyPress={(e) => {
                  if (e.key === 'Enter' && !e.shiftKey) {
                    e.preventDefault();
                    if (inputMessage.trim() && !isProcessing && fileId) {
                      handleSendMessage(inputMessage.trim());
                    }
                  }
                }}
                disabled={!fileId || isProcessing || backendStatus !== 'connected'}
                rows="1"
              />
              <button
                className="send-button"
                onClick={() => {
                  if (inputMessage.trim() && !isProcessing && fileId) {
                    handleSendMessage(inputMessage.trim());
                  }
                }}
                disabled={!fileId || isProcessing || !inputMessage.trim() || backendStatus !== 'connected'}
              >
                <i className="fas fa-paper-plane"></i>
              </button>
            </div>
          </div>
        </section>
      </main>
    </div>
  );
}

export default App;