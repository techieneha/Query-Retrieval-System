import axios from 'axios';

const API_BASE = 'http://127.0.0.1:8000';

class PolicyAIAPI {
  constructor() {
    this.client = axios.create({
      baseURL: API_BASE,
      timeout: 30000,
    });
  }

  async healthCheck() {
    try {
      const response = await this.client.get('/health');
      return response.data;
    } catch (error) {
      throw this.handleError(error);
    }
  }

  async uploadFile(file) {
    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await this.client.post('/api/v1/upload', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
        timeout: 120000, // 2 minutes for large files
      });
      return response.data;
    } catch (error) {
      throw this.handleError(error);
    }
  }

  async queryDocument(fileId, question) {
    try {
      const response = await this.client.post('/api/v1/query', {
        file_id: fileId,
        questions: [question],
      });
      return response.data;
    } catch (error) {
      throw this.handleError(error);
    }
  }

  async checkFileStatus(fileId) {
    try {
      const response = await this.client.get(`/api/v1/files/${fileId}`);
      return response.data;
    } catch (error) {
      throw this.handleError(error);
    }
  }

  handleError(error) {
    if (error.code === 'ECONNREFUSED') {
      return new Error('Backend server is offline. Please start the server with: python api/main.py');
    }
    if (error.response) {
      return new Error(error.response.data.detail || `Server error: ${error.response.status}`);
    }
    return new Error(error.message || 'Network error occurred');
  }
}

export const api = new PolicyAIAPI();