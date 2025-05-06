import axios from 'axios';
import { JobStatus, AnalysisResult, SystemStatus } from '../types';

// Create axios instance with base configuration
const api = axios.create({
  baseURL: process.env.REACT_APP_API_URL || 'http://localhost:8000',
  headers: {
    'Content-Type': 'application/json',
  },
});

// API client methods
export const apiClient = {
  // Submit code for analysis
  submitCode: async (code: string): Promise<JobStatus> => {
    console.log('Submitting code for analysis');
    const response = await api.post<JobStatus>('/api/submit', { code });
    console.log('Submit response:', response.data);
    return response.data;
  },
  
  // Get job results
  getJobResults: async (jobId: string): Promise<AnalysisResult> => {
    console.log('Fetching job results for:', jobId);
    const response = await api.get<AnalysisResult>(`/api/results/${jobId}`);
    console.log('Job results response:', response.data);
    return response.data;
  },
  
  // Get system status
  getSystemStatus: async (): Promise<SystemStatus> => {
    console.log('Fetching system status');
    const response = await api.get<SystemStatus>('/api/status');
    console.log('System status response:', response.data);
    return response.data;
  },
  
  // Get HTML annotation URL
  getHtmlAnnotationUrl: (filename: string): string => {
    return `${process.env.REACT_APP_API_URL || 'http://localhost:8000'}/api/html/${filename}`;
  },
  
  // Get detailed analysis URL
  getDetailedAnalysisUrl: (jobId: string): string => {
    return `${process.env.REACT_APP_API_URL || 'http://localhost:8000'}/api/analysis/${jobId}`;
  }
};

export default apiClient;