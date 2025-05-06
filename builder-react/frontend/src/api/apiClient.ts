// src/api/apiClient.ts

import axios from "axios";
import { API_BASE_URL } from "../config";

export const submitCode = (code: string) =>
  axios.post(`${API_BASE_URL}/submit`, { code }).then(r => r.data);

export const getJobResults = (jobId: string) =>
  axios.get(`${API_BASE_URL}/results/${jobId}`).then(r => r.data);

export const getSystemStatus = () =>
  axios.get(`${API_BASE_URL}/status`).then(r => r.data);

export const getDetailedAnalysisUrl = (jobId: string) =>
  `${API_BASE_URL}/analysis/${jobId}`;
