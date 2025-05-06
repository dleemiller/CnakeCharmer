// Get API URL from environment or use default
const getApiUrl = (): string => {
  return process.env.REACT_APP_API_URL || 'http://localhost:8000';
};

// Get Socket.io URL from environment or derive from API URL
const getSocketIoUrl = (): string => {
  const socketIoPort = process.env.REACT_APP_SOCKET_IO_PORT || '8001';
  // Replace the port in the API URL with the Socket.io port
  const apiUrl = getApiUrl();
  const apiUrlObject = new URL(apiUrl);
  const host = apiUrlObject.hostname;
  return `http://${host}:${socketIoPort}`;
};

// Export API URL for HTTP requests
export const API_URL = getApiUrl();

// Export Socket.io URL for real-time communication
export const SOCKET_IO_URL = getSocketIoUrl(); 