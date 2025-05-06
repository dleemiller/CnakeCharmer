import { useCallback, useEffect, useState } from 'react';
import useWebSocket, { ReadyState } from 'react-use-websocket';

// Generate a unique client ID
const generateClientId = () => {
  return `client_${Math.random().toString(36).substring(2, 9)}`;
};

// Get WebSocket URL based on API URL
const getWebSocketUrl = (): string => {
  const apiUrl = process.env.REACT_APP_API_URL || 'http://localhost:8000';
  const wsUrl = apiUrl.replace(/^http/, 'ws');
  return `${wsUrl}/ws/system`; // Connect to system channel for status updates
};

interface WebSocketMessage {
  type: string;
  job_id?: string;
  status?: string;
  result?: any;
  [key: string]: any;
}

interface UseWebSocketClientProps {
  onMessage?: (data: any) => void;
  onOpen?: () => void;
  onClose?: () => void;
  onError?: (event: Event) => void;
}

export const useWebSocketClient = ({
  onMessage,
  onOpen,
  onClose,
  onError,
}: UseWebSocketClientProps = {}) => {
  const [isReconnecting, setIsReconnecting] = useState(false);
  const [lastMessage, setLastMessage] = useState<WebSocketMessage | null>(null);
  const [pendingSubscriptions, setPendingSubscriptions] = useState<string[]>([]);
  
  // Initialize WebSocket connection
  const { 
    sendMessage,
    lastMessage: wsLastMessage,
    readyState 
  } = useWebSocket(getWebSocketUrl(), {
    onOpen: () => {
      console.log('WebSocket connection established');
      setIsReconnecting(false);
      if (onOpen) onOpen();
      
      // Resubscribe to any pending jobs
      if (pendingSubscriptions.length > 0) {
        pendingSubscriptions.forEach(jobId => {
          subscribeToJob(jobId);
        });
        setPendingSubscriptions([]);
      }
    },
    onClose: () => {
      console.log('WebSocket connection closed');
      if (onClose) onClose();
    },
    onError: (event) => {
      console.error('WebSocket error:', event);
      if (onError) onError(event);
    },
    // Reconnect if connection is closed
    shouldReconnect: () => true,
    reconnectAttempts: 10,
    reconnectInterval: 3000,
    onReconnectStop: () => {
      console.log('WebSocket reconnection stopped after maximum attempts');
    }
  });
  
  // Process received messages
  const processMessage = useCallback((message: any) => {
    try {
      console.log('Received WebSocket message:', message.data);
      const parsedData = JSON.parse(message.data) as WebSocketMessage;
      console.log('Parsed WebSocket message:', parsedData);
      
      // Only update lastMessage if it's different from the current one
      setLastMessage(prev => {
        if (JSON.stringify(prev) === JSON.stringify(parsedData)) {
          return prev;
        }
        return parsedData;
      });
      
      // Call the onMessage callback if provided
      if (onMessage) {
        console.log('Calling onMessage callback with:', parsedData);
        onMessage(parsedData);
      }
    } catch (error) {
      console.error('Error parsing WebSocket message:', error, 'Raw message:', message.data);
    }
  }, [onMessage]);
  
  // Process messages when they arrive
  useEffect(() => {
    if (wsLastMessage) {
      processMessage(wsLastMessage);
    }
  }, [wsLastMessage, processMessage]);
  
  // Subscribe to job updates
  const subscribeToJob = useCallback((jobId: string) => {
    if (readyState === ReadyState.OPEN) {
      console.log(`Subscribing to updates for job ${jobId}`);
      sendMessage(JSON.stringify({ 
        type: 'subscribe', 
        job_id: jobId 
      }));
    } else {
      console.warn(`Cannot subscribe to job ${jobId}, WebSocket not connected. Adding to pending subscriptions.`);
      setPendingSubscriptions(prev => [...prev, jobId]);
      setIsReconnecting(true);
    }
  }, [readyState, sendMessage]);
  
  // Connection status
  const connectionStatus = {
    [ReadyState.CONNECTING]: 'Connecting',
    [ReadyState.OPEN]: 'Connected',
    [ReadyState.CLOSING]: 'Closing',
    [ReadyState.CLOSED]: 'Disconnected',
    [ReadyState.UNINSTANTIATED]: 'Uninstantiated',
  }[readyState];
  
  return {
    sendMessage: (data: any) => sendMessage(JSON.stringify(data)),
    lastMessage,
    connectionStatus,
    isConnected: readyState === ReadyState.OPEN,
    isConnecting: readyState === ReadyState.CONNECTING || isReconnecting,
    subscribeToJob,
  };
};