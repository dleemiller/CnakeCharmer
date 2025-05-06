import React, { createContext, useContext, useEffect, useState, useRef, useCallback, ReactNode } from 'react';
import { API_URL } from '../config';
import isEqual from 'lodash/isEqual';

// We'll use API_URL to derive the WebSocket URL
const getWebSocketUrl = () => {
  // Replace http/https with ws/wss
  return API_URL.replace(/^http/, 'ws') + '/ws';
};

const API_WS_URL = getWebSocketUrl();

// Define WebSocketMessage type
export type WebSocketMessage = {
  type: string;
  [key: string]: any;
};

interface WebSocketContextType {
  isConnected: boolean;
  lastMessage: WebSocketMessage | null;
  error: string | null;
  subscribeToJob: (jobId: string) => void;
  unsubscribeFromJob: (jobId: string) => void;
  addMessageListener: (listener: (message: WebSocketMessage) => void) => void;
  removeMessageListener: (listener: (message: WebSocketMessage) => void) => void;
}

const WebSocketContext = createContext<WebSocketContextType | null>(null);

// Track processed messages globally to avoid duplicates
const processedMessages = new Set<string>();
const MAX_PROCESSED_MESSAGES = 200;

// Connection constants
const INITIAL_RECONNECT_DELAY = 2000;
const MAX_RECONNECT_DELAY = 30000;
const RECONNECT_DECAY = 1.3;
const MAX_RECONNECT_ATTEMPTS = 5;

// Helper function to generate a message ID for deduplication
const generateMessageId = (message: WebSocketMessage): string => {
  return `${message.type}-${message.job_id || ''}-${JSON.stringify(message)}`;
};

interface WebSocketProviderProps {
  children: ReactNode;
  url?: string;
}

export const WebSocketProvider: React.FC<WebSocketProviderProps> = ({
  children,
  url = API_WS_URL
}) => {
  const [isConnected, setIsConnected] = useState<boolean>(false);
  const [lastMessage, setLastMessage] = useState<WebSocketMessage | null>(null);
  const [error, setError] = useState<string | null>(null);
  
  // Track subscribed jobs and message listeners
  const subscribedJobsRef = useRef<Set<string>>(new Set());
  const messageListenersRef = useRef<Set<(message: WebSocketMessage) => void>>(new Set());
  const wsRef = useRef<WebSocket | null>(null);
  const reconnectTimeoutRef = useRef<NodeJS.Timeout | null>(null);
  const lastProcessedMessageRef = useRef<WebSocketMessage | null>(null);
  
  // Connection state tracking
  const reconnectAttemptsRef = useRef<number>(0);
  const reconnectDelayRef = useRef<number>(INITIAL_RECONNECT_DELAY);
  const isReconnectingRef = useRef<boolean>(false);
  const intentionalCloseRef = useRef<boolean>(false);
  const pingIntervalRef = useRef<NodeJS.Timeout | null>(null);

  // Prevent duplicate message processing
  const isDuplicateMessage = useCallback((message: WebSocketMessage): boolean => {
    if (!message) return true;
    
    // Generate a unique ID for this message
    const messageId = generateMessageId(message);
    
    // Check if we've already processed this message
    if (processedMessages.has(messageId)) {
      console.log('Skipping duplicate WebSocket message:', message.type);
      return true;
    }
    
    // Add this message to the processed set
    processedMessages.add(messageId);
    
    // Limit the size of the processed messages set
    if (processedMessages.size > MAX_PROCESSED_MESSAGES) {
      // Delete oldest message
      const iterator = processedMessages.values();
      const firstValue = iterator.next().value;
      if (firstValue) {
        processedMessages.delete(firstValue);
      }
    }
    
    // Check if the message is identical to the last one we processed
    if (lastProcessedMessageRef.current && 
        isEqual(lastProcessedMessageRef.current, message)) {
      console.log('Skipping identical consecutive WebSocket message');
      return true;
    }
    
    // Not a duplicate, update last processed message
    lastProcessedMessageRef.current = message;
    return false;
  }, []);
  
  // Function to parse WebSocket messages
  const parseMessage = useCallback((event: MessageEvent): WebSocketMessage | null => {
    try {
      console.log('WebSocket raw message:', event.data);
      const data = JSON.parse(event.data);
      console.log('WebSocket message parsed:', data);
      return data;
    } catch (err) {
      console.error('Error parsing WebSocket message:', err);
      return null;
    }
  }, []);
  
  // Function to handle WebSocket messages
  const handleMessage = useCallback((event: MessageEvent) => {
    // Reset reconnect attempts on successful message
    reconnectAttemptsRef.current = 0;
    reconnectDelayRef.current = INITIAL_RECONNECT_DELAY;
    
    const message = parseMessage(event);
    if (!message) return;
    
    // If we receive a pong message, don't process it further
    if (message.type === 'pong') {
      console.log('Received pong from server');
      return;
    }
    
    // Skip if this is a duplicate message
    if (isDuplicateMessage(message)) return;
    
    // Notify all listeners
    messageListenersRef.current.forEach(listener => {
      try {
        listener(message);
      } catch (err) {
        console.error('Error in WebSocket message listener:', err);
      }
    });
    
    // Check if this is a job update message and the job is subscribed
    if (message.type === 'job_update' && message.job_id) {
      if (subscribedJobsRef.current.has(message.job_id)) {
        console.log('WebSocket message for subscribed job:', message.job_id);
        setLastMessage(message);
      } else {
        console.log('WebSocket message for unsubscribed job, ignoring:', message.job_id);
      }
    } else {
      // For non-job messages, always update
      console.log('WebSocket message changed, updating state');
      setLastMessage(message);
    }
  }, [parseMessage, isDuplicateMessage]);
  
  // Setup ping to keep connection alive
  const setupPingInterval = useCallback(() => {
    if (pingIntervalRef.current) {
      clearInterval(pingIntervalRef.current);
    }
    
    pingIntervalRef.current = setInterval(() => {
      if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
        console.log('Sending ping to server');
        wsRef.current.send(JSON.stringify({ type: 'ping' }));
      }
    }, 30000); // Send ping every 30 seconds
    
    return () => {
      if (pingIntervalRef.current) {
        clearInterval(pingIntervalRef.current);
        pingIntervalRef.current = null;
      }
    };
  }, []);
  
  // Clean up existing connection
  const cleanupConnection = useCallback(() => {
    // Clean up any existing connection
    if (wsRef.current) {
      try {
        intentionalCloseRef.current = true; // Mark as intentional to avoid auto-reconnect
        wsRef.current.onclose = null; // Remove onclose handler to avoid triggers
        wsRef.current.onerror = null; // Remove error handler
        wsRef.current.close();
      } catch (e) {
        console.error('Error closing WebSocket:', e);
      }
      wsRef.current = null;
    }
    
    // Clear any existing reconnect timeout
    if (reconnectTimeoutRef.current) {
      clearTimeout(reconnectTimeoutRef.current);
      reconnectTimeoutRef.current = null;
    }
    
    // Clear any existing ping interval
    if (pingIntervalRef.current) {
      clearInterval(pingIntervalRef.current);
      pingIntervalRef.current = null;
    }
  }, []);
  
  // Create or reconnect WebSocket with exponential backoff
  const connectWebSocket = useCallback(() => {
    // Prevent reconnection if we're already connecting
    if (isReconnectingRef.current) {
      console.log('Already reconnecting, skipping duplicate attempt');
      return;
    }
    
    // Mark that we're reconnecting
    isReconnectingRef.current = true;
    
    // Clean up existing resources
    cleanupConnection();
    
    try {
      console.log(`Connecting to WebSocket (attempt ${reconnectAttemptsRef.current + 1})`);
      
      // Create new WebSocket connection
      const ws = new WebSocket(url);
      wsRef.current = ws;
      
      // Set up event handlers
      ws.onopen = () => {
        console.log('WebSocket connection established');
        setIsConnected(true);
        setError(null);
        isReconnectingRef.current = false;
        intentionalCloseRef.current = false;
        reconnectAttemptsRef.current = 0;
        reconnectDelayRef.current = INITIAL_RECONNECT_DELAY;
        
        // Setup ping interval to keep connection alive
        setupPingInterval();
        
        // Resubscribe to any jobs
        subscribedJobsRef.current.forEach(jobId => {
          ws.send(JSON.stringify({ action: 'subscribe', job_id: jobId }));
        });
      };
      
      ws.onclose = (event) => {
        console.log('WebSocket connection closed', event.code, event.reason);
        setIsConnected(false);
        
        // Clean up resources
        cleanupConnection();
        
        // Don't reconnect if closed intentionally or max attempts reached
        if (intentionalCloseRef.current) {
          console.log('WebSocket closed intentionally, not reconnecting');
          isReconnectingRef.current = false;
          return;
        }
        
        if (reconnectAttemptsRef.current >= MAX_RECONNECT_ATTEMPTS) {
          console.log(`Maximum reconnect attempts (${MAX_RECONNECT_ATTEMPTS}) reached, stopping reconnection`);
          setError('Maximum reconnection attempts reached');
          isReconnectingRef.current = false;
          return;
        }
        
        // Calculate delay with exponential backoff
        const delay = Math.min(
          reconnectDelayRef.current * Math.pow(RECONNECT_DECAY, reconnectAttemptsRef.current),
          MAX_RECONNECT_DELAY
        );
        
        console.log(`WebSocket reconnecting in ${delay}ms (attempt ${reconnectAttemptsRef.current + 1})`);
        
        // Schedule reconnection
        reconnectTimeoutRef.current = setTimeout(() => {
          reconnectAttemptsRef.current++;
          isReconnectingRef.current = false;
          connectWebSocket();
        }, delay);
      };
      
      ws.onerror = (event) => {
        console.error('WebSocket error:', event);
        setError('WebSocket connection error');
      };
      
      ws.onmessage = handleMessage;
    } catch (err) {
      console.error('Error creating WebSocket connection:', err);
      setError(err instanceof Error ? err.message : 'Unknown WebSocket error');
      isReconnectingRef.current = false;
      
      // Schedule reconnection
      const delay = Math.min(
        reconnectDelayRef.current * Math.pow(RECONNECT_DECAY, reconnectAttemptsRef.current),
        MAX_RECONNECT_DELAY
      );
      
      console.log(`WebSocket reconnecting after error in ${delay}ms (attempt ${reconnectAttemptsRef.current + 1})`);
      
      reconnectTimeoutRef.current = setTimeout(() => {
        reconnectAttemptsRef.current++;
        connectWebSocket();
      }, delay);
    }
  }, [url, handleMessage, setupPingInterval, cleanupConnection]);
  
  // Connect WebSocket on mount
  useEffect(() => {
    console.log('Initializing WebSocket connection');
    intentionalCloseRef.current = false;
    isReconnectingRef.current = false;
    reconnectAttemptsRef.current = 0;
    reconnectDelayRef.current = INITIAL_RECONNECT_DELAY;
    
    connectWebSocket();
    
    // Cleanup on unmount
    return () => {
      intentionalCloseRef.current = true; // Mark close as intentional
      cleanupConnection();
    };
  }, [connectWebSocket, cleanupConnection]);
  
  // Function to subscribe to job updates
  const subscribeToJob = useCallback((jobId: string) => {
    if (!jobId) return;
    
    // Check if already subscribed
    if (subscribedJobsRef.current.has(jobId)) {
      console.log(`Already subscribed to job ${jobId}`);
      return;
    }
    
    // Add to subscribed set
    subscribedJobsRef.current.add(jobId);
    console.log(`Subscribing to updates for job ${jobId}`);
    
    // Send subscription message if connected
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({ action: 'subscribe', job_id: jobId }));
    }
  }, []);
  
  // Function to unsubscribe from job updates
  const unsubscribeFromJob = useCallback((jobId: string) => {
    if (!jobId) return;
    
    // Check if not subscribed
    if (!subscribedJobsRef.current.has(jobId)) {
      return;
    }
    
    // Remove from subscribed set
    subscribedJobsRef.current.delete(jobId);
    console.log(`Unsubscribing from updates for job ${jobId}`);
    
    // Send unsubscription message if connected
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({ action: 'unsubscribe', job_id: jobId }));
    }
  }, []);
  
  // Add a message listener
  const addMessageListener = useCallback((listener: (message: WebSocketMessage) => void) => {
    console.log('Adding WebSocket message listener');
    messageListenersRef.current.add(listener);
  }, []);
  
  // Remove a message listener
  const removeMessageListener = useCallback((listener: (message: WebSocketMessage) => void) => {
    console.log('Removing WebSocket message listener');
    messageListenersRef.current.delete(listener);
  }, []);
  
  // Create context value
  const contextValue: WebSocketContextType = {
    isConnected,
    lastMessage,
    error,
    subscribeToJob,
    unsubscribeFromJob,
    addMessageListener,
    removeMessageListener
  };
  
  return (
    <WebSocketContext.Provider value={contextValue}>
      {children}
    </WebSocketContext.Provider>
  );
};

// Custom hook to use the WebSocket context
export const useWebSocketContext = (): WebSocketContextType => {
  const context = useContext(WebSocketContext);
  
  if (!context) {
    throw new Error('useWebSocketContext must be used within a WebSocketProvider');
  }
  
  return context;
};

export default WebSocketContext; 