import React, { createContext, useContext, useEffect, useState, useRef, useCallback, ReactNode } from 'react';
import { io, Socket } from 'socket.io-client';
import { SOCKET_IO_URL } from '../config';

// Message types
export interface SystemStatus {
  status: string;
  pending_jobs: string[];
  completed_jobs: string[];
  archived_jobs: string[];
  in_memory_status: Record<string, any>;
}

export interface JobUpdate {
  type: 'job_update';
  job_id: string;
  status: string;
  result: any;
}

export type SocketMessage = SystemStatus | JobUpdate;

// Context type
interface SocketContextType {
  isConnected: boolean;
  socket: Socket | null;
  lastMessage: SocketMessage | null;
  error: string | null;
  subscribeToJob: (jobId: string) => void;
  unsubscribeFromJob: (jobId: string) => void;
  addMessageListener: (event: string, listener: (data: any) => void) => () => void;
}

// Create the context
const SocketContext = createContext<SocketContextType | null>(null);

// Provider props
interface SocketProviderProps {
  children: ReactNode;
  url?: string;
}

export const SocketProvider: React.FC<SocketProviderProps> = ({ 
  children, 
  url = SOCKET_IO_URL 
}) => {
  const [socket, setSocket] = useState<Socket | null>(null);
  const [isConnected, setIsConnected] = useState<boolean>(false);
  const [lastMessage, setLastMessage] = useState<SocketMessage | null>(null);
  const [error, setError] = useState<string | null>(null);
  
  // Track subscribed jobs
  const subscribedJobsRef = useRef<Set<string>>(new Set());
  
  // Create and connect the socket
  useEffect(() => {
    console.log(`Connecting to Socket.io server at ${url}`);
    
    // Create socket with reconnection options
    const socketInstance = io(url, {
      reconnection: true,
      reconnectionAttempts: 5,
      reconnectionDelay: 2000,
      reconnectionDelayMax: 10000,
      timeout: 20000,
    });
    
    // Set up event handlers
    socketInstance.on('connect', () => {
      console.log('Socket.io connected');
      setIsConnected(true);
      setError(null);
      
      // Resubscribe to jobs
      subscribedJobsRef.current.forEach(jobId => {
        console.log(`Resubscribing to job ${jobId}`);
        socketInstance.emit('subscribe', { job_id: jobId });
      });
    });
    
    socketInstance.on('disconnect', (reason: string) => {
      console.log(`Socket.io disconnected: ${reason}`);
      setIsConnected(false);
    });
    
    socketInstance.on('connect_error', (err: Error) => {
      console.error('Socket.io connection error:', err);
      setError(`Connection error: ${err.message}`);
    });
    
    // Handle system status updates
    socketInstance.on('system_status', (data: SystemStatus) => {
      console.log('Received system status:', data);
      setLastMessage(data);
    });
    
    // Handle job updates
    socketInstance.on('job_update', (data: JobUpdate) => {
      console.log('Received job update:', data);
      setLastMessage(data);
    });
    
    // Save socket reference
    setSocket(socketInstance);
    
    // Clean up on unmount
    return () => {
      console.log('Disconnecting Socket.io');
      socketInstance.disconnect();
    };
  }, [url]);
  
  // Subscribe to job updates
  const subscribeToJob = useCallback((jobId: string) => {
    if (!jobId) return;
    
    // Skip if already subscribed
    if (subscribedJobsRef.current.has(jobId)) {
      console.log(`Already subscribed to job ${jobId}`);
      return;
    }
    
    console.log(`Subscribing to job ${jobId}`);
    subscribedJobsRef.current.add(jobId);
    
    // Send subscription if connected
    if (socket && isConnected) {
      socket.emit('subscribe', { job_id: jobId });
    }
  }, [socket, isConnected]);
  
  // Unsubscribe from job updates
  const unsubscribeFromJob = useCallback((jobId: string) => {
    if (!jobId) return;
    
    console.log(`Unsubscribing from job ${jobId}`);
    subscribedJobsRef.current.delete(jobId);
    
    // Send unsubscription if connected
    if (socket && isConnected) {
      socket.emit('unsubscribe', { job_id: jobId });
    }
  }, [socket, isConnected]);
  
  // Add a message listener
  const addMessageListener = useCallback((event: string, listener: (data: any) => void) => {
    if (!socket) {
      console.warn('Cannot add listener: socket not connected');
      return () => {}; // Return empty cleanup function
    }
    
    console.log(`Adding listener for ${event} events`);
    socket.on(event, listener);
    
    // Return cleanup function
    return () => {
      console.log(`Removing listener for ${event} events`);
      socket.off(event, listener);
    };
  }, [socket]);
  
  // Create context value
  const contextValue: SocketContextType = {
    isConnected,
    socket,
    lastMessage,
    error,
    subscribeToJob,
    unsubscribeFromJob,
    addMessageListener,
  };
  
  return (
    <SocketContext.Provider value={contextValue}>
      {children}
    </SocketContext.Provider>
  );
};

// Custom hook to use the socket context
export const useSocketContext = (): SocketContextType => {
  const context = useContext(SocketContext);
  
  if (!context) {
    throw new Error('useSocketContext must be used within a SocketProvider');
  }
  
  return context;
};

export default SocketContext; 