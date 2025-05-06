import React, { createContext, useContext, useReducer, useEffect, ReactNode, useCallback, useRef } from 'react';
import apiClient from '../api/apiClient';
import { Job, AnalysisResult, SystemStatus } from '../types';
import { useSocketContext, JobUpdate } from './SocketContext';
import isEqual from 'lodash/isEqual';

// State and action types
interface JobState {
  currentJobId: string | null;
  jobs: Record<string, Job>;
  isLoading: boolean;
  error: string | null;
  systemStatus: SystemStatus | null;
  lastUpdated: Record<string, number>; // Track last update time for each job
}

type JobAction =
  | { type: 'SET_CURRENT_JOB'; payload: string }
  | { type: 'ADD_JOB'; payload: Job }
  | { type: 'UPDATE_JOB_STATUS'; payload: { id: string; status: string } }
  | { type: 'SET_JOB_RESULT'; payload: { id: string; result: AnalysisResult } }
  | { type: 'SET_LOADING'; payload: boolean }
  | { type: 'SET_ERROR'; payload: string | null }
  | { type: 'SET_SYSTEM_STATUS'; payload: SystemStatus }
  | { type: 'CLEAR_CURRENT_JOB' }
  | { type: 'UPDATE_TIMESTAMP'; payload: { id: string; timestamp: number } };

// Initial state
const initialState: JobState = {
  currentJobId: null,
  jobs: {},
  isLoading: false,
  error: null,
  systemStatus: null,
  lastUpdated: {},
};

// Tracks processed WebSocket messages globally to avoid duplicates across component remounts
const processedMessages = new Set<string>();
const PROCESSED_MESSAGE_LIMIT = 100; // Prevent memory leaks by limiting size

// Throttle times (ms)
const API_THROTTLE_TIME = 1000; // Minimum time between API calls for the same resource
const THROTTLE_EXCEPTIONS = new Set<string>(['initial_load']); // IDs that bypass throttling

// Reducer
const jobReducer = (state: JobState, action: JobAction): JobState => {
  switch (action.type) {
    case 'SET_CURRENT_JOB':
      // Skip if already current job
      if (state.currentJobId === action.payload) {
        return state;
      }
      return {
        ...state,
        currentJobId: action.payload,
      };
    case 'ADD_JOB':
      // Skip if job already exists with the same data
      if (state.jobs[action.payload.id] && 
          isEqual(state.jobs[action.payload.id], action.payload)) {
        return state;
      }
      return {
        ...state,
        jobs: {
          ...state.jobs,
          [action.payload.id]: action.payload,
        },
        lastUpdated: {
          ...state.lastUpdated,
          [action.payload.id]: Date.now()
        }
      };
    case 'UPDATE_JOB_STATUS':
      // Skip if status hasn't changed
      if (state.jobs[action.payload.id]?.status === action.payload.status) {
        return state;
      }
      return {
        ...state,
        jobs: {
          ...state.jobs,
          [action.payload.id]: {
            ...state.jobs[action.payload.id],
            status: action.payload.status,
          },
        },
        lastUpdated: {
          ...state.lastUpdated,
          [action.payload.id]: Date.now()
        }
      };
    case 'SET_JOB_RESULT':
      // Create a completely new job object to ensure React detects the change
      const updatedJob = {
        ...state.jobs[action.payload.id],
        result: action.payload.result,
        status: action.payload.result.status || state.jobs[action.payload.id]?.status
      };
      
      // Skip if the job and result are identical (deep comparison)
      if (state.jobs[action.payload.id] && 
          isEqual(state.jobs[action.payload.id].result, updatedJob.result) &&
          state.jobs[action.payload.id].status === updatedJob.status) {
        console.log('Job result and status unchanged, skipping update');
        return state;
      }
      
      console.log('Updating job in state:', {
        id: action.payload.id,
        oldStatus: state.jobs[action.payload.id]?.status,
        newStatus: updatedJob.status
      });
      
      return {
        ...state,
        jobs: {
          ...state.jobs,
          [action.payload.id]: updatedJob
        },
        lastUpdated: {
          ...state.lastUpdated,
          [action.payload.id]: Date.now()
        }
      };
    case 'SET_LOADING':
      // Skip if loading state hasn't changed
      if (state.isLoading === action.payload) {
        return state;
      }
      return {
        ...state,
        isLoading: action.payload,
      };
    case 'SET_ERROR':
      // Skip if error state hasn't changed
      if (state.error === action.payload) {
        return state;
      }
      return {
        ...state,
        error: action.payload,
      };
    case 'SET_SYSTEM_STATUS':
      // Skip if system status hasn't changed (deep comparison)
      if (state.systemStatus && isEqual(state.systemStatus, action.payload)) {
        console.log('System status unchanged, skipping update');
        return state;
      }
      return {
        ...state,
        systemStatus: action.payload,
        lastUpdated: {
          ...state.lastUpdated,
          systemStatus: Date.now()
        }
      };
    case 'CLEAR_CURRENT_JOB':
      // Skip if already null
      if (state.currentJobId === null) {
        return state;
      }
      return {
        ...state,
        currentJobId: null,
      };
    case 'UPDATE_TIMESTAMP':
      return {
        ...state,
        lastUpdated: {
          ...state.lastUpdated,
          [action.payload.id]: action.payload.timestamp
        }
      };
    default:
      return state;
  }
};

// Context
interface JobContextType {
  state: JobState;
  submitCode: (code: string) => Promise<string>;
  loadJobResult: (jobId: string) => Promise<AnalysisResult>;
  refreshSystemStatus: () => Promise<SystemStatus>;
  setCurrentJob: (jobId: string | null) => void;
}

const JobContext = createContext<JobContextType | undefined>(undefined);

// Provider component
interface JobProviderProps {
  children: ReactNode;
}

// Global flag to track if system status has been loaded
let systemStatusLoaded = false;

export const JobProvider: React.FC<JobProviderProps> = ({ children }) => {
  const [state, dispatch] = useReducer(jobReducer, initialState);
  const { subscribeToJob, unsubscribeFromJob, addMessageListener, isConnected } = useSocketContext();
  const lastProcessedMessageRef = useRef<JobUpdate | null>(null);
  
  // Store in-progress requests to prevent duplicate API calls
  const pendingRequestsRef = useRef<Record<string, Promise<any>>>({});
  
  // Throttling logic
  const lastRequestTimestampsRef = useRef<Record<string, number>>({});
  const THROTTLE_TIME_MS = 2000; // 2 seconds
  
  const shouldThrottleRequest = useCallback((resourceId: string, bypassId?: string): boolean => {
    // If this is a bypass request, don't throttle
    if (bypassId && THROTTLE_EXCEPTIONS.has(bypassId)) {
      return false;
    }
    
    const now = Date.now();
    const lastRequest = lastRequestTimestampsRef.current[resourceId] || 0;
    const shouldThrottle = now - lastRequest < THROTTLE_TIME_MS;
    
    // Check if there's already a pending request for this resource
    if (pendingRequestsRef.current[resourceId] !== undefined) {
      console.log(`Throttling request for ${resourceId}: request already in progress`);
      return true;
    }
    
    if (!shouldThrottle) {
      lastRequestTimestampsRef.current[resourceId] = now;
    }
    
    return shouldThrottle;
  }, []);
  
  // Refresh system status
  const refreshSystemStatus = useCallback(async (): Promise<SystemStatus> => {
    if (shouldThrottleRequest('systemStatus')) {
      console.log('Throttling system status refresh');
      if (state.systemStatus) {
        return state.systemStatus;
      }
      throw new Error('System status refresh throttled');
    }
    
    try {
      dispatch({ type: 'SET_LOADING', payload: true });
      const status = await apiClient.getSystemStatus();
      systemStatusLoaded = true;
      dispatch({ type: 'SET_SYSTEM_STATUS', payload: status });
      return status;
    } catch (error) {
      console.error('Failed to refresh system status:', error);
      throw error;
    } finally {
      dispatch({ type: 'SET_LOADING', payload: false });
    }
  }, [shouldThrottleRequest, state.systemStatus]);
  
  // Handle Socket.io messages for system status
  useEffect(() => {
    const unsubscribe = addMessageListener('system_status', (data: SystemStatus) => {
      console.log('Received system status from Socket.io:', data);
      
      // Mark that system status has been loaded globally
      systemStatusLoaded = true;
      
      // Only update if the status has actually changed
      const currentStatus = state.systemStatus;
      const hasStatusChanged = !currentStatus || 
          currentStatus.status !== data.status ||
          !isEqual(currentStatus.pending_jobs, data.pending_jobs) ||
          !isEqual(currentStatus.completed_jobs, data.completed_jobs) ||
          !isEqual(currentStatus.archived_jobs, data.archived_jobs) ||
          !isEqual(currentStatus.in_memory_status, data.in_memory_status);
      
      if (hasStatusChanged) {
        dispatch({ type: 'SET_SYSTEM_STATUS', payload: data });
        
        // Update job list from system status
        const allJobs = [...data.pending_jobs, ...data.completed_jobs];
        allJobs.forEach(job => {
          const jobId = job.replace('.tar.gz', '').replace('.json', '');
          if (!state.jobs[jobId]) {
            const jobStatus = data.in_memory_status && data.in_memory_status[jobId]
              ? data.in_memory_status[jobId]
              : { 
                  status: job.endsWith('.json') ? 'completed' : 'pending', 
                  timestamp: Date.now() 
                };
            
            dispatch({
              type: 'ADD_JOB',
              payload: {
                id: jobId,
                status: jobStatus.status,
                timestamp: jobStatus.timestamp || Date.now(),
              },
            });
          }
        });
      } else {
        console.log('System status unchanged, skipping update');
      }
    });
    
    return unsubscribe;
  }, [addMessageListener, state.jobs, state.systemStatus]);
  
  // Handle Socket.io messages for job updates
  useEffect(() => {
    const unsubscribe = addMessageListener('job_update', (data: JobUpdate) => {
      console.log('Received job update from Socket.io:', data);
      
      // Skip if this is the same message we just processed
      if (lastProcessedMessageRef.current && 
          isEqual(lastProcessedMessageRef.current, data)) {
        console.log('Skipping duplicate job update');
        return;
      }
      
      lastProcessedMessageRef.current = data;
      
      const { job_id, status, result } = data;
      
      if (!job_id || !status) {
        console.error('Missing job_id or status in job_update message');
        return;
      }
      
      // Check if the job exists and if the status or result has changed
      const currentJob = state.jobs[job_id];
      if (currentJob) {
        // Check if status or result has changed
        const hasStatusChanged = currentJob.status !== status;
        const hasResultChanged = result && !isEqual(currentJob.result, result);
        
        if (hasStatusChanged || hasResultChanged) {
          console.log(`Updating job ${job_id} from status ${currentJob.status} to ${status}`);
          dispatch({
            type: 'SET_JOB_RESULT',
            payload: { 
              id: job_id, 
              result: {
                ...result,
                status: status
              }
            },
          });
        } else {
          console.log(`No changes detected for job ${job_id}, skipping update`);
        }
      } else {
        // Add job to state if it doesn't exist
        console.log(`Adding new job ${job_id} with status ${status}`);
        dispatch({
          type: 'ADD_JOB',
          payload: {
            id: job_id,
            status: status,
            timestamp: Date.now(),
            result: result
          },
        });
      }
    });
    
    return unsubscribe;
  }, [addMessageListener, state.jobs]);
  
  // Initial system status load - with fallback if Socket.io not connected
  useEffect(() => {
    // Skip if system status already loaded
    if (systemStatusLoaded) {
      console.log('JobContext: System status already loaded globally, skipping initial fetch');
      return;
    }
    
    // Try to load system status if not connected to Socket.io
    if (!isConnected) {
      console.log('JobContext: Not connected to Socket.io, fetching system status via HTTP');
      refreshSystemStatus().catch(error => {
        console.error('Failed to load initial system status:', error);
      });
    }
  }, [isConnected, refreshSystemStatus]);
  
  // Submit code for analysis
  const submitCode = async (code: string): Promise<string> => {
    try {
      dispatch({ type: 'SET_LOADING', payload: true });
      dispatch({ type: 'SET_ERROR', payload: null });
      
      const result = await apiClient.submitCode(code);
      console.log('Code submission result:', result);
      
      // Add job to state
      dispatch({
        type: 'ADD_JOB',
        payload: {
          id: result.job_id,
          status: result.status,
          timestamp: result.timestamp || Date.now(),
        },
      });
      
      // Set as current job
      dispatch({ type: 'SET_CURRENT_JOB', payload: result.job_id });
      
      // Subscribe to job updates
      console.log('Subscribing to new job:', result.job_id);
      subscribeToJob(result.job_id);
      
      return result.job_id;
    } catch (error) {
      let errorMessage = 'Failed to submit code';
      if (error instanceof Error) {
        errorMessage = error.message;
      }
      dispatch({ type: 'SET_ERROR', payload: errorMessage });
      throw error;
    } finally {
      dispatch({ type: 'SET_LOADING', payload: false });
    }
  };
  
  // Load job result with throttling and caching
  const loadJobResult = async (jobId: string, bypassThrottle?: string): Promise<AnalysisResult> => {
    try {
      // Check if we should throttle this request
      if (shouldThrottleRequest(jobId, bypassThrottle)) {
        console.log(`Throttling job result fetch for ${jobId} (too recent)`);
        // If we already have a result, return it
        if (state.jobs[jobId]?.result) {
          return state.jobs[jobId].result as AnalysisResult;
        }
        
        // If there's a pending request, wait for it instead of creating a new one
        const pendingRequest = pendingRequestsRef.current[jobId];
        if (pendingRequest !== undefined) {
          console.log(`Reusing in-flight request for ${jobId}`);
          return pendingRequest;
        }
      }
      
      console.log('Loading job result for:', jobId);
      dispatch({ type: 'SET_LOADING', payload: true });
      dispatch({ type: 'SET_ERROR', payload: null });
      
      // Create a promise and store it in the pending requests ref
      const requestPromise = apiClient.getJobResults(jobId);
      pendingRequestsRef.current[jobId] = requestPromise;
      
      try {
        const result = await requestPromise;
        console.log('Loaded job result:', result);
        
        // Check if job exists in state
        if (!state.jobs[jobId]) {
          console.log('Adding new job to state:', jobId);
          // Add job to state if it doesn't exist
          dispatch({
            type: 'ADD_JOB',
            payload: {
              id: jobId,
              status: result.status,
              timestamp: result.timestamp || Date.now(),
            },
          });
        }
        
        // Check if the result has actually changed before updating
        const currentJob = state.jobs[jobId];
        const hasResultChanged = !currentJob?.result || 
          !isEqual(currentJob.result, result);
        
        if (hasResultChanged) {
          // Update job result
          console.log('Updating job result in state');
          dispatch({
            type: 'SET_JOB_RESULT',
            payload: { id: jobId, result },
          });
        } else {
          console.log('Job result unchanged, skipping update');
        }
        
        // Update timestamp even if result hasn't changed
        dispatch({
          type: 'UPDATE_TIMESTAMP',
          payload: { id: jobId, timestamp: Date.now() },
        });
        
        return result;
      } finally {
        // Always remove the pending request when done
        delete pendingRequestsRef.current[jobId];
      }
    } catch (error) {
      console.error('Error loading job result:', error);
      let errorMessage = `Failed to load results for job ${jobId}`;
      if (error instanceof Error) {
        errorMessage = error.message;
      }
      dispatch({ type: 'SET_ERROR', payload: errorMessage });
      throw error;
    } finally {
      dispatch({ type: 'SET_LOADING', payload: false });
    }
  };
  
  // Set current job and subscribe to updates
  const setCurrentJob = (jobId: string | null) => {
    // Skip if there's no change
    if (jobId === state.currentJobId) {
      return;
    }
    
    // Unsubscribe from previous job if any
    if (state.currentJobId) {
      unsubscribeFromJob(state.currentJobId);
    }
    
    if (jobId === null) {
      dispatch({ type: 'CLEAR_CURRENT_JOB' });
    } else {
      // Only update if the job ID has changed
      if (state.currentJobId !== jobId) {
        dispatch({ type: 'SET_CURRENT_JOB', payload: jobId });
        
        // Load job details if not already loaded and not already loading
        if (!state.jobs[jobId]?.result && !state.isLoading) {
          // Only load if not throttled
          if (!shouldThrottleRequest(jobId)) {
            loadJobResult(jobId).catch(console.error);
          }
        }
        
        // Subscribe to job updates via Socket.io
        subscribeToJob(jobId);
      }
    }
  };
  
  // Context value
  const contextValue: JobContextType = {
    state,
    submitCode,
    loadJobResult,
    refreshSystemStatus,
    setCurrentJob,
  };
  
  return (
    <JobContext.Provider value={contextValue}>
      {children}
    </JobContext.Provider>
  );
};

// Custom hook to use the job context
export const useJobContext = (): JobContextType => {
  const context = useContext(JobContext);
  
  if (context === undefined) {
    throw new Error('useJobContext must be used within a JobProvider');
  }
  
  return context;
};

export default JobContext;