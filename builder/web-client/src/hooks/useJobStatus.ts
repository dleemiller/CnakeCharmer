import { useState, useEffect, useRef, useCallback } from 'react';
import { useJobContext } from '../contexts/JobContext';
import { JobResult } from '../types';
import { useSocketContext } from '../contexts/SocketContext';

// Set to track jobs that are currently being loading
const loadingJobs = new Set<string>();

export const useJobStatus = (jobId: string | null) => {
  // Call all hooks at the top level
  const { state, loadJobResult: fetchJobResults, setCurrentJob } = useJobContext();
  const { subscribeToJob, unsubscribeFromJob } = useSocketContext();
  const { jobs, currentJobId } = state;
  
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<JobResult | null>(null);
  
  const prevJobIdRef = useRef<string | null>(null);
  const initialLoadDoneRef = useRef<boolean>(false);
  
  // Function to load job results
  const loadJobResult = useCallback(async (id: string) => {
    if (!id || loadingJobs.has(id)) return;
    
    try {
      setLoading(true);
      loadingJobs.add(id);
      console.log(`useJobStatus: Loading initial result for job ${id}`);
      const result = await fetchJobResults(id);
      setResult(result);
      setError(null);
      return result;
    } catch (err: any) {
      console.error(`Error loading job result for ${id}:`, err);
      setError(err.message || 'Failed to load job results');
    } finally {
      setLoading(false);
      loadingJobs.delete(id);
    }
  }, [fetchJobResults]);
  
  // Function to refresh results manually
  const refresh = useCallback(async () => {
    if (!jobId || loadingJobs.has(jobId)) return;
    
    try {
      setLoading(true);
      loadingJobs.add(jobId);
      console.log(`useJobStatus: Manually refreshing result for job ${jobId}`);
      const result = await fetchJobResults(jobId);
      setResult(result);
      setError(null);
    } catch (err: any) {
      console.error(`Error refreshing job result:`, err);
      setError(err.message || 'Failed to refresh job results');
    } finally {
      setLoading(false);
      loadingJobs.delete(jobId);
    }
  }, [jobId, fetchJobResults]);
  
  // Subscribe to job updates via Socket.io
  useEffect(() => {
    if (!jobId) return;
    
    // Subscribe to job updates with Socket.io
    console.log(`useJobStatus: Subscribing to job ${jobId} via Socket.io`);
    subscribeToJob(jobId);
    
    // Cleanup function to unsubscribe when component unmounts or jobId changes
    return () => {
      if (jobId) {
        console.log(`useJobStatus: Unsubscribing from job ${jobId}`);
        unsubscribeFromJob(jobId);
      }
    };
  }, [jobId, subscribeToJob, unsubscribeFromJob]);
  
  // Only call setCurrentJob if the job ID differs from the currentJobId in context
  useEffect(() => {
    if (jobId && jobId !== currentJobId) {
      console.log(`useJobStatus: Setting current job to ${jobId} (currentJobId was ${currentJobId})`);
      setCurrentJob(jobId);
    }
  }, [jobId, currentJobId, setCurrentJob]);
  
  // Handle initial load and job ID changes
  useEffect(() => {
    // Clean up previous job if job ID changed
    if (prevJobIdRef.current && prevJobIdRef.current !== jobId) {
      loadingJobs.delete(prevJobIdRef.current);
      initialLoadDoneRef.current = false;
    }
    
    prevJobIdRef.current = jobId;
    
    // Only load initially if needed (job not in loading state and initial load not done)
    if (jobId && !initialLoadDoneRef.current && !loadingJobs.has(jobId)) {
      console.log(`useJobStatus: Initial load for job ${jobId}`);
      initialLoadDoneRef.current = true;
      loadJobResult(jobId);
    }
  }, [jobId, loadJobResult]);
  
  // Get the job from context state
  const job = jobId ? jobs[jobId] : null;
  
  // For compatibility with existing code
  const isLoading = loading;
  const isCompleted = job?.status === 'completed';
  const isError = job?.status === 'error';
  const isProcessing = job?.status === 'submitted' || job?.status === 'processing';
  const status = job?.status || 'unknown';
  
  return {
    job,
    loading,
    isLoading,
    error,
    result,
    refresh,
    status,
    isCompleted,
    isError,
    isProcessing
  };
};

export default useJobStatus;
