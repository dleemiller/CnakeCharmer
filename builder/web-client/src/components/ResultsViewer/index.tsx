// src/components/ResultsViewer/index.tsx

import React, { useEffect, useRef, useMemo, useCallback } from 'react';
import {
  Box,
  Stack,
  Typography,
  Paper,
  Chip,
  Alert,
  AlertTitle,
  CircularProgress,
  Tabs,
  Tab,
  Button,
  IconButton
} from '@mui/material';
import RefreshIcon from '@mui/icons-material/Refresh';
import { AnalysisResult } from '../../types';
import { TabPanel } from './TabPanel';
import { AnalysisDetails } from './AnalysisDetails';
import { CodeViewer } from './CodeViewer';
import { Tabs as ResultsTabs } from './Tabs';
import useJobStatus from '../../hooks/useJobStatus';

interface ResultsViewerProps {
  jobId: string | null;
}

export const ResultsViewer: React.FC<ResultsViewerProps> = ({ jobId }) => {
  const { job, loading, error, refresh, isCompleted } = useJobStatus(jobId);
  const [tabValue, setTabValue] = React.useState(0);
  const prevJobIdRef = useRef<string | null>(null);
  const prevJobStatusRef = useRef<string | null>(null);
  const refreshTriggeredRef = useRef<boolean>(false);

  // Memoize result to prevent unnecessary re-renders
  const result = useMemo(() => job?.result, [job?.result]);

  // Memoize tab change handler
  const handleTabChange = useCallback((_: unknown, newValue: number) => {
    console.log(`ResultsViewer: Changing tab to ${newValue}`);
    setTabValue(newValue);
  }, []);

  // Log job status changes for debugging (only when it changes)
  useEffect(() => {
    if (job?.status && job.status !== prevJobStatusRef.current) {
      console.log(`ResultsViewer: Job ${jobId} status changed to: ${job.status}, isCompleted: ${isCompleted}`);
      console.log(`ResultsViewer: Has result:`, !!result);
      prevJobStatusRef.current = job.status;
    }
  }, [job?.status, jobId, isCompleted, result]);

  // Reset tab value when job changes
  useEffect(() => {
    if (jobId !== prevJobIdRef.current) {
      console.log(`ResultsViewer: New job selected: ${jobId}`);
      setTabValue(0);
      // Reset refresh state when job changes
      refreshTriggeredRef.current = false;
      prevJobStatusRef.current = null;
      prevJobIdRef.current = jobId;
    }
  }, [jobId]);

  // Watch for job status changes to completed
  useEffect(() => {
    const currentStatus = job?.status || null;
    
    // Only log when status actually changes
    if (currentStatus !== prevJobStatusRef.current) {
      console.log(`ResultsViewer: Status change detected, current: ${currentStatus}, previous: ${prevJobStatusRef.current}`);
    }
    
    // Only refresh once when status changes to completed, and avoid refreshing if we already have the result
    if (currentStatus === 'completed' && 
        prevJobStatusRef.current !== 'completed' && 
        !refreshTriggeredRef.current && 
        !result) {
      console.log('ResultsViewer: Job completed, refreshing results');
      refreshTriggeredRef.current = true;
      refresh();
    }
    
    // Update previous status reference (when it changes)
    if (currentStatus !== prevJobStatusRef.current) {
      prevJobStatusRef.current = currentStatus;
    }
  }, [job?.status, refresh, result]);

  // Initial load on mount (only once)
  useEffect(() => {
    if (jobId && !refreshTriggeredRef.current) {
      console.log(`ResultsViewer: Initial load for job ${jobId}`);
      refreshTriggeredRef.current = true;
      refresh();
    }
  }, [jobId, refresh]);

  // Memoize the tab content to prevent unnecessary re-renders
  const summaryTab = useMemo(() => {
    if (!isCompleted || !result) return null;
    return <AnalysisDetails result={result as AnalysisResult} />;
  }, [isCompleted, result]);

  const detailedTab = useMemo(() => {
    if (!isCompleted || !result) return null;
    return <CodeViewer result={result as AnalysisResult} />;
  }, [isCompleted, result]);

  if (!jobId) {
    return (
      <Paper sx={{ p: 4, textAlign: 'center' }}>
        <Typography variant="h6" gutterBottom>
          No Job Selected
        </Typography>
        <Typography color="text.secondary">
          Submit code for analysis or select a job from the history to view results.
        </Typography>
      </Paper>
    );
  }

  if (loading) {
    return (
      <Paper sx={{ p: 4, textAlign: 'center' }}>
        <CircularProgress size={40} sx={{ mb: 2 }} />
        <Typography variant="h6" gutterBottom>
          Loading Results
        </Typography>
        <Typography color="text.secondary">
          Retrieving analysis results for job {jobId}...
        </Typography>
      </Paper>
    );
  }

  if (error) {
    return (
      <Alert severity="error" sx={{ mt: 2 }}>
        <AlertTitle>Error</AlertTitle>
        {error}
      </Alert>
    );
  }

  if (!job) {
    return (
      <Paper sx={{ p: 4, textAlign: 'center' }}>
        <Typography variant="h6" gutterBottom>
          Job not found
        </Typography>
      </Paper>
    );
  }

  const getStatusColor = (status: string): 'success' | 'info' | 'error' | 'default' => {
    switch (status) {
      case 'completed':
        return 'success';
      case 'submitted':
      case 'processing':
        return 'info';
      case 'error':
        return 'error';
      default:
        return 'default';
    }
  };

  return (
    <Paper sx={{ p: 3 }}>
      <Box sx={{ mb: 3, display: 'flex', justifyContent: 'space-between' }}>
        <Stack>
          <Typography variant="subtitle2" color="text.secondary">
            Job ID
          </Typography>
          <Typography variant="body2" fontFamily="monospace">
            {jobId}
          </Typography>
        </Stack>

        <Stack alignItems="flex-end">
          <Typography variant="subtitle2" color="text.secondary">
            Status
          </Typography>
          <Chip
            label={job.status}
            color={getStatusColor(job.status)}
            size="small"
            variant="outlined"
          />
        </Stack>
      </Box>

      {isCompleted && result ? (
        <>
          <ResultsTabs value={tabValue} onChange={handleTabChange} />

          <TabPanel value={tabValue} index={0}>
            {summaryTab}
          </TabPanel>

          <TabPanel value={tabValue} index={1}>
            {detailedTab}
          </TabPanel>
        </>
      ) : (
        <Box sx={{ mt: 3 }}>
          {job.status === 'submitted' || job.status === 'processing' ? (
            <Alert severity="info">
              <AlertTitle>Analysis in progress</AlertTitle>
              Your code is being analyzed. This may take a few moments.
            </Alert>
          ) : job.status === 'error' ? (
            <Alert severity="error">
              <AlertTitle>Analysis failed</AlertTitle>
              {result?.error || 'An unknown error occurred during analysis.'}
            </Alert>
          ) : (
            <Alert severity="warning">
              <AlertTitle>Unknown status</AlertTitle>
              Current status: {job.status}
            </Alert>
          )}
        </Box>
      )}
    </Paper>
  );
};

export default ResultsViewer;
