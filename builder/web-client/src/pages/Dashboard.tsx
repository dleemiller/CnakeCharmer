// src/pages/Dashboard.tsx

import React, { useState, useEffect, useCallback, useRef, useMemo } from 'react';
import {
  Box,
  Typography,
  Stack,
  Chip,
  useMediaQuery,
  useTheme,
  Drawer,
  Paper,
  IconButton,
  Divider,
  Button
} from '@mui/material';
import MenuIcon from '@mui/icons-material/Menu';
import RefreshIcon from '@mui/icons-material/Refresh';
import { useJobContext } from '../contexts/JobContext';
import CodeEditor from '../components/CodeEditor';
import ResultsViewer from '../components/ResultsViewer';
import JobHistory from '../components/JobHistory';
import { useSocketContext } from '../contexts/SocketContext';
import SystemStatus from '../components/SystemStatus';

const Dashboard: React.FC = () => {
  const { state, setCurrentJob, refreshSystemStatus } = useJobContext();
  const { currentJobId, jobs } = state;
  const [selectedJobId, setSelectedJobId] = useState<string | null>(currentJobId);
  const [forceRender, setForceRender] = useState(0);
  const debounceTimeoutRef = useRef<NodeJS.Timeout | null>(null);
  const prevJobStatusRef = useRef<Record<string, string>>({});
  const [isRefreshing, setIsRefreshing] = useState(false);
  const refreshTimeoutRef = useRef<NodeJS.Timeout | null>(null);

  // Socket.io connection status
  const { isConnected } = useSocketContext();
  const connectionStatus = isConnected ? 'Connected' : 'Disconnected';

  // Mobile drawer for job history
  const [drawerOpen, setDrawerOpen] = useState(false);

  // Theme and responsive layout
  const theme = useTheme();
  const isMobile = useMediaQuery(theme.breakpoints.down('lg'));

  // Debounced force update function
  const debouncedForceUpdate = useCallback(() => {
    if (debounceTimeoutRef.current) {
      clearTimeout(debounceTimeoutRef.current);
    }
    
    debounceTimeoutRef.current = setTimeout(() => {
      console.log('Dashboard: Debounced UI update triggered');
      setForceRender(prev => prev + 1);
      debounceTimeoutRef.current = null;
    }, 300); // Wait 300ms before updating
  }, []);

  // Handle code submission success (define before using in memoized components)
  const handleSubmitSuccess = useCallback((jobId: string) => {
    console.log(`Code submission success, selecting job: ${jobId}`);
    setSelectedJobId(jobId);
    // Trigger update with debounce
    debouncedForceUpdate();
  }, [debouncedForceUpdate]);

  // Memoize the CodeEditor component to prevent it from re-rendering
  const memoizedCodeEditor = useMemo(() => (
    <CodeEditor onSubmitSuccess={handleSubmitSuccess} />
  ), [handleSubmitSuccess]);

  // Memoize the ResultsViewer component
  const memoizedResultsViewer = useMemo(() => (
    <ResultsViewer jobId={selectedJobId} />
  ), [selectedJobId]);

  // Update selected job when current job changes
  useEffect(() => {
    if (currentJobId && currentJobId !== selectedJobId) {
      console.log(`Dashboard: Setting selected job to current job: ${currentJobId}`);
      setSelectedJobId(currentJobId);
    }
  }, [currentJobId, selectedJobId]);

  // Force an update when job status changes, with debounce
  useEffect(() => {
    // This effect will run when the jobs object changes
    if (selectedJobId && jobs[selectedJobId]) {
      const status = jobs[selectedJobId].status;
      const prevStatus = prevJobStatusRef.current[selectedJobId];
      
      // Only update if status has changed
      if (status !== prevStatus) {
        console.log(`Dashboard: Job status changed for ${selectedJobId}: ${prevStatus || 'unknown'} -> ${status}`);
        
        // Update previous status
        prevJobStatusRef.current[selectedJobId] = status;
        
        // If status is completed, force a re-render (debounced)
        if (status === 'completed') {
          console.log('Dashboard: Job completed, triggering debounced UI update');
          debouncedForceUpdate();
        }
      }
    }
  }, [selectedJobId, jobs, debouncedForceUpdate]);
  
  // Cleanup timeout on unmount
  useEffect(() => {
    return () => {
      if (debounceTimeoutRef.current) {
        clearTimeout(debounceTimeoutRef.current);
      }
      if (refreshTimeoutRef.current) {
        clearTimeout(refreshTimeoutRef.current);
      }
    };
  }, []);

  // Handle job selection
  const handleSelectJob = (jobId: string) => {
    console.log(`Selecting job: ${jobId}`);
    setSelectedJobId(jobId);
    setCurrentJob(jobId);
    // Trigger update with debounce
    debouncedForceUpdate();
    if (drawerOpen) setDrawerOpen(false);
  };
  
  // Handle manual refresh with debounce
  const handleRefresh = useCallback(() => {
    if (isRefreshing) return;
    
    console.log('Dashboard: Manual refresh triggered');
    setIsRefreshing(true);
    
    // Clear any existing timeout
    if (refreshTimeoutRef.current) {
      clearTimeout(refreshTimeoutRef.current);
    }
    
    refreshSystemStatus()
      .then(() => {
        // Wait a moment before allowing another refresh
        refreshTimeoutRef.current = setTimeout(() => {
          setIsRefreshing(false);
          refreshTimeoutRef.current = null;
        }, 2000); // 2 second cooldown
        
        // Update the UI
        debouncedForceUpdate();
      })
      .catch(error => {
        console.error('Failed to refresh system status:', error);
        setIsRefreshing(false);
      });
  }, [refreshSystemStatus, debouncedForceUpdate, isRefreshing]);

  // Get current job status
  const currentJob = selectedJobId ? jobs[selectedJobId] : null;
  const jobStatus = currentJob?.status || 'unknown';
  const jobStatusDisplay = (
    <Typography variant="body2" color="text.secondary">
      {selectedJobId
        ? `Selected Job: ${selectedJobId.substring(0, 8)}... (${jobStatus})`
        : 'No job selected'}
    </Typography>
  );

  // === Mobile layout ===
  if (isMobile) {
    return (
      <Box sx={{ p: 3 }}>
        <Stack spacing={3} width="100%">
          <Box display="flex" justifyContent="space-between" alignItems="center">
            <Typography variant="h5">Cython Analysis Dashboard</Typography>
            <IconButton edge="end" color="inherit" aria-label="menu" onClick={() => setDrawerOpen(true)}>
              <MenuIcon />
            </IconButton>
          </Box>

          <Box display="flex" alignItems="center" gap={1} justifyContent="space-between">
            <Box display="flex" alignItems="center" gap={1}>
              <Chip
                label={connectionStatus}
                color={isConnected ? 'success' : 'error'}
                size="small"
                variant="outlined"
              />
              {jobStatusDisplay}
            </Box>
            <Button 
              size="small" 
              startIcon={<RefreshIcon />} 
              onClick={handleRefresh}
              disabled={isRefreshing}
            >
              Refresh
            </Button>
          </Box>

          <SystemStatus />

          <Paper elevation={1} sx={{ p: 3 }}>
            {memoizedCodeEditor}
          </Paper>

          <Paper elevation={1} sx={{ p: 3 }}>
            {memoizedResultsViewer}
          </Paper>
        </Stack>

        <Drawer anchor="right" open={drawerOpen} onClose={() => setDrawerOpen(false)}>
          <Box sx={{ width: '100vw', p: 2 }}>
            <Box display="flex" justifyContent="space-between" alignItems="center" mb={2}>
              <Typography variant="h6">Job History</Typography>
              <IconButton onClick={() => setDrawerOpen(false)}>
                <MenuIcon />
              </IconButton>
            </Box>
            <Divider sx={{ mb: 2 }} />
            <JobHistory key={`history-${forceRender}`} onSelectJob={handleSelectJob} />
          </Box>
        </Drawer>
      </Box>
    );
  }

  // === Desktop layout ===
  return (
    <Box sx={{ p: 4 }}>
      <Stack spacing={3} width="100%">
        <Box display="flex" justifyContent="space-between" alignItems="center">
          <Typography variant="h4">Cython Analysis Dashboard</Typography>
          <SystemStatus />
        </Box>

        <Box display="flex" alignItems="center" gap={1} justifyContent="space-between">
          <Box display="flex" alignItems="center" gap={1}>
            <Chip
              label={connectionStatus}
              color={isConnected ? 'success' : 'error'}
              size="small"
              variant="outlined"
            />
            {jobStatusDisplay}
          </Box>
          <Button 
            size="small" 
            startIcon={<RefreshIcon />} 
            onClick={handleRefresh}
            disabled={isRefreshing}
          >
            Refresh
          </Button>
        </Box>

        <Box sx={{ display: 'flex', gap: 3 }}>
          {/* Main content area */}
          <Box sx={{ flex: 2 }}>
            <Stack spacing={3}>
              <Paper elevation={1} sx={{ p: 3 }}>
                {memoizedCodeEditor}
              </Paper>

              <Paper elevation={1} sx={{ p: 3 }}>
                {memoizedResultsViewer}
              </Paper>
            </Stack>
          </Box>

          {/* Sidebar */}
          <Box sx={{ flex: 1 }}>
            <Paper elevation={1} sx={{ p: 3, height: '100%' }}>
              <JobHistory key={`history-${forceRender}`} onSelectJob={handleSelectJob} />
            </Paper>
          </Box>
        </Box>
      </Stack>
    </Box>
  );
};

export default Dashboard;
