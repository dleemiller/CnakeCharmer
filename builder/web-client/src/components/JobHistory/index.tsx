import React, { useEffect, useMemo } from 'react';
import {
  Box,
  Stack,
  Typography,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Paper,
  Button,
  Chip,
  Tooltip,
  IconButton,
  Snackbar,
  Alert,
  CircularProgress
} from '@mui/material';
import RefreshIcon from '@mui/icons-material/Refresh';
import { useJobContext } from '../../contexts/JobContext';
import { Job } from '../../types';

interface JobHistoryProps {
  onSelectJob?: (jobId: string) => void;
}

const JobHistory: React.FC<JobHistoryProps> = ({ onSelectJob }) => {
  const {
    state: { jobs, systemStatus, currentJobId },
    refreshSystemStatus
  } = useJobContext();
  const [snackbar, setSnackbar] = React.useState({
    open: false,
    message: '',
    severity: 'info' as 'info' | 'success' | 'error'
  });
  const [isLoading, setIsLoading] = React.useState(false);
  
  // Load all jobs and fetch status
  useEffect(() => {
    const fetchStatus = async () => {
      try {
        setIsLoading(true);
        await refreshSystemStatus();
      } catch (err) {
        // Handle error but we don't need to store it
        console.error('Failed to fetch system status:', err);
      } finally {
        setIsLoading(false);
      }
    };

    fetchStatus();
  }, [refreshSystemStatus]);
  
  // Convert jobs object to array and sort by timestamp (newest first)
  const jobList = useMemo(() => {
    const list = Object.values(jobs) as Job[];
    return list.sort((a, b) => b.timestamp - a.timestamp);
  }, [jobs]);
  
  // Handle refresh
  const handleRefresh = () => {
    refreshSystemStatus()
      .then(() => {
        setSnackbar({
          open: true,
          message: 'Job history refreshed',
          severity: 'success'
        });
      })
      .catch((error: Error) => {
        setSnackbar({
          open: true,
          message: 'Failed to refresh job history',
          severity: 'error'
        });
        console.error('Error refreshing job history:', error);
      });
  };
  
  // Handle job selection
  const handleSelectJob = (jobId: string) => {
    if (onSelectJob) {
      onSelectJob(jobId);
    }
  };
  
  // Handle snackbar close
  const handleSnackbarClose = () => {
    setSnackbar({ ...snackbar, open: false });
  };
  
  // Status chip colors
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
  
  // Empty state
  if (jobList.length === 0 && !isLoading) {
    return (
      <Paper 
        elevation={2} 
        sx={{ 
          textAlign: 'center', 
          py: 5, 
          px: 3, 
          display: 'flex', 
          flexDirection: 'column',
          alignItems: 'center',
          gap: 2
        }}
      >
        <Typography variant="h6">No Jobs Found</Typography>
        <Typography color="text.secondary" sx={{ mb: 2 }}>
          Submit some Cython code for analysis to see job history.
        </Typography>
        <Button
          variant="contained"
          onClick={handleRefresh}
          startIcon={<RefreshIcon />}
          disabled={isLoading}
        >
          Refresh
        </Button>
      </Paper>
    );
  }
  
  return (
    <Stack spacing={2} width="100%">
      <Box display="flex" justifyContent="space-between" alignItems="center">
        <Typography variant="h6">Job History</Typography>
        <Tooltip title="Refresh job history">
          <IconButton
            aria-label="Refresh job history"
            onClick={handleRefresh}
            disabled={isLoading}
            size="small"
          >
            <RefreshIcon />
          </IconButton>
        </Tooltip>
      </Box>
      
      {isLoading && jobList.length === 0 ? (
        <Box display="flex" justifyContent="center" alignItems="center" py={5}>
          <CircularProgress />
          <Typography sx={{ ml: 2 }}>Loading job history...</Typography>
        </Box>
      ) : (
        <TableContainer component={Paper} variant="outlined">
          <Table size="small">
            <TableHead>
              <TableRow>
                <TableCell>Job ID</TableCell>
                <TableCell>Status</TableCell>
                <TableCell>Timestamp</TableCell>
                <TableCell>Actions</TableCell>
              </TableRow>
            </TableHead>
            <TableBody>
              {jobList.map((job) => (
                <TableRow 
                  key={job.id} 
                  sx={{ 
                    backgroundColor: job.id === currentJobId ? 'rgba(0, 0, 0, 0.04)' : undefined,
                    '&:hover': { backgroundColor: 'rgba(0, 0, 0, 0.08)' }
                  }}
                >
                  <TableCell sx={{ fontFamily: 'monospace', fontSize: '0.75rem' }}>
                    <Tooltip title={job.id}>
                      <span>{job.id.substring(0, 8)}...</span>
                    </Tooltip>
                  </TableCell>
                  <TableCell>
                    <Chip 
                      label={job.status}
                      color={getStatusColor(job.status)}
                      size="small"
                      variant="outlined"
                    />
                  </TableCell>
                  <TableCell sx={{ fontSize: '0.75rem' }}>
                    {new Date(job.timestamp).toLocaleString()}
                  </TableCell>
                  <TableCell>
                    <Button
                      size="small"
                      variant="outlined"
                      color="primary"
                      onClick={() => handleSelectJob(job.id)}
                      disabled={job.id === currentJobId}
                    >
                      View Details
                    </Button>
                  </TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </TableContainer>
      )}
      
      <Snackbar 
        open={snackbar.open} 
        autoHideDuration={3000} 
        onClose={handleSnackbarClose}
      >
        <Alert 
          onClose={handleSnackbarClose} 
          severity={snackbar.severity} 
          sx={{ width: '100%' }}
        >
          {snackbar.message}
        </Alert>
      </Snackbar>
    </Stack>
  );
};

export default JobHistory;