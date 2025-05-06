// src/components/JobHistory.tsx
import React, { useEffect, useMemo, useState } from 'react';
import {
  Paper, Typography, Button,
  Table, TableHead, TableBody, TableRow, TableCell, TableContainer,
  Stack, Box, Chip, IconButton, Tooltip, Snackbar, Alert, CircularProgress
} from '@mui/material';
import RefreshIcon from '@mui/icons-material/Refresh';
import { useJobContext } from '../contexts/JobContext';

interface JobHistoryProps {
  onSelectJob?: (jobId: string) => void;
}

const JobHistory: React.FC<JobHistoryProps> = ({ onSelectJob }) => {
  const { jobs, currentJobId, systemStatus, refreshSystemStatus } = useJobContext();
  const [loading, setLoading] = useState(false);
  const [snack, setSnack] = useState({ open: false, msg: '', sev: 'info' as 'info'|'success'|'error' });

  // initial load
  useEffect(() => {
    setLoading(true);
    refreshSystemStatus().finally(() => setLoading(false));
  }, [refreshSystemStatus]);

  const list = useMemo(() => Object.values(jobs).sort((a, b) => b.timestamp - a.timestamp), [jobs]);

  const handleRefresh = () => {
    setLoading(true);
    refreshSystemStatus()
      .then(() => setSnack({ open: true, msg: 'Refreshed', sev: 'success' }))
      .catch(() => setSnack({ open: true, msg: 'Failed to refresh', sev: 'error' }))
      .finally(() => setLoading(false));
  };

  if (!list.length && !loading) {
    return (
      <Paper sx={{ p: 4, textAlign: 'center' }}>
        <Typography>No jobs yet</Typography>
        <Button onClick={handleRefresh} startIcon={<RefreshIcon />} disabled={loading}>
          Refresh
        </Button>
      </Paper>
    );
  }

  return (
    <Stack spacing={2}>
      <Box display="flex" justifyContent="space-between">
        <Typography variant="h6">Job History</Typography>
        <Tooltip title="Refresh">
          <IconButton onClick={handleRefresh} disabled={loading}>
            <RefreshIcon />
          </IconButton>
        </Tooltip>
      </Box>

      {loading && !list.length ? (
        <CircularProgress />
      ) : (
        <TableContainer component={Paper}>
          <Table size="small">
            <TableHead>
              <TableRow>
                <TableCell>Job</TableCell>
                <TableCell>Status</TableCell>
                <TableCell>When</TableCell>
                <TableCell>Action</TableCell>
              </TableRow>
            </TableHead>
            <TableBody>
              {list.map(job => (
                <TableRow key={job.id} selected={job.id === currentJobId} hover>
                  <TableCell sx={{ fontFamily: 'monospace' }}>{job.id.slice(0, 8)}…</TableCell>
                  <TableCell>
                    <Chip label={job.status} size="small" />
                  </TableCell>
                  <TableCell>{new Date(job.timestamp).toLocaleString()}</TableCell>
                  <TableCell>
                    <Button
                      size="small"
                      onClick={() => onSelectJob?.(job.id)}
                      disabled={job.id === currentJobId}
                    >
                      View
                    </Button>
                  </TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </TableContainer>
      )}

      <Snackbar
        open={snack.open}
        autoHideDuration={3000}
        onClose={() => setSnack(s => ({ ...s, open: false }))}
      >
        <Alert severity={snack.sev}>{snack.msg}</Alert>
      </Snackbar>
    </Stack>
  );
};

export default JobHistory;
