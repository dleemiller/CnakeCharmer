// src/components/ResultsViewer.tsx

import React, { useEffect, useMemo } from 'react';
import {
  Paper,
  Box,
  Stack,
  Typography,
  Chip,
  Alert,
  CircularProgress
} from '@mui/material';
import { Tabs as ResultsTabs } from './Tabs';
import { TabPanel } from './TabPanel';
import { AnalysisDetails } from './AnalysisDetails';
import { CodeViewer } from './CodeViewer';
import useJobStatus from '../hooks/useJobStatus';

interface ResultsViewerProps {
  jobId: string | null;
}

const ResultsViewer: React.FC<ResultsViewerProps> = ({ jobId }) => {
  const { job, result, loading, error, refresh, isCompleted } = useJobStatus(jobId);
  const [tab, setTab] = React.useState(0);

  useEffect(() => {
    if (jobId) refresh();
  }, [jobId, refresh]);

  useEffect(() => {
    if (job?.status === 'completed') {
      refresh();
    }
  }, [job?.status, refresh]);

  if (!jobId) {
    return <Paper sx={{ p: 4 }}>Select a job to view results.</Paper>;
  }

  if (loading) {
    return (
      <Paper sx={{ p: 4, textAlign: 'center' }}>
        <CircularProgress />
        <Typography>Loading results…</Typography>
      </Paper>
    );
  }

  if (error) {
    return <Alert severity="error">{error}</Alert>;
  }

  const status = job?.status ?? 'unknown';
  const color = status === 'completed' ? 'success' : status === 'error' ? 'error' : 'info';

  return (
    <Paper sx={{ p: 3 }}>
      <Stack direction="row" justifyContent="space-between" alignItems="center" mb={2}>
        <Typography variant="subtitle2">Job ID</Typography>
        <Typography variant="body2">{jobId}</Typography>
        <Chip label={status} color={color as any} size="small" />
      </Stack>

      {isCompleted && result ? (
        <>
          <ResultsTabs value={tab} onChange={(_, v) => setTab(v)} />
          <TabPanel value={tab} index={0}>
            <AnalysisDetails result={result} />
          </TabPanel>
          <TabPanel value={tab} index={1}>
            <CodeViewer result={result} />
          </TabPanel>
        </>
      ) : (
        <Alert severity="info">Analysis in progress…</Alert>
      )}
    </Paper>
  );
};

export default ResultsViewer;
