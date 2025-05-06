// src/pages/Dashboard.tsx

import React from 'react';
import { Box, Stack, Typography } from '@mui/material';
import { useJobContext } from '../contexts/JobContext';
import CodeEditor from '../components/CodeEditor';
import JobHistory from '../components/JobHistory';
import ResultsViewer from '../components/ResultsViewer';
import SystemStatus from '../components/SystemStatus';

const Dashboard: React.FC = () => {
  const { currentJobId } = useJobContext();

  return (
    <Box p={4}>
      <Stack spacing={3}>
        <Typography variant="h4">Cython Analysis Dashboard</Typography>
        <SystemStatus />

        <Stack direction="row" spacing={3}>
          <Box flex={2}>
            <CodeEditor onSubmitSuccess={() => {/* no-op */}} />
          </Box>
          <Box flex={1}>
            <JobHistory onSelectJob={() => {/* no-op */}} />
          </Box>
        </Stack>

        <ResultsViewer jobId={currentJobId} />
      </Stack>
    </Box>
  );
};

export default Dashboard;
