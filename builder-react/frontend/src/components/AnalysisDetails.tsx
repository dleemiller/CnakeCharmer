// src/components/AnalysisDetails.tsx

import React, { useCallback } from 'react';
import { Box, Paper, Typography, Button } from '@mui/material';
import OpenInNewIcon from '@mui/icons-material/OpenInNew';
import { AnalysisResult } from '../types';
import { getDetailedAnalysisUrl } from '../api/apiClient';

interface AnalysisDetailsProps {
  result: AnalysisResult;
}

export const AnalysisDetails: React.FC<AnalysisDetailsProps> = ({ result }) => {
  const getCount = (category: 'excellent' | 'good') =>
    result.score_distribution?.[category]?.count ?? 0;

  const handleViewFull = useCallback(
    (e: React.MouseEvent) => {
      e.preventDefault();
      window.open(getDetailedAnalysisUrl(result.job_id), '_blank', 'noopener');
    },
    [result.job_id]
  );

  return (
    <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 2 }}>
      {/* Excellent */}
      <Box sx={{ flex: '1 1 calc(25% - 8px)' }}>
        <Paper sx={{ p: 2 }}>
          <Typography variant="h6">Excellent Lines</Typography>
          <Typography variant="h4">{getCount('excellent')}</Typography>
          <Typography variant="body2">Highly optimized</Typography>
        </Paper>
      </Box>

      {/* Good */}
      <Box sx={{ flex: '1 1 calc(25% - 8px)' }}>
        <Paper sx={{ p: 2 }}>
          <Typography variant="h6">Good Lines</Typography>
          <Typography variant="h4">{getCount('good')}</Typography>
          <Typography variant="body2">Well optimized</Typography>
        </Paper>
      </Box>

      {/* Yellow */}
      <Box sx={{ flex: '1 1 calc(25% - 8px)' }}>
        <Paper sx={{ p: 2 }}>
          <Typography variant="h6">Yellow Lines</Typography>
          <Typography variant="h4">{result.yellow_lines}</Typography>
          <Typography variant="body2">Could optimize</Typography>
        </Paper>
      </Box>

      {/* Red */}
      <Box sx={{ flex: '1 1 calc(25% - 8px)' }}>
        <Paper sx={{ p: 2 }}>
          <Typography variant="h6">Red Lines</Typography>
          <Typography variant="h4">{result.red_lines}</Typography>
          <Typography variant="body2">Hard to optimize</Typography>
        </Paper>
      </Box>

      {/* Cython Lint */}
      <Box sx={{ flex: '1 1 calc(50% - 8px)' }}>
        <Paper sx={{ p: 2 }}>
          <Typography variant="h6">Cython Lint Issues</Typography>
          <Typography variant="h4">{result.cython_lint}</Typography>
        </Paper>
      </Box>

      {/* PEP8 */}
      <Box sx={{ flex: '1 1 calc(50% - 8px)' }}>
        <Paper sx={{ p: 2 }}>
          <Typography variant="h6">PEP8 Issues</Typography>
          <Typography variant="h4">{result.pep8_issues}</Typography>
        </Paper>
      </Box>

      {/* Full Button */}
      <Box sx={{ width: '100%', mt: 2 }}>
        <Button
          variant="outlined"
          startIcon={<OpenInNewIcon />}
          onClick={handleViewFull}
        >
          View Full Analysis
        </Button>
      </Box>
    </Box>
  );
};
