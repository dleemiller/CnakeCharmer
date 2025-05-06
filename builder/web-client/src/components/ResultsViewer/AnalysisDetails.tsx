import React, { useCallback } from 'react';
import { Box, Paper, Typography, Button } from '@mui/material';
import OpenInNewIcon from '@mui/icons-material/OpenInNew';
import { AnalysisResult } from '../../types';
import apiClient from '../../api/apiClient';

interface AnalysisDetailsProps {
  result: AnalysisResult;
}

export const AnalysisDetails: React.FC<AnalysisDetailsProps> = ({ result }) => {
  const getScoreCount = (category: string) => {
    return result.score_distribution?.[category]?.count || 0;
  };

  // Use callback to open the detailed analysis in a new tab without causing parent re-renders
  const handleViewDetailedAnalysis = useCallback((e: React.MouseEvent) => {
    e.preventDefault();
    // Use window.open so the browser opens a new tab without navigating the current page
    window.open(apiClient.getDetailedAnalysisUrl(result.job_id), '_blank', 'noopener');
  }, [result.job_id]);

  return (
    <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 3 }}>
      <Box sx={{ flex: '1 1 calc(25% - 12px)', minWidth: '200px' }}>
        <Paper
          elevation={1}
          sx={{
            p: 2,
            bgcolor: 'success.light',
            color: 'success.contrastText',
            height: '100%',
          }}
        >
          <Typography variant="h6">Excellent Lines</Typography>
          <Typography variant="h3">{getScoreCount('excellent')}</Typography>
          <Typography variant="body2">Highly optimized Cython code</Typography>
        </Paper>
      </Box>

      <Box sx={{ flex: '1 1 calc(25% - 12px)', minWidth: '200px' }}>
        <Paper
          elevation={1}
          sx={{
            p: 2,
            bgcolor: '#e6ffcc',
            color: 'text.primary',
            height: '100%',
          }}
        >
          <Typography variant="h6">Good Lines</Typography>
          <Typography variant="h3">{getScoreCount('good')}</Typography>
          <Typography variant="body2">Well-optimized Cython code</Typography>
        </Paper>
      </Box>

      <Box sx={{ flex: '1 1 calc(25% - 12px)', minWidth: '200px' }}>
        <Paper
          elevation={1}
          sx={{
            p: 2,
            bgcolor: 'warning.light',
            color: 'warning.contrastText',
            height: '100%',
          }}
        >
          <Typography variant="h6">Yellow Lines</Typography>
          <Typography variant="h3">{result.yellow_lines}</Typography>
          <Typography variant="body2">Python operations that could be optimized</Typography>
        </Paper>
      </Box>

      <Box sx={{ flex: '1 1 calc(25% - 12px)', minWidth: '200px' }}>
        <Paper
          elevation={1}
          sx={{
            p: 2,
            bgcolor: 'error.light',
            color: 'error.contrastText',
            height: '100%',
          }}
        >
          <Typography variant="h6">Red Lines</Typography>
          <Typography variant="h3">{result.red_lines}</Typography>
          <Typography variant="body2">Python operations that are difficult to optimize</Typography>
        </Paper>
      </Box>

      <Box sx={{ flex: '1 1 calc(50% - 12px)', minWidth: '200px' }}>
        <Paper elevation={1} sx={{ p: 2, bgcolor: 'grey.100', height: '100%' }}>
          <Typography variant="h6">Cython Lint Issues</Typography>
          <Typography variant="h3">{result.cython_lint}</Typography>
        </Paper>
      </Box>

      <Box sx={{ flex: '1 1 calc(50% - 12px)', minWidth: '200px' }}>
        <Paper elevation={1} sx={{ p: 2, bgcolor: 'grey.100', height: '100%' }}>
          <Typography variant="h6">PEP8 Issues</Typography>
          <Typography variant="h3">{result.pep8_issues}</Typography>
        </Paper>
      </Box>

      <Box sx={{ mt: 4, width: '100%' }}>
        <Button
          onClick={handleViewDetailedAnalysis}
          variant="text"
          color="primary"
          startIcon={<OpenInNewIcon fontSize="small" />}
          sx={{ textTransform: 'none' }}
        >
          View Full Detailed Analysis
        </Button>
      </Box>
    </Box>
  );
}; 