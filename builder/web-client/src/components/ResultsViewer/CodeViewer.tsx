import React, { useEffect } from 'react';
import { Box, Paper, Typography, Table, TableBody, TableCell, TableContainer, TableHead, TableRow, LinearProgress, Accordion, AccordionSummary, AccordionDetails, Stack } from '@mui/material';
import ExpandMoreIcon from '@mui/icons-material/ExpandMore';
import { AnalysisResult, DetailedAnalysisRecord, DetailedAnalysisMessage } from '../../types';

interface CodeViewerProps {
  result: AnalysisResult;
}

export const CodeViewer: React.FC<CodeViewerProps> = ({ result }) => {
  // Type guard to check if detailed_analysis is a Record object and not a message object
  const isDetailedAnalysisRecord = (obj: any): obj is DetailedAnalysisRecord => 
    obj !== null && 
    typeof obj === 'object' && 
    !('message' in obj);
    
  // Type guard for message object
  const isDetailedAnalysisMessage = (obj: any): obj is DetailedAnalysisMessage =>
    obj !== null &&
    typeof obj === 'object' &&
    'message' in obj &&
    typeof obj.message === 'string';
  
  // Get the first file from detailed_analysis if it exists
  const detailedAnalysis = result.detailed_analysis;
  const firstFileKey = detailedAnalysis && isDetailedAnalysisRecord(detailedAnalysis)
    ? Object.keys(detailedAnalysis)[0] 
    : null;
    
  const fileAnalysis = firstFileKey && isDetailedAnalysisRecord(detailedAnalysis) 
    ? detailedAnalysis[firstFileKey] 
    : null;
    
  const scoreDistribution = fileAnalysis?.score_distribution || result.score_distribution;
  const efficiencyMetrics = fileAnalysis?.efficiency_metrics || result.efficiency_metrics;
  
  useEffect(() => {
    console.log('CodeViewer: Result data', { 
      result, 
      firstFileKey,
      fileAnalysis,
      scoreDistribution
    });
  }, [result, firstFileKey, fileAnalysis, scoreDistribution]);
  
  const categoryNames = {
    excellent: 'Excellent',
    good: 'Good',
    yellow: 'Yellow',
    red: 'Red',
  };

  // If there's a detailed_analysis message but not a record, show it
  if (detailedAnalysis && isDetailedAnalysisMessage(detailedAnalysis)) {
    return <Typography>{detailedAnalysis.message}</Typography>;
  }

  if (!scoreDistribution && !result.manual_analysis) {
    return <Typography>No detailed analysis data available</Typography>;
  }

  return (
    <Stack spacing={4}>
      {/* Score Distribution */}
      {scoreDistribution && (
        <Box>
          <Typography variant="h6" gutterBottom>
            Score Distribution
          </Typography>
          <TableContainer component={Paper} variant="outlined">
            <Table size="small">
              <TableHead>
                <TableRow>
                  <TableCell>Category</TableCell>
                  <TableCell align="right">Count</TableCell>
                  <TableCell align="right">Percentage</TableCell>
                  <TableCell>Visualization</TableCell>
                </TableRow>
              </TableHead>
              <TableBody>
                {Object.entries(categoryNames).map(([key, label]) => {
                  const data = scoreDistribution[key as keyof typeof scoreDistribution];
                  if (!data) return null;

                  const colorMap: Record<string, string> = {
                    excellent: '#CCFFCC',
                    good: '#E6FFCC',
                    yellow: '#FFFF99',
                    red: '#FFCCCC',
                  };

                  return (
                    <TableRow key={key}>
                      <TableCell>{label}</TableCell>
                      <TableCell align="right">{data.count}</TableCell>
                      <TableCell align="right">{data.percentage}%</TableCell>
                      <TableCell>
                        <LinearProgress
                          variant="determinate"
                          value={data.percentage}
                          sx={{
                            height: 10,
                            borderRadius: 5,
                            backgroundColor: 'rgba(0,0,0,0.1)',
                            '& .MuiLinearProgress-bar': {
                              backgroundColor: colorMap[key] || '#ccc',
                            },
                          }}
                        />
                      </TableCell>
                    </TableRow>
                  );
                })}
              </TableBody>
            </Table>
          </TableContainer>
        </Box>
      )}

      {/* Efficiency Metrics */}
      {efficiencyMetrics && (
        <Box>
          <Typography variant="h6" gutterBottom>
            Efficiency Metrics
          </Typography>
          <TableContainer component={Paper} variant="outlined">
            <Table size="small">
              <TableBody>
                {efficiencyMetrics.avg_score !== undefined && (
                  <TableRow>
                    <TableCell component="th" scope="row" sx={{ fontWeight: 'medium' }}>
                      Average Score
                    </TableCell>
                    <TableCell>{efficiencyMetrics.avg_score.toFixed(2)}</TableCell>
                  </TableRow>
                )}

                {efficiencyMetrics.python_api_heavy_lines !== undefined && (
                  <TableRow>
                    <TableCell component="th" scope="row" sx={{ fontWeight: 'medium' }}>
                      Python API Heavy Lines
                    </TableCell>
                    <TableCell>{efficiencyMetrics.python_api_heavy_lines}</TableCell>
                  </TableRow>
                )}

                {efficiencyMetrics.exception_handling_lines !== undefined && (
                  <TableRow>
                    <TableCell component="th" scope="row" sx={{ fontWeight: 'medium' }}>
                      Exception Handling Lines
                    </TableCell>
                    <TableCell>{efficiencyMetrics.exception_handling_lines}</TableCell>
                  </TableRow>
                )}
              </TableBody>
            </Table>
          </TableContainer>
        </Box>
      )}

      {/* Collapsible Sections */}
      <Stack spacing={2}>
        {/* Yellow Lines */}
        {result.manual_analysis?.yellow_lines && result.manual_analysis.yellow_lines.length > 0 && (
          <Accordion defaultExpanded>
            <AccordionSummary expandIcon={<ExpandMoreIcon />} sx={{ bgcolor: 'warning.light' }}>
              <Typography fontWeight="medium">
                Yellow Lines ({result.manual_analysis.yellow_lines.length})
              </Typography>
            </AccordionSummary>
            <AccordionDetails>
              <TableContainer>
                <Table size="small">
                  <TableHead>
                    <TableRow>
                      <TableCell>File</TableCell>
                      <TableCell>Line</TableCell>
                      <TableCell>Content</TableCell>
                    </TableRow>
                  </TableHead>
                  <TableBody>
                    {result.manual_analysis.yellow_lines.map((line, index) => (
                      <TableRow
                        key={`yellow-${index}`}
                        sx={{ bgcolor: 'warning.light', opacity: 0.7 }}
                      >
                        <TableCell>{line.file}</TableCell>
                        <TableCell>{line.line_num}</TableCell>
                        <TableCell sx={{ fontFamily: 'monospace', fontSize: '0.75rem' }}>
                          {line.content}
                        </TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              </TableContainer>
            </AccordionDetails>
          </Accordion>
        )}

        {/* Red Lines */}
        {result.manual_analysis?.red_lines && result.manual_analysis.red_lines.length > 0 && (
          <Accordion defaultExpanded>
            <AccordionSummary expandIcon={<ExpandMoreIcon />} sx={{ bgcolor: 'error.light' }}>
              <Typography fontWeight="medium">
                Red Lines ({result.manual_analysis.red_lines.length})
              </Typography>
            </AccordionSummary>
            <AccordionDetails>
              <TableContainer>
                <Table size="small">
                  <TableHead>
                    <TableRow>
                      <TableCell>File</TableCell>
                      <TableCell>Line</TableCell>
                      <TableCell>Content</TableCell>
                    </TableRow>
                  </TableHead>
                  <TableBody>
                    {result.manual_analysis.red_lines.map((line, index) => (
                      <TableRow
                        key={`red-${index}`}
                        sx={{ bgcolor: 'error.light', opacity: 0.7 }}
                      >
                        <TableCell>{line.file}</TableCell>
                        <TableCell>{line.line_num}</TableCell>
                        <TableCell sx={{ fontFamily: 'monospace', fontSize: '0.75rem' }}>
                          {line.content}
                        </TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              </TableContainer>
            </AccordionDetails>
          </Accordion>
        )}
      </Stack>
    </Stack>
  );
}; 