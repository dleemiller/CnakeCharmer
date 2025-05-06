// src/components/CodeViewer.tsx

import React from 'react';
import {
  Box,
  Typography,
  TableContainer,
  Table,
  TableHead,
  TableRow,
  TableCell,
  TableBody,
  LinearProgress,
  Accordion,
  AccordionSummary,
  AccordionDetails
} from '@mui/material';
import ExpandMoreIcon from '@mui/icons-material/ExpandMore';
import {
  AnalysisResult,
  DetailedAnalysis,
  DetailedAnalysisMessage,
  ManualLine
} from '../types';

interface CodeViewerProps {
  result: AnalysisResult;
}

export const CodeViewer: React.FC<CodeViewerProps> = ({ result }) => {
  const da = result.detailed_analysis as DetailedAnalysis | undefined;

  if (!da) {
    return <Typography>No detailed analysis available</Typography>;
  }

  // Narrow to message vs record
  const isMessage = (v: DetailedAnalysis): v is DetailedAnalysisMessage =>
    typeof (v as DetailedAnalysisMessage).message === 'string';

  if (isMessage(da)) {
    return <Typography>{da.message}</Typography>;
  }

  // It's a record keyed by filename
  const record = da as Record<
    string,
    {
      score_distribution?: Record<string, { count: number; percentage: number }>;
      efficiency_metrics?: Record<string, number>;
      red_line_reasons?: Array<{ line: number; content: string; reason: string }>;
      yellow_line_reasons?: Array<{ line: number; content: string; reason: string }>;
    }
  >;

  const fileKey = Object.keys(record)[0];
  const data = record[fileKey];
  const sd = data.score_distribution ?? {};
  const em = data.efficiency_metrics ?? {};

  // Safely extract manual lines arrays
  const yellowLines: ManualLine[] = result.manual_analysis?.yellow_lines ?? [];
  const redLines: ManualLine[]    = result.manual_analysis?.red_lines    ?? [];

  return (
    <Box>
      {/* Score Distribution */}
      <Typography variant="h6" gutterBottom>
        Score Distribution — {fileKey}
      </Typography>
      <TableContainer>
        <Table size="small">
          <TableHead>
            <TableRow>
              <TableCell>Category</TableCell>
              <TableCell align="right">Count</TableCell>
              <TableCell align="right">%</TableCell>
              <TableCell>Bar</TableCell>
            </TableRow>
          </TableHead>
          <TableBody>
            {(['excellent', 'good', 'yellow', 'red'] as const).map((k) => {
              const row = sd[k];
              if (!row) return null;
              return (
                <TableRow key={k}>
                  <TableCell>{k.charAt(0).toUpperCase() + k.slice(1)}</TableCell>
                  <TableCell align="right">{row.count}</TableCell>
                  <TableCell align="right">{row.percentage}</TableCell>
                  <TableCell sx={{ width: 100 }}>
                    <LinearProgress variant="determinate" value={row.percentage} />
                  </TableCell>
                </TableRow>
              );
            })}
          </TableBody>
        </Table>
      </TableContainer>

      {/* Efficiency Metrics */}
      <Typography variant="h6" gutterBottom sx={{ mt: 2 }}>
        Efficiency Metrics
      </Typography>
      <TableContainer>
        <Table size="small">
          <TableBody>
            {em.avg_score !== undefined && (
              <TableRow>
                <TableCell>Average Score</TableCell>
                <TableCell>{em.avg_score.toFixed(2)}</TableCell>
              </TableRow>
            )}
            {em.python_api_heavy_lines !== undefined && (
              <TableRow>
                <TableCell>Python API Heavy Lines</TableCell>
                <TableCell>{em.python_api_heavy_lines}</TableCell>
              </TableRow>
            )}
            {em.exception_handling_lines !== undefined && (
              <TableRow>
                <TableCell>Exception Handling Lines</TableCell>
                <TableCell>{em.exception_handling_lines}</TableCell>
              </TableRow>
            )}
          </TableBody>
        </Table>
      </TableContainer>

      {/* Yellow Lines */}
      {yellowLines.length > 0 && (
        <Accordion defaultExpanded sx={{ mt: 2 }}>
          <AccordionSummary expandIcon={<ExpandMoreIcon />}>
            <Typography>Yellow Lines ({yellowLines.length})</Typography>
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
                  {yellowLines.map((l, i) => (
                    <TableRow key={i}>
                      <TableCell>{l.file}</TableCell>
                      <TableCell>{l.line_num}</TableCell>
                      <TableCell>
                        <pre>{l.content}</pre>
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
      {redLines.length > 0 && (
        <Accordion defaultExpanded sx={{ mt: 2 }}>
          <AccordionSummary expandIcon={<ExpandMoreIcon />}>
            <Typography>Red Lines ({redLines.length})</Typography>
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
                  {redLines.map((l, i) => (
                    <TableRow key={i}>
                      <TableCell>{l.file}</TableCell>
                      <TableCell>{l.line_num}</TableCell>
                      <TableCell>
                        <pre>{l.content}</pre>
                      </TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </TableContainer>
          </AccordionDetails>
        </Accordion>
      )}
    </Box>
  );
};
