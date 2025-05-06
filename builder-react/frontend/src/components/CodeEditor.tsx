// src/components/CodeEditor.tsx
import React, { useState, useRef, useEffect, useCallback, memo } from 'react';
import Editor from '@monaco-editor/react';
import { Stack, Typography, Snackbar, Alert, Paper, Button } from '@mui/material';
import { useJobContext } from '../contexts/JobContext';

interface CodeEditorProps {
  onSubmitSuccess?: (jobId: string) => void;
}

const SubmittingMessage = memo(({ isLoading }: { isLoading: boolean }) =>
  isLoading ? (
    <Typography variant="body2" color="text.secondary">
      Submitting code for analysis...
    </Typography>
  ) : null
);
SubmittingMessage.displayName = 'SubmittingMessage';

const CodeEditor: React.FC<CodeEditorProps> = memo(({ onSubmitSuccess }) => {
  const [code, setCode] = useState<string>('');
  const [snackbar, setSnackbar] = useState<{
    open: boolean;
    message: string;
    severity: 'success' | 'error' | 'info';
  }>({ open: false, message: '', severity: 'info' });
  const [loading, setLoading] = useState<boolean>(false);
  const editorRef = useRef<any>(null);

  const { submitCode, setCurrentJob } = useJobContext();

  const handleMount = (editor: any) => {
    editorRef.current = editor;
  };

  const handleChange = (value: string | undefined) => {
    if (value !== undefined) {
      setCode(value);
    }
  };

  const loadExample = useCallback(() => {
    const example = `# Simple Cython example
def add(int a, int b):
    return a + b`;
    setCode(example);
    editorRef.current?.setValue(example);
  }, []);

  const loadComplex = useCallback(() => {
    const complex = `# Complex Cython with regex
import re
def process(data):
    output = []
    for item in data:
        # replace digits
        new = re.sub(r'\\d+', lambda m: str(int(m.group(0)) * 2), item)
        output.append(new)
    return output`;
    setCode(complex);
    editorRef.current?.setValue(complex);
  }, []);

  const handleSubmit = useCallback(async () => {
    if (!code.trim()) {
      setSnackbar({ open: true, message: 'Please enter code', severity: 'error' });
      return;
    }
    try {
      setLoading(true);
      const jobId = await submitCode(code);
      setCurrentJob(jobId);
      setSnackbar({ open: true, message: `Submitted. Job ID: ${jobId}`, severity: 'success' });
      if (onSubmitSuccess) setTimeout(() => onSubmitSuccess(jobId), 100);
    } catch (err: any) {
      setSnackbar({ open: true, message: err.message ?? 'Submission failed', severity: 'error' });
    } finally {
      setLoading(false);
    }
  }, [code, submitCode, setCurrentJob, onSubmitSuccess]);

  return (
    <Stack spacing={2}>
      <Typography variant="h6">Enter Your Cython Code</Typography>

      <Paper elevation={3} sx={{ height: 400, overflow: 'hidden' }}>
        <Editor
          height="100%"
          defaultLanguage="python"
          value={code}
          onChange={handleChange}
          onMount={handleMount}
          options={{ minimap: { enabled: false }, automaticLayout: true }}
        />
      </Paper>

      <Stack direction="row" spacing={2}>
        <Button variant="contained" onClick={handleSubmit} disabled={loading}>
          {loading ? 'Submitting…' : 'Analyze Code'}
        </Button>
        <Button variant="outlined" onClick={loadExample} disabled={loading}>
          Load Example
        </Button>
        <Button variant="outlined" onClick={loadComplex} disabled={loading}>
          Load Complex
        </Button>
      </Stack>

      <SubmittingMessage isLoading={loading} />

      <Snackbar
        open={snackbar.open}
        autoHideDuration={4000}
        onClose={() => setSnackbar((s) => ({ ...s, open: false }))}
      >
        <Alert severity={snackbar.severity}>{snackbar.message}</Alert>
      </Snackbar>
    </Stack>
  );
});

CodeEditor.displayName = 'CodeEditor';
export default CodeEditor;
