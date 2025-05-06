import React, { useState, useRef, useEffect, memo, useCallback } from 'react';
import Editor from '@monaco-editor/react';
import { 
  Stack, 
  Typography, 
  Snackbar,
  Alert,
  Paper,
  Button
} from '@mui/material';
import { useJobContext } from '../../contexts/JobContext';

interface CodeEditorProps {
  onSubmitSuccess?: (jobId: string) => void;
}

// Create a separate component for the loading message to avoid re-renders of the entire CodeEditor
const SubmittingMessage = memo(({ isLoading }: { isLoading: boolean }) => {
  if (!isLoading) return null;
  
  return (
    <Typography variant="body2" color="text.secondary">
      Submitting code for analysis...
    </Typography>
  );
});

SubmittingMessage.displayName = 'SubmittingMessage';

const CodeEditor: React.FC<CodeEditorProps> = memo(({ onSubmitSuccess }) => {
  const [code, setCode] = useState<string>('');
  const [snackbar, setSnackbar] = useState<{
    open: boolean;
    message: string;
    severity: 'success' | 'error' | 'info';
  }>({
    open: false,
    message: '',
    severity: 'info',
  });
  
  // Create a local loading state that is completely independent of the job context
  const [localLoading, setLocalLoading] = useState<boolean>(false);
  
  const editorRef = useRef<any>(null);
  
  const { submitCode, state, setCurrentJob } = useJobContext();
  // We'll only use the error from the context, not the loading state
  const { error } = state;
  
  // Handle editor mount
  function handleEditorDidMount(editor: any) {
    editorRef.current = editor;
  }
  
  // Handle code change
  function handleEditorChange(value: string | undefined) {
    if (value !== undefined) {
      setCode(value);
    }
  }
  
  // Load example code
  const loadExample = useCallback(() => {
    const exampleCode = `# Example Cython code with optimization opportunities
import numpy as np
from libc.math cimport sqrt

def distance_calculator(points_list):
    """Calculate distances between consecutive points."""
    distances = []
    
    for i in range(len(points_list) - 1):
        # This will be highlighted as slower Python operation
        point1 = points_list[i]
        point2 = points_list[i + 1]
        
        # Non-typed math operations
        dx = point2[0] - point1[0]
        dy = point2[1] - point1[1]
        
        # Slow function call without static typing
        dist = sqrt(dx*dx + dy*dy)
        distances.append(dist)
    
    return distances`;
    
    setCode(exampleCode);
    if (editorRef.current) {
      editorRef.current.setValue(exampleCode);
    }
  }, []);
  
  // Load simple example
  const loadSimpleExample = useCallback(() => {
    const simpleCode = `# Simple Cython function
def add(int a, int b):
    """Add two integers."""
    return a + b`;
    
    setCode(simpleCode);
    if (editorRef.current) {
      editorRef.current.setValue(simpleCode);
    }
  }, []);
  
  // Load complex example
  const loadComplexExample = useCallback(() => {
    const complexCode = `# Complex Cython example with inefficient Python operations
import numpy as np
from libc.math cimport sin, cos, sqrt
import random
import re
import json

def highly_inefficient_function(data, iteration_count=100):
    """This function contains operations that are very difficult for Cython to optimize."""
    
    # Dynamic typing - red flag for Cython
    results = {}
    
    # Recursion - difficult to optimize
    def recursive_process(item, depth):
        if depth <= 0:
            return item
        # String operations - red flag for Cython
        if isinstance(item, str):
            # Regular expressions - red flag for Cython
            return re.sub(r'\\d+', lambda m: str(int(m.group(0)) * 2), item)
        elif isinstance(item, (list, tuple)):
            # List comprehension with dynamic typing - red flag
            return [recursive_process(x, depth-1) for x in item]
        elif isinstance(item, dict):
            # Dictionary operations - red flag
            return {k: recursive_process(v, depth-1) for k, v in item.items()}
        else:
            return item
    
    # Eval usage - extreme red flag for Cython
    def unsafe_calculation(expr):
        try:
            return eval(expr)
        except:
            return 0
    
    # Using Python's dir() function - red flag
    dynamic_attributes = dir(data)
    
    for i in range(iteration_count):
        # Converting between Python types - red flag
        key = str(i)
        
        # Unpredictable branching - red flag
        if random.random() > 0.5:
            # JSON operations - red flag
            results[key] = json.dumps({"value": i, "squared": i*i})
        else:
            # Exception handling - red flag
            try:
                # Math with dynamic typing - red flag
                value = data[i % len(data)] if isinstance(data, (list, tuple)) else i
                # String formatting - red flag
                results[key] = f"Value: {value} processed: {sin(float(value))}"
            except Exception as e:
                results[key] = str(e)
    
    # Global namespace operations - red flag
    for i in range(min(5, len(data))):
        item = data[i] if isinstance(data, (list, tuple)) else i
        # Eval with string formatting - extreme red flag
        operation = f"{item} * 2 + {random.random()}"
        results[f"calc_{i}"] = unsafe_calculation(operation)
    
    return results`;
    
    setCode(complexCode);
    if (editorRef.current) {
      editorRef.current.setValue(complexCode);
    }
  }, []);
  
  // Handle submit
  const handleSubmit = useCallback(async () => {
    if (!code.trim()) {
      setSnackbar({
        open: true,
        message: 'Please enter some Cython code',
        severity: 'error',
      });
      return;
    }
    
    try {
      // Use our local loading state instead of the context's loading state
      setLocalLoading(true);
      
      const jobId = await submitCode(code);
      console.log('Job submitted with ID:', jobId);
      
      // Set current job in the context
      setCurrentJob(jobId);
      
      setSnackbar({
        open: true,
        message: `Code submitted for analysis. Job ID: ${jobId}`,
        severity: 'success',
      });
      
      // Call onSubmitSuccess callback if provided
      if (onSubmitSuccess) {
        setTimeout(() => {
          onSubmitSuccess(jobId);
        }, 100); // Add a small delay to ensure state updates properly
      }
    } catch (err) {
      console.error('Error submitting code:', err);
    } finally {
      setLocalLoading(false);
    }
  }, [code, onSubmitSuccess, setCurrentJob, submitCode]);
  
  // Show error snackbar when there's an error in the context
  useEffect(() => {
    if (error) {
      setSnackbar({
        open: true,
        message: error,
        severity: 'error',
      });
    }
  }, [error]);
  
  // Handle snackbar close
  const handleSnackbarClose = useCallback(() => {
    setSnackbar(prev => ({ ...prev, open: false }));
  }, []);
  
  return (
    <Stack spacing={2} width="100%">
      <Typography variant="h6">Enter Your Cython Code</Typography>
      
      <Paper 
        elevation={3} 
        sx={{ 
          height: 500, 
          position: 'relative',
          overflow: 'hidden'
        }}
      >
        <Editor
          height="100%"
          defaultLanguage="python"
          value={code}
          onChange={handleEditorChange}
          onMount={handleEditorDidMount}
          options={{
            minimap: { enabled: false },
            fontSize: 14,
            scrollBeyondLastLine: false,
            automaticLayout: true,
          }}
        />
      </Paper>
      
      <Stack direction="row" spacing={2}>
        <span>
          <Button 
            variant="contained" 
            color="primary" 
            onClick={handleSubmit} 
            disabled={localLoading}
          >
            {localLoading ? 'Submitting...' : 'Analyze Code'}
          </Button>
        </span>
        <span>
          <Button 
            variant="outlined" 
            onClick={loadExample} 
            disabled={localLoading}
          >
            Load Example
          </Button>
        </span>
        <span>
          <Button 
            variant="outlined" 
            onClick={loadSimpleExample} 
            disabled={localLoading}
          >
            Load Simple Example
          </Button>
        </span>
        <span>
          <Button 
            variant="outlined" 
            onClick={loadComplexExample} 
            disabled={localLoading}
          >
            Load Complex Example
          </Button>
        </span>
      </Stack>
      
      {/* Use the isolated message component */}
      <SubmittingMessage isLoading={localLoading} />
      
      <Snackbar 
        open={snackbar.open} 
        autoHideDuration={5000} 
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
});

CodeEditor.displayName = 'CodeEditor';

export default CodeEditor;