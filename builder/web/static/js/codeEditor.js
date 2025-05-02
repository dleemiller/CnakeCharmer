// Submit code and start polling for results
function submitCode() {
  const code = document.getElementById("code-editor").value;
  if (!code.trim()) {
      alert("Please enter some Cython code!");
      return;
  }
  
  // Clear any existing timers
  if (pollTimer) {
      clearTimeout(pollTimer);
      pollTimer = null;
  }
  
  document.getElementById("submit-btn").disabled = true;
  document.getElementById("status").innerText = "Submitting...";
  document.getElementById("status").style.color = "blue";
  
  // Clear previous annotation links
  document.getElementById("annotation-links").innerHTML = "";
  document.getElementById("detailed-view").innerHTML = "";
  
  logMessage("Submitting code for analysis...");
  
  fetch('/api/submit', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ code: code })
  })
  .then(response => {
      if (!response.ok) {
          throw new Error(`Server returned ${response.status}: ${response.statusText}`);
      }
      return response.json();
  })
  .then(data => {
      // Store the job ID for future reference
      currentJobId = data.job_id;
      
      document.getElementById("job-id").innerText = currentJobId;
      document.getElementById("status").innerText = "Processing...";
      logMessage(`Job submitted successfully. Job ID: ${currentJobId}`);
      
      // Reset result values
      document.getElementById("yellow-lines").innerText = "0";
      document.getElementById("red-lines").innerText = "0";
      document.getElementById("lint-issues").innerText = "0";
      document.getElementById("pep8-issues").innerText = "0";
      
      // Refresh job history
      checkSystemStatus();
      
      // Start polling for results
      pollStartTime = Date.now();
      pollTimer = setTimeout(() => updateResults(currentJobId), POLL_INTERVAL);
  })
  .catch(error => {
      logMessage(`Error submitting job: ${error}`, 'error');
      document.getElementById("status").innerText = "Error submitting job";
      document.getElementById("status").style.color = "red";
      document.getElementById("submit-btn").disabled = false;
  });
}

// Example code for quick testing
function loadExample() {
  document.getElementById("code-editor").value = `# Example Cython code with optimization opportunities
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
  
  logMessage("Loaded example code");
}

// Load a simple example
function loadSimpleExample() {
  document.getElementById("code-editor").value = `# Simple Cython function
def add(int a, int b):
  """Add two integers."""
  return a + b`;
  
  logMessage("Loaded simple example code");
}

// Load complex example with red lines
function loadComplexExample() {
  document.getElementById("code-editor").value = `# Complex Cython example with inefficient Python operations
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
  
  logMessage("Loaded complex example with inefficient operations");
}