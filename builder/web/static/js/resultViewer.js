// Update results display
// JavaScript to poll for updates and update the dashboard
function updateResults(jobId) {
  // Safeguard against undefined job IDs
  if (!jobId || jobId === 'undefined' || jobId === 'null') {
      logMessage('Invalid job ID, stopping polling', 'error');
      clearTimeout(pollTimer);
      document.getElementById("submit-btn").disabled = false;
      return;
  }
  
  // Check if we've been polling too long
  const currentTime = Date.now();
  if (currentTime - pollStartTime > MAX_POLL_TIME) {
      logMessage(`Polling timeout after ${MAX_POLL_TIME/1000} seconds`, 'error');
      document.getElementById("status").innerText = "Timeout - check logs";
      document.getElementById("status").style.color = "red";
      clearTimeout(pollTimer);
      document.getElementById("submit-btn").disabled = false;
      return;
  }
  
  fetch(`/api/results/${jobId}`)
      .then(response => {
          if (!response.ok) {
              throw new Error(`Server returned ${response.status}: ${response.statusText}`);
          }
          return response.json();
      })
      .then(data => {
          logMessage(`Received update for job ${jobId}: status=${data.status}`);
          
          updateResultsDisplay(data);
          
          if (data.status === "completed" || data.status === "error") {
              document.getElementById("submit-btn").disabled = false;
              clearTimeout(pollTimer);
              
              // Refresh job history
              checkSystemStatus();
              
              if (data.status === "completed") {
                  logMessage("Analysis completed successfully", 'success');
              }
          } else {
              document.getElementById("status").innerText = `Status: ${data.status}`;
              document.getElementById("status").style.color = "blue";
              logMessage(`Job ${jobId} still processing: ${data.status}`);
              pollTimer = setTimeout(() => updateResults(jobId), POLL_INTERVAL);
          }
      })
      .catch(error => {
          logMessage(`Error fetching results: ${error}`, 'error');
          document.getElementById("status").innerText = "Error fetching results";
          document.getElementById("status").style.color = "red";
          
          // Retry a few times before giving up
          if (currentTime - pollStartTime < MAX_POLL_TIME/2) {
              pollTimer = setTimeout(() => updateResults(jobId), POLL_INTERVAL*2);
          } else {
              document.getElementById("submit-btn").disabled = false;
          }
      });
}

function updateResultsDisplay(data) {
  logMessage(`Updating display with data: status=${data.status}`);
  
  if (data.status === "completed") {
      // Update score metrics
      if (data.detailed_analysis) {
          const firstFile = Object.keys(data.detailed_analysis)[0];
          if (firstFile) {
              const fileAnalysis = data.detailed_analysis[firstFile];
              if (fileAnalysis.score_distribution) {
                  document.getElementById("excellent-lines").innerText = fileAnalysis.score_distribution.excellent?.count || 0;
                  document.getElementById("good-lines").innerText = fileAnalysis.score_distribution.good?.count || 0;
                  document.getElementById("yellow-lines").innerText = fileAnalysis.score_distribution.yellow?.count || 0;
                  document.getElementById("red-lines").innerText = fileAnalysis.score_distribution.red?.count || 0;
              }
          }
      }
      
      document.getElementById("lint-issues").innerText = data.cython_lint;
      document.getElementById("pep8-issues").innerText = data.pep8_issues;
      document.getElementById("status").innerText = "Completed";
      document.getElementById("status").style.color = "green";
      
      // Set up HTML annotation links if available
      if (data.html_files && data.html_files.length > 0) {
          const annotationLinks = document.getElementById("annotation-links");
          annotationLinks.innerHTML = "";
          data.html_files.forEach(file => {
              const link = document.createElement("a");
              link.href = `/api/html/${file}`;
              link.textContent = file;
              link.target = "_blank";
              link.className = "annotation-link";
              annotationLinks.appendChild(link);
              annotationLinks.appendChild(document.createElement("br"));
          });
      }
      
      // Set up detailed analysis view link
      const detailedView = document.getElementById("detailed-view");
      detailedView.innerHTML = "";
      const analysisLink = document.createElement("a");
      analysisLink.href = `/api/analysis/${data.job_id}`;
      analysisLink.textContent = "View Detailed Analysis";
      analysisLink.target = "_blank";
      analysisLink.className = "annotation-link";
      detailedView.appendChild(analysisLink);
      
      // Update detailed analysis tab
      updateDetailedAnalysis(data);
      
      logMessage("Analysis data updated successfully");
      
// Update detailed analysis tab
function updateDetailedAnalysis(data) {
  console.log("Updating detailed analysis with data:", data);
  
  if (!data.detailed_analysis) {
      logMessage("No detailed analysis data available");
      return;
  }
  
  // Check if we have a message instead of analysis data
  if (data.detailed_analysis.message) {
      const scoreDistribution = document.getElementById('score-distribution');
      scoreDistribution.innerHTML = `<p class="info-message">${data.detailed_analysis.message}</p>`;
      
      const efficiencyMetrics = document.getElementById('efficiency-metrics');
      efficiencyMetrics.innerHTML = '';
      
      const excellentLines = document.getElementById('excellent-lines-details');
      excellentLines.innerHTML = '';
      
      const goodLines = document.getElementById('good-lines-details');
      goodLines.innerHTML = '';
      
      const redReasons = document.getElementById('red-reasons');
      redReasons.innerHTML = '';
      
      const yellowReasons = document.getElementById('yellow-reasons');
      yellowReasons.innerHTML = '';
      
      return;
  }
  
  const firstFile = Object.keys(data.detailed_analysis)[0];
  if (!firstFile) {
      logMessage("No files in detailed analysis");
      return;
  }
  
  const fileAnalysis = data.detailed_analysis[firstFile];
  console.log("File analysis data:", fileAnalysis);
  
  // Update score distribution
  const scoreDistribution = document.getElementById('score-distribution');
  scoreDistribution.innerHTML = '';
  
  if (fileAnalysis.score_distribution) {
      const categories = ['excellent', 'good', 'yellow', 'red'];
      const labels = {
          'excellent': 'Excellent (80-255)',
          'good': 'Good (40-79)',
          'yellow': 'Yellow (20-39)',
          'red': 'Red (0-19)'
      };
      
      const table = document.createElement('table');
      table.className = 'score-table';
      
      // Add header
      const header = table.createTHead();
      const headerRow = header.insertRow();
      headerRow.insertCell().textContent = 'Category';
      headerRow.insertCell().textContent = 'Count';
      headerRow.insertCell().textContent = 'Percentage';
      headerRow.insertCell().textContent = 'Visualization';
      
      // Add rows for each category
      const body = table.createTBody();
      categories.forEach(category => {
          if (fileAnalysis.score_distribution[category]) {
              const row = body.insertRow();
              row.className = category;
              
              const count = fileAnalysis.score_distribution[category].count;
              const percentage = fileAnalysis.score_distribution[category].percentage;
              
              row.insertCell().textContent = labels[category];
              row.insertCell().textContent = count;
              row.insertCell().textContent = `${percentage}%`;
              
              const visCell = row.insertCell();
              const bar = document.createElement('div');
              bar.className = 'score-bar';
              bar.style.width = `${percentage}%`;
              bar.style.backgroundColor = getCategoryColor(category);
              visCell.appendChild(bar);
          }
      });
      
      scoreDistribution.appendChild(table);
  } else {
      scoreDistribution.innerHTML = '<p class="info-message">No score distribution data available</p>';
  }
  
  // Update efficiency metrics
  const efficiencyMetrics = document.getElementById('efficiency-metrics');
  efficiencyMetrics.innerHTML = '';
  
  if (fileAnalysis.efficiency_metrics) {
      const table = document.createElement('table');
      table.className = 'metrics-table';
      
      const body = table.createTBody();
      
      // Add average score
      if (fileAnalysis.efficiency_metrics.avg_score !== undefined) {
          const row = body.insertRow();
          row.insertCell().textContent = 'Average Score';
          row.insertCell().textContent = fileAnalysis.efficiency_metrics.avg_score.toFixed(2);
      }
      
      // Add Python API heavy lines
      if (fileAnalysis.efficiency_metrics.python_api_heavy_lines !== undefined) {
          const row = body.insertRow();
          row.insertCell().textContent = 'Python API Heavy Lines';
          row.insertCell().textContent = fileAnalysis.efficiency_metrics.python_api_heavy_lines;
      }
      
      // Add exception handling lines
      if (fileAnalysis.efficiency_metrics.exception_handling_lines !== undefined) {
          const row = body.insertRow();
          row.insertCell().textContent = 'Exception Handling Lines';
          row.insertCell().textContent = fileAnalysis.efficiency_metrics.exception_handling_lines;
      }
      
      efficiencyMetrics.appendChild(table);
  } else {
      efficiencyMetrics.innerHTML = '<p class="info-message">No efficiency metrics available</p>';
  }
  
  // Update excellent lines details
  const excellentLines = document.getElementById('excellent-lines-details');
  excellentLines.innerHTML = '';
  
  if (fileAnalysis.score_distribution?.excellent?.lines) {
      const table = document.createElement('table');
      table.className = 'reasons-table';
      
      const header = table.createTHead();
      const headerRow = header.insertRow();
      headerRow.insertCell().textContent = 'Line';
      headerRow.insertCell().textContent = 'Content';
      
      const body = table.createTBody();
      fileAnalysis.score_distribution.excellent.lines.forEach(([lineNum, content]) => {
          const row = body.insertRow();
          row.className = 'excellent';
          row.insertCell().textContent = lineNum;
          row.insertCell().textContent = content;
      });
      
      excellentLines.appendChild(table);
  } else {
      excellentLines.innerHTML = '<p class="info-message">No excellent lines found</p>';
  }
  
  // Update good lines details
  const goodLines = document.getElementById('good-lines-details');
  goodLines.innerHTML = '';
  
  if (fileAnalysis.score_distribution?.good?.lines) {
      const table = document.createElement('table');
      table.className = 'reasons-table';
      
      const header = table.createTHead();
      const headerRow = header.insertRow();
      headerRow.insertCell().textContent = 'Line';
      headerRow.insertCell().textContent = 'Content';
      
      const body = table.createTBody();
      fileAnalysis.score_distribution.good.lines.forEach(([lineNum, content]) => {
          const row = body.insertRow();
          row.className = 'good';
          row.insertCell().textContent = lineNum;
          row.insertCell().textContent = content;
      });
      
      goodLines.appendChild(table);
  } else {
      goodLines.innerHTML = '<p class="info-message">No good lines found</p>';
  }
  
  // Update red line reasons
  const redReasons = document.getElementById('red-reasons');
  redReasons.innerHTML = '';
  
  if (data.manual_analysis && data.manual_analysis.red_lines && data.manual_analysis.red_lines.length > 0) {
      const table = document.createElement('table');
      table.className = 'reasons-table';
      
      const header = table.createTHead();
      const headerRow = header.insertRow();
      headerRow.insertCell().textContent = 'File';
      headerRow.insertCell().textContent = 'Line';
      headerRow.insertCell().textContent = 'Content';
      
      const body = table.createTBody();
      data.manual_analysis.red_lines.forEach(line => {
          const row = body.insertRow();
          row.className = 'red';
          row.insertCell().textContent = line.file;
          row.insertCell().textContent = line.line_num;
          row.insertCell().textContent = line.content;
      });
      
      redReasons.appendChild(table);
  } else {
      redReasons.innerHTML = '<p class="info-message">No red lines found</p>';
  }
  
  // Update yellow line reasons
  const yellowReasons = document.getElementById('yellow-reasons');
  yellowReasons.innerHTML = '';
  
  if (data.manual_analysis && data.manual_analysis.yellow_lines && data.manual_analysis.yellow_lines.length > 0) {
      const table = document.createElement('table');
      table.className = 'reasons-table';
      
      const header = table.createTHead();
      const headerRow = header.insertRow();
      headerRow.insertCell().textContent = 'File';
      headerRow.insertCell().textContent = 'Line';
      headerRow.insertCell().textContent = 'Content';
      
      const body = table.createTBody();
      data.manual_analysis.yellow_lines.forEach(line => {
          const row = body.insertRow();
          row.className = 'yellow';
          row.insertCell().textContent = line.file;
          row.insertCell().textContent = line.line_num;
          row.insertCell().textContent = line.content;
      });
      
      yellowReasons.appendChild(table);
  } else {
      yellowReasons.innerHTML = '<p class="info-message">No yellow lines found</p>';
  }
  
  // Initialize collapsible elements
  initializeCollapsibles();
  logMessage("Detailed analysis updated successfully");
}
  } 
  else if (data.status === "error") {
      document.getElementById("status").innerText = "Error: " + data.error;
      document.getElementById("status").style.color = "red";
      logMessage(`Analysis failed: ${data.error}`, 'error');
  }
  else {
      document.getElementById("status").innerText = `Status: ${data.status}`;
      document.getElementById("status").style.color = "blue";
      logMessage(`Job ${data.job_id} status: ${data.status}`);
  }
}