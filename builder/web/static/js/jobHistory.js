// Update job history table
function updateJobHistory(statusData) {
  const historyTable = document.getElementById('job-history-tbody');
  if (!historyTable) return;
  
  // Clear current table content
  historyTable.innerHTML = '';
  
  // Combine pending and completed jobs
  const allJobs = [];
  
  // Add pending jobs
  statusData.pending_jobs.forEach(job => {
      const jobId = job.replace('.tar.gz', '');
      allJobs.push({
          id: jobId,
          status: 'pending',
          timestamp: statusData.in_memory_status[jobId]?.timestamp || Date.now()
      });
  });
  
  // Add completed jobs
  statusData.completed_jobs.forEach(job => {
      const jobId = job.replace('.json', '');
      allJobs.push({
          id: jobId,
          status: 'completed',
          timestamp: statusData.in_memory_status[jobId]?.timestamp || Date.now() - 3600000 // Older if not in memory
      });
  });
  
  // Sort by timestamp (newest first)
  allJobs.sort((a, b) => b.timestamp - a.timestamp);
  
  // Cache job history for detailed view
  jobHistory = allJobs;
  
  // Create table rows
  allJobs.forEach(job => {
      const row = document.createElement('tr');
      
      // Job ID cell
      const idCell = document.createElement('td');
      idCell.textContent = job.id.substring(0, 8) + '...'; // Show shortened ID
      idCell.title = job.id; // Full ID on hover
      row.appendChild(idCell);
      
      // Status cell
      const statusCell = document.createElement('td');
      statusCell.textContent = job.status;
      statusCell.className = `status-${job.status}`;
      row.appendChild(statusCell);
      
      // Timestamp cell
      const timeCell = document.createElement('td');
      timeCell.textContent = new Date(job.timestamp).toLocaleString();
      row.appendChild(timeCell);
      
      // Actions cell
      const actionsCell = document.createElement('td');
      const detailsBtn = document.createElement('button');
      detailsBtn.textContent = 'View Details';
      detailsBtn.className = 'job-details-btn';
      detailsBtn.onclick = function() { loadJobDetails(job.id); };
      actionsCell.appendChild(detailsBtn);
      row.appendChild(actionsCell);
      
      historyTable.appendChild(row);
  });
  
  // Update count in tab
  const historyCount = document.getElementById('history-count');
  if (historyCount) {
      historyCount.textContent = allJobs.length;
  }
}

// Load job details
function loadJobDetails(jobId) {
  logMessage(`Loading details for job ${jobId}`);
  
  // Set as current job and update results
  currentJobId = jobId;
  document.getElementById("job-id").innerText = currentJobId;
  
  // Fetch results
  fetch(`/api/results/${jobId}`)
      .then(response => response.json())
      .then(data => {
          updateResultsDisplay(data);
          
          // Switch to summary tab
          const summaryTab = document.querySelector('.tablinks[data-tab="summary-view"]');
          if (summaryTab) {
              openTab({ currentTarget: summaryTab }, 'summary-view');
          }
      })
      .catch(error => {
          logMessage(`Error fetching job details: ${error}`, 'error');
      });
}

// Refresh job history
function refreshJobHistory() {
  checkSystemStatus();
  logMessage("Job history refreshed");
}