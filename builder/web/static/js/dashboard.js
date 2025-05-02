// Global variables
const POLL_INTERVAL = 1000; // 1 second
const MAX_POLL_TIME = 60000; // 60 seconds
let pollTimer = null;
let pollStartTime = 0;
let currentJobId = null;
let jobHistory = [];

// Initialize when DOM is fully loaded
document.addEventListener('DOMContentLoaded', function() {
    initializeApp();
});

// Initialize the application
function initializeApp() {
    logMessage("Dashboard initialized");
    checkSystemStatus();
    
    // Add event listeners to collapsible elements
    initializeCollapsibles();
    
    // Setup periodic refresh of job history
    setInterval(checkSystemStatus, 30000); // Refresh every 30 seconds
}

// Log messages to the console
function logMessage(message, type = 'info') {
    const logPanel = document.getElementById('log-panel');
    const timestamp = new Date().toLocaleTimeString();
    const logEntry = document.createElement('div');
    logEntry.className = type;
    logEntry.textContent = `[${timestamp}] ${message}`;
    logPanel.appendChild(logEntry);
    logPanel.scrollTop = logPanel.scrollHeight;
    
    console.log(`[${type}] ${message}`);
}

// Check system status and update job history
function checkSystemStatus() {
    fetch('/api/status')
        .then(response => response.json())
        .then(data => {
            logMessage(`System status: ${data.status}`);
            logMessage(`Pending jobs: ${data.pending_jobs.length}`);
            logMessage(`Completed jobs: ${data.completed_jobs.length}`);
            
            // Update job history view
            updateJobHistory(data);
        })
        .catch(error => {
            logMessage(`Error checking system status: ${error}`, 'error');
        });
}

// Initialize collapsible elements
function initializeCollapsibles() {
    const collapsibles = document.getElementsByClassName("collapsible");
    for (let i = 0; i < collapsibles.length; i++) {
        collapsibles[i].addEventListener("click", function() {
            this.classList.toggle("active");
            const content = this.nextElementSibling;
            if (content.style.maxHeight) {
                content.style.maxHeight = null;
                content.classList.remove("active");
            } else {
                content.style.maxHeight = content.scrollHeight + "px";
                content.classList.add("active");
            }
        });
    }
}

// Tab switching
function openTab(evt, tabName) {
    // Hide all tab content
    const tabcontent = document.getElementsByClassName("tabcontent");
    for (let i = 0; i < tabcontent.length; i++) {
        tabcontent[i].style.display = "none";
    }
    
    // Remove "active" class from all tab buttons
    const tablinks = document.getElementsByClassName("tablinks");
    for (let i = 0; i < tablinks.length; i++) {
        tablinks[i].className = tablinks[i].className.replace(" active", "");
    }
    
    // Show current tab and mark button as active
    document.getElementById(tabName).style.display = "block";
    evt.currentTarget.className += " active";
}

// Cancel current job polling
function cancelPolling() {
    if (pollTimer) {
        clearTimeout(pollTimer);
        pollTimer = null;
        document.getElementById("submit-btn").disabled = false;
        document.getElementById("status").innerText = "Cancelled";
        document.getElementById("status").style.color = "orange";
        logMessage("Polling cancelled by user", "info");
    }
}

// Helper function to get category color
function getCategoryColor(category) {
    switch(category) {
        case 'excellent': return '#CCFFCC';
        case 'good': return '#E6FFCC';
        case 'yellow': return '#FFFF99';
        case 'red': return '#FFCCCC';
        default: return '#FFFFFF';
    }
}