const express = require('express');
const http = require('http');
const { Server } = require('socket.io');
const cors = require('cors');
const axios = require('axios');

// Configuration
const API_HOST = process.env.API_HOST || 'http://localhost:8000';
const PORT = process.env.SOCKET_PORT || 8001;

// Set up Express app
const app = express();
app.use(cors());

// Create HTTP server and Socket.io instance
const server = http.createServer(app);
const io = new Server(server, {
  cors: {
    origin: "*",
    methods: ["GET", "POST"]
  }
});

// Track active jobs and subscriptions
const activeJobs = new Set();
const jobSubscriptions = new Map(); // jobId -> Set of socket IDs
const lastJobStatuses = new Map(); // jobId -> last status

// Function to fetch system status from FastAPI
async function fetchSystemStatus() {
  try {
    const response = await axios.get(`${API_HOST}/api/status`);
    return response.data;
  } catch (error) {
    console.error('Error fetching system status:', error.message);
    return null;
  }
}

// Function to fetch job results from FastAPI
async function fetchJobResults(jobId) {
  try {
    const response = await axios.get(`${API_HOST}/api/results/${jobId}`);
    return response.data;
  } catch (error) {
    console.error(`Error fetching results for job ${jobId}:`, error.message);
    return null;
  }
}

// Poll for updates to jobs 
const POLL_INTERVAL = 2000; // 2 seconds
setInterval(async () => {
  try {
    // Fetch system status
    const status = await fetchSystemStatus();
    if (!status) return;
    
    // Broadcast system status to all clients
    io.emit('system_status', status);
    
    // For each active job, check if status has changed
    for (const jobId of activeJobs) {
      // Only fetch if there are subscribers
      if (jobSubscriptions.has(jobId) && jobSubscriptions.get(jobId).size > 0) {
        const jobResult = await fetchJobResults(jobId);
        
        if (jobResult) {
          const lastStatus = lastJobStatuses.get(jobId);
          
          // Check if status changed or if this is a new job
          if (!lastStatus || lastStatus !== jobResult.status) {
            console.log(`Job ${jobId} status changed from ${lastStatus || 'unknown'} to ${jobResult.status}`);
            
            // Update last known status
            lastJobStatuses.set(jobId, jobResult.status);
            
            // Send update to all subscribers
            const message = {
              type: 'job_update',
              job_id: jobId,
              status: jobResult.status,
              result: jobResult
            };
            
            // Get all subscribed sockets and emit to them
            const subscribers = jobSubscriptions.get(jobId) || new Set();
            subscribers.forEach(socketId => {
              const socket = io.sockets.sockets.get(socketId);
              if (socket) {
                socket.emit('job_update', message);
              }
            });
          }
        }
      }
    }
  } catch (error) {
    console.error('Error in polling loop:', error);
  }
}, POLL_INTERVAL);

// Socket.io connection handler
io.on('connection', (socket) => {
  console.log(`New client connected: ${socket.id}`);
  
  // Handle socket disconnection
  socket.on('disconnect', () => {
    console.log(`Client disconnected: ${socket.id}`);
    
    // Remove socket from all job subscriptions
    for (const [jobId, subscribers] of jobSubscriptions.entries()) {
      subscribers.delete(socket.id);
      
      // Clean up empty subscription sets
      if (subscribers.size === 0) {
        jobSubscriptions.delete(jobId);
      }
    }
  });
  
  // Handle subscription to job updates
  socket.on('subscribe', ({ job_id }) => {
    if (!job_id) return;
    
    console.log(`Client ${socket.id} subscribing to job ${job_id}`);
    
    // Add job to active jobs
    activeJobs.add(job_id);
    
    // Add socket to job subscribers
    if (!jobSubscriptions.has(job_id)) {
      jobSubscriptions.set(job_id, new Set());
    }
    jobSubscriptions.get(job_id).add(socket.id);
    
    // Send initial job status if available
    fetchJobResults(job_id).then(result => {
      if (result) {
        // Update last known status
        lastJobStatuses.set(job_id, result.status);
        
        // Send update to subscriber
        socket.emit('job_update', {
          type: 'job_update',
          job_id,
          status: result.status,
          result
        });
      }
    }).catch(error => {
      console.error(`Error fetching initial status for job ${job_id}:`, error);
    });
  });
  
  // Handle unsubscription from job updates
  socket.on('unsubscribe', ({ job_id }) => {
    if (!job_id) return;
    
    console.log(`Client ${socket.id} unsubscribing from job ${job_id}`);
    
    // Remove socket from job subscribers
    if (jobSubscriptions.has(job_id)) {
      jobSubscriptions.get(job_id).delete(socket.id);
      
      // Clean up empty subscription sets
      if (jobSubscriptions.get(job_id).size === 0) {
        jobSubscriptions.delete(job_id);
      }
    }
  });
  
  // Send initial system status
  fetchSystemStatus().then(status => {
    if (status) {
      socket.emit('system_status', status);
    }
  }).catch(error => {
    console.error('Error fetching initial system status:', error);
  });
  
  // Handle ping messages
  socket.on('ping', () => {
    socket.emit('pong', { timestamp: Date.now() });
  });
});

// Start the server
server.listen(PORT, () => {
  console.log(`Socket.io server running on port ${PORT}`);
});

// Basic health endpoint
app.get('/health', (req, res) => {
  res.json({ 
    status: 'healthy',
    activeJobs: Array.from(activeJobs),
    connections: io.engine.clientsCount
  });
}); 