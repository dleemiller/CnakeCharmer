import React, { memo, useEffect, useState, useRef } from 'react';
import { useJobContext } from '../../contexts/JobContext';
import { Box, Typography, CircularProgress, Button, Tooltip } from '@mui/material';
import RefreshIcon from '@mui/icons-material/Refresh';
import { useSocketContext } from '../../contexts/SocketContext';

const REFRESH_COOLDOWN = 5000; // 5 seconds between manual refreshes

// Track if the app has ever fetched system status to avoid duplicate initial loads
const hasInitialLoadOccurred = { value: false };

const SystemStatus: React.FC = memo(() => {
  const { state, refreshSystemStatus } = useJobContext();
  const { systemStatus } = state;
  const { isConnected } = useSocketContext();
  const [lastRefreshTime, setLastRefreshTime] = useState<number>(0);
  const [isRefreshing, setIsRefreshing] = useState<boolean>(false);
  const initialLoadAttemptedRef = useRef<boolean>(false);

  // Handle initial load - once and only once per app session
  useEffect(() => {
    // Only attempt to load status if:
    // 1. We don't already have data, and
    // 2. We haven't attempted an initial load in this component instance, and
    // 3. The app as a whole hasn't successfully loaded system status yet
    if (!systemStatus && !initialLoadAttemptedRef.current && !hasInitialLoadOccurred.value && !isRefreshing) {
      console.log('SystemStatus: Initial load attempt');
      initialLoadAttemptedRef.current = true;
      
      // Wait a moment to see if Socket.io connects first (let JobContext handle initial load)
      const initialLoadTimeout = setTimeout(() => {
        // Only proceed if we still don't have data
        if (!systemStatus && !isRefreshing) {
          console.log('SystemStatus: Performing initial load via HTTP');
          setIsRefreshing(true);
          refreshSystemStatus()
            .then(() => {
              // Mark that the app has loaded system status
              hasInitialLoadOccurred.value = true;
            })
            .catch(error => {
              console.error('Failed to load initial system status:', error);
            })
            .finally(() => {
              setIsRefreshing(false);
              setLastRefreshTime(Date.now());
            });
        }
      }, 3000); // Wait 3 seconds before attempting, allowing JobContext to handle it
      
      return () => clearTimeout(initialLoadTimeout);
    }
    
    // If we already have data, mark that initial load has occurred
    if (systemStatus && !hasInitialLoadOccurred.value) {
      hasInitialLoadOccurred.value = true;
    }
  }, [systemStatus, isRefreshing, refreshSystemStatus]);
  
  // Only set up backup polling if Socket.io is disconnected for extended period
  useEffect(() => {
    // Skip if we don't have initial data yet
    if (!systemStatus) return;
    
    // Only set up polling if Socket.io is disconnected
    if (!isConnected) {
      console.log('SystemStatus: Setting up fallback polling (Socket.io disconnected)');
      
      // Backup polling with long interval (60 seconds)
      const pollingInterval = setInterval(() => {
        const now = Date.now();
        // Only refresh if not already refreshing and sufficient time has passed
        if (!isRefreshing && now - lastRefreshTime > 60000) {
          console.log('SystemStatus: Fallback polling via HTTP');
          setIsRefreshing(true);
          refreshSystemStatus()
            .catch(error => {
              console.error('Failed to refresh system status:', error);
            })
            .finally(() => {
              setIsRefreshing(false);
              setLastRefreshTime(now);
            });
        }
      }, 60000); // Check every 60 seconds
      
      return () => clearInterval(pollingInterval);
    }
  }, [isConnected, systemStatus, lastRefreshTime, isRefreshing, refreshSystemStatus]);

  // Update last refresh time when system status is updated via Socket.io
  useEffect(() => {
    // Listen for system status updates from Socket.io
    if (systemStatus) {
      setLastRefreshTime(Date.now());
    }
  }, [systemStatus]);

  // Handle manual refresh with cooldown
  const handleRefresh = () => {
    if (isRefreshing) return;
    
    const now = Date.now();
    // Prevent rapid refreshes
    if (now - lastRefreshTime < REFRESH_COOLDOWN) {
      console.log(`SystemStatus: Refresh on cooldown (${Math.floor((REFRESH_COOLDOWN - (now - lastRefreshTime))/1000)}s remaining)`);
      return;
    }
    
    console.log('SystemStatus: Manual refresh via HTTP');
    setIsRefreshing(true);
    refreshSystemStatus()
      .catch(error => {
        console.error('Failed to refresh system status:', error);
      })
      .finally(() => {
        setIsRefreshing(false);
        setLastRefreshTime(Date.now());
      });
  };

  // Show loading state if we don't have data yet
  if (!systemStatus) {
    return (
      <Box display="flex" alignItems="center" gap={1}>
        <CircularProgress size={16} />
        <Typography variant="body2">Loading system status...</Typography>
      </Box>
    );
  }

  return (
    <Box display="flex" alignItems="center" gap={1}>
      <Typography variant="body2">
        System Status: {systemStatus.status}
      </Typography>
      <Typography variant="body2" color="text.secondary">
        ({systemStatus.pending_jobs.length} pending, {systemStatus.completed_jobs.length} completed)
      </Typography>
      <Tooltip title={isRefreshing ? "Refreshing..." : "Refresh status"}>
        <span>
          <Button 
            onClick={handleRefresh} 
            size="small" 
            sx={{ minWidth: '24px', p: 0.5 }}
            disabled={isRefreshing}
          >
            <RefreshIcon fontSize="small" />
          </Button>
        </span>
      </Tooltip>
    </Box>
  );
});

SystemStatus.displayName = 'SystemStatus';

export default SystemStatus; 