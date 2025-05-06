// frontend/src/components/SystemStatus.tsx
import React from "react";
import { Box, Typography, CircularProgress } from "@mui/material";
import { useSocket } from "../contexts/SocketContext";
import { useJobContext } from "../contexts/JobContext";

const SystemStatus: React.FC = () => {
  const { isConnected } = useSocket();
  const { systemStatus, refreshSystemStatus } = useJobContext();

  React.useEffect(() => {
    if (!isConnected) {
      refreshSystemStatus();
    }
  }, [isConnected, refreshSystemStatus]);

  if (!systemStatus) {
    return (
      <Box display="flex" alignItems="center" gap={1}>
        <CircularProgress size={16} /> <Typography>Loading status...</Typography>
      </Box>
    );
  }

  return (
    <Typography variant="body2">
      {isConnected ? "🟢 Connected" : "🔴 Disconnected"} —{" "}
      {systemStatus.pending_jobs.length} pending,{" "}
      {systemStatus.completed_jobs.length} completed
    </Typography>
  );
};

export default SystemStatus;
