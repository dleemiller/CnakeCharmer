import React, { createContext, useContext, useEffect, useState } from "react";
import { io, Socket } from "socket.io-client";
import { SOCKET_IO_PATH } from "../config";

interface SocketContextType {
  socket: Socket | null;
  isConnected: boolean;
  subscribe: (jobId: string) => void;
  unsubscribe: (jobId: string) => void;
  onSystemStatus: (cb: (data: any) => void) => () => void;
  onJobUpdate: (cb: (data: any) => void) => () => void;
}

const SocketContext = createContext<SocketContextType | undefined>(undefined);

export const SocketProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const [socket, setSocket] = useState<Socket | null>(null);
  const [isConnected, setConnected] = useState(false);

  useEffect(() => {
    // connect back to same origin, at the /socket.io path
    const sock = io({
      path: SOCKET_IO_PATH,
      transports: ["websocket"],
    });

    sock.on("connect",    () => setConnected(true));
    sock.on("disconnect", () => setConnected(false));

    setSocket(sock);
    return () => { sock.disconnect(); };
  }, []);

  const subscribe   = (jobId: string) => { socket?.emit("subscribe",   { job_id: jobId }); };
  const unsubscribe = (jobId: string) => { socket?.emit("unsubscribe", { job_id: jobId }); };

  const onSystemStatus = (cb: (data: any) => void) => {
    socket?.on("system_status", cb);
    return () => socket?.off("system_status", cb);
  };
  const onJobUpdate = (cb: (data: any) => void) => {
    socket?.on("job_update", cb);
    return () => socket?.off("job_update", cb);
  };

  return (
    <SocketContext.Provider
      value={{ socket, isConnected, subscribe, unsubscribe, onSystemStatus, onJobUpdate }}
    >
      {children}
    </SocketContext.Provider>
  );
};

export const useSocket = () => {
  const ctx = useContext(SocketContext);
  if (!ctx) throw new Error("useSocket must be used within SocketProvider");
  return ctx;
};
