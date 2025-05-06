// frontend/src/contexts/JobContext.tsx
import React, { createContext, useContext, useReducer, useEffect, useCallback } from "react";
import * as api from "../api/apiClient";
import { useSocket } from "./SocketContext";

export interface Job {
  id: string;
  status: string;
  timestamp: number;
  result?: any;
}
export interface SystemStatus {
  status: string;
  pending_jobs: string[];
  completed_jobs: string[];
  in_memory_status: Record<string, { status: string; timestamp: number }>;
}

interface State {
  currentJobId: string | null;
  jobs: Record<string, Job>;
  systemStatus: SystemStatus | null;
}
type Action =
  | { type: "SET_SYSTEM_STATUS"; payload: SystemStatus }
  | { type: "ADD_JOB"; payload: Job }
  | { type: "UPDATE_JOB_RESULT"; payload: { job_id: string; result: any } }
  | { type: "SET_CURRENT_JOB"; payload: string | null };

const initialState: State = {
  currentJobId: null,
  jobs: {},
  systemStatus: null,
};

function reducer(state: State, action: Action): State {
  switch (action.type) {
    case "SET_SYSTEM_STATUS":
      return { ...state, systemStatus: action.payload };
    case "ADD_JOB":
      return {
        ...state,
        jobs: { ...state.jobs, [action.payload.id]: action.payload },
      };
    case "UPDATE_JOB_RESULT":
      const job = state.jobs[action.payload.job_id];
      if (!job) return state;
      return {
        ...state,
        jobs: {
          ...state.jobs,
          [action.payload.job_id]: {
            ...job,
            ...action.payload.result,
          },
        },
      };
    case "SET_CURRENT_JOB":
      return { ...state, currentJobId: action.payload };
    default:
      return state;
  }
}

interface ContextValue extends State {
  submitCode: (code: string) => Promise<string>;
  loadJobResult: (jobId: string) => Promise<any>;
  refreshSystemStatus: () => Promise<SystemStatus>;
  setCurrentJob: (jobId: string | null) => void;
}

const JobContext = createContext<ContextValue | undefined>(undefined);

export const JobProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const [state, dispatch] = useReducer(reducer, initialState);
  const { subscribe, unsubscribe, onSystemStatus, onJobUpdate } = useSocket();

  // Handle real-time system status
  useEffect(() => {
    const off = onSystemStatus((data: SystemStatus) => {
      dispatch({ type: "SET_SYSTEM_STATUS", payload: data });
      // ensure all jobs exist in state
      data.pending_jobs.concat(data.completed_jobs).forEach(fn => {
        const id = fn.replace(/\.(tar\.gz|json)$/, "");
        dispatch({
          type: "ADD_JOB",
          payload: {
            id,
            status: data.in_memory_status[id]?.status || (data.completed_jobs.includes(fn) ? "completed" : "submitted"),
            timestamp: data.in_memory_status[id]?.timestamp || Date.now(),
          },
        });
      });
    });
    return off;
  }, [onSystemStatus]);

  // Handle per-job updates
  useEffect(() => {
    const off = onJobUpdate((data: any) => {
      dispatch({ type: "UPDATE_JOB_RESULT", payload: { job_id: data.job_id, result: data.result } });
      dispatch({
        type: "ADD_JOB",
        payload: {
          id: data.job_id,
          status: data.status,
          timestamp: data.result.timestamp || Date.now(),
          result: data.result,
        },
      });
    });
    return off;
  }, [onJobUpdate]);

  // API actions
  const submitCode = useCallback(async (code: string) => {
    const res = await api.submitCode(code);
    dispatch({ type: "ADD_JOB", payload: { id: res.job_id, status: res.status, timestamp: res.timestamp } });
    dispatch({ type: "SET_CURRENT_JOB", payload: res.job_id });
    subscribe(res.job_id);
    return res.job_id;
  }, [subscribe]);

  const loadJobResult = useCallback(async (jobId: string) => {
    const result = await api.getJobResults(jobId);
    dispatch({ type: "UPDATE_JOB_RESULT", payload: { job_id: jobId, result } });
    return result;
  }, []);

  const refreshSystemStatus = useCallback(async () => {
    const status = await api.getSystemStatus();
    dispatch({ type: "SET_SYSTEM_STATUS", payload: status });
    return status;
  }, []);

  const setCurrentJob = useCallback((jobId: string | null) => {
    if (state.currentJobId && state.currentJobId !== jobId) {
      unsubscribe(state.currentJobId);
    }
    if (jobId) subscribe(jobId);
    dispatch({ type: "SET_CURRENT_JOB", payload: jobId });
  }, [state.currentJobId, subscribe, unsubscribe]);

  return (
    <JobContext.Provider value={{ ...state, submitCode, loadJobResult, refreshSystemStatus, setCurrentJob }}>
      {children}
    </JobContext.Provider>
  );
};

export const useJobContext = () => {
  const ctx = useContext(JobContext);
  if (!ctx) throw new Error("useJobContext must be used within JobProvider");
  return ctx;
};
