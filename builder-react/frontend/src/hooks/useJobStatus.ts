// src/hooks/useJobStatus.ts
import { useState, useEffect, useCallback } from 'react';
import { useJobContext } from '../contexts/JobContext';

export const useJobStatus = (jobId: string | null) => {
  const { jobs, loadJobResult } = useJobContext();
  const [loading, setLoading] = useState(false);
  const [error,   setError]   = useState<string | null>(null);
  const [result,  setResult]  = useState<any>(null);

  const refresh = useCallback(async () => {
    if (!jobId) return;
    setLoading(true);
    try {
      const res = await loadJobResult(jobId);
      setResult(res);
      setError(null);
      return res;
    } catch (e: any) {
      setError(e.message);
      throw e;
    } finally {
      setLoading(false);
    }
  }, [jobId, loadJobResult]);

  useEffect(() => {
    if (jobId) refresh();
  }, [jobId, refresh]);

  const job = jobId ? jobs[jobId] : null;
  const isCompleted = job?.status === 'completed';

  return { job, result, loading, error, refresh, isCompleted };
};

export default useJobStatus;
