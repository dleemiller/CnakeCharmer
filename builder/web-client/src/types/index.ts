// Job Types
export interface JobStatus {
  job_id: string;
  status: string;
  timestamp?: number;
}

export interface Job {
  id: string;
  status: string;
  timestamp: number;
  result?: AnalysisResult;
}

// Analysis Results
export type FileAnalysisData = {
  score_distribution?: {
    [key: string]: {
      count: number;
      percentage: number;
      lines?: Array<[number, string]>;
      range?: [number, number];
    };
  };
  efficiency_metrics?: {
    avg_score?: number;
    python_api_heavy_lines?: number;
    exception_handling_lines?: number;
  };
};

export type DetailedAnalysisMessage = {
  message: string;
};

export type DetailedAnalysisRecord = Record<string, FileAnalysisData>;

export interface AnalysisResult {
  job_id: string;
  status: string;
  timestamp?: number;
  yellow_lines: number;
  red_lines: number;
  cython_lint: number;
  pep8_issues: number;
  score_distribution?: {
    [key: string]: {
      count: number;
      percentage: number;
    };
  };
  efficiency_metrics?: {
    avg_score?: number;
    python_api_heavy_lines?: number;
    exception_handling_lines?: number;
  };
  manual_analysis?: {
    yellow_lines?: Array<{
      file: string;
      line_num: number;
      content: string;
    }>;
    red_lines?: Array<{
      file: string;
      line_num: number;
      content: string;
    }>;
  };
  detailed_analysis?: DetailedAnalysisRecord | DetailedAnalysisMessage;
  html_files?: string[];
  error?: string;
  summary?: string;
}

export interface JobResult {
  status: string;
  summary?: string;
  detailed_analysis?: any;
  timestamp?: number;
  [key: string]: any;
}

export interface ManualAnalysisLine {
  file: string;
  line_num: number;
  content: string;
}

export interface ScoreDistribution {
  count: number;
  percentage: number;
  lines?: [number, string][];
}

export interface FileAnalysis {
  score_distribution?: {
    excellent?: ScoreDistribution;
    good?: ScoreDistribution;
    yellow?: ScoreDistribution;
    red?: ScoreDistribution;
  };
  efficiency_metrics?: EfficiencyMetrics;
  red_line_reasons?: Array<{
    line: number;
    content: string;
    reason: string;
  }>;
  yellow_line_reasons?: Array<{
    line: number;
    content: string;
    reason: string;
  }>;
}

export interface EfficiencyMetrics {
  avg_score?: number;
  python_api_heavy_lines?: number;
  exception_handling_lines?: number;
}

// System Status
export interface SystemStatus {
  status: string;
  pending_jobs: string[];
  completed_jobs: string[];
  archived_jobs: string[];
  in_memory_status: Record<string, JobStatus>;
}

// Component Props
export interface CodeEditorProps {
  onSubmitSuccess?: (jobId: string) => void;
}

export interface ResultsViewerProps {
  jobId: string | null;
}

export interface JobHistoryProps {
  onSelectJob?: (jobId: string) => void;
}