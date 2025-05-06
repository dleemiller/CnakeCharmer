// src/types.ts

export interface ScoreDist {
  count: number;
  percentage: number;
}

export interface EfficiencyMetrics {
  avg_score?: number;
  python_api_heavy_lines?: number;
  exception_handling_lines?: number;
}

export interface ManualLine {
  file: string;
  line_num: number;
  content: string;
}

export type DetailedAnalysisRecord = Record<
  string,
  {
    score_distribution?: Record<string, ScoreDist>;
    efficiency_metrics?: EfficiencyMetrics;
    red_line_reasons?: Array<{ line: number; content: string; reason: string }>;
    yellow_line_reasons?: Array<{ line: number; content: string; reason: string }>;
  }
>;

export interface DetailedAnalysisMessage {
  message: string;
}

export type DetailedAnalysis = DetailedAnalysisRecord | DetailedAnalysisMessage;

export interface AnalysisResult {
  job_id: string;
  status: string;
  timestamp?: number;
  yellow_lines: number;
  red_lines: number;
  cython_lint: number;
  pep8_issues: number;
  score_distribution?: Record<string, ScoreDist>;
  efficiency_metrics?: EfficiencyMetrics;
  manual_analysis?: {
    yellow_lines?: ManualLine[];
    red_lines?: ManualLine[];
  };
  detailed_analysis?: DetailedAnalysis;
  html_files?: string[];
  error?: string;
}
