export interface SocialJob {
  id: string;
  name: string;
  url: string;
  screenshot_selector: string;
  text_selector: string;
  platforms: string[];
  schedule_time: string;
  is_active: boolean;
  created_at: string;
  last_run: string | null;
  last_status: string | null;
}

export interface PlatformResult {
  platform: string;
  success: boolean;
  message: string;
  postId?: string;
  url?: string;
}

export interface PostHistory {
  id: string;
  job_id: string;
  job_name: string;
  caption: string | null;
  hashtags: string | null;
  screenshot_path: string | null;
  extracted_text: string | null;
  platforms: string[];
  status: 'success' | 'partial' | 'pending' | 'error';
  platform_results: PlatformResult[] | null;
  created_at: string;
}

export interface CreateJobInput {
  name: string;
  url: string;
  screenshot_selector: string;
  text_selector: string;
  platforms: string[];
  schedule_time: string;
}

export interface RunJobResult {
  success: boolean;
  postId?: string;
  caption?: string;
  hashtags?: string;
  screenshotPath?: string | null;
  extractedText?: string | null;
  platformResults?: PlatformResult[];
  status?: string;
  error?: string;
}
