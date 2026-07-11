// Client for backend/service (Phase 1 FastAPI service). Talks directly to
// localhost — the Tauri shell only manages the backend process lifecycle
// (see src-tauri/src/lib.rs), it doesn't proxy requests.

const BASE_URL = "http://127.0.0.1:8000";

export type JobStatus = "queued" | "running" | "completed" | "failed";

export interface JobResponse {
  id: string;
  status: JobStatus;
  progress_step: number;
  progress_total: number;
  error: string | null;
  video_url: string | null;
}

export interface GenerateParams {
  prompt: string;
  negativePrompt?: string;
  width: number;
  height: number;
  length: number;
  steps: number;
  image?: File;
}

export async function generate(params: GenerateParams): Promise<JobResponse> {
  const form = new FormData();
  form.set("prompt", params.prompt);
  if (params.negativePrompt) form.set("negative_prompt", params.negativePrompt);
  form.set("width", String(params.width));
  form.set("height", String(params.height));
  form.set("length", String(params.length));
  form.set("steps", String(params.steps));
  if (params.image) form.set("image", params.image);

  const res = await fetch(`${BASE_URL}/generate`, { method: "POST", body: form });
  if (!res.ok) throw new Error(`generate failed: ${res.status} ${await res.text()}`);
  return res.json();
}

export async function getJob(id: string): Promise<JobResponse> {
  const res = await fetch(`${BASE_URL}/jobs/${id}`);
  if (!res.ok) throw new Error(`job fetch failed: ${res.status} ${await res.text()}`);
  return res.json();
}

export function videoUrl(job: JobResponse): string | null {
  return job.video_url ? `${BASE_URL}${job.video_url}` : null;
}
