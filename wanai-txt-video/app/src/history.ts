// Small JSON index of past generations, persisted to localStorage. Video
// files themselves stay on disk in backend/comfyui/output/ (served by the
// backend service); this only indexes metadata for the history list.

export interface HistoryEntry {
  jobId: string;
  prompt: string;
  createdAt: number;
  status: "completed" | "failed";
  videoUrl: string | null;
}

const STORAGE_KEY = "wanai.history";

export function loadHistory(): HistoryEntry[] {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    return raw ? (JSON.parse(raw) as HistoryEntry[]) : [];
  } catch {
    return [];
  }
}

export function addHistoryEntry(entry: HistoryEntry): HistoryEntry[] {
  const updated = [entry, ...loadHistory()].slice(0, 50);
  localStorage.setItem(STORAGE_KEY, JSON.stringify(updated));
  return updated;
}
