import { useEffect, useRef, useState } from "react";
import "./App.css";
import { generate, getJob, videoUrl, type JobResponse } from "./api";
import { addHistoryEntry, loadHistory, type HistoryEntry } from "./history";

// Defaults mirror backend/service/config.py — the only settings actually
// validated for speed/memory/quality on the reference hardware (see
// docs/IMPLEMENT.md Phase 0). Users can override, but these are the floor.
const DEFAULT_WIDTH = 640;
const DEFAULT_HEIGHT = 384;
const DEFAULT_LENGTH = 49;
const DEFAULT_STEPS = 20;

function App() {
  const [prompt, setPrompt] = useState("");
  const [image, setImage] = useState<File | null>(null);
  const [width, setWidth] = useState(DEFAULT_WIDTH);
  const [height, setHeight] = useState(DEFAULT_HEIGHT);
  const [length, setLength] = useState(DEFAULT_LENGTH);
  const [steps, setSteps] = useState(DEFAULT_STEPS);

  const [job, setJob] = useState<JobResponse | null>(null);
  const [submitError, setSubmitError] = useState<string | null>(null);
  const [history, setHistory] = useState<HistoryEntry[]>(() => loadHistory());
  const [isDragging, setIsDragging] = useState(false);

  const pollRef = useRef<number | null>(null);

  useEffect(() => {
    return () => {
      if (pollRef.current !== null) window.clearInterval(pollRef.current);
    };
  }, []);

  function startPolling(jobId: string) {
    if (pollRef.current !== null) window.clearInterval(pollRef.current);
    pollRef.current = window.setInterval(async () => {
      try {
        const updated = await getJob(jobId);
        setJob(updated);
        if (updated.status === "completed" || updated.status === "failed") {
          if (pollRef.current !== null) window.clearInterval(pollRef.current);
          if (updated.status === "completed") {
            setHistory(
              addHistoryEntry({
                jobId: updated.id,
                prompt,
                createdAt: Date.now(),
                status: "completed",
                videoUrl: videoUrl(updated),
              }),
            );
          }
        }
      } catch (err) {
        if (pollRef.current !== null) window.clearInterval(pollRef.current);
        setSubmitError(err instanceof Error ? err.message : String(err));
      }
    }, 3000);
  }

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    setSubmitError(null);
    setJob(null);
    try {
      const created = await generate({
        prompt,
        width,
        height,
        length,
        steps,
        image: image ?? undefined,
      });
      setJob(created);
      startPolling(created.id);
    } catch (err) {
      setSubmitError(err instanceof Error ? err.message : String(err));
    }
  }

  function handleDrop(e: React.DragEvent<HTMLDivElement>) {
    e.preventDefault();
    setIsDragging(false);
    const file = e.dataTransfer.files?.[0];
    if (file) setImage(file);
  }

  const isBusy = job !== null && (job.status === "queued" || job.status === "running");
  const progressPct =
    job && job.progress_total > 0 ? Math.round((job.progress_step / job.progress_total) * 100) : 0;

  return (
    <main className="app">
      <h1>wanai-txt-video</h1>
      <p className="subtitle">
        Local Wan2.2 text/image-to-video generation. Generation is slow (~30–40 min per clip at
        the default resolution) — this runs entirely on-device, so start a job and check back.
      </p>

      <form onSubmit={handleSubmit} className="form">
        <label>
          Prompt
          <textarea
            value={prompt}
            onChange={(e) => setPrompt(e.target.value)}
            required
            rows={3}
            placeholder="a small red boat floating on a calm lake, gentle ripples"
          />
        </label>

        <div
          className={`drop-zone ${isDragging ? "dragging" : ""}`}
          onDragOver={(e) => {
            e.preventDefault();
            setIsDragging(true);
          }}
          onDragLeave={() => setIsDragging(false)}
          onDrop={handleDrop}
        >
          {image ? (
            <span>
              {image.name}{" "}
              <button type="button" onClick={() => setImage(null)}>
                remove
              </button>
            </span>
          ) : (
            <label>
              Drop a start image here (optional, image-to-video), or{" "}
              <input
                type="file"
                accept="image/*"
                onChange={(e) => setImage(e.target.files?.[0] ?? null)}
              />
            </label>
          )}
        </div>

        <div className="controls-row">
          <label>
            Width
            <input type="number" step={32} value={width} onChange={(e) => setWidth(Number(e.target.value))} />
          </label>
          <label>
            Height
            <input type="number" step={32} value={height} onChange={(e) => setHeight(Number(e.target.value))} />
          </label>
          <label>
            Frames
            <input type="number" step={4} value={length} onChange={(e) => setLength(Number(e.target.value))} />
          </label>
          <label>
            Steps
            <input type="number" value={steps} onChange={(e) => setSteps(Number(e.target.value))} />
          </label>
        </div>

        <button type="submit" disabled={isBusy}>
          {isBusy ? "Generating..." : "Generate"}
        </button>
      </form>

      {submitError && <p className="error">{submitError}</p>}

      {job && (
        <section className="job-status">
          <p>
            Job {job.id.slice(0, 8)} — {job.status}
          </p>
          {(job.status === "running" || job.status === "queued") && (
            <div className="progress-bar">
              <div className="progress-fill" style={{ width: `${progressPct}%` }} />
              <span>
                {job.progress_step}/{job.progress_total || "?"} steps
              </span>
            </div>
          )}
          {job.status === "failed" && <p className="error">{job.error}</p>}
          {job.status === "completed" && videoUrl(job) && (
            <video controls src={videoUrl(job)!} className="output-video" />
          )}
        </section>
      )}

      {history.length > 0 && (
        <section className="history">
          <h2>History</h2>
          <ul>
            {history.map((entry) => (
              <li key={entry.jobId}>
                <span className="history-prompt">{entry.prompt}</span>
                {entry.videoUrl && (
                  <video controls src={entry.videoUrl} className="history-video" />
                )}
              </li>
            ))}
          </ul>
        </section>
      )}
    </main>
  );
}

export default App;
