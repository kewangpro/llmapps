import { useState, useMemo, useEffect } from "react";
import { motion } from "framer-motion";

export default function GPUArchitectureAnimation() {
  // --- Selection state (click to lock selection) ---
  const [selectedKey, setSelectedKey] = useState("Scheduler");

  const smComponents = [
    { key: "Scheduler", desc: "Distributes warps to execution units and hides latency by switching between ready warps." },
    { key: "CUDA Cores", desc: "General-purpose ALUs executing FP/INT instructions from the current warp." },
    { key: "Tensor Cores", desc: "Specialized units for matrix multiply–accumulate, accelerating AI/HPC." },
    { key: "Load/Store Units", desc: "Move data between registers, shared memory/L1, L2, and global memory." },
    { key: "Register File", desc: "Per-thread registers with extremely low latency and high bandwidth." },
    { key: "Shared Memory / L1 Cache", desc: "On‑chip SRAM shared by threads in a block; also serves as L1 cache." },
  ];

  return (
    <div className="w-full h-full flex flex-col items-center bg-gray-900 text-white p-6 space-y-6">
      <motion.h1
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.6 }}
        className="text-3xl font-bold text-cyan-400 text-center"
      >
        Streaming Multiprocessor (SM) Components
      </motion.h1>

      {/* SM tiles (click to select) */}
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ duration: 0.6 }}
        className="w-full max-w-6xl grid md:grid-cols-3 gap-4"
      >
        {smComponents.map((c) => (
          <button
            key={c.key}
            onClick={() => setSelectedKey(c.key)}
            className={`text-left p-4 rounded-2xl border transition-all duration-200 focus:outline-none focus:ring-2 focus:ring-cyan-400 ${
              selectedKey === c.key
                ? "border-cyan-400 bg-gray-800 shadow-lg"
                : "border-gray-700 bg-gray-800/70 hover:border-cyan-600 hover:bg-gray-800"
            }`}
            aria-pressed={selectedKey === c.key}
          >
            <h3 className="text-lg font-semibold text-cyan-300 mb-2">{c.key}</h3>
            <p className="text-sm text-gray-300">{c.desc}</p>
          </button>
        ))}
      </motion.div>

      {/* Detail panel renders a dedicated demo for the selected component */}
      <DetailPanel selectedKey={selectedKey} />

      <p className="text-center text-gray-400 text-xs max-w-3xl mt-2">
        Click a component tile to lock selection and view a focused demo for that SM unit. This makes it clear how schedulers, CUDA cores, Tensor Cores, and the memory hierarchy cooperate.
      </p>
    </div>
  );
}

// ===================== Detail Panel =====================
function DetailPanel({ selectedKey }) {
  return (
    <motion.div
      key={selectedKey}
      initial={{ opacity: 0, y: 12 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.35 }}
      className="w-full max-w-6xl mt-2"
    >
      {selectedKey === "Scheduler" && <SchedulerDemo />}
      {selectedKey === "CUDA Cores" && <CUDACoresDemo />}
      {selectedKey === "Tensor Cores" && <TensorCoresDemo />}
      {selectedKey === "Load/Store Units" && <LoadStoreDemo />}
      {selectedKey === "Register File" && <RegisterFileDemo />}
      {selectedKey === "Shared Memory / L1 Cache" && <SharedMemoryDemo />}
    </motion.div>
  );
}

// ===================== Scheduler Demo =====================
function SchedulerDemo() {
  const WARPS = 12; // display a dozen warps
  const THREADS_PER_WARP = 32;
  const SCHEDULERS = 4;
  const [tick, setTick] = useState(0);
  const [stallPct, setStallPct] = useState(30);
  useEffect(() => {
    const id = setInterval(() => setTick((t) => t + 1), 900);
    return () => clearInterval(id);
  }, []);
  const stalled = (w, t) => ((w * 17 + (t % 13) * 7) % 100) < stallPct; // ~stallPct%
  const activeWarpFor = (sched, t) => {
    let base = (t + sched * 3) % WARPS;
    for (let i = 0; i < WARPS; i++) {
      const cand = (base + i) % WARPS;
      if (!stalled(cand, t)) return cand;
    }
    return base;
  };
  const active = Array.from({ length: SCHEDULERS }).map((_, s) => activeWarpFor(s, tick));
  return (
    <div className="bg-gray-800 rounded-2xl p-6 border border-blue-700 space-y-4">
      <div className="flex items-center justify-between">
        <h2 className="text-xl font-semibold text-blue-300">Warp Scheduling and Parallel Execution</h2>
        <div className="flex items-center gap-3">
          <label className="text-xs text-gray-300">Memory pressure (stall %)</label>
          <input type="range" min={0} max={100} step={5} value={stallPct} onChange={(e) => setStallPct(parseInt(e.target.value))} />
          <span className="text-xs text-gray-200 w-10 text-right">{stallPct}%</span>
        </div>
      </div>

      {/* Scheduler lanes */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
        {Array.from({ length: SCHEDULERS }).map((_, sched) => (
          <div key={sched} className="bg-gray-900 rounded-md p-3 border border-gray-700">
            <div className="flex items-center justify-between text-[11px] text-gray-300 mb-1">
              <span>Scheduler {sched}</span>
              <span>Issuing: <span className="text-emerald-400 font-semibold">Warp {active[sched]}</span></span>
            </div>
            <div className="relative w-full h-2 bg-gray-700 rounded">
              <motion.div className="absolute left-0 top-0 h-2 bg-emerald-500 rounded" animate={{ x: ["0%", "80%"] }} transition={{ duration: 1.0, repeat: Infinity, ease: "easeInOut" }} style={{ width: "22%" }} />
            </div>
          </div>
        ))}
      </div>

      {/* Warps grid */}
      <div className="mt-2 grid gap-1" style={{ gridTemplateColumns: "1.7rem 1fr" }}>
        {Array.from({ length: WARPS }).map((_, w) => {
          const isIssued = active.includes(w);
          const isStalled = stalled(w, tick);
          return (
            <div key={w} className="contents">
              <div className={`text-[10px] px-1 py-[2px] rounded ${isIssued ? "bg-emerald-900/50 text-emerald-300" : isStalled ? "bg-blue-950/60 text-blue-300" : "text-gray-400"}`}>W{w}</div>
              <div className={`grid gap-[2px] items-center rounded ${isIssued ? "ring-1 ring-emerald-500/60" : ""}`} style={{ gridTemplateColumns: `repeat(${THREADS_PER_WARP}, minmax(0, 1fr))` }}>
                {Array.from({ length: THREADS_PER_WARP }).map((__, t) => (
                  <div key={t} className={`h-3 rounded-sm ${isStalled ? "bg-blue-900" : isIssued ? "bg-blue-500" : "bg-blue-700"}`} />
                ))}
              </div>
            </div>
          );
        })}
      </div>

      <p className="text-[11px] text-gray-400">Rows are <span className="text-gray-200">warps</span> (32 threads). Schedulers pick ready warps each cycle; stalled warps dim while another warp issues.</p>
    </div>
  );
}

// ===================== CUDA Cores Demo =====================
function CUDACoresDemo() {
  return (
    <div className="bg-gray-800 rounded-2xl p-6 border border-cyan-700 space-y-3">
      <h2 className="text-xl font-semibold text-cyan-300">CUDA Cores (ALUs)</h2>
      <p className="text-sm text-gray-300">General-purpose scalar/vector pipelines that execute FP/INT instructions issued from the active warp.</p>
      <div className="grid grid-cols-12 gap-[2px]">
        {Array.from({ length: 144 }).map((_, i) => (
          <motion.div key={i} className="h-3 rounded-sm bg-cyan-500" animate={{ opacity: [0.35, 1, 0.35] }} transition={{ duration: 1.2, repeat: Infinity, delay: (i % 12) * 0.05 }} />
        ))}
      </div>
      <p className="text-[11px] text-gray-400">Brightness suggests activity; real SMs have multiple ALU pipelines and issue slots.</p>
    </div>
  );
}

// ===================== Tensor Cores Demo =====================
function TensorCoresDemo() {
  const tileDim = 16;
  const cells = useMemo(() => Array.from({ length: tileDim * tileDim }, (_, i) => i), []);
  const [sweep, setSweep] = useState(0);
  useEffect(() => {
    const id = setInterval(() => setSweep((s) => (s + 1) % tileDim), 300);
    return () => clearInterval(id);
  }, []);
  return (
    <div className="bg-gray-800 rounded-2xl p-6 border border-fuchsia-700 space-y-4">
      <div className="flex items-center justify-between">
        <h2 className="text-xl font-semibold text-fuchsia-300">Tensor Cores (MMA: D = A×B + C)</h2>
        <span className="text-xs text-gray-400">Tiles: 16×16 per warp (illustrative)</span>
      </div>
      <div className="grid md:grid-cols-3 gap-6">
        {/* A tile */}
        <div>
          <div className="text-sm text-gray-300 mb-1">Matrix A (16×16)</div>
          <div className="grid gap-[2px] p-2 bg-gray-900 rounded-xl border border-gray-700" style={{ gridTemplateColumns: "repeat(16, minmax(0, 1fr))" }}>
            {cells.map((i) => {
              const r = Math.floor(i / tileDim);
              const c = i % tileDim;
              const active = c === sweep;
              return (
                <motion.div key={i} className={`w-3 h-3 rounded-sm ${active ? "bg-fuchsia-400" : "bg-gray-700"}`} animate={{ opacity: active ? [0.6, 1, 0.6] : 1 }} transition={{ duration: 0.6, repeat: active ? Infinity : 0 }} />
              );
            })}
          </div>
        </div>
        {/* B tile */}
        <div>
          <div className="text-sm text-gray-300 mb-1">Matrix B (16×16)</div>
          <div className="grid gap-[2px] p-2 bg-gray-900 rounded-xl border border-gray-700" style={{ gridTemplateColumns: "repeat(16, minmax(0, 1fr))" }}>
            {cells.map((i) => {
              const r = Math.floor(i / tileDim);
              const c = i % tileDim;
              const active = r === sweep;
              return (
                <motion.div key={i} className={`w-3 h-3 rounded-sm ${active ? "bg-emerald-400" : "bg-gray-700"}`} animate={{ opacity: active ? [0.6, 1, 0.6] : 1 }} transition={{ duration: 0.6, repeat: active ? Infinity : 0 }} />
              );
            })}
          </div>
        </div>
        {/* Accumulator tile */}
        <div>
          <div className="text-sm text-gray-300 mb-1">Accumulator C → D (16×16)</div>
          <div className="grid gap-[2px] p-2 bg-gray-900 rounded-xl border border-gray-700" style={{ gridTemplateColumns: "repeat(16, minmax(0, 1fr))" }}>
            {cells.map((i) => {
              const r = Math.floor(i / tileDim);
              const c = i % tileDim;
              const active = (r + c) % tileDim === sweep;
              return (
                <motion.div key={i} className={`w-3 h-3 rounded-sm ${active ? "bg-indigo-400" : "bg-gray-700"}`} animate={{ scale: active ? [1, 1.35, 1] : 1, opacity: active ? [0.7, 1, 0.7] : 1 }} transition={{ duration: 0.6, repeat: active ? Infinity : 0 }} />
              );
            })}
          </div>
        </div>
      </div>
      <p className="text-[11px] text-gray-400">A column × B row streams into the tensor unit; partial sums accumulate into C → D.</p>
    </div>
  );
}

// ===================== Load/Store Demo =====================
function LoadStoreDemo() {
  const stages = ["Registers", "Shared/L1", "L2", "Global (VRAM)"];
  const [tick, setTick] = useState(0);
  useEffect(() => {
    const id = setInterval(() => setTick((t) => t + 1), 1200);
    return () => clearInterval(id);
  }, []);
  return (
    <div className="bg-gray-800 rounded-2xl p-6 border border-yellow-700 space-y-3">
      <h2 className="text-xl font-semibold text-yellow-300">Load/Store Units & Memory Path</h2>
      <div className="relative w-full h-3 bg-gray-700 rounded">
        <motion.div className="absolute top-[-6px] w-4 h-4 rounded-full bg-yellow-400" animate={{ x: ["0%", "30%", "60%", "90%"] }} transition={{ duration: 2, repeat: Infinity }} />
      </div>
      <div className="grid grid-cols-4 gap-2 text-center text-xs">
        {stages.map((s) => (
          <div key={s} className="bg-gray-900 p-2 rounded-lg border border-gray-700 text-gray-300">{s}</div>
        ))}
      </div>
      <p className="text-[11px] text-gray-400">Coalesced accesses hit L1/L2 and return quickly; scattered patterns fall back to VRAM with higher latency.</p>
    </div>
  );
}

// ===================== Register File Demo =====================
function RegisterFileDemo() {
  return (
    <div className="bg-gray-800 rounded-2xl p-6 border border-emerald-700 space-y-3">
      <h2 className="text-xl font-semibold text-emerald-300">Register File</h2>
      <p className="text-sm text-gray-300">Per-thread registers provide the fastest storage; occupancy is limited by total registers per SM and per-thread usage.</p>
      <div className="grid grid-cols-16 gap-[2px]">
        {Array.from({ length: 16 * 8 }).map((_, i) => (
          <motion.div key={i} className="h-3 rounded-sm bg-emerald-500" animate={{ opacity: [0.6, 1, 0.6] }} transition={{ duration: 1.4, repeat: Infinity, delay: (i % 16) * 0.03 }} />
        ))}
      </div>
      <p className="text-[11px] text-gray-400">Heavier register usage can reduce resident warps (occupancy), affecting latency hiding.</p>
    </div>
  );
}

// ===================== Shared Memory Demo =====================
function SharedMemoryDemo() {
  const BANKS = 32;
  const [mode, setMode] = useState("coalesced");
  return (
    <div className="bg-gray-800 rounded-2xl p-6 border border-pink-700 space-y-3">
      <div className="flex items-center justify-between">
        <h2 className="text-xl font-semibold text-pink-300">Shared Memory / L1</h2>
        <div className="flex items-center gap-2 text-xs text-gray-300">
          <span>Access pattern:</span>
          <select className="bg-gray-900 border border-gray-700 rounded px-2 py-1" value={mode} onChange={(e) => setMode(e.target.value)}>
            <option value="coalesced">Coalesced</option>
            <option value="strided">Strided (conflicts)</option>
          </select>
        </div>
      </div>
      <div className="grid" style={{ gridTemplateColumns: `repeat(${BANKS}, minmax(0, 1fr))` }}>
        {Array.from({ length: BANKS }).map((_, i) => (
          <div key={i} className="h-8 bg-gray-900 border border-gray-700 rounded-sm relative overflow-hidden">
            <motion.div
              className="absolute inset-y-0 left-0 bg-pink-500"
              style={{ width: mode === "coalesced" ? "100%" : "40%" }}
              animate={{ opacity: mode === "coalesced" ? [0.5, 1, 0.5] : [0.7, 1, 0.7] }}
              transition={{ duration: 1.2, repeat: Infinity, delay: i * (mode === "coalesced" ? 0.01 : 0.02) }}
            />
          </div>
        ))}
      </div>
      <p className="text-[11px] text-gray-400">Coalesced accesses distribute evenly across banks; strided patterns cause bank conflicts and serialization.</p>
    </div>
  );
}
