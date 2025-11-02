// --- CLSTM-PPO Training & Trading Visualization (TypeScript) -----------------
// A self-contained React component for visualizing CLSTM-PPO training and trading simulation

import { useEffect, useMemo, useState } from 'react';
import { motion } from 'framer-motion';
import { Play, Pause, RotateCcw, ChevronLeft } from 'lucide-react';
import { Card, CardContent } from '../components/ui/card';

export const metadata = {
  name: "CLSTM-PPO Simulation",
  icon: "TrendingUp"
};

// Type definitions
interface Point {
  x: number;
  y: number;
}

interface XYLineChartProps {
  points: Point[];
  width?: number;
  height?: number;
  color?: string;
  showXAxis?: boolean;
  showYAxis?: boolean;
  xTicks?: number;
  yTicks?: number;
  label?: string;
  yLabel?: string;
}

interface SparklineProps {
  values: number[];
  width?: number;
  height?: number;
  color?: string;
}

interface Padding {
  top: number;
  right: number;
  bottom: number;
  left: number;
}

// Date formatting utility
const fmt = new Intl.DateTimeFormat(undefined, { month: 'short', day: '2-digit' });
const fmtDate = (ms: number): string => fmt.format(new Date(ms));

// ---------------- Architecture Visualization Utilities ----------------
const clamp = (v: number, a: number, b: number) => Math.max(a, Math.min(b, v));
function randn() {
  const u = 1 - Math.random();
  const v = 1 - Math.random();
  return Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v);
}
function nextPrice(p: number, vol = 0.6, drift = 0.02) {
  const r = drift / 100 + (vol / 100) * randn();
  return Math.max(0.1, p * Math.exp(r));
}

// Architecture Layout
interface NodeLayout {
  x: number;
  y: number;
  w: number;
  h: number;
  label: string;
}

const LAYOUT = {
  nodes: {
    market: { x: 20, y: 350, w: 180, h: 100, label: 'Market Data' },
    extractor: { x: 260, y: 350, w: 210, h: 110, label: 'LSTM Extractor' },
    actor: { x: 560, y: 200, w: 190, h: 100, label: 'Actor' },
    ppo: { x: 560, y: 380, w: 190, h: 80, label: 'PPO Update' },
    critic: { x: 560, y: 540, w: 190, h: 100, label: 'Critic' },
    actions: { x: 830, y: 200, w: 220, h: 100, label: 'Actions (Buy / Sell / Hold)' },
    value: { x: 830, y: 540, w: 220, h: 100, label: 'State Value E[V|s]' },
  },
  edges: [
    { from: 'market' as const, to: 'extractor' as const },
    { from: 'extractor' as const, to: 'actor' as const },
    { from: 'extractor' as const, to: 'critic' as const },
    { from: 'actor' as const, to: 'ppo' as const },
    { from: 'ppo' as const, to: 'critic' as const },
    { from: 'actor' as const, to: 'actions' as const },
    { from: 'critic' as const, to: 'value' as const },
  ],
} as const;

type NodeKey = keyof typeof LAYOUT.nodes;

interface CandleData {
  o: number;
  h: number;
  l: number;
  c: number;
}

// Architecture Visualization Components
function ArchNode({
  node,
  focused,
  onClick,
}: {
  node: NodeLayout;
  focused: boolean;
  onClick: () => void;
}) {
  return (
    <g onClick={onClick} style={{ cursor: 'pointer' }}>
      <rect
        x={node.x}
        y={node.y}
        rx={16}
        ry={16}
        width={node.w}
        height={node.h}
        fill={focused ? '#dbeafe' : '#eff6ff'}
        stroke={focused ? '#2563eb' : '#60a5fa'}
        strokeWidth={2}
      />
      <text
        x={node.x + node.w / 2}
        y={node.y + node.h / 2}
        textAnchor="middle"
        dominantBaseline="middle"
        fill="#1e40af"
        style={{ fontSize: 14, fontWeight: 500 }}
      >
        {node.label}
      </text>
    </g>
  );
}

function ArchEdge({ ax, ay, bx, by }: { ax: number; ay: number; bx: number; by: number }) {
  const id = Math.random().toString(36).slice(2);
  return (
    <g>
      <defs>
        <marker
          id={`arrow-${id}`}
          markerWidth="10"
          markerHeight="10"
          refX="8"
          refY="3"
          orient="auto"
          markerUnits="strokeWidth"
        >
          <path d="M0,0 L0,6 L9,3 z" fill="#3b82f6" />
        </marker>
      </defs>
      <line
        x1={ax}
        y1={ay}
        x2={bx}
        y2={by}
        stroke="#3b82f6"
        strokeWidth={2}
        markerEnd={`url(#arrow-${id})`}
      />
    </g>
  );
}

function FlowDot({
  ax,
  ay,
  bx,
  by,
  running,
}: {
  ax: number;
  ay: number;
  bx: number;
  by: number;
  running: boolean;
}) {
  const [t, setT] = useState(0);
  useEffect(() => {
    if (!running) return;
    const id = setInterval(() => setT((p) => (p + 0.01) % 1), 16);
    return () => clearInterval(id);
  }, [running]);
  const x = ax + (bx - ax) * t;
  const y = ay + (by - ay) * t;
  return <circle cx={x} cy={y} r={5} fill="#2563eb" opacity={0.8} />;
}

function CandleMini({ data }: { data: CandleData[] }) {
  const w = 300,
    h = 130,
    pad = 10;
  const highs = data.map((d) => d.h),
    lows = data.map((d) => d.l);
  const min = Math.min(...lows),
    max = Math.max(...highs);
  const scaleY = (v: number) => h - pad - ((v - min) / (max - min + 1e-6)) * (h - 2 * pad);
  const dx = (w - 2 * pad) / Math.max(1, data.length - 1);
  const sma = data.map((_, i) => {
    const start = Math.max(0, i - 11);
    const slice = data.slice(start, i + 1);
    return slice.reduce((a, b) => a + b.c, 0) / slice.length;
  });
  const smaPts = sma.map((v, i) => `${pad + i * dx},${scaleY(v)}`).join(' ');
  return (
    <svg width={w} height={h} className="rounded-xl bg-gray-50 border border-gray-200">
      {data.map((d, i) => {
        const x = pad + i * dx;
        const yH = scaleY(d.h),
          yL = scaleY(d.l);
        const yO = scaleY(d.o),
          yC = scaleY(d.c);
        const up = d.c >= d.o;
        return (
          <g key={i}>
            <line x1={x} y1={yH} x2={x} y2={yL} stroke="#6b7280" strokeWidth={1} />
            <rect
              x={x - 3}
              y={Math.min(yO, yC)}
              width={6}
              height={Math.max(2, Math.abs(yO - yC))}
              fill={up ? '#10b981' : '#ef4444'}
              rx={2}
            />
          </g>
        );
      })}
      <polyline points={smaPts} fill="none" stroke="#3b82f6" strokeWidth={2} />
    </svg>
  );
}

function PolicyBars({ probs }: { probs: number[] }) {
  const labels = ['Buy', 'Hold', 'Sell'];
  return (
    <div className="grid grid-cols-3 gap-3 select-none">
      {probs.map((p, i) => (
        <div key={i} className="space-y-1">
          <div className="text-xs text-gray-700 font-medium text-center">{labels[i]}</div>
          <div className="h-24 rounded-xl bg-gray-100 border border-gray-200 overflow-hidden flex items-end">
            <motion.div
              initial={{ height: 0 }}
              animate={{ height: `${clamp(p * 100, 1, 100)}%` }}
              transition={{ type: 'spring', stiffness: 120 }}
              className="w-full bg-blue-500"
            />
          </div>
          <div className="text-xs text-gray-600 text-center">{(p * 100).toFixed(1)}%</div>
        </div>
      ))}
    </div>
  );
}

function ValueSpark({ series }: { series: number[] }) {
  const min = Math.min(...series),
    max = Math.max(...series);
  const w = 260,
    h = 70;
  const pts = series.map((v, i) => [
    (i / Math.max(1, series.length - 1)) * w,
    h - ((v - min) / (max - min + 1e-6)) * h,
  ]);
  const d = `M ${pts.map((p) => p.join(',')).join(' L ')}`;
  return (
    <svg width={w} height={h} className="rounded-xl bg-gray-50 border border-gray-200">
      <path d={d} fill="none" stroke="#10b981" strokeWidth={2} />
    </svg>
  );
}

function ClipGraph({ clip = 0.2, ratio = 1.1 }: { clip?: number; ratio?: number }) {
  const w = 280,
    h = 120;
  const cx = w / 2,
    cy = h / 2;
  const scale = 80;
  const poly = [
    [cx - clip * scale, cy - (1 + clip) * 10],
    [cx + clip * scale, cy - (1 + clip) * 10],
    [cx + clip * scale, cy + 40],
    [cx - clip * scale, cy + 40],
  ];
  return (
    <svg width={w} height={h} className="rounded-xl bg-gray-50 border border-gray-200">
      <line x1={20} y1={cy} x2={w - 20} y2={cy} stroke="#6b7280" strokeWidth={1} />
      <line x1={cx} y1={20} x2={cx} y2={h - 20} stroke="#6b7280" strokeWidth={1} />
      <text x={w - 26} y={cy - 6} fill="#6b7280" fontSize={10}>
        r_t
      </text>
      <text x={cx + 6} y={26} fill="#6b7280" fontSize={10}>
        L^clip
      </text>
      <polygon
        points={poly.map((p) => p.join(',')).join(' ')}
        fill="rgba(59,130,246,0.2)"
        stroke="#3b82f6"
      />
      <circle cx={cx + (ratio - 1) * scale} cy={cy - 8} r={5} fill="#f59e0b" />
    </svg>
  );
}

function DetailPanel({ focus }: { focus: NodeKey | null }) {
  const [candles, setCandles] = useState<CandleData[]>(() => {
    const arr: CandleData[] = [];
    let p = 100;
    for (let i = 0; i < 60; i++) {
      const o = p;
      p = nextPrice(p);
      const c = p;
      const h = Math.max(o, c) * (1 + Math.random() * 0.01);
      const l = Math.min(o, c) * (1 - Math.random() * 0.01);
      arr.push({ o, h, l, c });
    }
    return arr;
  });
  const [probs, setProbs] = useState([0.33, 0.34, 0.33]);
  const [vals, setVals] = useState([0]);

  useEffect(() => {
    const id = setInterval(() => {
      setCandles((prev) => {
        const last = prev[prev.length - 1];
        const o = last.c,
          c = nextPrice(o);
        const h = Math.max(o, c) * (1 + Math.random() * 0.01);
        const l = Math.min(o, c) * (1 - Math.random() * 0.01);
        return [...prev.slice(-59), { o, h, l, c }];
      });
      setProbs((p) => {
        const d = p.map(() => (Math.random() - 0.5) * 0.06);
        let np = p.map((v, i) => clamp(v + d[i], 0.05, 0.9));
        const s = np.reduce((a, b) => a + b, 0);
        return np.map((v) => v / s);
      });
      setVals((arr) => [...arr.slice(-49), (arr.at(-1) ?? 0) + (Math.random() - 0.4) * 0.3]);
    }, 500);
    return () => clearInterval(id);
  }, []);

  if (focus === 'market') {
    return (
      <div>
        <CandleMini data={candles} />
        <div className="text-xs text-gray-600 mt-2">
          Streaming mock candles (auto-updating every 0.5s).
        </div>
      </div>
    );
  }
  if (focus === 'extractor') {
    return (
      <div className="space-y-3">
        <div className="h-24 w-full grid grid-cols-12 gap-1">
          {Array.from({ length: 24 }).map((_, i) => (
            <motion.div
              key={i}
              className="rounded bg-blue-400/40"
              initial={{ opacity: 0.3 }}
              animate={{ opacity: [0.3, 0.95, 0.3] }}
              transition={{ duration: 1.1, repeat: Infinity, delay: i * 0.03 }}
            />
          ))}
        </div>
        <div className="text-xs text-gray-600">
          Temporal features pulsing through the LSTM encoder (mocked embeddings).
        </div>
      </div>
    );
  }
  if (focus === 'actor') {
    return (
      <div className="space-y-3">
        <PolicyBars probs={probs} />
        <div className="text-xs text-gray-600">
          Simulated action probabilities π(a|s) evolving over time.
        </div>
      </div>
    );
  }
  if (focus === 'critic') {
    return (
      <div className="space-y-3">
        <ValueSpark series={vals} />
        <div className="text-xs text-gray-600">
          Toy critic value estimate V(s) series (illustrative).
        </div>
      </div>
    );
  }
  if (focus === 'ppo') {
    const ratio = 1.0 + Math.sin(vals.length / 10) * 0.4;
    return (
      <div className="space-y-3">
        <ClipGraph clip={0.2} ratio={ratio} />
        <div className="text-xs text-gray-600">
          PPO clipped surrogate objective. Orange marker ≈ probability ratio rₜ.
        </div>
      </div>
    );
  }
  if (focus === 'actions') {
    const labels = ['Buy', 'Hold', 'Sell'];
    const maxIdx = probs.indexOf(Math.max(...probs));
    return (
      <div className="space-y-3">
        <div className="flex gap-2">
          {labels.map((k, i) => (
            <motion.div
              key={k}
              className="px-3 py-2 rounded-xl bg-gray-100 border border-gray-300 text-gray-700 font-medium"
              animate={{ scale: i === maxIdx ? [1, 1.08, 1] : 1 }}
              transition={{ duration: 0.9, repeat: Infinity }}
            >
              {k}
            </motion.div>
          ))}
        </div>
        <div className="text-xs text-gray-600">
          Policy-preferred action pulses. (In a live sim, this would update positions and reward.)
        </div>
      </div>
    );
  }
  if (focus === 'value') {
    const min = Math.min(...vals),
      max = Math.max(...vals);
    const w = 260,
      h = 70;
    const pts = vals.map((v, i) => [
      (i / Math.max(1, vals.length - 1)) * w,
      h - ((v - min) / (max - min + 1e-6)) * h,
    ]);
    const d = `M ${pts.map((p) => p.join(',')).join(' L ')}`;
    return (
      <div className="space-y-3">
        <svg width={w} height={h} className="rounded-xl bg-gray-50 border border-gray-200">
          <path d={d} fill="none" stroke="#3b82f6" strokeWidth={2} />
        </svg>
        <div className="text-xs text-gray-600">State value E[V|s] over time (mocked).</div>
      </div>
    );
  }
  return (
    <div className="text-gray-600 text-sm">Click a node to view its simulation.</div>
  );
}

// XYLineChart component - renders a line chart with optional axes
function XYLineChart({
  points,
  width = 540,
  height = 220,
  color = '#60a5fa',
  showXAxis = true,
  showYAxis = false,
  xTicks = 6,
  yTicks = 4,
  label,
  yLabel,
}: XYLineChartProps) {
  const padding: Padding = {
    top: 10,
    right: 12,
    bottom: showXAxis ? 28 : 10,
    left: showYAxis ? 54 : 12,
  };

  const innerW = Math.max(1, width - padding.left - padding.right);
  const innerH = Math.max(1, height - padding.top - padding.bottom);

  if (!points || points.length === 0) {
    return (
      <svg width={width} height={height}>
        <text x={8} y={16} fill="#666">
          No data
        </text>
      </svg>
    );
  }

  const xs = points.map((p) => p.x);
  const ys = points.map((p) => p.y);
  const xMin = Math.min(...xs);
  const xMax = Math.max(...xs);
  const yMinRaw = Math.min(...ys);
  const yMaxRaw = Math.max(...ys);
  const yPad = (yMaxRaw - yMinRaw) * 0.08 || 1;
  const yMin = yMinRaw - yPad;
  const yMax = yMaxRaw + yPad;

  const sx = (x: number): number =>
    padding.left + (xMax === xMin ? 0 : (x - xMin) / (xMax - xMin)) * innerW;
  const sy = (y: number): number =>
    padding.top + (1 - (y - yMin) / (yMax - yMin || 1)) * innerH;

  const pathD = points
    .map((p, i) => `${i ? 'L' : 'M'}${sx(p.x)},${sy(p.y)}`)
    .join(' ');

  const ticksX = Array.from(
    { length: xTicks },
    (_, i) => xMin + (i / (xTicks - 1)) * (xMax - xMin || 1)
  );
  const ticksY = Array.from(
    { length: yTicks },
    (_, i) => yMin + (i / (yTicks - 1)) * (yMax - yMin || 1)
  );

  return (
    <svg width={width} height={height} role="img" aria-label={label ?? 'line chart'}>
      {showXAxis && (
        <g>
          <line
            x1={padding.left}
            y1={height - padding.bottom}
            x2={width - padding.right}
            y2={height - padding.bottom}
            stroke="#d1d5db"
            strokeWidth={1}
          />
          {ticksX.map((tx, i) => (
            <g key={i}>
              <line
                x1={sx(tx)}
                y1={height - padding.bottom}
                x2={sx(tx)}
                y2={height - padding.bottom + 4}
                stroke="#9ca3af"
              />
              <text
                x={sx(tx)}
                y={height - 6}
                fill="#6b7280"
                fontSize="10"
                textAnchor="middle"
              >
                {fmtDate(tx)}
              </text>
            </g>
          ))}
        </g>
      )}
      {showYAxis && (
        <g>
          <line
            x1={padding.left}
            y1={padding.top}
            x2={padding.left}
            y2={height - padding.bottom}
            stroke="#d1d5db"
            strokeWidth={1}
          />
          {ticksY.map((ty, i) => (
            <g key={i}>
              <line
                x1={padding.left - 3}
                y1={sy(ty)}
                x2={padding.left}
                y2={sy(ty)}
                stroke="#9ca3af"
              />
              <text
                x={padding.left - 10}
                y={sy(ty) + 3}
                fill="#6b7280"
                fontSize="10"
                textAnchor="end"
              >
                {Math.round(ty)}
              </text>
            </g>
          ))}
          {yLabel && (
            <text
              transform={`rotate(-90)`}
              x={-height / 2}
              y={16}
              fill="#6b7280"
              fontSize="11"
              textAnchor="middle"
            >
              {yLabel}
            </text>
          )}
        </g>
      )}
      <path d={pathD} fill="none" stroke={color} strokeWidth={2} />
    </svg>
  );
}

// Sparkline component - renders a minimal line chart
function Sparkline({
  values,
  width = 540,
  height = 50,
  color = '#4ade80',
}: SparklineProps) {
  if (!values || values.length === 0) {
    return <svg width={width} height={height} />;
  }

  const padding = 2;
  const innerW = width - padding * 2;
  const innerH = height - padding * 2;
  const xMax = Math.max(1, values.length - 1);
  const yMinRaw = Math.min(...values);
  const yMaxRaw = Math.max(...values);
  const yPad = (yMaxRaw - yMinRaw) * 0.1 || 1;
  const yMin = yMinRaw - yPad;
  const yMax = yMaxRaw + yPad;

  const sx = (x: number): number => padding + (x / xMax) * innerW;
  const sy = (y: number): number =>
    padding + (1 - (y - yMin) / (yMax - yMin || 1)) * innerH;

  const d = values.map((v, i) => `${i ? 'L' : 'M'}${sx(i)},${sy(v)}`).join(' ');

  return (
    <svg width={width} height={height}>
      <path d={d} fill="none" stroke={color} strokeWidth={1.5} />
    </svg>
  );
}

// Main visualization component
export default function ClstmPpoViz() {
  const [frame, setFrame] = useState<number>(0);
  const [isPlaying, setIsPlaying] = useState<boolean>(false);
  const [archFocus, setArchFocus] = useState<NodeKey | null>(null);
  const [archRunning, setArchRunning] = useState(true);

  const FRAMES = 50;
  const TICK_MS = 500;
  const seed = 42;

  // Simple LCG random number generator for reproducible results
  const rng = useMemo(() => {
    let s = seed >>> 0;
    return (): number => {
      s = (1664525 * s + 1013904223) >>> 0;
      return (s & 0xfffffff) / 0xfffffff;
    };
  }, []);

  const startDate = useMemo(() => new Date('2025-01-01T00:00:00Z'), []);
  const timestamps = useMemo(
    () =>
      Array.from(
        { length: FRAMES },
        (_, i) => new Date(startDate.getTime() + i * 86400000)
      ),
    [FRAMES, startDate]
  );
  const timestampsMs = useMemo(() => timestamps.map((t) => t.getTime()), [timestamps]);

  // Generate reward series data
  const rewardsSeries = useMemo(
    () =>
      timestampsMs.map((ms, i) => ({
        x: ms,
        y: -10 + 5 * Math.sqrt(i + 1) + (rng() - 0.5) * 5,
      })),
    [timestampsMs, rng]
  );

  // Generate portfolio value series
  const portfolioSeries = useMemo(() => {
    let value = 10000;
    const data: Point[] = [];
    for (let i = 0; i < timestampsMs.length; i++) {
      const shock = (rng() - 0.5) * 600;
      const drift = 120 + shock;
      value = Math.max(9000, value + drift);
      data.push({ x: timestampsMs[i], y: Math.round(value) });
    }
    return data;
  }, [timestampsMs, rng]);

  // Generate stock price series
  const stockSeries = useMemo(() => {
    let price = 100;
    const data: Point[] = [];
    for (let i = 0; i < timestampsMs.length; i++) {
      const eps = (rng() - 0.5) * 2;
      const r = 0.0003 + eps * 0.02;
      price = Math.max(1, price * (1 + r));
      data.push({ x: timestampsMs[i], y: Number(price.toFixed(2)) });
    }
    return data;
  }, [timestampsMs, rng]);

  // Animation effect
  useEffect(() => {
    let timer: ReturnType<typeof setInterval> | undefined;
    if (isPlaying) {
      timer = setInterval(() => {
        setFrame((f) => (f < FRAMES - 1 ? f + 1 : FRAMES - 1));
      }, TICK_MS);
    }
    return () => {
      if (timer) clearInterval(timer);
    };
  }, [isPlaying, FRAMES, TICK_MS]);

  // Calculate daily returns
  const dailyReturns = useMemo(() => {
    const r: number[] = [];
    for (let i = 1; i <= frame; i++) {
      const prev = portfolioSeries[i - 1].y;
      const curr = portfolioSeries[i].y;
      r.push(((curr - prev) / prev) * 100);
    }
    return r;
  }, [portfolioSeries, frame]);

  // Calculate performance metrics
  const cumulativeReturn =
    ((portfolioSeries[frame].y - portfolioSeries[0].y) / portfolioSeries[0].y) * 100;
  const mean = dailyReturns.reduce((a, b) => a + b, 0) / (dailyReturns.length || 1);
  const std = Math.sqrt(
    dailyReturns.reduce((a, b) => a + (b - mean) ** 2, 0) / (dailyReturns.length || 1)
  );
  const volatility = std * Math.sqrt(252);
  const sharpe = (mean / (std || 1e-9)) * Math.sqrt(252);
  const peak = Math.max(...portfolioSeries.slice(0, frame + 1).map((p) => p.y));
  const drawdown = ((peak - portfolioSeries[frame].y) / peak) * 100;
  const sparkValues = dailyReturns.slice(-10);

  // Slice data to current frame
  const rewardPoints = rewardsSeries.slice(0, frame + 1);
  const portfolioPoints = portfolioSeries.slice(0, frame + 1);
  const stockPoints = stockSeries.slice(0, frame + 1);

  return (
    <div className="max-w-7xl mx-auto p-8">
      <h1 className="text-3xl font-bold mb-6 text-center text-gray-900">
        CLSTM-PPO Training & Trading Simulation
      </h1>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6 min-h-[280px]">
        {/* Left: Reward + Stock charts */}
        <Card className="bg-white shadow-lg border-gray-200 overflow-hidden">
          <CardContent className="pt-4">
            <XYLineChart
              points={rewardPoints}
              width={540}
              height={220}
              color="#3b82f6"
              showXAxis
              showYAxis
              yLabel="Reward"
              label="Training Reward"
            />
            <p className="text-center mt-2 text-gray-600">Training Reward Over Time</p>
          </CardContent>
          <CardContent className="pt-4 pb-4">
            <XYLineChart
              points={stockPoints}
              width={540}
              height={220}
              color="#f59e0b"
              showXAxis
              showYAxis
              yLabel="Stock Price ($)"
              label="Underlying Stock Price"
            />
            <p className="text-center mt-2 text-gray-600">Underlying Stock Price</p>
          </CardContent>
        </Card>

        {/* Right: Portfolio chart + Stats */}
        <Card className="bg-white shadow-lg border-gray-200 overflow-hidden pb-8">
          <CardContent className="pt-4 h-[260px]">
            <XYLineChart
              points={portfolioPoints}
              width={540}
              height={260}
              color="#10b981"
              showXAxis
              showYAxis
              yLabel="Portfolio Value ($)"
              label="Portfolio Value"
            />
            <p className="text-center mt-2 text-gray-600">
              Portfolio Value with Drawdowns
            </p>
          </CardContent>

          {/* KPIs */}
          <div className="mt-12" />
          <CardContent className="pt-3 pb-3 text-sm min-h-[190px]">
            <div className="space-y-1">
              <div className="flex justify-between">
                <span className="text-gray-700 font-medium">Cumulative Return</span>
                <span
                  className={`font-mono tabular-nums w-28 text-right font-semibold ${
                    cumulativeReturn >= 0 ? 'text-green-600' : 'text-red-600'
                  }`}
                >
                  {cumulativeReturn.toFixed(2)}%
                </span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-700 font-medium">Daily Return</span>
                <span
                  className={`font-mono tabular-nums w-28 text-right font-semibold ${
                    (dailyReturns[dailyReturns.length - 1] || 0) >= 0
                      ? 'text-green-600'
                      : 'text-red-600'
                  }`}
                >
                  {(dailyReturns[dailyReturns.length - 1] || 0).toFixed(2)}%
                </span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-700 font-medium">Max Drawdown</span>
                <span
                  className={`font-mono tabular-nums w-28 text-right font-semibold ${
                    drawdown > 5 ? 'text-red-600' : 'text-gray-600'
                  }`}
                >
                  {drawdown.toFixed(2)}%
                </span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-700 font-medium">Volatility (ann.)</span>
                <span className="font-mono tabular-nums w-28 text-right font-semibold text-blue-600">
                  {volatility.toFixed(2)}%
                </span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-700 font-medium">Sharpe (ann.)</span>
                <span
                  className={`font-mono tabular-nums w-28 text-right font-semibold ${
                    sharpe > 1
                      ? 'text-green-600'
                      : sharpe > 0
                      ? 'text-yellow-600'
                      : 'text-red-600'
                  }`}
                >
                  {sharpe.toFixed(2)}
                </span>
              </div>
              <div className="h-[54px] mt-2 pt-2 border-t border-gray-300">
                <Sparkline
                  values={sparkValues}
                  width={540}
                  height={50}
                  color={
                    (dailyReturns[dailyReturns.length - 1] ?? 0) >= 0
                      ? '#10b981'
                      : '#ef4444'
                  }
                />
              </div>
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Architecture Visualization */}
      <div className="mt-8">
        <h2 className="text-2xl font-bold mb-4 text-gray-900">CLSTM-PPO Architecture</h2>
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
          <div className="lg:col-span-8">
            <Card className="bg-white shadow-lg border-gray-200">
              <CardContent className="pt-4">
                <div className="flex justify-between mb-4">
                  <h3 className="text-lg font-semibold text-gray-800">Architecture Flow</h3>
                  <div className="flex gap-2">
                    <button
                      onClick={() => setArchRunning((r) => !r)}
                      className="inline-flex items-center gap-2 px-3 py-2 rounded-lg border border-gray-300 bg-white text-gray-700 hover:bg-gray-50 font-medium transition-colors"
                    >
                      {archRunning ? (
                        <>
                          <Pause className="w-4 h-4" /> Pause
                        </>
                      ) : (
                        <>
                          <Play className="w-4 h-4" /> Play
                        </>
                      )}
                    </button>
                    <button
                      onClick={() => setArchFocus(null)}
                      className="inline-flex items-center gap-2 px-3 py-2 rounded-lg border border-gray-300 bg-white text-gray-700 hover:bg-gray-50 font-medium transition-colors"
                    >
                      <RotateCcw className="w-4 h-4" /> Reset
                    </button>
                  </div>
                </div>
                <svg viewBox="0 0 1200 800" className="w-full h-[500px] bg-gray-50 rounded-xl">
                  {LAYOUT.edges.map((e, i) => {
                    const from = LAYOUT.nodes[e.from];
                    const to = LAYOUT.nodes[e.to];
                    return (
                      <ArchEdge
                        key={i}
                        ax={from.x + from.w / 2}
                        ay={from.y + from.h / 2}
                        bx={to.x + to.w / 2}
                        by={to.y + to.h / 2}
                      />
                    );
                  })}
                  {LAYOUT.edges.map((e, i) => {
                    const from = LAYOUT.nodes[e.from];
                    const to = LAYOUT.nodes[e.to];
                    return (
                      <FlowDot
                        key={`dot-${i}`}
                        ax={from.x + from.w / 2}
                        ay={from.y + from.h / 2}
                        bx={to.x + to.w / 2}
                        by={to.y + to.h / 2}
                        running={archRunning}
                      />
                    );
                  })}
                  {Object.entries(LAYOUT.nodes).map(([k, node]) => (
                    <ArchNode
                      key={k}
                      node={node}
                      focused={archFocus === k}
                      onClick={() => setArchFocus(k as NodeKey)}
                    />
                  ))}
                </svg>
              </CardContent>
            </Card>
          </div>
          <div className="lg:col-span-4">
            <Card className="bg-white shadow-lg border-gray-200">
              <CardContent className="pt-4">
                <h4 className="text-lg font-semibold text-gray-800 mb-4">Detail Panel</h4>
                <DetailPanel focus={archFocus} />
                {archFocus && (
                  <button
                    onClick={() => setArchFocus(null)}
                    className="mt-4 inline-flex items-center gap-2 px-3 py-2 rounded-lg border border-gray-300 bg-white text-gray-700 hover:bg-gray-50 font-medium transition-colors"
                  >
                    <ChevronLeft className="w-4 h-4" /> Back
                  </button>
                )}
              </CardContent>
            </Card>
          </div>
        </div>
      </div>

      {/* Controls */}
      <div className="flex flex-wrap items-center justify-center gap-4 mt-6">
        <button
          onClick={() => setIsPlaying((p) => !p)}
          className="px-6 py-2 bg-blue-600 text-white rounded-lg font-medium hover:bg-blue-700 transition-colors shadow-md"
        >
          {isPlaying ? 'Pause' : 'Play'}
        </button>
        <div className="w-full md:w-1/2 px-4">
          <input
            type="range"
            min={0}
            max={FRAMES - 1}
            step={1}
            value={frame}
            onChange={(e) => setFrame(Number(e.target.value))}
            className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer"
          />
        </div>
        <span className="tabular-nums text-gray-700 font-medium">
          Frame: {frame + 1}/{FRAMES}
        </span>
        <button
          onClick={() => {
            setFrame(0);
            setIsPlaying(false);
          }}
          className="px-6 py-2 bg-gray-200 text-gray-700 rounded-lg font-medium hover:bg-gray-300 transition-colors shadow-md"
        >
          Reset
        </button>
      </div>
    </div>
  );
}
