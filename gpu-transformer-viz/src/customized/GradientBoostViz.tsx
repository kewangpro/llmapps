import { useEffect, useMemo, useState } from "react";
import { motion } from "framer-motion";
import { Zap } from 'lucide-react';

// Required metadata export
export const metadata = {
  name: "Gradient Boost",
  icon: "Zap"
};

// ===================== Small GBM-on-Residuals Sandbox =====================
// Baseline is the mean (μ). Weak learners are stumps / depth-2 / depth-3 trees.
// Splits prefer lag1; index feature is disabled by default to avoid flat forecasts.
// Forecast rollout is stage-wise: each tree contributes η * leaf(..., lag),
// and the running prediction becomes the next lag for the same horizon.

type SplitFeature = "idx" | "lag1";

// ---------------------- Weak learner: Stump ----------------------
function trainStump(
  residuals: number[],
  xIdx: number[],
  lag1: number[],
  cfg: { idxPenalty: number; minGainFrac: number; forbidIdx: boolean } = {
    idxPenalty: 1.25,
    minGainFrac: 0.01,
    forbidIdx: true,
  }
) {
  const n = residuals.length;
  const baseSSE = residuals.reduce((s, r) => s + r * r, 0);
  const minGainAbs = cfg.minGainFrac * Math.max(1e-12, baseSSE);

  let best = {
    type: "stump" as const,
    feature: "lag1" as SplitFeature,
    thr: 0,
    leftMean: 0,
    rightMean: 0,
    sse: Infinity,
    preds: new Array(n).fill(0) as number[],
  };

  const tryFeature = (feat: number[], name: SplitFeature, thresholds: number[]) => {
    for (const t of thresholds) {
      const L: number[] = [], R: number[] = [];
      for (let i = 0; i < n; i++) (feat[i] <= t ? L : R).push(i);
      if (!L.length || !R.length) continue;
      const lMean = L.reduce((s, i) => s + residuals[i], 0) / L.length;
      const rMean = R.reduce((s, i) => s + residuals[i], 0) / R.length;
      let sse = 0;
      const preds = new Array(n) as number[];
      for (let i = 0; i < n; i++) {
        const p = feat[i] <= t ? lMean : rMean;
        preds[i] = p;
        sse += (residuals[i] - p) ** 2;
      }
      const gain = baseSSE - sse;
      if (gain < minGainAbs) continue;
      const penalty = name === "idx" ? cfg.idxPenalty : 1.0;
      const sseAdj = sse * penalty;
      if (sseAdj < best.sse) best = { type: "stump", feature: name, thr: t, leftMean: lMean, rightMean: rMean, sse: sseAdj, preds };
    }
  };

  // index thresholds
  const idxThr: number[] = [];
  for (let t = 0.5; t < n - 0.5; t += 1) idxThr.push(t);
  if (!cfg.forbidIdx) tryFeature(xIdx, "idx", idxThr);

  // lag1 thresholds (midpoints between unique sorted lag values)
  const uniqLag = Array.from(new Set(lag1)).sort((a, b) => a - b);
  const mids: number[] = [];
  for (let i = 0; i < uniqLag.length - 1; i++) mids.push((uniqLag[i] + uniqLag[i + 1]) / 2);
  const stepThin = Math.max(1, Math.ceil(mids.length / 24));
  const midsThin = mids.filter((_, i) => i % stepThin === 0);
  if (midsThin.length) tryFeature(lag1, "lag1", midsThin);
  else if (mids.length) tryFeature(lag1, "lag1", mids);

  return best;
}

// ---------------------- Weak learner: Depth 2 ----------------------
function trainTreeDepth2(
  residuals: number[],
  xIdx: number[],
  lag1: number[],
  cfg: { idxPenalty: number; minGainFrac: number; forbidIdx: boolean }
) {
  const n = residuals.length;
  const root = trainStump(residuals, xIdx, lag1, cfg);
  const goLeft = (i: number) => (root.feature === "idx" ? xIdx[i] : lag1[i]) <= root.thr;
  const leftMask = xIdx.map((_, i) => goLeft(i));
  const rightMask = leftMask.map((v) => !v);
  const sub = (arr: number[], m: boolean[]) => arr.filter((_, i) => m[i]);
  const L = { idx: sub(xIdx, leftMask), lag1: sub(lag1, leftMask), res: sub(residuals, leftMask) };
  const R = { idx: sub(xIdx, rightMask), lag1: sub(lag1, rightMask), res: sub(residuals, rightMask) };
  const childLeft = L.res.length >= 2 ? trainStump(L.res, L.idx, L.lag1, cfg) : null;
  const childRight = R.res.length >= 2 ? trainStump(R.res, R.idx, R.lag1, cfg) : null;

  const preds = new Array(n) as number[];
  let sse = 0;
  for (let i = 0; i < n; i++) {
    const left = goLeft(i);
    let p: number;
    if (left) {
      if (childLeft) {
        const f = childLeft.feature === "idx" ? xIdx[i] : lag1[i];
        p = f <= childLeft.thr ? childLeft.leftMean : childLeft.rightMean;
      } else {
        const m = L.res.reduce((a, b) => a + b, 0) / Math.max(1, L.res.length);
        p = m;
      }
    } else {
      if (childRight) {
        const f = childRight.feature === "idx" ? xIdx[i] : lag1[i];
        p = f <= childRight.thr ? childRight.leftMean : childRight.rightMean;
      } else {
        const m = R.res.reduce((a, b) => a + b, 0) / Math.max(1, R.res.length);
        p = m;
      }
    }
    preds[i] = p;
    sse += (residuals[i] - p) ** 2;
  }
  return { type: "tree2" as const, root, childLeft, childRight, preds, sse };
}

// ---------------------- Weak learner: Depth 3 ----------------------
function trainTreeDepth3(
  residuals: number[],
  xIdx: number[],
  lag1: number[],
  cfg: { idxPenalty: number; minGainFrac: number; forbidIdx: boolean }
) {
  const n = residuals.length;
  const root = trainStump(residuals, xIdx, lag1, cfg);
  const goLeft = (i: number) => (root.feature === "idx" ? xIdx[i] : lag1[i]) <= root.thr;
  const leftMask = xIdx.map((_, i) => goLeft(i));
  const rightMask = leftMask.map((v) => !v);
  const sub = (arr: number[], m: boolean[]) => arr.filter((_, i) => m[i]);
  const L = { idx: sub(xIdx, leftMask), lag1: sub(lag1, leftMask), res: sub(residuals, leftMask) };
  const R = { idx: sub(xIdx, rightMask), lag1: sub(lag1, rightMask), res: sub(residuals, rightMask) };

  const childLeft = L.res.length >= 2 ? trainStump(L.res, L.idx, L.lag1, cfg) : null;
  const childRight = R.res.length >= 2 ? trainStump(R.res, R.idx, R.lag1, cfg) : null;

  // grandchildren
  let gLL: ReturnType<typeof trainStump> | null = null;
  let gLR: ReturnType<typeof trainStump> | null = null;
  let gRL: ReturnType<typeof trainStump> | null = null;
  let gRR: ReturnType<typeof trainStump> | null = null;

  if (childLeft) {
    const goLeft2 = (i: number) => (childLeft.feature === "idx" ? xIdx[i] : lag1[i]) <= childLeft.thr;
    const llMask = xIdx.map((_, i) => leftMask[i] && goLeft2(i));
    const lrMask = xIdx.map((_, i) => leftMask[i] && !goLeft2(i));
    const LL = { idx: sub(xIdx, llMask), lag1: sub(lag1, llMask), res: sub(residuals, llMask) };
    const LR = { idx: sub(xIdx, lrMask), lag1: sub(lag1, lrMask), res: sub(residuals, lrMask) };
    gLL = LL.res.length >= 2 ? trainStump(LL.res, LL.idx, LL.lag1, cfg) : null;
    gLR = LR.res.length >= 2 ? trainStump(LR.res, LR.idx, LR.lag1, cfg) : null;
  }
  if (childRight) {
    const goRight2 = (i: number) => (childRight.feature === "idx" ? xIdx[i] : lag1[i]) <= childRight.thr;
    const rlMask = xIdx.map((_, i) => rightMask[i] && goRight2(i));
    const rrMask = xIdx.map((_, i) => rightMask[i] && !goRight2(i));
    const RL = { idx: sub(xIdx, rlMask), lag1: sub(lag1, rlMask), res: sub(residuals, rlMask) };
    const RR = { idx: sub(xIdx, rrMask), lag1: sub(lag1, rrMask), res: sub(residuals, rrMask) };
    gRL = RL.res.length >= 2 ? trainStump(RL.res, RL.idx, RL.lag1, cfg) : null;
    gRR = RR.res.length >= 2 ? trainStump(RR.res, RR.idx, RR.lag1, cfg) : null;
  }

  const preds = new Array(n) as number[];
  let sse = 0;
  for (let i = 0; i < n; i++) {
    const left = (root.feature === "idx" ? xIdx[i] : lag1[i]) <= root.thr;
    let p: number;
    if (left) {
      if (childLeft) {
        const f2 = childLeft.feature === "idx" ? xIdx[i] : lag1[i];
        const left2 = f2 <= childLeft.thr;
        if (left2) {
          if (gLL) {
            const f3 = gLL.feature === "idx" ? xIdx[i] : lag1[i];
            p = f3 <= gLL.thr ? gLL.leftMean : gLL.rightMean;
          } else {
            p = childLeft.leftMean;
          }
        } else {
          if (gLR) {
            const f3 = gLR.feature === "idx" ? xIdx[i] : lag1[i];
            p = f3 <= gLR.thr ? gLR.leftMean : gLR.rightMean;
          } else {
            p = childLeft.rightMean;
          }
        }
      } else {
        p = root.leftMean;
      }
    } else {
      if (childRight) {
        const f2 = childRight.feature === "idx" ? xIdx[i] : lag1[i];
        const left2 = f2 <= childRight.thr;
        if (left2) {
          if (gRL) {
            const f3 = gRL.feature === "idx" ? xIdx[i] : lag1[i];
            p = f3 <= gRL.thr ? gRL.leftMean : gRL.rightMean;
          } else {
            p = childRight.leftMean;
          }
        } else {
          if (gRR) {
            const f3 = gRR.feature === "idx" ? xIdx[i] : lag1[i];
            p = f3 <= gRR.thr ? gRR.leftMean : gRR.rightMean;
          } else {
            p = childRight.rightMean;
          }
        }
      } else {
        p = root.rightMean;
      }
    }
    preds[i] = p;
    sse += (residuals[i] - p) ** 2;
  }

  return { type: "tree3" as const, root, childLeft, childRight, gLL, gLR, gRL, gRR, preds, sse };
}

// ---------------------- Prediction (walks any model) ----------------------
function predictLeaf(
  model: ReturnType<typeof trainStump> | ReturnType<typeof trainTreeDepth2> | ReturnType<typeof trainTreeDepth3>,
  xf: number,
  lagf: number
) {
  if (!model) return 0;
  const kind = (model as any).type as "stump" | "tree2" | "tree3";
  if (kind === "stump") {
    const stump = model as ReturnType<typeof trainStump>;
    const feat = stump.feature === "idx" ? xf : lagf;
    return feat <= stump.thr ? stump.leftMean : stump.rightMean;
  }
  if (kind === "tree2") {
    const tree = model as ReturnType<typeof trainTreeDepth2>;
    const { root, childLeft, childRight } = tree;
    const left = (root.feature === "idx" ? xf : lagf) <= root.thr;
    if (left) {
      if (!childLeft) return root.leftMean;
      const f = childLeft.feature === "idx" ? xf : lagf;
      return f <= childLeft.thr ? childLeft.leftMean : childLeft.rightMean;
    } else {
      if (!childRight) return root.rightMean;
      const f = childRight.feature === "idx" ? xf : lagf;
      return f <= childRight.thr ? childRight.leftMean : childRight.rightMean;
    }
  }
  // tree3
  const t3 = model as ReturnType<typeof trainTreeDepth3>;
  const { root, childLeft, childRight, gLL, gLR, gRL, gRR } = t3;
  const left = (root.feature === "idx" ? xf : lagf) <= root.thr;
  if (left) {
    if (!childLeft) return root.leftMean;
    const f2 = childLeft.feature === "idx" ? xf : lagf;
    const left2 = f2 <= childLeft.thr;
    if (left2) {
      if (!gLL) return childLeft.leftMean;
      const f3 = gLL.feature === "idx" ? xf : lagf;
      return f3 <= gLL.thr ? gLL.leftMean : gLL.rightMean;
    } else {
      if (!gLR) return childLeft.rightMean;
      const f3 = gLR.feature === "idx" ? xf : lagf;
      return f3 <= gLR.thr ? gLR.leftMean : gLR.rightMean;
    }
  } else {
    if (!childRight) return root.rightMean;
    const f2 = childRight.feature === "idx" ? xf : lagf;
    const left2 = f2 <= childRight.thr;
    if (left2) {
      if (!gRL) return childRight.leftMean;
      const f3 = gRL.feature === "idx" ? xf : lagf;
      return f3 <= gRL.thr ? gRL.leftMean : gRL.rightMean;
    } else {
      if (!gRR) return childRight.rightMean;
      const f3 = gRR.feature === "idx" ? xf : lagf;
      return f3 <= gRR.thr ? gRR.leftMean : gRR.rightMean;
    }
  }
}

// ---------------------- Number formatters ----------------------
const fmtCompact = new Intl.NumberFormat("en-US", { notation: "compact", maximumFractionDigits: 2 });
const fmtFull = new Intl.NumberFormat("en-US", { maximumFractionDigits: 2 });

// ===================== Component =====================
export default function GradientBoostViz() {
  const [inputData, setInputData] = useState("5,6,7,6,8,9,10,9,11,10");
  const [eta, setEta] = useState(0.4);
  const [boostSteps, setBoostSteps] = useState(6);
  const [depth, setDepth] = useState(2);
  const [forecastCount, setForecastCount] = useState(1);
  const [step, setStep] = useState(0);
  const [scrollWidth, setScrollWidth] = useState(640);
  const [inspectIdx, setInspectIdx] = useState(0);

  // Core compute
  const { yTrue, mu, fits, residualsByStep, models, forecastsByStep } = useMemo(() => {
    const y = inputData.split(",").map((v) => parseFloat(v.trim())).filter((v) => !isNaN(v));
    const n = y.length;
    const xIdx = Array.from({ length: n }, (_, i) => i);
    if (!n)
      return { yTrue: [] as number[], mu: 0, fits: [] as number[][], residualsByStep: [] as number[][], models: [] as any[], forecastsByStep: [] as number[][] };

    const mu = y.reduce((a, b) => a + b, 0) / n;
    const lag1 = y.map((_, i) => (i === 0 ? mu : y[i - 1]));

    let yPred = new Array(n).fill(mu) as number[];
    const fits: number[][] = [];
    const resAll: number[][] = [];
    const models: any[] = [];
    const cfg = { idxPenalty: 1.25, minGainFrac: 0.01, forbidIdx: true };

    for (let s = 0; s < boostSteps; s++) {
      fits.push([...yPred]);
      const res = y.map((yi, i) => yi - yPred[i]);
      resAll.push(res);
      const model = depth === 3 ? trainTreeDepth3(res, xIdx, lag1, cfg) : depth === 2 ? trainTreeDepth2(res, xIdx, lag1, cfg) : trainStump(res, xIdx, lag1, cfg);
      models.push(model);
      yPred = yPred.map((v, i) => v + eta * (model as any).preds[i]);
    }

    // Forecasts: stage-wise rollout per tree, recursive across horizons
    const forecasts: number[][] = [];
    for (let s = 0; s < boostSteps; s++) {
      const fut: number[] = [];
      let lastLag = y[n - 1];
      for (let j = 0; j < forecastCount; j++) {
        const xf = n + j;
        let pred = mu;
        let tempLag = lastLag;
        for (let k = 0; k <= s; k++) {
          const leaf = predictLeaf(models[k], xf, tempLag);
          pred += eta * leaf;
          tempLag = pred;
        }
        fut.push(pred);
        lastLag = pred;
      }
      forecasts.push(fut);
    }

    return { yTrue: y, mu, fits, residualsByStep: resAll, models, forecastsByStep: forecasts };
  }, [inputData, eta, boostSteps, depth, forecastCount]);

  useEffect(() => {
    const base = 560;
    const spacing = 38;
    const w = Math.min(980, 50 + (yTrue.length + forecastCount + 1) * spacing);
    setScrollWidth(Math.max(base, w));
    setStep(0);
    setInspectIdx(0);
  }, [yTrue.length, forecastCount, eta, boostSteps, depth]);

  const currentFit = fits[step];
  const currentForecasts = forecastsByStep[step] || [];
  const currentModel: any = models[step];
  const currentResiduals = residualsByStep[step] || [];

  // ---------- Dynamic Y scaling ----------
  const yScale = useMemo(() => {
    const canvasTop = 40;
    const canvasBottom = 270;
    const collect: number[] = [];
    for (const v of yTrue) collect.push(v);
    if (currentFit) for (const v of currentFit) collect.push(v);
    for (const v of currentForecasts) collect.push(v);
    if (!collect.length) return { toCY: (_y: number) => 150, ticks: [] as number[], yMin: 0, yMax: 1 };
    let yMin = Math.min(...collect);
    let yMax = Math.max(...collect);
    if (yMax === yMin) {
      yMax = yMin + 1;
    }
    const pad = (yMax - yMin) * 0.08;
    yMin -= pad;
    yMax += pad;
    const toCY = (y: number) => {
      const t = (y - yMin) / (yMax - yMin);
      const cy = canvasBottom - t * (canvasBottom - canvasTop);
      return Math.max(canvasTop, Math.min(canvasBottom, cy));
    };
    const k = (yMax - yMin) / 4;
    const ticks = [0, 1, 2, 3, 4].map((i) => yMin + i * k);
    return { toCY, ticks, yMin, yMax };
  }, [yTrue, currentFit, currentForecasts]);

  // Per-forecast breakdown table
  const breakdown = useMemo(() => {
    if (!yTrue.length || !currentForecasts.length) return null as null | any;
    const n = yTrue.length;
    const S = step;
    const j = Math.max(0, Math.min(inspectIdx, currentForecasts.length - 1));

    let lastLag = yTrue[n - 1];
    for (let t = 0; t < j; t++) {
      const xfTmp = n + t;
      let predTmp = mu;
      let tmpLag = lastLag;
      for (let k = 0; k <= S; k++) {
        const leaf = predictLeaf(models[k], xfTmp, tmpLag);
        predTmp += eta * leaf;
        tmpLag = predTmp;
      }
      lastLag = predTmp;
    }

    const xf = n + j;
    let cum = mu;
    let tmpLag = lastLag;
    const rows: { round: number; feature: string; thr: number; leaf: number; contrib: number; total: number }[] = [];
    for (let k = 0; k <= S; k++) {
      const model = models[k];
      const leaf = predictLeaf(model, xf, tmpLag);
      const contrib = eta * leaf;
      cum += contrib;
      rows.push({
        round: k + 1,
        feature: (model as any).type === "stump" ? (model as any).feature : (model as any).root.feature + "(root)",
        thr: (model as any).type === "stump" ? (model as any).thr : (model as any).root.thr,
        leaf,
        contrib,
        total: cum,
      });
      tmpLag = cum;
    }
    return { j, rows, total: cum, lagUsed: lastLag };
  }, [yTrue, models, mu, eta, step, inspectIdx, currentForecasts.length]);

  // Test cases
  const [testReport, setTestReport] = useState<string>("");
  useEffect(() => {
    try {
      const series = [10, 10, 10, 10, 10];
      const n = series.length;
      const xIdx = Array.from({ length: n }, (_, i) => i);
      const mu0 = series.reduce((a, b) => a + b, 0) / n;
      const lag1 = series.map((_, i) => (i === 0 ? mu0 : series[i - 1]));
      const stump = trainStump(series.map((v) => v - mu0), xIdx, lag1, { idxPenalty: 1.25, minGainFrac: 0.01, forbidIdx: true });
      if (!stump) throw new Error("stump null");
      const pv = predictLeaf(stump as any, n, series[n - 1]);
      if (!Number.isFinite(pv)) throw new Error("predictLeaf not finite");

      const series2 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
      const n2 = series2.length;
      const xIdx2 = Array.from({ length: n2 }, (_, i) => i);
      const mu2 = series2.reduce((a, b) => a + b, 0) / n2;
      const lag2 = series2.map((_, i) => (i === 0 ? mu2 : series2[i - 1]));
      const res2 = series2.map((v) => v - mu2);
      const model2 = trainTreeDepth2(res2, xIdx2, lag2, { idxPenalty: 1.25, minGainFrac: 0.01, forbidIdx: true });
      let lastLag2 = series2[n2 - 1];
      const fut2: number[] = [];
      for (let j = 0; j < 4; j++) {
        let pred = mu2;
        let tmpLag = lastLag2;
        for (let k = 0; k <= 0; k++) {
          const leaf = predictLeaf(model2 as any, n2 + j, tmpLag);
          pred += 0.4 * leaf;
          tmpLag = pred;
        }
        fut2.push(pred);
        lastLag2 = pred;
      }
      const allEqual = fut2.every((v) => Math.abs(v - fut2[0]) < 1e-9);
      if (allEqual) throw new Error("trend test failed");

      const big = [1_200_000, 1_201_000, 1_199_500, 1_205_000];
      const n3 = big.length;
      const mu3 = big.reduce((a,b)=>a+b,0)/n3;
      const lag3 = big.map((_,i)=> i===0? mu3: big[i-1]);
      const stump3 = trainStump(big.map(v=> v-mu3), Array.from({length:n3},(_,i)=>i), lag3, { idxPenalty: 1.1, minGainFrac: 0, forbidIdx: false });
      if (!stump3) throw new Error("big numbers stump failed");

      const d3 = trainTreeDepth3(res2, xIdx2, lag2, { idxPenalty: 1.25, minGainFrac: 0.01, forbidIdx: true });
      const leaf3 = predictLeaf(d3 as any, n2, series2[n2-1]);
      if (!Number.isFinite(leaf3)) throw new Error("depth3 predict not finite");

      setTestReport("✓ tests passed");
    } catch (e: any) {
      setTestReport("TEST FAILURE: " + e?.message);
    }
  }, []);

  return (
    <div className="flex flex-col items-center p-6 space-y-4 text-gray-900 w-full max-w-7xl mx-auto">
      <div className="flex items-center gap-3 mb-2">
        <Zap className="text-blue-600" size={32} />
        <h2 className="text-2xl font-bold text-gray-900">Gradient Boosting Visualization</h2>
      </div>
      <p className="text-sm text-gray-600 text-center max-w-3xl">
        Enter numeric data, tweak <b>η</b>, <b>steps</b>, <b>depth</b>, then step through boosting. Click a green dot to see its calculation table.
      </p>

      {/* Controls */}
      <div className="flex items-center gap-3 flex-wrap justify-center bg-gray-50 p-4 rounded-lg">
        <input
          value={inputData}
          onChange={(e) => setInputData(e.target.value)}
          className="w-96 px-3 py-2 text-center bg-white border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
        />
        <div className="flex items-center gap-2">
          <span className="text-xs text-gray-600 font-medium">η</span>
          <input
            type="number"
            step="0.05"
            min="0.05"
            max="1"
            value={eta}
            onChange={(e) => setEta(Number(e.target.value))}
            className="w-20 px-2 py-2 bg-white border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500"
          />
        </div>
        <div className="flex items-center gap-2">
          <span className="text-xs text-gray-600 font-medium">Steps</span>
          <input
            type="number"
            min="1"
            max="20"
            value={boostSteps}
            onChange={(e) => setBoostSteps(Math.max(1, Math.min(20, Number(e.target.value || 1))))}
            className="w-20 px-2 py-2 bg-white border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500"
          />
        </div>
        <div className="flex items-center gap-2">
          <span className="text-xs text-gray-600 font-medium">Depth</span>
          <input
            type="number"
            min="1"
            max="3"
            value={depth}
            onChange={(e) => setDepth(Math.max(1, Math.min(3, Number(e.target.value || 1))))}
            className="w-20 px-2 py-2 bg-white border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500"
          />
        </div>
        <div className="flex items-center gap-2">
          <span className="text-xs text-gray-600 font-medium">Forecast</span>
          <input
            type="number"
            min="1"
            max="24"
            value={forecastCount}
            onChange={(e) => setForecastCount(Math.max(1, Math.min(24, Number(e.target.value || 1))))}
            className="w-24 px-2 py-2 bg-white border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500"
          />
        </div>
      </div>

      {/* Chart */}
      <div className="overflow-x-auto w-full flex justify-center bg-white rounded-lg shadow-lg p-4">
        <svg width={scrollWidth} height={360} viewBox={`0 0 ${scrollWidth} 360`} style={{ backgroundColor: "#f9fafb", borderRadius: 8 }}>
          <line x1={50} y1={300} x2={scrollWidth - 50} y2={300} stroke="#d1d5db" strokeWidth={1} />
          <line x1={50} y1={300} x2={50} y2={40} stroke="#d1d5db" strokeWidth={1} />
          {yScale.ticks.map((t, i) => (
            <g key={i}>
              <line x1={48} y1={yScale.toCY(t)} x2={scrollWidth - 50} y2={yScale.toCY(t)} stroke="#e5e7eb" />
              <text x={44} y={yScale.toCY(t) + 4} fontSize={10} fill="#6b7280" textAnchor="end">
                {fmtCompact.format(t)}
              </text>
            </g>
          ))}

          {yTrue.map((y, i) => (
            <circle key={`t-${i}`} cx={50 + i * 38} cy={yScale.toCY(y)} r={3.5} fill="#3b82f6" />
          ))}

          {currentFit && (
            <motion.path
              d={`M ${yTrue.map((_, i) => `${50 + i * 38},${yScale.toCY(currentFit[i])}`).join(" L ")}`}
              fill="none"
              stroke="#f59e0b"
              strokeWidth={2.5}
              initial={{ pathLength: 0 }}
              animate={{ pathLength: 1 }}
              transition={{ duration: 0.45 }}
            />
          )}

          {currentForecasts.map((v, idx) => {
            const cx = 50 + (yTrue.length + idx) * 38;
            const cy = yScale.toCY(v);
            const isSelected = inspectIdx === idx;
            return (
              <g key={`p-${idx}`} style={{ cursor: "pointer" }} onClick={() => setInspectIdx(idx)}>
                {/* Larger invisible click target */}
                <circle cx={cx} cy={cy} r={12} fill="transparent" pointerEvents="all" />
                {/* Visible dot */}
                <circle
                  cx={cx}
                  cy={cy}
                  r={isSelected ? 8 : 6}
                  fill="#10b981"
                  stroke={isSelected ? "#ffffff" : "none"}
                  strokeWidth={isSelected ? 2 : 0}
                  pointerEvents="none"
                />
              </g>
            );
          })}
        </svg>
      </div>

      {/* Iteration control */}
      <div className="flex items-center space-x-4">
        <button
          onClick={() => setStep((s) => (s + 1) % Math.max(1, boostSteps))}
          className="px-6 py-2 bg-blue-600 text-white rounded-lg font-medium hover:bg-blue-700 transition-colors"
        >
          Next Step
        </button>
        <span className="font-mono text-sm text-gray-700">Step {Math.min(step + 1, boostSteps)}/{boostSteps}</span>
        <span className="text-xs text-gray-500">{testReport}</span>
      </div>

      {/* Details */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4 w-full">
        <div className="text-sm text-gray-700 space-y-2 p-4 rounded-lg bg-white shadow">
          <div><b>μ (baseline):</b> {fmtFull.format(mu)}</div>
          <div><b>Learning rate (η):</b> <span className="text-blue-600">{eta}</span></div>
          <div><b>Residuals:</b> <span className="text-amber-600">{currentResiduals.map((r) => fmtFull.format(r)).join(", ")}</span></div>
          <div><b>Forecasts:</b> <span className="text-green-600">{currentForecasts.map((v) => fmtFull.format(v)).join(", ")}</span></div>
        </div>

        <div className="p-4 rounded-lg bg-white shadow text-sm text-gray-700">
          <div className="font-semibold mb-2">{depth === 1 ? "Stump" : `Tree (depth ${depth})`} — Step {Math.min(step + 1, boostSteps)}</div>
          {!currentModel ? (
            <div className="text-gray-400">No model</div>
          ) : (
            <div className="space-y-1">
              <div><b>Feature:</b> {(currentModel as any).type === "stump" ? (currentModel as any).feature : (currentModel as any).root.feature}</div>
              <div><b>Threshold:</b> {((currentModel as any).type === "stump" ? (currentModel as any).thr : (currentModel as any).root.thr).toFixed(2)}</div>
              <div><b>SSE:</b> {currentModel.sse?.toFixed(3)}</div>
            </div>
          )}
        </div>
      </div>

      {/* Breakdown table */}
      {breakdown && (
        <div className="w-full p-4 rounded-lg bg-white shadow text-sm">
          <div className="font-semibold mb-2">Forecast Calculation (j = {breakdown.j})</div>
          <div className="overflow-x-auto">
            <table className="w-full text-left">
              <thead>
                <tr className="text-gray-600 border-b">
                  <th className="px-2 py-1">Round</th>
                  <th className="px-2 py-1">Feature / Thr</th>
                  <th className="px-2 py-1">Leaf</th>
                  <th className="px-2 py-1">η × Leaf</th>
                  <th className="px-2 py-1">Total</th>
                </tr>
              </thead>
              <tbody>
                <tr className="border-b">
                  <td className="px-2 py-1">0</td>
                  <td className="px-2 py-1">Baseline</td>
                  <td className="px-2 py-1">—</td>
                  <td className="px-2 py-1">—</td>
                  <td className="px-2 py-1"><b>{fmtFull.format(mu)}</b></td>
                </tr>
                {breakdown.rows.map((r: any) => (
                  <tr key={r.round} className="border-b">
                    <td className="px-2 py-1">{r.round}</td>
                    <td className="px-2 py-1">{r.feature} ≤ {Number(r.thr).toFixed(2)}</td>
                    <td className="px-2 py-1">{fmtFull.format(r.leaf)}</td>
                    <td className="px-2 py-1">{fmtFull.format(eta * r.leaf)}</td>
                    <td className="px-2 py-1"><b>{fmtFull.format(r.total)}</b></td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
}
