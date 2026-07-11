import { useState, useMemo } from 'react';
import { motion } from 'framer-motion';
import { ArrowRight, Zap } from 'lucide-react';

const AttentionVisualizer = () => {
  const [queryIdx, setQueryIdx] = useState(0);
  const [temperature, setTemperature] = useState(1.0);
  const [dModel, setDModel] = useState(4);

  const tokens = ['The', 'cat', 'sat', 'on', 'mat'];

  const embeddings = useMemo(() => {
    const seed = [
      [0.5, 0.2, 0.8, 0.1, 0.6, 0.3, 0.9, 0.4],
      [0.9, 0.1, 0.3, 0.7, 0.4, 0.8, 0.2, 0.6],
      [0.2, 0.6, 0.4, 0.9, 0.7, 0.1, 0.5, 0.3],
      [0.7, 0.4, 0.6, 0.3, 0.2, 0.9, 0.1, 0.8],
      [0.3, 0.8, 0.2, 0.5, 0.9, 0.4, 0.7, 0.1],
    ];
    return seed.map(row => row.slice(0, dModel));
  }, [dModel]);

  const Wq = useMemo(() => {
    const seed = [
      [0.8, 0.2, 0.5, 0.3, 0.7, 0.4, 0.6, 0.1],
      [0.1, 0.9, 0.4, 0.6, 0.3, 0.8, 0.2, 0.5],
      [0.6, 0.3, 0.7, 0.2, 0.5, 0.1, 0.9, 0.4],
      [0.4, 0.5, 0.1, 0.8, 0.2, 0.6, 0.3, 0.7],
      [0.3, 0.6, 0.9, 0.4, 0.8, 0.2, 0.5, 0.1],
      [0.7, 0.1, 0.4, 0.9, 0.6, 0.3, 0.8, 0.2],
      [0.2, 0.8, 0.3, 0.5, 0.1, 0.7, 0.4, 0.6],
      [0.5, 0.4, 0.6, 0.1, 0.9, 0.5, 0.2, 0.8],
    ];
    const subset = seed.slice(0, dModel);
    return subset.map(row => row.slice(0, dModel));
  }, [dModel]);

  const Wk = useMemo(() => {
    const seed = [
      [0.7, 0.4, 0.2, 0.6, 0.5, 0.8, 0.3, 0.1],
      [0.3, 0.8, 0.5, 0.1, 0.6, 0.2, 0.9, 0.4],
      [0.5, 0.2, 0.9, 0.4, 0.1, 0.7, 0.6, 0.3],
      [0.2, 0.6, 0.3, 0.7, 0.4, 0.1, 0.8, 0.5],
      [0.8, 0.1, 0.6, 0.3, 0.9, 0.5, 0.2, 0.7],
      [0.4, 0.9, 0.1, 0.8, 0.3, 0.6, 0.7, 0.2],
      [0.6, 0.3, 0.7, 0.2, 0.8, 0.4, 0.1, 0.9],
      [0.1, 0.7, 0.4, 0.5, 0.2, 0.9, 0.5, 0.6],
    ];
    const subset = seed.slice(0, dModel);
    return subset.map(row => row.slice(0, dModel));
  }, [dModel]);

  const Wv = useMemo(() => {
    const seed = [
      [0.6, 0.5, 0.3, 0.4, 0.8, 0.2, 0.7, 0.1],
      [0.4, 0.7, 0.2, 0.8, 0.1, 0.6, 0.3, 0.9],
      [0.8, 0.1, 0.6, 0.3, 0.4, 0.9, 0.5, 0.2],
      [0.3, 0.4, 0.5, 0.9, 0.2, 0.7, 0.1, 0.6],
      [0.7, 0.2, 0.9, 0.1, 0.6, 0.3, 0.8, 0.4],
      [0.1, 0.8, 0.4, 0.6, 0.3, 0.5, 0.2, 0.7],
      [0.9, 0.3, 0.1, 0.7, 0.5, 0.4, 0.6, 0.8],
      [0.2, 0.6, 0.8, 0.5, 0.7, 0.1, 0.4, 0.3],
    ];
    const subset = seed.slice(0, dModel);
    return subset.map(row => row.slice(0, dModel));
  }, [dModel]);

  const matMul = (a: number[][], b: number[][]) => {
    return a.map(row =>
      b[0].map((_: number, j: number) =>
        row.reduce((sum: number, val: number, k: number) => sum + val * b[k][j], 0)
      )
    );
  };

  const Q = useMemo(() => matMul(embeddings, Wq), [embeddings, Wq]);
  const K = useMemo(() => matMul(embeddings, Wk), [embeddings, Wk]);
  const V = useMemo(() => matMul(embeddings, Wv), [embeddings, Wv]);

  const rawScores = useMemo(() => {
    const q = Q[queryIdx];
    return K.map((k: number[]) => {
      const dotProduct = q.reduce((sum: number, val: number, i: number) => sum + val * k[i], 0);
      return dotProduct;
    });
  }, [Q, K, queryIdx]);

  const scaledScores = useMemo(() => {
    return rawScores.map((score: number) => score / Math.sqrt(dModel));
  }, [rawScores, dModel]);

  const temperatureScores = useMemo(() => {
    return scaledScores.map((score: number) => score / temperature);
  }, [scaledScores, temperature]);

  const attentionWeights = useMemo(() => {
    const maxScore = Math.max(...temperatureScores);
    const expScores = temperatureScores.map((s: number) => Math.exp(s - maxScore));
    const sumExp = expScores.reduce((a: number, b: number) => a + b, 0);
    return expScores.map((e: number) => e / sumExp);
  }, [temperatureScores]);

  const output = useMemo(() => {
    return V[0].map((_: number, i: number) =>
      attentionWeights.reduce((sum: number, weight: number, j: number) => sum + weight * V[j][i], 0)
    );
  }, [V, attentionWeights]);

  const maxWeight = Math.max(...attentionWeights);

  return (
    <div className="w-full h-full flex flex-col items-center bg-gray-900 text-white p-6 space-y-6">

      {/* Header */}
      <motion.h1
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.6 }}
        className="text-3xl font-bold text-cyan-400 text-center"
      >
        Transformer Self-Attention Mechanism
      </motion.h1>

      {/* Formula */}
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ duration: 0.6, delay: 0.2 }}
        className="bg-gray-800 rounded-2xl p-6 border border-cyan-700 max-w-4xl w-full"
      >
        <p className="text-gray-300 mb-4 text-center">
          Scaled dot-product attention computes how much each token should attend to every other token:
        </p>
        <div className="bg-gray-900 p-4 rounded-lg border border-cyan-600 text-center font-mono text-lg text-cyan-300">
          Attention(Q, K, V) = softmax(QK<sup>T</sup> / √d<sub>k</sub>) V
        </div>
      </motion.div>

      {/* Interactive Controls */}
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ duration: 0.6, delay: 0.3 }}
        className="bg-gray-800 rounded-2xl p-6 border border-purple-700 max-w-4xl w-full"
      >
        <h2 className="text-xl font-semibold text-purple-300 mb-4">Interactive Controls</h2>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">

          <div>
            <label className="block text-sm font-semibold mb-2 text-gray-300">
              Query Token: <span className="text-cyan-400 text-lg font-bold">{tokens[queryIdx]}</span>
            </label>
            <input
              type="range"
              min="0"
              max={tokens.length - 1}
              value={queryIdx}
              onChange={(e) => setQueryIdx(parseInt(e.target.value))}
              className="w-full h-2 bg-gray-700 rounded-lg appearance-none cursor-pointer"
            />
            <div className="flex justify-between text-xs text-gray-400 mt-1">
              {tokens.map((t, i) => <span key={i}>{t}</span>)}
            </div>
          </div>

          <div>
            <label className="block text-sm font-semibold mb-2 text-gray-300">
              Dimensions (d<sub>model</sub>): <span className="text-fuchsia-400 text-lg font-bold">{dModel}</span>
            </label>
            <input
              type="range"
              min="2"
              max="8"
              step="1"
              value={dModel}
              onChange={(e) => setDModel(parseInt(e.target.value))}
              className="w-full h-2 bg-gray-700 rounded-lg appearance-none cursor-pointer"
            />
            <div className="flex justify-between text-xs text-gray-400 mt-1">
              <span>2</span>
              <span>4</span>
              <span>6</span>
              <span>8</span>
            </div>
            <p className="text-xs text-gray-500 mt-2">
              Real models: 512, 768, 1024
            </p>
          </div>

          <div>
            <label className="block text-sm font-semibold mb-2 text-gray-300">
              Temperature: <span className="text-emerald-400 text-lg font-bold">{temperature.toFixed(2)}</span>
            </label>
            <input
              type="range"
              min="0.1"
              max="2"
              step="0.1"
              value={temperature}
              onChange={(e) => setTemperature(parseFloat(e.target.value))}
              className="w-full h-2 bg-gray-700 rounded-lg appearance-none cursor-pointer"
            />
            <div className="flex justify-between text-xs text-gray-400 mt-1">
              <span>0.1</span>
              <span>1.0</span>
              <span>2.0</span>
            </div>
            <p className="text-xs text-gray-500 mt-2">
              Lower = sharper, Higher = uniform
            </p>
          </div>

        </div>
      </motion.div>

      {/* Complete Flow */}
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ duration: 0.6, delay: 0.4 }}
        className="bg-gray-800 rounded-2xl p-6 border border-blue-700 max-w-6xl w-full space-y-6"
      >
        <h2 className="text-xl font-semibold text-blue-300 text-center">Complete Attention Flow</h2>

        {/* Step 1: Input Embeddings */}
        <div>
          <div className="text-center mb-3">
            <span className="bg-blue-600 text-white px-4 py-2 rounded-full font-bold text-sm">
              STEP 1: Input Embeddings
            </span>
          </div>
          <div className="flex justify-center gap-2 mb-4 flex-wrap">
            {tokens.map((token, i) => (
              <div key={i} className={`p-3 rounded-lg border-2 transition-all ${
                i === queryIdx
                  ? 'border-cyan-400 bg-gray-700 shadow-lg ring-2 ring-cyan-500'
                  : 'border-gray-600 bg-gray-800/70'
              }`}>
                <div className="font-bold text-center mb-1 text-cyan-300">{token}</div>
                <div className="text-xs font-mono text-gray-400">
                  [{embeddings[i].map((v: number) => v.toFixed(1)).join(', ')}]
                </div>
              </div>
            ))}
          </div>
          <div className="flex justify-center">
            <ArrowRight className="text-cyan-500" size={32} />
          </div>
        </div>

        {/* Step 2: Linear Projections */}
        <div className="bg-gray-900 p-6 rounded-xl border border-purple-600">
          <div className="text-center mb-4">
            <span className="bg-purple-600 text-white px-4 py-2 rounded-full font-bold text-sm">
              STEP 2: Linear Projections (Q, K, V)
            </span>
          </div>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div className="bg-gray-800 p-4 rounded-lg border border-orange-500">
              <div className="text-center font-bold text-orange-400 mb-2">Query (Q)</div>
              <div className="text-xs font-mono bg-gray-900 p-2 rounded text-center mb-2 text-gray-400">
                X × W<sub>Q</sub>
              </div>
              <div className="text-xs text-gray-300 mb-1 text-center font-semibold">
                Q[{queryIdx}] ({tokens[queryIdx]}):
              </div>
              <div className="text-xs font-mono bg-orange-950/30 p-2 rounded border border-orange-700 text-orange-300">
                [{Q[queryIdx].map((v: number) => v.toFixed(2)).join(', ')}]
              </div>
            </div>
            <div className="bg-gray-800 p-4 rounded-lg border border-emerald-500">
              <div className="text-center font-bold text-emerald-400 mb-2">Key (K)</div>
              <div className="text-xs font-mono bg-gray-900 p-2 rounded text-center mb-2 text-gray-400">
                X × W<sub>K</sub>
              </div>
              <div className="text-xs text-gray-300 mb-1 text-center font-semibold">All Keys:</div>
              <div className="space-y-1 max-h-32 overflow-y-auto">
                {K.map((k: number[], i: number) => (
                  <div key={i} className={`text-xs font-mono p-1 rounded ${
                    i === queryIdx ? 'bg-emerald-900 font-bold text-emerald-300' : 'bg-gray-900 text-gray-400'
                  }`}>
                    {tokens[i]}: [{k.map((v: number) => v.toFixed(2)).join(', ')}]
                  </div>
                ))}
              </div>
            </div>
            <div className="bg-gray-800 p-4 rounded-lg border border-blue-500">
              <div className="text-center font-bold text-blue-400 mb-2">Value (V)</div>
              <div className="text-xs font-mono bg-gray-900 p-2 rounded text-center mb-2 text-gray-400">
                X × W<sub>V</sub>
              </div>
              <div className="text-xs text-gray-300 mb-1 text-center font-semibold">All Values:</div>
              <div className="space-y-1 max-h-32 overflow-y-auto">
                {V.map((v: number[], i: number) => (
                  <div key={i} className={`text-xs font-mono p-1 rounded ${
                    i === queryIdx ? 'bg-blue-900 font-bold text-blue-300' : 'bg-gray-900 text-gray-400'
                  }`}>
                    {tokens[i]}: [{v.map((val: number) => val.toFixed(2)).join(', ')}]
                  </div>
                ))}
              </div>
            </div>
          </div>
          <div className="flex justify-center mt-4">
            <ArrowRight className="text-purple-500" size={32} />
          </div>
        </div>

        {/* Step 3: Attention Scores */}
        <div className="bg-gray-900 p-6 rounded-xl border border-yellow-600">
          <div className="text-center mb-4">
            <span className="bg-yellow-600 text-white px-4 py-2 rounded-full font-bold text-sm">
              STEP 3: Attention Scores (Q · K<sup>T</sup>)
            </span>
          </div>
          <div className="bg-gray-800 p-4 rounded-lg">
            <div className="text-sm mb-3 text-center text-gray-300">
              Raw dot product: Query "<span className="font-bold text-cyan-400">{tokens[queryIdx]}</span>" · each Key
            </div>
            <div className="grid grid-cols-5 gap-2">
              {rawScores.map((score: number, i: number) => (
                <div key={i} className="text-center">
                  <div className="text-xs text-gray-400 mb-1">{tokens[i]}</div>
                  <div className={`p-2 rounded font-mono text-sm font-bold ${
                    i === queryIdx
                      ? 'bg-yellow-900/50 border-2 border-yellow-500 text-yellow-300'
                      : 'bg-gray-900 text-gray-300'
                  }`}>
                    {score.toFixed(2)}
                  </div>
                </div>
              ))}
            </div>
          </div>
          <div className="flex justify-center mt-4">
            <ArrowRight className="text-yellow-500" size={32} />
          </div>
        </div>

        {/* Step 4: Scaling */}
        <div className="bg-gray-900 p-6 rounded-xl border border-indigo-600">
          <div className="text-center mb-4">
            <span className="bg-indigo-600 text-white px-4 py-2 rounded-full font-bold text-sm">
              STEP 4: Scale by √d<sub>k</sub> = {Math.sqrt(dModel).toFixed(2)} and Temperature = {temperature.toFixed(2)}
            </span>
          </div>
          <div className="bg-gray-800 p-4 rounded-lg mb-3">
            <div className="text-sm mb-3 text-center text-gray-300 font-semibold">
              Scaled scores = Raw scores / √{dModel}
            </div>
            <div className="grid grid-cols-5 gap-2">
              {scaledScores.map((score: number, i: number) => (
                <div key={i} className="text-center">
                  <div className="text-xs text-gray-400 mb-1">{tokens[i]}</div>
                  <div className={`p-2 rounded font-mono text-sm font-bold ${
                    i === queryIdx
                      ? 'bg-indigo-900/50 border-2 border-indigo-500 text-indigo-300'
                      : 'bg-gray-900 text-gray-300'
                  }`}>
                    {score.toFixed(2)}
                  </div>
                </div>
              ))}
            </div>
          </div>
          <div className="text-center my-2">
            <span className="text-indigo-400 font-bold text-xl">↓</span>
          </div>
          <div className="bg-gray-800 p-4 rounded-lg">
            <div className="text-sm mb-3 text-center text-gray-300 font-semibold">
              Final scores = Scaled scores / {temperature.toFixed(2)}
            </div>
            <div className="grid grid-cols-5 gap-2">
              {temperatureScores.map((score: number, i: number) => (
                <div key={i} className="text-center">
                  <div className="text-xs text-gray-400 mb-1">{tokens[i]}</div>
                  <div className={`p-2 rounded font-mono text-sm font-bold ${
                    i === queryIdx
                      ? 'bg-purple-900/50 border-2 border-purple-500 text-purple-300'
                      : 'bg-gray-900 text-gray-300'
                  }`}>
                    {score.toFixed(2)}
                  </div>
                </div>
              ))}
            </div>
          </div>
          <div className="flex justify-center mt-4">
            <ArrowRight className="text-indigo-500" size={32} />
          </div>
        </div>

        {/* Step 5: Softmax */}
        <div className="bg-gray-900 p-6 rounded-xl border border-emerald-600">
          <div className="text-center mb-4">
            <span className="bg-emerald-600 text-white px-4 py-2 rounded-full font-bold text-sm inline-flex items-center gap-2">
              <Zap size={16} />
              STEP 5: Attention Weights (Softmax)
            </span>
          </div>
          <div className="bg-gray-800 p-4 rounded-lg space-y-2">
            {tokens.map((token, i) => {
              const weight = attentionWeights[i];
              const barWidth = (weight / maxWeight) * 100;
              return (
                <div key={i} className="flex items-center gap-3">
                  <div className="w-16 text-sm font-semibold text-gray-300">{token}</div>
                  <div className="flex-1 bg-gray-700 rounded-full h-8 relative overflow-hidden">
                    <motion.div
                      className="bg-gradient-to-r from-emerald-500 to-emerald-600 h-8 rounded-full flex items-center justify-end pr-2"
                      initial={{ width: 0 }}
                      animate={{ width: `${barWidth}%` }}
                      transition={{ duration: 0.5 }}
                    >
                      <span className="text-white text-xs font-bold">
                        {(weight * 100).toFixed(1)}%
                      </span>
                    </motion.div>
                  </div>
                  <div className="w-20 text-right font-mono text-sm font-bold text-emerald-400">
                    {weight.toFixed(4)}
                  </div>
                </div>
              );
            })}
          </div>
          <div className="flex justify-center mt-4">
            <ArrowRight className="text-emerald-500" size={32} />
          </div>
        </div>

        {/* Step 6: Weighted Sum */}
        <div className="bg-gray-900 p-6 rounded-xl border border-pink-600">
          <div className="text-center mb-4">
            <span className="bg-pink-600 text-white px-4 py-2 rounded-full font-bold text-sm">
              STEP 6: Weighted Sum (Output)
            </span>
          </div>
          <div className="bg-gray-800 p-4 rounded-lg">
            <div className="text-sm mb-3 text-center text-gray-300">
              Output = Σ (weight<sub>i</sub> × value<sub>i</sub>)
            </div>
            <div className="bg-gray-900 p-4 rounded-lg border border-pink-700">
              <div className="font-mono text-center text-lg font-bold text-pink-300">
                [{output.map((v: number) => v.toFixed(3)).join(', ')}]
              </div>
            </div>
            <div className="mt-4 flex justify-center gap-2">
              {output.map((val: number, i: number) => (
                <div key={i} className="flex flex-col items-center">
                  <div className="w-16 bg-gradient-to-t from-pink-500 to-pink-400 rounded-t"
                       style={{ height: `${Math.max(5, Math.abs(val) * 80)}px` }}>
                  </div>
                  <div className="text-xs mt-1 font-semibold text-gray-400">d{i}</div>
                </div>
              ))}
            </div>
          </div>
        </div>

        {/* Final Result */}
        <div className="text-center bg-gradient-to-r from-cyan-900 to-blue-900 text-white p-6 rounded-xl border border-cyan-600">
          <div className="text-xl font-bold mb-2">✓ Final Contextualized Representation</div>
          <div className="text-sm text-cyan-200">
            Token "{tokens[queryIdx]}" now contains information from all relevant tokens
          </div>
        </div>

      </motion.div>

      {/* Key Insights */}
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ duration: 0.6, delay: 0.5 }}
        className="bg-gray-800 rounded-2xl p-6 border border-emerald-700 max-w-4xl w-full"
      >
        <h2 className="text-xl font-semibold text-emerald-300 mb-4">Key Insights</h2>
        <ul className="space-y-2 text-sm text-gray-300">
          <li className="flex items-start">
            <span className="text-emerald-500 mr-2 text-lg">•</span>
            <span>Each token learns to attend to relevant positions in the sequence</span>
          </li>
          <li className="flex items-start">
            <span className="text-emerald-500 mr-2 text-lg">•</span>
            <span>Q, K, V projections allow the model to learn different representations</span>
          </li>
          <li className="flex items-start">
            <span className="text-emerald-500 mr-2 text-lg">•</span>
            <span>Softmax ensures attention weights sum to 1.0</span>
          </li>
          <li className="flex items-start">
            <span className="text-emerald-500 mr-2 text-lg">•</span>
            <span>Higher dimensions allow richer token representations</span>
          </li>
        </ul>
      </motion.div>

      <p className="text-center text-gray-400 text-xs max-w-3xl mt-2">
        Adjust the controls above to see how different parameters affect attention patterns. Temperature controls the sharpness of attention distribution.
      </p>

    </div>
  );
};

export default AttentionVisualizer;
