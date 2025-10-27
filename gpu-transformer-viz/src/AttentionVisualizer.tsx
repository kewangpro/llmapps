import React, { useState, useMemo } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from './components/ui/card';
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
  
  const matMul = (a, b) => {
    return a.map(row => 
      b[0].map((_, j) => 
        row.reduce((sum, val, k) => sum + val * b[k][j], 0)
      )
    );
  };
  
  const Q = useMemo(() => matMul(embeddings, Wq), [embeddings, Wq]);
  const K = useMemo(() => matMul(embeddings, Wk), [embeddings, Wk]);
  const V = useMemo(() => matMul(embeddings, Wv), [embeddings, Wv]);
  
  const scores = useMemo(() => {
    const q = Q[queryIdx];
    return K.map(k => {
      const dotProduct = q.reduce((sum, val, i) => sum + val * k[i], 0);
      const scaled = dotProduct / Math.sqrt(dModel);
      return scaled / temperature;
    });
  }, [Q, K, queryIdx, dModel, temperature]);
  
  const attentionWeights = useMemo(() => {
    const maxScore = Math.max(...scores);
    const expScores = scores.map(s => Math.exp(s - maxScore));
    const sumExp = expScores.reduce((a, b) => a + b, 0);
    return expScores.map(e => e / sumExp);
  }, [scores]);
  
  const output = useMemo(() => {
    return V[0].map((_, i) => 
      attentionWeights.reduce((sum, weight, j) => sum + weight * V[j][i], 0)
    );
  }, [V, attentionWeights]);
  
  const maxWeight = Math.max(...attentionWeights);
  
  return (
    <div className="w-full max-w-7xl mx-auto p-4 space-y-6 bg-gradient-to-br from-slate-50 to-blue-50 min-h-screen">
      
      <Card className="border-2 border-blue-200 shadow-lg">
        <CardHeader className="bg-gradient-to-r from-blue-500 to-indigo-600 text-white">
          <CardTitle className="text-2xl">Transformer Self-Attention Mechanism</CardTitle>
        </CardHeader>
        <CardContent className="pt-6">
          <p className="text-gray-700 mb-4">
            This visualization shows how attention is calculated in a Transformer model using the scaled dot-product attention formula:
          </p>
          <div className="bg-white p-4 rounded-lg border-2 border-indigo-200 text-center font-mono text-lg mb-4">
            Attention(Q, K, V) = softmax(QK<sup>T</sup> / √d<sub>k</sub>) V
          </div>
        </CardContent>
      </Card>

      <Card className="shadow-lg border-2 border-indigo-300">
        <CardHeader className="bg-gradient-to-r from-indigo-500 to-purple-600 text-white">
          <CardTitle className="text-xl">Interactive Controls</CardTitle>
        </CardHeader>
        <CardContent className="pt-6">
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            
            <div>
              <label className="block text-sm font-semibold mb-2 text-gray-700">
                Query Token: <span className="text-blue-600 text-lg font-bold">{tokens[queryIdx]}</span>
              </label>
              <input
                type="range"
                min="0"
                max={tokens.length - 1}
                value={queryIdx}
                onChange={(e) => setQueryIdx(parseInt(e.target.value))}
                className="w-full"
              />
              <div className="flex justify-between text-xs text-gray-600 mt-1">
                {tokens.map((t, i) => <span key={i}>{t}</span>)}
              </div>
            </div>
            
            <div>
              <label className="block text-sm font-semibold mb-2 text-gray-700">
                Dimensions (d<sub>model</sub>): <span className="text-purple-600 text-lg font-bold">{dModel}</span>
              </label>
              <input
                type="range"
                min="2"
                max="8"
                step="1"
                value={dModel}
                onChange={(e) => setDModel(parseInt(e.target.value))}
                className="w-full"
              />
              <div className="flex justify-between text-xs text-gray-600 mt-1">
                <span>2</span>
                <span>4</span>
                <span>6</span>
                <span>8</span>
              </div>
              <p className="text-xs text-gray-600 mt-2">
                Real models: 512, 768, 1024
              </p>
            </div>
            
            <div>
              <label className="block text-sm font-semibold mb-2 text-gray-700">
                Temperature: <span className="text-green-600 text-lg font-bold">{temperature.toFixed(2)}</span>
              </label>
              <input
                type="range"
                min="0.1"
                max="2"
                step="0.1"
                value={temperature}
                onChange={(e) => setTemperature(parseFloat(e.target.value))}
                className="w-full"
              />
              <div className="flex justify-between text-xs text-gray-600 mt-1">
                <span>0.1</span>
                <span>1.0</span>
                <span>2.0</span>
              </div>
              <p className="text-xs text-gray-600 mt-2">
                Lower = sharper, Higher = uniform
              </p>
            </div>
            
          </div>
        </CardContent>
      </Card>

      <Card className="border-2 border-purple-200 shadow-xl">
        <CardHeader className="bg-gradient-to-r from-purple-500 to-pink-600 text-white">
          <CardTitle className="text-xl">Complete Attention Flow Diagram</CardTitle>
        </CardHeader>
        <CardContent className="pt-6 space-y-6">
          
          <div>
            <div className="text-center mb-3">
              <span className="bg-blue-600 text-white px-4 py-2 rounded-full font-bold text-sm">
                STEP 1: Input Embeddings
              </span>
            </div>
            <div className="flex justify-center gap-2 mb-4 flex-wrap">
              {tokens.map((token, i) => (
                <div key={i} className={`p-3 rounded-lg border-2 transition-all ${
                  i === queryIdx ? 'border-blue-500 bg-blue-100 scale-110 shadow-lg' : 'border-gray-300 bg-white'
                }`}>
                  <div className="font-bold text-center mb-1">{token}</div>
                  <div className="text-xs font-mono text-gray-600">
                    [{embeddings[i].map(v => v.toFixed(1)).join(', ')}]
                  </div>
                </div>
              ))}
            </div>
            <div className="flex justify-center">
              <ArrowRight className="text-blue-500" size={32} />
            </div>
          </div>

          <div className="bg-gradient-to-r from-purple-50 to-pink-50 p-6 rounded-lg border-2 border-purple-200">
            <div className="text-center mb-4">
              <span className="bg-purple-600 text-white px-4 py-2 rounded-full font-bold text-sm">
                STEP 2: Linear Projections
              </span>
            </div>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              <div className="bg-white p-4 rounded-lg border-2 border-orange-300 shadow">
                <div className="text-center font-bold text-orange-600 mb-2">Query (Q)</div>
                <div className="text-xs font-mono bg-orange-50 p-2 rounded text-center mb-2">
                  X × W<sub>Q</sub>
                </div>
                <div className="text-xs text-gray-600 mb-1 text-center font-semibold">
                  Q[{queryIdx}] ({tokens[queryIdx]}):
                </div>
                <div className="text-xs font-mono bg-orange-100 p-2 rounded border border-orange-200">
                  [{Q[queryIdx].map(v => v.toFixed(2)).join(', ')}]
                </div>
              </div>
              <div className="bg-white p-4 rounded-lg border-2 border-green-300 shadow">
                <div className="text-center font-bold text-green-600 mb-2">Key (K)</div>
                <div className="text-xs font-mono bg-green-50 p-2 rounded text-center mb-2">
                  X × W<sub>K</sub>
                </div>
                <div className="text-xs text-gray-600 mb-1 text-center font-semibold">All Keys:</div>
                <div className="space-y-1 max-h-32 overflow-y-auto">
                  {K.map((k, i) => (
                    <div key={i} className={`text-xs font-mono p-1 rounded ${
                      i === queryIdx ? 'bg-green-200 font-bold' : 'bg-green-50'
                    }`}>
                      {tokens[i]}: [{k.map(v => v.toFixed(2)).join(', ')}]
                    </div>
                  ))}
                </div>
              </div>
              <div className="bg-white p-4 rounded-lg border-2 border-blue-300 shadow">
                <div className="text-center font-bold text-blue-600 mb-2">Value (V)</div>
                <div className="text-xs font-mono bg-blue-50 p-2 rounded text-center mb-2">
                  X × W<sub>V</sub>
                </div>
                <div className="text-xs text-gray-600 mb-1 text-center font-semibold">All Values:</div>
                <div className="space-y-1 max-h-32 overflow-y-auto">
                  {V.map((v, i) => (
                    <div key={i} className={`text-xs font-mono p-1 rounded ${
                      i === queryIdx ? 'bg-blue-200 font-bold' : 'bg-blue-50'
                    }`}>
                      {tokens[i]}: [{v.map(val => val.toFixed(2)).join(', ')}]
                    </div>
                  ))}
                </div>
              </div>
            </div>
            <div className="flex justify-center mt-4">
              <ArrowRight className="text-purple-500" size={32} />
            </div>
          </div>

          <div className="bg-gradient-to-r from-yellow-50 to-orange-50 p-6 rounded-lg border-2 border-yellow-300">
            <div className="text-center mb-4">
              <span className="bg-yellow-600 text-white px-4 py-2 rounded-full font-bold text-sm">
                STEP 3: Attention Scores (Q · K<sup>T</sup>)
              </span>
            </div>
            <div className="bg-white p-4 rounded-lg shadow">
              <div className="text-sm mb-3 text-center text-gray-700">
                Query "<span className="font-bold text-blue-600">{tokens[queryIdx]}</span>" attending to:
              </div>
              <div className="grid grid-cols-5 gap-2">
                {scores.map((score, i) => (
                  <div key={i} className="text-center">
                    <div className="text-xs text-gray-600 mb-1">{tokens[i]}</div>
                    <div className={`p-2 rounded font-mono text-sm font-bold ${
                      i === queryIdx ? 'bg-yellow-200 border-2 border-yellow-500' : 'bg-gray-100'
                    }`}>
                      {score.toFixed(2)}
                    </div>
                  </div>
                ))}
              </div>
            </div>
            <div className="flex justify-center mt-4">
              <ArrowRight className="text-yellow-600" size={32} />
            </div>
          </div>

          <div className="bg-gradient-to-r from-indigo-50 to-blue-50 p-4 rounded-lg border-2 border-indigo-300">
            <div className="text-center mb-3">
              <span className="bg-indigo-600 text-white px-4 py-2 rounded-full font-bold text-sm">
                STEP 4: Scale by √d<sub>k</sub> = {Math.sqrt(dModel).toFixed(2)}
              </span>
            </div>
            <div className="flex justify-center mt-3">
              <ArrowRight className="text-indigo-500" size={32} />
            </div>
          </div>

          <div className="bg-gradient-to-r from-green-50 to-emerald-50 p-6 rounded-lg border-2 border-green-300">
            <div className="text-center mb-4">
              <span className="bg-green-600 text-white px-4 py-2 rounded-full font-bold text-sm inline-flex items-center gap-2">
                <Zap size={16} />
                STEP 5: Softmax
              </span>
            </div>
            <div className="bg-white p-4 rounded-lg shadow space-y-2">
              {tokens.map((token, i) => {
                const weight = attentionWeights[i];
                const barWidth = (weight / maxWeight) * 100;
                return (
                  <div key={i} className="flex items-center gap-3">
                    <div className="w-16 text-sm font-semibold">{token}</div>
                    <div className="flex-1 bg-gray-200 rounded-full h-8 relative overflow-hidden">
                      <div
                        className="bg-gradient-to-r from-green-500 to-emerald-600 h-8 rounded-full transition-all duration-500 flex items-center justify-end pr-2"
                        style={{ width: `${barWidth}%` }}
                      >
                        <span className="text-white text-xs font-bold">
                          {(weight * 100).toFixed(1)}%
                        </span>
                      </div>
                    </div>
                    <div className="w-20 text-right font-mono text-sm font-bold text-green-700">
                      {weight.toFixed(4)}
                    </div>
                  </div>
                );
              })}
            </div>
            <div className="flex justify-center mt-4">
              <ArrowRight className="text-green-600" size={32} />
            </div>
          </div>

          <div className="bg-gradient-to-r from-red-50 to-pink-50 p-6 rounded-lg border-2 border-red-300">
            <div className="text-center mb-4">
              <span className="bg-red-600 text-white px-4 py-2 rounded-full font-bold text-sm">
                STEP 6: Weighted Sum
              </span>
            </div>
            <div className="bg-white p-4 rounded-lg shadow">
              <div className="text-sm mb-3 text-center text-gray-700">
                Output = Σ (weight<sub>i</sub> × value<sub>i</sub>)
              </div>
              <div className="bg-gradient-to-r from-red-100 to-pink-100 p-4 rounded-lg border-2 border-red-300">
                <div className="font-mono text-center text-lg font-bold text-red-700">
                  [{output.map(v => v.toFixed(3)).join(', ')}]
                </div>
              </div>
              <div className="mt-4 flex justify-center gap-2">
                {output.map((val, i) => (
                  <div key={i} className="flex flex-col items-center">
                    <div className="w-16 bg-gradient-to-t from-red-500 to-pink-400 rounded-t" 
                         style={{ height: `${Math.max(5, Math.abs(val) * 80)}px` }}>
                    </div>
                    <div className="text-xs mt-1 font-semibold">d{i}</div>
                  </div>
                ))}
              </div>
            </div>
          </div>

          <div className="text-center bg-gradient-to-r from-gray-700 to-gray-900 text-white p-6 rounded-lg shadow-xl">
            <div className="text-xl font-bold mb-2">✓ Final Contextualized Representation</div>
            <div className="text-sm opacity-90">
              Token "{tokens[queryIdx]}" now contains information from all relevant tokens
            </div>
          </div>

        </CardContent>
      </Card>

      <Card className="border-2 border-green-200 shadow-md">
        <CardHeader className="bg-green-100">
          <CardTitle className="text-lg">Key Insights</CardTitle>
        </CardHeader>
        <CardContent className="pt-4">
          <ul className="space-y-2 text-sm text-gray-700">
            <li className="flex items-start">
              <span className="text-green-600 mr-2 text-lg">•</span>
              <span>Each token learns to attend to relevant positions in the sequence</span>
            </li>
            <li className="flex items-start">
              <span className="text-green-600 mr-2 text-lg">•</span>
              <span>Q, K, V projections allow the model to learn different representations</span>
            </li>
            <li className="flex items-start">
              <span className="text-green-600 mr-2 text-lg">•</span>
              <span>Softmax ensures attention weights sum to 1.0</span>
            </li>
            <li className="flex items-start">
              <span className="text-green-600 mr-2 text-lg">•</span>
              <span>Higher dimensions allow richer token representations</span>
            </li>
          </ul>
        </CardContent>
      </Card>

    </div>
  );
};

export default AttentionVisualizer;