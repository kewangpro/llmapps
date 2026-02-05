import { useState, useEffect, useRef, useCallback } from 'react';
import { Play, Pause, RotateCcw, Info, ChevronLeft, ChevronRight, Search } from 'lucide-react';

export const metadata = {
  name: "Similarity Search",
  icon: "Search"
};

type AlgorithmType = 'lsh' | 'hnsw' | 'ivf';

interface Point {
  x: number;
  y: number;
  id: number;
  cluster: number;
}

interface QueryPoint {
  x: number;
  y: number;
}

interface ClusterCenter {
  x: number;
  y: number;
}

interface StepDescription {
  title: string;
  watch: string;
}

interface AlgorithmConfig {
  name: string;
  description: string;
  steps: number;
  stepDescriptions: StepDescription[];
}

interface SearchState {
  visited: Set<number>;
  candidates: Set<number>;
  result: number | null;
  buckets: number[][];
  layers: number[][];
  clusters: ClusterCenter[];
  targetCluster?: number;
}

const algorithms: Record<AlgorithmType, AlgorithmConfig> = {
  lsh: {
    name: 'Locality Sensitive Hashing (LSH)',
    description: 'Uses hash functions to map similar points to the same buckets',
    steps: 8,
    stepDescriptions: [
      { title: 'Initialization', watch: 'Points are scattered randomly in 2D space' },
      { title: 'Hash Bucket Creation', watch: 'Space is divided into 4 hash buckets based on coordinates' },
      { title: 'Bucket 0 Check', watch: 'Algorithm checks first bucket - notice it only examines points in this region' },
      { title: 'Bucket 1 Check', watch: 'Moving to second bucket - skipping all other points' },
      { title: 'Bucket 2 Check', watch: 'Third bucket scan - see how much of the space is ignored' },
      { title: 'Bucket 3 Check', watch: 'Final bucket - all points are now assigned to buckets' },
      { title: 'Query Bucket Selection', watch: 'Query point hashes to a specific bucket - only these candidates matter!' },
      { title: 'Best Match Found', watch: 'Found nearest neighbor by searching just ONE bucket instead of all 100 points' }
    ]
  },
  hnsw: {
    name: 'Hierarchical Navigable Small World (HNSW)',
    description: 'Multi-layer graph structure for efficient approximate search',
    steps: 12,
    stepDescriptions: [
      { title: 'Initialization', watch: 'Points scattered across space, ready for hierarchical search' },
      { title: 'Layer Structure Created', watch: 'Three layers built: sparse top layer, medium middle, dense bottom' },
      { title: 'Top Layer Search', watch: 'Starting at coarse layer - making big jumps to get close quickly' },
      { title: 'Middle Layer Search', watch: 'Refined search in medium granularity layer - notice the blue path' },
      { title: 'Bottom Layer Entry', watch: 'Entering densest layer where most points live' },
      { title: 'Local Navigation 1', watch: 'Following graph edges to nearby neighbors - greedy hill climbing' },
      { title: 'Local Navigation 2', watch: 'Expanding search radius - examining immediate neighborhood' },
      { title: 'Local Navigation 3', watch: 'Graph connections guide the search - only visits connected nodes' },
      { title: 'Local Navigation 4', watch: 'Getting closer to query - graph structure makes this efficient' },
      { title: 'Local Navigation 5', watch: 'Fine-tuning the search in local region' },
      { title: 'Local Navigation 6', watch: 'Final exploration of nearby candidates' },
      { title: 'Best Match Found', watch: 'Found excellent neighbor using graph navigation - avoided distant regions entirely!' }
    ]
  },
  ivf: {
    name: 'Inverted File Index (IVF)',
    description: 'Partitions space into clusters using k-means for faster search',
    steps: 10,
    stepDescriptions: [
      { title: 'Initialization', watch: 'Random points ready for clustering' },
      { title: 'Cluster Centers', watch: '5 cluster centers placed - these are like "region headquarters"' },
      { title: 'Point Assignment', watch: 'Each point assigned to nearest cluster - see the Voronoi partitioning' },
      { title: 'Query Cluster Selection', watch: 'Query point finds its nearest cluster - ONE cluster highlighted in blue' },
      { title: 'Cluster Search 1/5', watch: 'Searching ONLY within the selected cluster - ignoring 4 other clusters!' },
      { title: 'Cluster Search 2/5', watch: 'Continuing within cluster - see how many points are skipped in other regions' },
      { title: 'Cluster Search 3/5', watch: 'More candidates from target cluster - focused search area' },
      { title: 'Cluster Search 4/5', watch: 'Nearly done with cluster - typically 20% of total points' },
      { title: 'Cluster Search 5/5', watch: 'Completed cluster scan - avoided 80% of the dataset!' },
      { title: 'Best Match Found', watch: 'Found nearest neighbor from just ONE cluster - massive efficiency gain!' }
    ]
  }
};

export default function SimilarityViz() {
  const [algorithm, setAlgorithm] = useState<AlgorithmType>('lsh');
  const [isAnimating, setIsAnimating] = useState(false);
  const [step, setStep] = useState(0);
  const [points, setPoints] = useState<Point[]>([]);
  const [queryPoint, setQueryPoint] = useState<QueryPoint | null>(null);
  const [searchState, setSearchState] = useState<SearchState>({
    visited: new Set(),
    candidates: new Set(),
    result: null,
    buckets: [],
    layers: [],
    clusters: []
  });

  const canvasRef = useRef<HTMLCanvasElement>(null);
  const animationRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  const resetSearch = useCallback(() => {
    setStep(0);
    setIsAnimating(false);
    setSearchState({
      visited: new Set(),
      candidates: new Set(),
      result: null,
      buckets: [],
      layers: [],
      clusters: []
    });
  }, []);

  // Initialize points
  useEffect(() => {
    const newPoints: Point[] = [];
    for (let i = 0; i < 100; i++) {
      newPoints.push({
        x: Math.random() * 700 + 50,
        y: Math.random() * 400 + 50,
        id: i,
        cluster: Math.floor(Math.random() * 5)
      });
    }
    setPoints(newPoints);
    setQueryPoint({ x: 400, y: 250 });
    resetSearch();
  }, [algorithm, resetSearch]);

  const updateSearchState = useCallback((currentStep: number) => {
    if (!queryPoint || points.length === 0) return;

    setSearchState(prevState => {
      const newState: SearchState = {
        ...prevState,
        visited: new Set(prevState.visited),
        candidates: new Set(prevState.candidates)
      };

      if (algorithm === 'lsh') {
        if (currentStep === 1) {
          const buckets: number[][] = Array(4).fill(null).map(() => []);
          points.forEach(p => {
            const hash = Math.floor((p.x + p.y) / 200) % 4;
            buckets[hash].push(p.id);
          });
          newState.buckets = buckets;
        } else if (currentStep >= 2 && currentStep <= 5) {
          const bucketIdx = currentStep - 2;
          if (newState.buckets[bucketIdx]) {
            newState.buckets[bucketIdx].forEach(id => newState.visited.add(id));
          }
        } else if (currentStep === 6) {
          const queryHash = Math.floor((queryPoint.x + queryPoint.y) / 200) % 4;
          if (newState.buckets[queryHash]) {
            newState.buckets[queryHash].forEach(id => newState.candidates.add(id));
          }
        } else if (currentStep === 7) {
          let bestDist = Infinity;
          let best: number | null = null;
          newState.candidates.forEach(id => {
            const p = points[id];
            const dist = Math.sqrt((p.x - queryPoint.x) ** 2 + (p.y - queryPoint.y) ** 2);
            if (dist < bestDist) {
              bestDist = dist;
              best = id;
            }
          });
          newState.result = best;
        }
      } else if (algorithm === 'hnsw') {
        if (currentStep === 1) {
          newState.layers = [
            points.filter((_, i) => i % 10 === 0).map(p => p.id),
            points.filter((_, i) => i % 3 === 0).map(p => p.id),
            points.map(p => p.id)
          ];
        } else if (currentStep >= 2 && currentStep <= 4) {
          const layer = 2 - (currentStep - 2);
          let current = newState.layers[layer]?.[0];
          if (current !== undefined) {
            newState.visited.add(current);

            let improved = true;
            let iterations = 0;
            while (improved && iterations < 3) {
              improved = false;
              const neighbors = newState.layers[layer].filter(id =>
                Math.abs(points[id].x - points[current!].x) < 150 &&
                Math.abs(points[id].y - points[current!].y) < 150
              );

              const currentDist = Math.sqrt(
                (points[current].x - queryPoint.x) ** 2 +
                (points[current].y - queryPoint.y) ** 2
              );

              for (const neighbor of neighbors) {
                newState.visited.add(neighbor);
                const neighborDist = Math.sqrt(
                  (points[neighbor].x - queryPoint.x) ** 2 +
                  (points[neighbor].y - queryPoint.y) ** 2
                );
                if (neighborDist < currentDist) {
                  current = neighbor;
                  improved = true;
                  break;
                }
              }
              iterations++;
            }
          }
        } else if (currentStep >= 5 && currentStep <= 10) {
          const recentlyVisited = Array.from(newState.visited);
          const lastVisited = recentlyVisited[recentlyVisited.length - 1];
          if (lastVisited !== undefined) {
            const neighbors = points.filter(p =>
              !newState.visited.has(p.id) &&
              Math.sqrt((p.x - points[lastVisited].x) ** 2 + (p.y - points[lastVisited].y) ** 2) < 100
            ).slice(0, 3);

            neighbors.forEach(p => {
              newState.visited.add(p.id);
              newState.candidates.add(p.id);
            });
          }
        } else if (currentStep === 11) {
          let bestDist = Infinity;
          let best: number | null = null;
          newState.candidates.forEach(id => {
            const p = points[id];
            const dist = Math.sqrt((p.x - queryPoint.x) ** 2 + (p.y - queryPoint.y) ** 2);
            if (dist < bestDist) {
              bestDist = dist;
              best = id;
            }
          });
          newState.result = best;
        }
      } else if (algorithm === 'ivf') {
        if (currentStep === 1) {
          const clusterCenters: ClusterCenter[] = [
            { x: 150, y: 150 }, { x: 450, y: 150 },
            { x: 650, y: 150 }, { x: 150, y: 400 },
            { x: 450, y: 400 }
          ];
          newState.clusters = clusterCenters;
        } else if (currentStep === 2) {
          points.forEach(p => {
            let minDist = Infinity;
            let cluster = 0;
            newState.clusters.forEach((c, i) => {
              const dist = Math.sqrt((p.x - c.x) ** 2 + (p.y - c.y) ** 2);
              if (dist < minDist) {
                minDist = dist;
                cluster = i;
              }
            });
            p.cluster = cluster;
          });
        } else if (currentStep === 3) {
          let minDist = Infinity;
          let nearestCluster = 0;
          newState.clusters.forEach((c, i) => {
            const dist = Math.sqrt((queryPoint.x - c.x) ** 2 + (queryPoint.y - c.y) ** 2);
            if (dist < minDist) {
              minDist = dist;
              nearestCluster = i;
            }
          });
          newState.targetCluster = nearestCluster;
        } else if (currentStep >= 4 && currentStep <= 8) {
          const targetPoints = points.filter(p => p.cluster === newState.targetCluster);
          const batchSize = Math.ceil(targetPoints.length / 5);
          const startIdx = (currentStep - 4) * batchSize;
          const batch = targetPoints.slice(startIdx, startIdx + batchSize);

          batch.forEach(p => {
            newState.visited.add(p.id);
            newState.candidates.add(p.id);
          });
        } else if (currentStep === 9) {
          let bestDist = Infinity;
          let best: number | null = null;
          newState.candidates.forEach(id => {
            const p = points[id];
            const dist = Math.sqrt((p.x - queryPoint.x) ** 2 + (p.y - queryPoint.y) ** 2);
            if (dist < bestDist) {
              bestDist = dist;
              best = id;
            }
          });
          newState.result = best;
        }
      }

      return newState;
    });
  }, [algorithm, points, queryPoint]);

  // Animation loop
  useEffect(() => {
    if (isAnimating && step < algorithms[algorithm].steps) {
      animationRef.current = setTimeout(() => {
        setStep(s => s + 1);
        updateSearchState(step + 1);
      }, 1200);
    } else if (step >= algorithms[algorithm].steps) {
      setIsAnimating(false);
    }
    return () => {
      if (animationRef.current) {
        clearTimeout(animationRef.current);
      }
    };
  }, [isAnimating, step, algorithm, updateSearchState]);

  // Draw canvas
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || points.length === 0) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    ctx.clearRect(0, 0, 800, 500);

    // Draw clusters for IVF with enhanced visuals
    if (algorithm === 'ivf' && searchState.clusters && searchState.clusters.length > 0) {
      searchState.clusters.forEach((c, i) => {
        ctx.beginPath();
        ctx.arc(c.x, c.y, 120, 0, 2 * Math.PI);

        if (i === searchState.targetCluster) {
          ctx.strokeStyle = '#3b82f6';
          ctx.lineWidth = 3;
          ctx.setLineDash([]);
          ctx.stroke();

          ctx.strokeStyle = 'rgba(59, 130, 246, 0.2)';
          ctx.lineWidth = 8;
          ctx.stroke();
        } else {
          ctx.strokeStyle = '#e5e7eb';
          ctx.lineWidth = 1;
          ctx.setLineDash([5, 5]);
          ctx.stroke();
        }
        ctx.setLineDash([]);

        ctx.beginPath();
        ctx.arc(c.x, c.y, 6, 0, 2 * Math.PI);
        ctx.fillStyle = i === searchState.targetCluster ? '#3b82f6' : '#9ca3af';
        ctx.fill();

        ctx.fillStyle = '#6b7280';
        ctx.font = '12px sans-serif';
        ctx.fillText(`C${i}`, c.x + 10, c.y - 10);
      });
    }

    // Draw hash buckets for LSH
    if (algorithm === 'lsh' && step >= 2 && step <= 7) {
      for (let i = 0; i < 4; i++) {
        const x = (i % 2) * 400;
        const y = Math.floor(i / 2) * 250;

        ctx.strokeStyle = step >= 2 && step <= 5 && i === step - 2 ? '#3b82f6' : '#e5e7eb';
        ctx.lineWidth = step >= 2 && step <= 5 && i === step - 2 ? 3 : 1;
        ctx.strokeRect(x, y, 400, 250);

        ctx.fillStyle = '#6b7280';
        ctx.font = 'bold 14px sans-serif';
        ctx.fillText(`Bucket ${i}`, x + 10, y + 20);
      }
    }

    // Draw connections for HNSW with gradient
    if (algorithm === 'hnsw' && searchState.visited && searchState.visited.size > 1) {
      const visitedArray = Array.from(searchState.visited);

      for (let i = 1; i < visitedArray.length; i++) {
        const p1 = points[visitedArray[i - 1]];
        const p2 = points[visitedArray[i]];
        if (p1 && p2) {
          const gradient = ctx.createLinearGradient(p1.x, p1.y, p2.x, p2.y);
          gradient.addColorStop(0, '#93c5fd');
          gradient.addColorStop(1, '#3b82f6');

          ctx.strokeStyle = gradient;
          ctx.lineWidth = 2;
          ctx.beginPath();
          ctx.moveTo(p1.x, p1.y);
          ctx.lineTo(p2.x, p2.y);
          ctx.stroke();

          const midX = (p1.x + p2.x) / 2;
          const midY = (p1.y + p2.y) / 2;
          const angle = Math.atan2(p2.y - p1.y, p2.x - p1.x);

          ctx.fillStyle = '#3b82f6';
          ctx.beginPath();
          ctx.moveTo(midX, midY);
          ctx.lineTo(midX - 8 * Math.cos(angle - Math.PI / 6), midY - 8 * Math.sin(angle - Math.PI / 6));
          ctx.lineTo(midX - 8 * Math.cos(angle + Math.PI / 6), midY - 8 * Math.sin(angle + Math.PI / 6));
          ctx.closePath();
          ctx.fill();
        }
      }
    }

    // Draw points with size variation
    points.forEach(p => {
      let pointSize = 4;
      let strokeWidth = 0;

      ctx.beginPath();

      if (searchState.result === p.id) {
        ctx.fillStyle = '#22c55e';
        pointSize = 10;
        strokeWidth = 3;
        ctx.strokeStyle = '#fff';
      } else if (searchState.candidates && searchState.candidates.has(p.id)) {
        ctx.fillStyle = '#fbbf24';
        pointSize = 5;
      } else if (searchState.visited && searchState.visited.has(p.id)) {
        ctx.fillStyle = '#93c5fd';
        pointSize = 5;
      } else {
        ctx.fillStyle = '#cbd5e1';
      }

      ctx.arc(p.x, p.y, pointSize, 0, 2 * Math.PI);
      ctx.fill();

      if (strokeWidth > 0) {
        ctx.lineWidth = strokeWidth;
        ctx.stroke();
      }
    });

    // Draw query point with pulse effect
    if (queryPoint) {
      if (step > 0) {
        ctx.beginPath();
        ctx.arc(queryPoint.x, queryPoint.y, 15 + (step % 3) * 3, 0, 2 * Math.PI);
        ctx.strokeStyle = 'rgba(239, 68, 68, 0.3)';
        ctx.lineWidth = 2;
        ctx.stroke();
      }

      ctx.beginPath();
      ctx.arc(queryPoint.x, queryPoint.y, 10, 0, 2 * Math.PI);
      ctx.fillStyle = '#ef4444';
      ctx.fill();
      ctx.strokeStyle = '#fff';
      ctx.lineWidth = 3;
      ctx.stroke();

      ctx.fillStyle = '#1f2937';
      ctx.font = 'bold 12px sans-serif';
      ctx.fillText('QUERY', queryPoint.x - 22, queryPoint.y - 18);
    }

    // Draw result connection line
    if (searchState.result !== null && queryPoint) {
      const resultPoint = points[searchState.result];
      if (resultPoint) {
        ctx.setLineDash([5, 5]);
        ctx.strokeStyle = '#22c55e';
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.moveTo(queryPoint.x, queryPoint.y);
        ctx.lineTo(resultPoint.x, resultPoint.y);
        ctx.stroke();
        ctx.setLineDash([]);

        const distance = Math.sqrt(
          (resultPoint.x - queryPoint.x) ** 2 +
          (resultPoint.y - queryPoint.y) ** 2
        );
        const midX = (queryPoint.x + resultPoint.x) / 2;
        const midY = (queryPoint.y + resultPoint.y) / 2;

        ctx.fillStyle = '#22c55e';
        ctx.font = 'bold 11px sans-serif';
        ctx.fillText(`d=${distance.toFixed(0)}`, midX + 5, midY - 5);
      }
    }
  }, [points, queryPoint, searchState, algorithm, step]);

  const handleStepBack = () => {
    if (step > 0) {
      setStep(step - 1);
      updateSearchState(step - 1);
    }
  };

  const handleStepForward = () => {
    if (step < algorithms[algorithm].steps) {
      setStep(step + 1);
      updateSearchState(step + 1);
    }
  };

  return (
    <div className="w-full max-w-5xl mx-auto p-6 bg-white rounded-lg shadow-lg">
      <div className="mb-6">
        <div className="flex items-center gap-3 mb-2">
          <Search className="w-8 h-8 text-blue-600" />
          <h2 className="text-3xl font-bold text-gray-800">
            Approximate Nearest Neighbor Algorithms
          </h2>
        </div>
        <p className="text-gray-600">
          Visualizing how different ANN algorithms find similar points efficiently
        </p>
      </div>

      {/* Algorithm selector */}
      <div className="grid grid-cols-3 gap-4 mb-6">
        {(Object.entries(algorithms) as [AlgorithmType, AlgorithmConfig][]).map(([key, algo]) => (
          <button
            key={key}
            onClick={() => {
              setAlgorithm(key);
              resetSearch();
            }}
            className={`p-4 rounded-lg border-2 transition-all ${
              algorithm === key
                ? 'border-blue-500 bg-blue-50'
                : 'border-gray-200 hover:border-gray-300'
            }`}
          >
            <div className="font-semibold text-sm mb-1">{algo.name}</div>
            <div className="text-xs text-gray-600">{algo.description}</div>
          </button>
        ))}
      </div>

      {/* What to Watch Panel */}
      <div className="bg-gradient-to-r from-purple-50 to-blue-50 rounded-lg p-5 mb-6 border-2 border-purple-200">
        <div className="flex items-start gap-3">
          <div className="bg-purple-500 text-white rounded-full p-2 mt-1">
            <Info size={20} />
          </div>
          <div className="flex-1">
            <div className="text-lg font-bold text-gray-800 mb-2">
              {step === 0 ? 'Ready to Start' : algorithms[algorithm].stepDescriptions[step - 1]?.title || 'Search Complete'}
            </div>
            <div className="text-sm text-gray-700 leading-relaxed">
              {step === 0 ? (
                <span>Click <strong>Play</strong> to see how <strong>{algorithms[algorithm].name}</strong> finds the nearest neighbor without checking all 100 points. Watch the point colors and counts change as the algorithm searches.</span>
              ) : (
                <span className="font-medium">{algorithms[algorithm].stepDescriptions[step - 1]?.watch || 'Search complete!'}</span>
              )}
            </div>
            {step > 0 && (
              <div className="mt-2 flex items-center gap-4 text-xs">
                <span className="px-2 py-1 bg-white rounded-md border border-purple-200">
                  Step {step} of {algorithms[algorithm].steps}
                </span>
                {searchState.visited && searchState.visited.size > 0 && (
                  <span className="px-2 py-1 bg-blue-100 text-blue-700 rounded-md font-semibold">
                    {searchState.visited.size} points examined
                  </span>
                )}
                {searchState.candidates && searchState.candidates.size > 0 && (
                  <span className="px-2 py-1 bg-yellow-100 text-yellow-700 rounded-md font-semibold">
                    {searchState.candidates.size} candidates
                  </span>
                )}
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Canvas */}
      <div className="bg-gray-50 rounded-lg p-4 mb-6">
        <canvas
          ref={canvasRef}
          width={800}
          height={500}
          className="w-full border border-gray-200 rounded"
        />
      </div>

      {/* Controls */}
      <div className="flex items-center justify-between mb-6">
        <div className="flex gap-2">
          <button
            onClick={() => setIsAnimating(!isAnimating)}
            disabled={step >= algorithms[algorithm].steps}
            className="px-4 py-2 bg-blue-500 text-white rounded-lg hover:bg-blue-600 disabled:bg-gray-300 disabled:cursor-not-allowed flex items-center gap-2"
          >
            {isAnimating ? <Pause size={16} /> : <Play size={16} />}
            {isAnimating ? 'Pause' : 'Play'}
          </button>
          <button
            onClick={handleStepBack}
            disabled={step === 0}
            className="px-3 py-2 bg-gray-200 text-gray-700 rounded-lg hover:bg-gray-300 disabled:bg-gray-100 disabled:text-gray-400 disabled:cursor-not-allowed"
          >
            <ChevronLeft size={16} />
          </button>
          <button
            onClick={handleStepForward}
            disabled={step >= algorithms[algorithm].steps}
            className="px-3 py-2 bg-gray-200 text-gray-700 rounded-lg hover:bg-gray-300 disabled:bg-gray-100 disabled:text-gray-400 disabled:cursor-not-allowed"
          >
            <ChevronRight size={16} />
          </button>
          <button
            onClick={resetSearch}
            className="px-4 py-2 bg-gray-200 text-gray-700 rounded-lg hover:bg-gray-300 flex items-center gap-2"
          >
            <RotateCcw size={16} />
            Reset
          </button>
        </div>
        <div className="text-sm text-gray-600 font-medium">
          Step {step} / {algorithms[algorithm].steps}
        </div>
      </div>

      {/* Legend */}
      <div className="bg-blue-50 rounded-lg p-4 border border-blue-200">
        <div className="flex items-start gap-2 mb-3">
          <Info size={20} className="text-blue-600 mt-0.5" />
          <div className="flex-1">
            <div className="font-semibold text-gray-800 mb-3">Legend</div>
            <div className="grid grid-cols-2 gap-3 text-sm">
              <div className="flex items-center gap-2">
                <div className="relative">
                  <div className="w-4 h-4 bg-red-500 rounded-full"></div>
                  {step > 0 && (
                    <div className="absolute inset-0 bg-red-500 rounded-full animate-ping opacity-75"></div>
                  )}
                </div>
                <span className="font-medium">Query Point (what we're searching for)</span>
              </div>
              <div className="flex items-center gap-2">
                <div className="w-4 h-4 bg-green-500 rounded-full"></div>
                <span className={searchState.result !== null ? 'font-bold text-green-700' : ''}>
                  Result (Nearest Neighbor Found!)
                </span>
              </div>
              <div className="flex items-center gap-2">
                <div className="w-4 h-4 bg-blue-300 rounded-full"></div>
                <span className={searchState.visited && searchState.visited.size > 0 ? 'font-medium' : ''}>
                  Visited Points (examined by algorithm)
                </span>
              </div>
              <div className="flex items-center gap-2">
                <div className="w-4 h-4 bg-yellow-400 rounded-full"></div>
                <span className={searchState.candidates && searchState.candidates.size > 0 ? 'font-medium' : ''}>
                  Candidate Points (potential matches)
                </span>
              </div>
              <div className="flex items-center gap-2">
                <div className="w-4 h-4 bg-gray-300 rounded-full"></div>
                <span className="text-gray-600">Unvisited Points (ignored/skipped)</span>
              </div>
              {algorithm === 'ivf' && searchState.clusters && searchState.clusters.length > 0 && (
                <div className="flex items-center gap-2">
                  <div className="w-4 h-4 border-2 border-blue-500 rounded-sm"></div>
                  <span className="font-medium text-blue-700">Selected Cluster Region</span>
                </div>
              )}
            </div>
          </div>
        </div>
        <div className="text-xs text-gray-700 mt-3 border-t border-blue-200 pt-3 bg-white rounded px-2 py-2">
          <strong>Key Insight:</strong> {algorithms[algorithm].description}.
          {' '}The <strong className="text-gray-300">gray points</strong> are never examined—this is where the speedup comes from!
        </div>
      </div>

      {/* Stats */}
      <div className="mt-4 grid grid-cols-4 gap-4">
        <div className="bg-gray-50 p-3 rounded-lg border border-gray-200">
          <div className="text-xs text-gray-600 mb-1">Points Visited</div>
          <div className="text-2xl font-bold text-blue-600">
            {searchState.visited ? searchState.visited.size : 0}
          </div>
          <div className="text-xs text-gray-500 mt-1">out of 100 total</div>
        </div>
        <div className="bg-gray-50 p-3 rounded-lg border border-gray-200">
          <div className="text-xs text-gray-600 mb-1">Candidates</div>
          <div className="text-2xl font-bold text-yellow-600">
            {searchState.candidates ? searchState.candidates.size : 0}
          </div>
          <div className="text-xs text-gray-500 mt-1">under consideration</div>
        </div>
        <div className="bg-gradient-to-br from-green-50 to-emerald-50 p-3 rounded-lg border-2 border-green-300">
          <div className="text-xs text-gray-700 mb-1 font-semibold">Speedup vs Brute Force</div>
          <div className="text-2xl font-bold text-green-600">
            {searchState.visited && searchState.visited.size > 0
              ? (100 / searchState.visited.size).toFixed(1) + 'x'
              : '—'}
          </div>
          <div className="text-xs text-gray-600 mt-1">faster than linear scan</div>
        </div>
        <div className="bg-gradient-to-br from-purple-50 to-pink-50 p-3 rounded-lg border-2 border-purple-300">
          <div className="text-xs text-gray-700 mb-1 font-semibold">Points Skipped</div>
          <div className="text-2xl font-bold text-purple-600">
            {searchState.visited ? 100 - searchState.visited.size : 100}
          </div>
          <div className="text-xs text-gray-600 mt-1">
            {searchState.visited ? Math.round((1 - searchState.visited.size / points.length) * 100) : 0}% of dataset
          </div>
        </div>
      </div>
    </div>
  );
}
