import { useState, useRef, useEffect, useCallback } from 'react';
import { Brain, Play, RotateCcw, Zap } from 'lucide-react';

export const metadata = {
  name: "Deep Learning",
  icon: "Brain"
};

type Algorithm = 'feedforward' | 'backprop' | 'activation' | 'gradient' | 'dropout';

interface FlowParticle {
  fromLayer: number;
  toLayer: number;
  fromNode: number;
  toNode: number;
  progress: number;
  speed: number;
}

const LAYERS = [3, 5, 4, 2]; // input, hidden1, hidden2, output

const DESCRIPTIONS: Record<Algorithm, string> = {
  feedforward: "A feedforward neural network where information moves in one direction from input to output through hidden layers. <strong>Watch:</strong> Blue glowing particles flow forward along connections, showing how data propagates. Forward arrows (→) indicate information direction. Connection brightness shows signal strength.",
  backprop: "Backpropagation calculates gradients by propagating errors backward through the network to update weights. <strong>Watch:</strong> Red highlighted connections show gradient flow - brighter/thicker lines indicate stronger gradients being backpropagated.",
  activation: "Activation functions introduce non-linearity, allowing networks to learn complex patterns. Common functions include ReLU, Sigmoid, and Tanh. <strong>Watch:</strong> Node colors change from red (low activation) to green (high activation). Compare the three function graphs at the bottom.",
  gradient: "Gradient descent optimizes weights by iteratively moving in the direction that minimizes the loss function. <strong>Watch:</strong> Blue = positive weights, Red = negative weights. Line thickness shows weight magnitude. Bottom graphs track loss reduction and weight changes over time.",
  dropout: "Dropout randomly deactivates neurons during training to prevent overfitting and improve generalization. <strong>Watch:</strong> Gray nodes are temporarily 'dropped out' (deactivated). Each training step randomly selects different neurons to drop."
};

const ALGORITHM_TITLES: Record<Algorithm, string> = {
  feedforward: 'Feedforward Network',
  backprop: 'Backpropagation',
  activation: 'Activation Functions',
  gradient: 'Gradient Descent',
  dropout: 'Dropout Regularization'
};

// Activation functions
const relu = (x: number) => Math.max(0, x);
const sigmoid = (x: number) => 1 / (1 + Math.exp(-x));
const tanh = (x: number) => Math.tanh(x);
const sigmoidDerivative = (y: number) => y * (1 - y);

export default function DeepLearningViz() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const animationRef = useRef<number | undefined>(undefined);

  const [algorithm, setAlgorithm] = useState<Algorithm>('feedforward');
  const [learningRate, setLearningRate] = useState(0.1);
  const [epoch, setEpoch] = useState(0);
  const [loss, setLoss] = useState(1.0);
  const [accuracy, setAccuracy] = useState(0);
  const [autoTraining, setAutoTraining] = useState(false);

  // Network state refs (to avoid re-renders during animation)
  const weightsRef = useRef<number[][][]>([]);
  const activationsRef = useRef<number[][]>([]);
  const gradientsRef = useRef<number[][]>([]);
  const dropoutMaskRef = useRef<number[][]>([]);
  const forwardFlowParticlesRef = useRef<FlowParticle[]>([]);
  const currentTargetRef = useRef([0.9, 0.1]);
  const weightUpdateHistoryRef = useRef<number[]>([]);
  const lossHistoryRef = useRef<number[]>([]);
  const epochRef = useRef(0);
  const lossRef = useRef(1.0);
  const accuracyRef = useRef(0);

  const initNetwork = useCallback(() => {
    const weights: number[][][] = [];
    const activations: number[][] = [];
    const gradients: number[][] = [];
    const dropoutMask: number[][] = [];

    for (let i = 0; i < LAYERS.length; i++) {
      activations[i] = new Array(LAYERS[i]).fill(0).map(() => Math.random());
      dropoutMask[i] = new Array(LAYERS[i]).fill(1);
      gradients[i] = new Array(LAYERS[i]).fill(0);

      if (i < LAYERS.length - 1) {
        weights[i] = [];
        for (let j = 0; j < LAYERS[i]; j++) {
          weights[i][j] = [];
          for (let k = 0; k < LAYERS[i + 1]; k++) {
            weights[i][j][k] = Math.random() * 2 - 1;
          }
        }
      }
    }

    weightsRef.current = weights;
    activationsRef.current = activations;
    gradientsRef.current = gradients;
    dropoutMaskRef.current = dropoutMask;
    forwardFlowParticlesRef.current = [];
    weightUpdateHistoryRef.current = [];
    lossHistoryRef.current = [];
  }, []);

  const forward = useCallback((currentAlgorithm: Algorithm) => {
    const weights = weightsRef.current;
    const activations = activationsRef.current;
    const dropoutMask = dropoutMaskRef.current;

    // Create flow particles for feedforward visualization
    if (currentAlgorithm === 'feedforward') {
      forwardFlowParticlesRef.current = [];
      for (let layer = 0; layer < LAYERS.length - 1; layer++) {
        for (let i = 0; i < Math.min(3, LAYERS[layer]); i++) {
          for (let j = 0; j < Math.min(3, LAYERS[layer + 1]); j++) {
            forwardFlowParticlesRef.current.push({
              fromLayer: layer,
              toLayer: layer + 1,
              fromNode: i,
              toNode: j,
              progress: 0,
              speed: 0.05 + Math.random() * 0.03
            });
          }
        }
      }
    }

    for (let layer = 1; layer < LAYERS.length; layer++) {
      for (let j = 0; j < LAYERS[layer]; j++) {
        let sum = 0;
        for (let i = 0; i < LAYERS[layer - 1]; i++) {
          sum += activations[layer - 1][i] * weights[layer - 1][i][j] * dropoutMask[layer - 1][i];
        }
        activations[layer][j] = sigmoid(sum);
      }
    }
  }, []);

  const backward = useCallback((currentAlgorithm: Algorithm, currentLearningRate: number) => {
    const weights = weightsRef.current;
    const activations = activationsRef.current;
    const gradients = gradientsRef.current;

    // Vary target slightly each iteration
    currentTargetRef.current[0] = 0.7 + Math.random() * 0.3;
    currentTargetRef.current[1] = Math.random() * 0.3;

    // Store old weights for gradient descent visualization
    const oldWeights: number[][][] = [];
    if (currentAlgorithm === 'gradient') {
      for (let layer = 0; layer < weights.length; layer++) {
        oldWeights[layer] = [];
        for (let i = 0; i < weights[layer].length; i++) {
          oldWeights[layer][i] = [...weights[layer][i]];
        }
      }
    }

    // Calculate output layer gradients
    const outputLayer = LAYERS.length - 1;
    for (let i = 0; i < LAYERS[outputLayer]; i++) {
      const error = currentTargetRef.current[i] - activations[outputLayer][i];
      gradients[outputLayer][i] = error * sigmoidDerivative(activations[outputLayer][i]);
    }

    // Backpropagate
    for (let layer = LAYERS.length - 2; layer >= 0; layer--) {
      for (let i = 0; i < LAYERS[layer]; i++) {
        let error = 0;
        for (let j = 0; j < LAYERS[layer + 1]; j++) {
          error += (gradients[layer + 1][j] || 0) * weights[layer][i][j];
        }
        gradients[layer][i] = error * sigmoidDerivative(activations[layer][i]);
      }
    }

    // Update weights with noise for visual variation
    for (let layer = 0; layer < weights.length; layer++) {
      for (let i = 0; i < weights[layer].length; i++) {
        for (let j = 0; j < weights[layer][i].length; j++) {
          const gradient = gradients[layer + 1]?.[j] || 0;
          const update = currentLearningRate * gradient * activations[layer][i];
          weights[layer][i][j] += update + (Math.random() - 0.5) * 0.02;
          weights[layer][i][j] = Math.max(-3, Math.min(3, weights[layer][i][j]));
        }
      }
    }

    // Track weight changes for gradient descent
    if (currentAlgorithm === 'gradient') {
      let totalWeightChange = 0;
      for (let layer = 0; layer < weights.length; layer++) {
        for (let i = 0; i < weights[layer].length; i++) {
          for (let j = 0; j < weights[layer][i].length; j++) {
            totalWeightChange += Math.abs(weights[layer][i][j] - oldWeights[layer][i][j]);
          }
        }
      }
      weightUpdateHistoryRef.current.push(totalWeightChange);
      if (weightUpdateHistoryRef.current.length > 50) weightUpdateHistoryRef.current.shift();

      lossHistoryRef.current.push(lossRef.current);
      if (lossHistoryRef.current.length > 50) lossHistoryRef.current.shift();
    }
  }, []);

  const applyDropout = useCallback(() => {
    const dropoutMask = dropoutMaskRef.current;
    for (let i = 0; i < dropoutMask.length - 1; i++) {
      for (let j = 0; j < dropoutMask[i].length; j++) {
        dropoutMask[i][j] = Math.random() > 0.5 ? 1 : 0;
      }
    }
  }, []);

  const drawArrow = (
    ctx: CanvasRenderingContext2D,
    fromX: number,
    fromY: number,
    toX: number,
    toY: number,
    intensity: number,
    colorPrefix: string
  ) => {
    const headLength = 8;
    const angle = Math.atan2(toY - fromY, toX - fromX);
    const position = colorPrefix.includes('255, 50') ? 0.3 : 0.7;
    const arrowX = fromX + (toX - fromX) * position;
    const arrowY = fromY + (toY - fromY) * position;

    ctx.save();
    ctx.fillStyle = `${colorPrefix}${intensity})`;
    ctx.beginPath();
    ctx.moveTo(arrowX, arrowY);
    ctx.lineTo(
      arrowX - headLength * Math.cos(angle - Math.PI / 6),
      arrowY - headLength * Math.sin(angle - Math.PI / 6)
    );
    ctx.lineTo(
      arrowX - headLength * Math.cos(angle + Math.PI / 6),
      arrowY - headLength * Math.sin(angle + Math.PI / 6)
    );
    ctx.closePath();
    ctx.fill();
    ctx.restore();
  };

  const drawForwardFlowParticles = (
    ctx: CanvasRenderingContext2D,
    nodePositions: { x: number; y: number }[][]
  ) => {
    forwardFlowParticlesRef.current.forEach(particle => {
      const fromPos = nodePositions[particle.fromLayer][particle.fromNode];
      const toPos = nodePositions[particle.toLayer][particle.toNode];

      particle.progress += particle.speed;
      if (particle.progress > 1) {
        particle.progress = 0;
      }

      const x = fromPos.x + (toPos.x - fromPos.x) * particle.progress;
      const y = fromPos.y + (toPos.y - fromPos.y) * particle.progress;

      const gradient = ctx.createRadialGradient(x, y, 0, x, y, 6);
      gradient.addColorStop(0, 'rgba(66, 135, 245, 0.9)');
      gradient.addColorStop(0.5, 'rgba(66, 135, 245, 0.5)');
      gradient.addColorStop(1, 'rgba(66, 135, 245, 0)');

      ctx.fillStyle = gradient;
      ctx.beginPath();
      ctx.arc(x, y, 6, 0, Math.PI * 2);
      ctx.fill();
    });
  };

  const drawActivationFunctions = (ctx: CanvasRenderingContext2D, canvasWidth: number) => {
    const graphY = 480;
    const graphWidth = 140;
    const graphHeight = 80;
    const functions = [
      { name: 'ReLU', fn: relu, color: '#ff6b6b' },
      { name: 'Sigmoid', fn: sigmoid, color: '#4ecdc4' },
      { name: 'Tanh', fn: tanh, color: '#95e1d3' }
    ];

    const totalWidth = functions.length * (graphWidth + 60);
    const startOffset = (canvasWidth - totalWidth) / 2 + 30;

    functions.forEach((func, idx) => {
      const startX = startOffset + idx * (graphWidth + 60);

      ctx.fillStyle = '#333';
      ctx.font = 'bold 12px Arial';
      ctx.textAlign = 'left';
      ctx.fillText(func.name, startX, graphY - 10);

      ctx.strokeStyle = '#ddd';
      ctx.lineWidth = 1;
      ctx.strokeRect(startX, graphY, graphWidth, graphHeight);

      const zeroX = startX + graphWidth / 2;

      ctx.strokeStyle = '#bbb';
      ctx.setLineDash([3, 3]);
      ctx.beginPath();
      ctx.moveTo(zeroX, graphY);
      ctx.lineTo(zeroX, graphY + graphHeight);
      ctx.stroke();

      let horizontalZeroY: number;
      if (func.name === 'Tanh') {
        horizontalZeroY = graphY + graphHeight / 2;
      } else {
        horizontalZeroY = graphY + graphHeight;
      }

      ctx.beginPath();
      ctx.moveTo(startX, horizontalZeroY);
      ctx.lineTo(startX + graphWidth, horizontalZeroY);
      ctx.stroke();
      ctx.setLineDash([]);

      ctx.fillStyle = '#999';
      ctx.font = '10px Arial';
      ctx.textAlign = 'center';
      ctx.fillText('0', zeroX, graphY + graphHeight + 12);

      ctx.beginPath();
      ctx.strokeStyle = func.color;
      ctx.lineWidth = 2;

      for (let x = 0; x <= graphWidth; x++) {
        const inputVal = (x / graphWidth) * 10 - 5;
        const outputVal = func.fn(inputVal);
        let plotY: number;

        if (func.name === 'Tanh') {
          plotY = graphY + graphHeight / 2 - (outputVal * graphHeight / 2);
        } else if (func.name === 'ReLU') {
          const maxOutput = 5;
          plotY = graphY + graphHeight - (Math.min(outputVal, maxOutput) / maxOutput * graphHeight);
        } else {
          plotY = graphY + graphHeight - (outputVal * graphHeight);
        }

        if (x === 0) {
          ctx.moveTo(startX + x, plotY);
        } else {
          ctx.lineTo(startX + x, plotY);
        }
      }
      ctx.stroke();
    });
  };

  const drawErrorVisualization = (
    ctx: CanvasRenderingContext2D,
    canvasWidth: number
  ) => {
    const activations = activationsRef.current;
    const outputLayer = LAYERS.length - 1;
    const startY = 480;
    const boxWidth = 350;
    const boxHeight = 100;
    const startX = (canvasWidth - boxWidth) / 2;

    ctx.fillStyle = 'rgba(255, 255, 255, 0.95)';
    ctx.strokeStyle = '#e74c3c';
    ctx.lineWidth = 2;
    ctx.fillRect(startX, startY, boxWidth, boxHeight);
    ctx.strokeRect(startX, startY, boxWidth, boxHeight);

    ctx.fillStyle = '#e74c3c';
    ctx.font = 'bold 14px Arial';
    ctx.textAlign = 'left';
    ctx.fillText('Error Signal Flow (Backward)', startX + 15, startY + 25);

    ctx.fillStyle = '#333';
    ctx.font = '11px Arial';
    ctx.fillText(`Target: [${currentTargetRef.current[0].toFixed(2)}, ${currentTargetRef.current[1].toFixed(2)}]`, startX + 15, startY + 48);
    ctx.fillText(`Output: [${activations[outputLayer][0].toFixed(2)}, ${activations[outputLayer][1].toFixed(2)}]`, startX + 15, startY + 65);

    let totalError = 0;
    for (let i = 0; i < LAYERS[outputLayer]; i++) {
      const error = currentTargetRef.current[i] - activations[outputLayer][i];
      totalError += error * error;
    }
    totalError = Math.sqrt(totalError);

    ctx.fillStyle = '#e74c3c';
    ctx.font = 'bold 12px Arial';
    ctx.fillText(`Total Error: ${totalError.toFixed(4)}`, startX + 15, startY + 85);

    ctx.fillStyle = '#e74c3c';
    ctx.font = 'bold 16px Arial';
    ctx.textAlign = 'right';
    ctx.fillText('← Gradient Flow', startX + boxWidth - 15, startY + 55);
  };

  const drawFeedforwardVisualization = (
    ctx: CanvasRenderingContext2D,
    canvasWidth: number
  ) => {
    const activations = activationsRef.current;
    const outputLayer = LAYERS.length - 1;
    const startY = 480;
    const boxWidth = 400;
    const boxHeight = 100;
    const startX = (canvasWidth - boxWidth) / 2;

    ctx.fillStyle = 'rgba(255, 255, 255, 0.95)';
    ctx.strokeStyle = '#4285f4';
    ctx.lineWidth = 2;
    ctx.fillRect(startX, startY, boxWidth, boxHeight);
    ctx.strokeRect(startX, startY, boxWidth, boxHeight);

    ctx.fillStyle = '#4285f4';
    ctx.font = 'bold 14px Arial';
    ctx.textAlign = 'left';
    ctx.fillText('Information Flow (Forward)', startX + 15, startY + 25);

    ctx.fillStyle = '#333';
    ctx.font = '11px Arial';
    ctx.fillText('Each neuron computes: σ(Σ(weight × input))', startX + 15, startY + 48);

    const inputStr = activations[0].map(v => v.toFixed(2)).join(', ');
    const outputStr = activations[outputLayer].map(v => v.toFixed(2)).join(', ');

    ctx.fillText(`Input: [${inputStr}]`, startX + 15, startY + 68);
    ctx.fillText(`Output: [${outputStr}]`, startX + 15, startY + 85);

    ctx.fillStyle = '#4285f4';
    ctx.font = 'bold 16px Arial';
    ctx.textAlign = 'right';
    ctx.fillText('Data Flow →', startX + boxWidth - 15, startY + 55);
  };

  const drawGradientDescentVisualization = (
    ctx: CanvasRenderingContext2D,
    canvasWidth: number,
    currentLearningRate: number,
    currentEpoch: number
  ) => {
    const startY = 450;
    const boxWidth = 500;
    const boxHeight = 140;
    const startX = (canvasWidth - boxWidth) / 2;

    ctx.fillStyle = 'rgba(255, 255, 255, 0.95)';
    ctx.strokeStyle = '#ff9800';
    ctx.lineWidth = 2;
    ctx.fillRect(startX, startY, boxWidth, boxHeight);
    ctx.strokeRect(startX, startY, boxWidth, boxHeight);

    ctx.fillStyle = '#ff9800';
    ctx.font = 'bold 14px Arial';
    ctx.textAlign = 'left';
    ctx.fillText('Gradient Descent Optimization', startX + 15, startY + 20);

    const graphX = startX + 15;
    const graphY = startY + 35;
    const graphWidth = 220;
    const graphHeight = 80;

    ctx.strokeStyle = '#ddd';
    ctx.lineWidth = 1;
    ctx.strokeRect(graphX, graphY, graphWidth, graphHeight);

    ctx.fillStyle = '#666';
    ctx.font = '10px Arial';
    ctx.textAlign = 'left';
    ctx.fillText('Loss over time', graphX, graphY - 5);

    const lossHistory = lossHistoryRef.current;
    if (lossHistory.length > 1) {
      ctx.beginPath();
      ctx.strokeStyle = '#ff5722';
      ctx.lineWidth = 2;

      const maxLoss = Math.max(...lossHistory, 0.1);
      lossHistory.forEach((lossVal, idx) => {
        const x = graphX + (idx / Math.max(lossHistory.length - 1, 1)) * graphWidth;
        const y = graphY + graphHeight - (lossVal / maxLoss) * graphHeight;

        if (idx === 0) {
          ctx.moveTo(x, y);
        } else {
          ctx.lineTo(x, y);
        }
      });
      ctx.stroke();

      const lastIdx = lossHistory.length - 1;
      const lastX = graphX + (lastIdx / Math.max(lossHistory.length - 1, 1)) * graphWidth;
      const lastY = graphY + graphHeight - (lossHistory[lastIdx] / maxLoss) * graphHeight;
      ctx.fillStyle = '#ff5722';
      ctx.beginPath();
      ctx.arc(lastX, lastY, 4, 0, Math.PI * 2);
      ctx.fill();
    }

    const graph2X = startX + 255;
    const graph2Y = startY + 35;

    ctx.strokeStyle = '#ddd';
    ctx.lineWidth = 1;
    ctx.strokeRect(graph2X, graph2Y, graphWidth, graphHeight);

    ctx.fillStyle = '#666';
    ctx.font = '10px Arial';
    ctx.fillText('Weight update size', graph2X, graph2Y - 5);

    const weightUpdateHistory = weightUpdateHistoryRef.current;
    if (weightUpdateHistory.length > 1) {
      ctx.beginPath();
      ctx.strokeStyle = '#ffd700';
      ctx.lineWidth = 2;

      const maxUpdate = Math.max(...weightUpdateHistory, 0.01);
      weightUpdateHistory.forEach((update, idx) => {
        const x = graph2X + (idx / Math.max(weightUpdateHistory.length - 1, 1)) * graphWidth;
        const y = graph2Y + graphHeight - (update / maxUpdate) * graphHeight;

        if (idx === 0) {
          ctx.moveTo(x, y);
        } else {
          ctx.lineTo(x, y);
        }
      });
      ctx.stroke();

      const lastIdx = weightUpdateHistory.length - 1;
      const lastX = graph2X + (lastIdx / Math.max(weightUpdateHistory.length - 1, 1)) * graphWidth;
      const lastY = graph2Y + graphHeight - (weightUpdateHistory[lastIdx] / maxUpdate) * graphHeight;
      ctx.fillStyle = '#ffd700';
      ctx.beginPath();
      ctx.arc(lastX, lastY, 4, 0, Math.PI * 2);
      ctx.fill();
    }

    ctx.fillStyle = '#333';
    ctx.font = '11px Arial';
    ctx.textAlign = 'left';
    ctx.fillText(`Learning Rate: ${currentLearningRate.toFixed(3)} | Steps: ${currentEpoch}`, startX + 15, startY + 130);
  };

  const draw = useCallback((currentAlgorithm: Algorithm, currentLearningRate: number, currentEpoch: number) => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const weights = weightsRef.current;
    const activations = activationsRef.current;
    const gradients = gradientsRef.current;
    const dropoutMask = dropoutMaskRef.current;

    ctx.clearRect(0, 0, canvas.width, canvas.height);

    const layerSpacing = canvas.width / (LAYERS.length + 1);
    const nodePositions: { x: number; y: number }[][] = [];

    // Calculate positions
    for (let l = 0; l < LAYERS.length; l++) {
      nodePositions[l] = [];
      const x = layerSpacing * (l + 1);
      const verticalSpacing = canvas.height / (LAYERS[l] + 1);
      for (let n = 0; n < LAYERS[l]; n++) {
        const y = verticalSpacing * (n + 1);
        nodePositions[l][n] = { x, y };
      }
    }

    // Draw connections
    for (let l = 0; l < LAYERS.length - 1; l++) {
      for (let i = 0; i < LAYERS[l]; i++) {
        for (let j = 0; j < LAYERS[l + 1]; j++) {
          const weight = weights[l][i][j];
          const gradient = Math.abs(gradients[l + 1]?.[j] || 0);

          ctx.beginPath();
          ctx.moveTo(nodePositions[l][i].x, nodePositions[l][i].y);
          ctx.lineTo(nodePositions[l + 1][j].x, nodePositions[l + 1][j].y);

          if (currentAlgorithm === 'backprop') {
            const normalizedGradient = Math.min(1, gradient * 3);
            ctx.strokeStyle = `rgba(255, 100, 100, ${Math.max(0.2, normalizedGradient)})`;
            ctx.lineWidth = normalizedGradient * 4 + 0.5;
            ctx.stroke();

            if (normalizedGradient > 0.3) {
              drawArrow(
                ctx,
                nodePositions[l + 1][j].x,
                nodePositions[l + 1][j].y,
                nodePositions[l][i].x,
                nodePositions[l][i].y,
                normalizedGradient,
                'rgba(255, 50, 50, '
              );
            }
          } else if (currentAlgorithm === 'feedforward') {
            const inputActivation = activations[l][i];
            const flowStrength = inputActivation * Math.abs(weight) * 0.5 + 0.2;
            const color = weight > 0 ? '66, 135, 245' : '245, 100, 100';
            ctx.strokeStyle = `rgba(${color}, ${Math.min(1, flowStrength)})`;
            ctx.lineWidth = Math.abs(weight) * 1.5 + 0.5;
            ctx.stroke();

            if (flowStrength > 0.4 && i < 3 && j < 3) {
              drawArrow(
                ctx,
                nodePositions[l][i].x,
                nodePositions[l][i].y,
                nodePositions[l + 1][j].x,
                nodePositions[l + 1][j].y,
                Math.min(1, flowStrength * 1.2),
                'rgba(66, 135, 245, '
              );
            }
          } else if (currentAlgorithm === 'gradient') {
            const color = weight > 0 ? '100, 150, 255' : '255, 100, 100';
            const absWeight = Math.abs(weight);
            const alpha = Math.min(1, absWeight * 0.4 + 0.3);
            ctx.strokeStyle = `rgba(${color}, ${alpha})`;
            ctx.lineWidth = absWeight * 2.5 + 1;
            ctx.stroke();

            if (gradient > 0.2) {
              ctx.strokeStyle = `rgba(255, 215, 0, ${gradient * 0.6})`;
              ctx.lineWidth = absWeight * 3 + 2;
              ctx.stroke();
            }

            if (gradient > 0.3 && i < 2 && j < 2) {
              const midX = (nodePositions[l][i].x + nodePositions[l + 1][j].x) / 2;
              const midY = (nodePositions[l][i].y + nodePositions[l + 1][j].y) / 2;

              ctx.fillStyle = `rgba(255, 215, 0, ${gradient})`;
              ctx.beginPath();
              ctx.arc(midX, midY, 4, 0, Math.PI * 2);
              ctx.fill();

              const updateSign = gradient * ((gradients[l + 1]?.[j] || 0) > 0 ? 1 : -1);
              ctx.fillStyle = '#333';
              ctx.font = 'bold 10px Arial';
              ctx.textAlign = 'center';
              ctx.fillText(updateSign > 0 ? '↑' : '↓', midX, midY + 15);
            }
          } else {
            ctx.strokeStyle = weight > 0 ? 'rgba(100, 150, 255, 0.3)' : 'rgba(255, 100, 100, 0.3)';
            ctx.lineWidth = 1;
            ctx.stroke();
          }
        }
      }
    }

    // Draw flowing particles for feedforward mode
    if (currentAlgorithm === 'feedforward') {
      drawForwardFlowParticles(ctx, nodePositions);
    }

    // Draw nodes
    for (let l = 0; l < LAYERS.length; l++) {
      for (let n = 0; n < LAYERS[l]; n++) {
        const pos = nodePositions[l][n];
        const activation = activations[l][n];
        const isDropped = currentAlgorithm === 'dropout' && dropoutMask[l][n] === 0;
        const nodeGradient = Math.abs(gradients[l]?.[n] || 0);

        ctx.beginPath();
        ctx.arc(pos.x, pos.y, 18, 0, Math.PI * 2);

        if (isDropped) {
          ctx.fillStyle = '#ccc';
          ctx.strokeStyle = '#999';
        } else {
          if (currentAlgorithm === 'activation') {
            const intensity = Math.floor(activation * 255);
            ctx.fillStyle = `rgb(${255 - intensity}, ${intensity}, 150)`;
          } else if (currentAlgorithm === 'backprop') {
            const gradIntensity = Math.min(1, nodeGradient * 4);
            const baseColor = `rgba(102, 126, 234, ${activation * 0.5 + 0.3})`;
            const errorColor = `rgba(255, 100, 100, ${gradIntensity * 0.7})`;
            ctx.fillStyle = gradIntensity > 0.2 ? errorColor : baseColor;

            if (gradIntensity > 0.4) {
              ctx.shadowBlur = 15;
              ctx.shadowColor = 'rgba(255, 100, 100, 0.8)';
            }
          } else {
            ctx.fillStyle = `rgba(102, 126, 234, ${activation})`;
          }
          ctx.strokeStyle = '#667eea';
        }

        ctx.lineWidth = 2;
        ctx.fill();
        ctx.stroke();
        ctx.shadowBlur = 0;

        if (currentAlgorithm === 'activation' && !isDropped) {
          ctx.fillStyle = '#000';
          ctx.font = '10px Arial';
          ctx.textAlign = 'center';
          ctx.fillText(activation.toFixed(2), pos.x, pos.y + 32);
        }

        if (currentAlgorithm === 'backprop' && nodeGradient > 0.1 && !isDropped) {
          ctx.fillStyle = '#d63031';
          ctx.font = 'bold 9px Arial';
          ctx.textAlign = 'center';
          ctx.fillText('∇' + nodeGradient.toFixed(2), pos.x, pos.y + 32);
        }
      }

      // Layer labels
      ctx.fillStyle = '#333';
      ctx.font = 'bold 14px Arial';
      ctx.textAlign = 'center';
      const labels = ['Input', 'Hidden 1', 'Hidden 2', 'Output'];
      ctx.fillText(labels[l], layerSpacing * (l + 1), 30);
    }

    // Draw algorithm-specific visualizations
    if (currentAlgorithm === 'activation') {
      drawActivationFunctions(ctx, canvas.width);
    }

    if (currentAlgorithm === 'backprop') {
      drawErrorVisualization(ctx, canvas.width);
    }

    if (currentAlgorithm === 'feedforward') {
      drawFeedforwardVisualization(ctx, canvas.width);
    }

    if (currentAlgorithm === 'gradient') {
      drawGradientDescentVisualization(ctx, canvas.width, currentLearningRate, currentEpoch);
    }
  }, []);

  const trainStep = useCallback(() => {
    const activations = activationsRef.current;

    // Generate new random input
    for (let i = 0; i < LAYERS[0]; i++) {
      activations[0][i] = Math.random();
    }

    if (algorithm === 'dropout') {
      applyDropout();
    }

    forward(algorithm);

    if (algorithm === 'backprop' || algorithm === 'gradient') {
      backward(algorithm, learningRate);
    } else {
      // For other algorithms, add some weight noise
      const weights = weightsRef.current;
      for (let layer = 0; layer < weights.length; layer++) {
        for (let i = 0; i < weights[layer].length; i++) {
          for (let j = 0; j < weights[layer][i].length; j++) {
            weights[layer][i][j] += (Math.random() - 0.5) * 0.02;
            weights[layer][i][j] = Math.max(-3, Math.min(3, weights[layer][i][j]));
          }
        }
      }
    }

    epochRef.current += 1;
    lossRef.current = Math.max(0.01, lossRef.current * 0.98);
    accuracyRef.current = Math.min(100, accuracyRef.current + 0.5);

    setEpoch(epochRef.current);
    setLoss(lossRef.current);
    setAccuracy(accuracyRef.current);

    draw(algorithm, learningRate, epochRef.current);
  }, [algorithm, learningRate, forward, backward, applyDropout, draw]);

  const handleReset = useCallback(() => {
    epochRef.current = 0;
    lossRef.current = 1.0;
    accuracyRef.current = 0;

    setEpoch(0);
    setLoss(1.0);
    setAccuracy(0);
    setAutoTraining(false);

    initNetwork();
    draw(algorithm, learningRate, 0);
  }, [algorithm, learningRate, initNetwork, draw]);

  const toggleAutoTrain = useCallback(() => {
    setAutoTraining(prev => !prev);
  }, []);

  // Initialize network on mount
  useEffect(() => {
    initNetwork();
  }, [initNetwork]);

  // Animation loop
  useEffect(() => {
    let lastTime = 0;
    const animate = (time: number) => {
      if (autoTraining && time - lastTime > 200) {
        trainStep();
        lastTime = time;
      } else if (!autoTraining) {
        draw(algorithm, learningRate, epochRef.current);
      }
      animationRef.current = requestAnimationFrame(animate);
    };

    animationRef.current = requestAnimationFrame(animate);

    return () => {
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current);
      }
    };
  }, [algorithm, learningRate, autoTraining, trainStep, draw]);

  // Redraw when algorithm changes
  useEffect(() => {
    draw(algorithm, learningRate, epochRef.current);
  }, [algorithm, learningRate, draw]);

  return (
    <div className="min-h-screen bg-gradient-to-br from-indigo-500 to-purple-600 p-5">
      <div className="max-w-6xl mx-auto">
        <h1 className="text-center text-white text-4xl font-bold mb-8 drop-shadow-lg flex items-center justify-center gap-3">
          <Brain className="w-10 h-10" />
          Artificial Neural Network Visualizations
        </h1>

        {/* Controls */}
        <div className="bg-white rounded-xl p-5 mb-5 shadow-lg">
          <div className="flex flex-wrap gap-4 items-center mb-4">
            <label className="font-semibold text-gray-700">Algorithm:</label>
            <select
              value={algorithm}
              onChange={(e) => setAlgorithm(e.target.value as Algorithm)}
              className="px-4 py-2 rounded-md border-2 border-indigo-500 text-sm cursor-pointer transition-all hover:border-indigo-600 focus:outline-none focus:ring-2 focus:ring-indigo-300"
            >
              <option value="feedforward">Feedforward Network</option>
              <option value="backprop">Backpropagation</option>
              <option value="activation">Activation Functions</option>
              <option value="gradient">Gradient Descent</option>
              <option value="dropout">Dropout Regularization</option>
            </select>

            <label className="font-semibold text-gray-700">Learning Rate:</label>
            <input
              type="range"
              min="0.01"
              max="1"
              step="0.01"
              value={learningRate}
              onChange={(e) => setLearningRate(parseFloat(e.target.value))}
              className="w-32 cursor-pointer"
            />
            <span className="text-gray-600 font-mono">{learningRate.toFixed(2)}</span>

            <button
              onClick={trainStep}
              className="px-4 py-2 bg-indigo-500 text-white rounded-md font-semibold hover:bg-indigo-600 hover:-translate-y-0.5 hover:shadow-lg transition-all flex items-center gap-2"
            >
              <Play size={16} />
              Train Step
            </button>

            <button
              onClick={handleReset}
              className="px-4 py-2 bg-gray-200 text-gray-700 rounded-md font-semibold hover:bg-gray-300 hover:-translate-y-0.5 hover:shadow-lg transition-all flex items-center gap-2"
            >
              <RotateCcw size={16} />
              Reset
            </button>

            <button
              onClick={toggleAutoTrain}
              className={`px-4 py-2 rounded-md font-semibold hover:-translate-y-0.5 hover:shadow-lg transition-all flex items-center gap-2 ${
                autoTraining
                  ? 'bg-red-500 text-white hover:bg-red-600'
                  : 'bg-indigo-500 text-white hover:bg-indigo-600'
              }`}
            >
              <Zap size={16} />
              {autoTraining ? 'Stop Training' : 'Auto Train'}
            </button>
          </div>
        </div>

        {/* Canvas */}
        <div className="bg-white rounded-xl p-5 mb-5 shadow-lg">
          <canvas
            ref={canvasRef}
            width={1100}
            height={600}
            className="block mx-auto border border-gray-200 rounded-md"
          />
        </div>

        {/* Info Panel */}
        <div className="bg-white rounded-xl p-5 shadow-lg">
          <h3 className="text-indigo-500 text-xl font-bold mb-3">
            {ALGORITHM_TITLES[algorithm]}
          </h3>
          <p
            className="text-gray-600 leading-relaxed"
            dangerouslySetInnerHTML={{ __html: DESCRIPTIONS[algorithm] }}
          />

          {/* Metrics */}
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mt-4">
            <div className="bg-gray-50 p-4 rounded-md border-l-4 border-indigo-500">
              <div className="text-xs text-gray-500 mb-1">Epoch</div>
              <div className="text-2xl font-bold text-gray-800">{epoch}</div>
            </div>
            <div className="bg-gray-50 p-4 rounded-md border-l-4 border-indigo-500">
              <div className="text-xs text-gray-500 mb-1">Loss</div>
              <div className="text-2xl font-bold text-gray-800">{loss.toFixed(3)}</div>
            </div>
            <div className="bg-gray-50 p-4 rounded-md border-l-4 border-indigo-500">
              <div className="text-xs text-gray-500 mb-1">Accuracy</div>
              <div className="text-2xl font-bold text-gray-800">{accuracy.toFixed(1)}%</div>
            </div>
            <div className="bg-gray-50 p-4 rounded-md border-l-4 border-indigo-500">
              <div className="text-xs text-gray-500 mb-1">Learning Rate</div>
              <div className="text-2xl font-bold text-gray-800">{learningRate.toFixed(2)}</div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
