import { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import { Rocket, Play, RotateCcw } from 'lucide-react';
import { Card, CardHeader, CardTitle, CardContent } from '../components/ui/card';

export const metadata = {
  name: "Example Viz",
  icon: "Rocket"
};

export default function ExampleViz() {
  const [count, setCount] = useState(0);
  const [speed, setSpeed] = useState(5);
  const [autoIncrement, setAutoIncrement] = useState(false);

  // Auto increment example
  useEffect(() => {
    if (autoIncrement) {
      const interval = setInterval(() => {
        setCount(c => c + 1);
      }, 1000 / speed);
      return () => clearInterval(interval);
    }
  }, [autoIncrement, speed]);

  return (
    <div className="max-w-4xl mx-auto p-8">
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-3">
            <Rocket className="text-blue-600" size={32} />
            Example Custom Visualization
          </CardTitle>
        </CardHeader>
        <CardContent>
          <p className="text-gray-600 mb-6">
            This demonstrates the correct way to build custom visualizations using native HTML controls styled with Tailwind CSS.
          </p>

          {/* Animated Counter */}
          <div className="flex flex-col items-center gap-6 mb-8">
            <motion.div
              key={count}
              initial={{ scale: 0, rotate: -180 }}
              animate={{ scale: 1, rotate: 0 }}
              transition={{ type: "spring", stiffness: 260, damping: 20 }}
              className="w-32 h-32 bg-gradient-to-br from-blue-500 to-purple-600 rounded-lg flex items-center justify-center text-white text-4xl font-bold shadow-xl"
            >
              {count}
            </motion.div>

            {/* Buttons - native HTML with Tailwind */}
            <div className="flex gap-3">
              <button
                onClick={() => setCount(count + 1)}
                className="px-6 py-3 bg-blue-600 text-white rounded-lg font-medium hover:bg-blue-700 transition-colors shadow-md flex items-center gap-2"
              >
                <Play size={18} />
                Increment
              </button>

              <button
                onClick={() => setCount(0)}
                className="px-6 py-3 bg-gray-200 text-gray-700 rounded-lg font-medium hover:bg-gray-300 transition-colors flex items-center gap-2"
              >
                <RotateCcw size={18} />
                Reset
              </button>
            </div>
          </div>

          {/* Controls Section */}
          <div className="space-y-4 p-4 bg-gray-50 rounded-lg">
            <h3 className="font-semibold text-gray-900 mb-3">Interactive Controls</h3>

            {/* Slider - native HTML with Tailwind */}
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Speed: {speed}x
              </label>
              <input
                type="range"
                min="1"
                max="10"
                value={speed}
                onChange={(e) => setSpeed(Number(e.target.value))}
                className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer"
              />
            </div>

            {/* Checkbox - native HTML with Tailwind */}
            <label className="flex items-center gap-2 cursor-pointer">
              <input
                type="checkbox"
                checked={autoIncrement}
                onChange={(e) => setAutoIncrement(e.target.checked)}
                className="w-4 h-4 text-blue-600 rounded"
              />
              <span className="text-sm text-gray-700">Auto-increment</span>
            </label>
          </div>

          {/* Info Box */}
          <div className="mt-6 p-4 bg-blue-50 rounded-lg border border-blue-200">
            <p className="text-sm text-blue-900">
              <strong>Tip:</strong> Use native HTML elements (&lt;button&gt;, &lt;input type="range"&gt;, &lt;select&gt;)
              styled with Tailwind CSS. Don't import slider/button components that don't exist!
            </p>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
