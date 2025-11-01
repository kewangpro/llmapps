import { useState } from 'react';
import { motion } from 'framer-motion';
import { ArrowUpDown, Play, RotateCcw } from 'lucide-react';

export const metadata = {
  name: "Bubble Sort",
  icon: "ArrowUpDown"
};

interface Step {
  index1: number;
  index2: number;
  array: number[];
}

export default function BubbleSortViz() {
  const [array, setArray] = useState([5, 2, 9, 1, 5, 6]);
  const [currentStep, setCurrentStep] = useState(0);
  const [steps, setSteps] = useState<Step[]>([]);
  const [isSorting, setIsSorting] = useState(false);
  const [isSorted, setIsSorted] = useState(false);

  const bubbleSort = () => {
    const arr = [...array];
    const newSteps: Step[] = [];

    for (let i = 0; i < arr.length - 1; i++) {
      for (let j = 0; j < arr.length - 1 - i; j++) {
        if (arr[j] > arr[j + 1]) {
          // Swap
          const temp = arr[j];
          arr[j] = arr[j + 1];
          arr[j + 1] = temp;

          newSteps.push({
            index1: j,
            index2: j + 1,
            array: [...arr],
          });
        }
      }
    }

    setSteps(newSteps);
    setCurrentStep(0);
    setIsSorting(true);
    setIsSorted(false);
  };

  const handleReset = () => {
    setArray([5, 2, 9, 1, 5, 6]);
    setSteps([]);
    setCurrentStep(0);
    setIsSorting(false);
    setIsSorted(false);
  };

  const displayArray = isSorting && steps.length > 0 && currentStep < steps.length
    ? steps[currentStep].array
    : array;

  return (
    <div className="max-w-4xl mx-auto p-8">
      <div className="bg-white rounded-lg shadow-lg p-8">
        <div className="flex items-center gap-3 mb-6">
          <ArrowUpDown className="text-blue-600" size={32} />
          <h2 className="text-3xl font-bold text-gray-900">Bubble Sort Visualization</h2>
        </div>

        <p className="text-gray-600 mb-6">
          Watch how bubble sort compares and swaps adjacent elements to sort the array.
        </p>

        {/* Array Visualization */}
        <div className="flex justify-center gap-2 mb-8">
          {displayArray.map((value, index) => {
            const isComparing = isSorting && steps[currentStep]?.index1 === index || steps[currentStep]?.index2 === index;

            return (
              <motion.div
                key={`${index}-${value}`}
                animate={{
                  scale: isComparing ? 1.1 : 1,
                  backgroundColor: isComparing ? '#3b82f6' : '#e5e7eb'
                }}
                className="w-16 h-24 rounded-lg flex items-center justify-center text-2xl font-bold text-gray-900 shadow-md"
              >
                {value}
              </motion.div>
            );
          })}
        </div>

        {/* Progress */}
        {isSorting && steps.length > 0 && (
          <div className="mb-6">
            <p className="text-sm text-gray-600 mb-2">
              Step {currentStep + 1} of {steps.length}
            </p>
            <div className="w-full bg-gray-200 rounded-full h-2">
              <div
                className="bg-blue-600 h-2 rounded-full transition-all duration-300"
                style={{ width: `${((currentStep + 1) / steps.length) * 100}%` }}
              />
            </div>
          </div>
        )}

        {/* Controls */}
        <div className="flex gap-3 justify-center">
          <button
            onClick={bubbleSort}
            disabled={isSorting}
            className="px-6 py-3 bg-blue-600 text-white rounded-lg font-medium hover:bg-blue-700 disabled:bg-gray-300 disabled:cursor-not-allowed flex items-center gap-2 transition-colors"
          >
            <Play size={18} />
            Start Sort
          </button>

          <button
            onClick={handleReset}
            className="px-6 py-3 bg-gray-200 text-gray-700 rounded-lg font-medium hover:bg-gray-300 flex items-center gap-2 transition-colors"
          >
            <RotateCcw size={18} />
            Reset
          </button>

          {isSorting && currentStep < steps.length - 1 && (
            <button
              onClick={() => setCurrentStep(currentStep + 1)}
              className="px-6 py-3 bg-green-600 text-white rounded-lg font-medium hover:bg-green-700 transition-colors"
            >
              Next Step
            </button>
          )}
        </div>

        {isSorted && (
          <div className="mt-6 p-4 bg-green-50 border border-green-200 rounded-lg">
            <p className="text-green-800 text-center font-medium">
              ✅ Array sorted successfully!
            </p>
          </div>
        )}
      </div>
    </div>
  );
}