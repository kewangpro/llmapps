import { useState } from 'react'
import GPUArchitectureAnimation from './GPUArchitectureAnimation'
import AttentionVisualizer from './AttentionVisualizer'
import { Cpu, Brain } from 'lucide-react'

function App() {
  const [activeView, setActiveView] = useState<'gpu' | 'attention'>('gpu')

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Navigation */}
      <nav className="bg-white shadow-lg sticky top-0 z-50">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center h-16">
            <h1 className="text-2xl font-bold text-gray-900">
              GPU & Transformer Visualizations
            </h1>
            <div className="flex gap-2">
              <button
                onClick={() => setActiveView('gpu')}
                className={`flex items-center gap-2 px-4 py-2 rounded-lg font-medium transition-all ${
                  activeView === 'gpu'
                    ? 'bg-blue-600 text-white shadow-md'
                    : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
                }`}
              >
                <Cpu size={20} />
                GPU Architecture
              </button>
              <button
                onClick={() => setActiveView('attention')}
                className={`flex items-center gap-2 px-4 py-2 rounded-lg font-medium transition-all ${
                  activeView === 'attention'
                    ? 'bg-purple-600 text-white shadow-md'
                    : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
                }`}
              >
                <Brain size={20} />
                Attention Mechanism
              </button>
            </div>
          </div>
        </div>
      </nav>

      {/* Content */}
      <main>
        {activeView === 'gpu' ? (
          <GPUArchitectureAnimation />
        ) : (
          <AttentionVisualizer />
        )}
      </main>
    </div>
  )
}

export default App
