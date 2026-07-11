import { useState, useEffect, Suspense } from 'react'
import { AnimatePresence } from 'framer-motion'
import GPUArchitectureAnimation from './GPUArchitectureAnimation'
import AttentionVisualizer from './AttentionVisualizer'
import CustomDialog from './components/CustomDialog'
import CustomVizMenu from './components/CustomVizMenu'
import ErrorBoundary from './components/ErrorBoundary'
import { Cpu, Brain, Sparkles, Loader2 } from 'lucide-react'
import { loadCustomVisualizations } from './utils/customVizLoader'

type ViewType = 'gpu' | 'attention' | string;

interface CustomViz {
  id: string;
  name: string;
  icon: string;
  component: React.LazyExoticComponent<React.ComponentType>;
}

function App() {
  const [activeView, setActiveView] = useState<ViewType>('gpu')
  const [customVizs, setCustomVizs] = useState<CustomViz[]>([])
  const [isDialogOpen, setIsDialogOpen] = useState(false)
  const [isMenuOpen, setIsMenuOpen] = useState(false)

  // Load custom visualizations on mount and when new ones are created
  const loadVizs = async () => {
    try {
      console.log('⚙️ Loading visualizations...');
      const vizs = await loadCustomVisualizations();
      console.log('💾 Setting state with', vizs.length, 'visualizations:', vizs.map(v => v.name));
      setCustomVizs(vizs);
    } catch (error) {
      console.error('Failed to load custom visualizations:', error);
    }
  }

  useEffect(() => {
    loadVizs()
  }, [])

  const handleVisualizationCreated = async () => {
    console.log('🔄 Visualization created, reloading list...');

    // Wait for file to be written, then reload visualizations
    await new Promise(resolve => setTimeout(resolve, 1500));

    // Reload visualizations from backend
    console.log('📥 Fetching updated visualization list...');
    await loadVizs();
    console.log('✅ Visualization list reloaded');
  }

  const renderActiveView = () => {
    if (activeView === 'gpu') {
      return <GPUArchitectureAnimation />
    }

    if (activeView === 'attention') {
      return <AttentionVisualizer />
    }

    // Render custom visualization
    const customViz = customVizs.find(v => v.id === activeView)
    if (customViz) {
      const CustomComponent = customViz.component
      return (
        <ErrorBoundary fallbackTitle={`Error in ${customViz.name}`}>
          <Suspense fallback={
            <div className="flex items-center justify-center min-h-[400px]">
              <Loader2 className="animate-spin text-blue-600" size={48} />
            </div>
          }>
            <CustomComponent />
          </Suspense>
        </ErrorBoundary>
      )
    }

    return <GPUArchitectureAnimation />
  }

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Navigation */}
      <nav className="bg-white shadow-lg sticky top-0 z-50">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center h-16">
            <h1 className="text-2xl font-bold text-gray-900">
              Visualizations
            </h1>
            <div className="flex gap-2">
              {/* Built-in visualizations */}
              <button
                onClick={() => setActiveView('gpu')}
                className={`flex items-center gap-2 px-4 py-2 rounded-lg font-medium transition-all whitespace-nowrap ${
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
                className={`flex items-center gap-2 px-4 py-2 rounded-lg font-medium transition-all whitespace-nowrap ${
                  activeView === 'attention'
                    ? 'bg-purple-600 text-white shadow-md'
                    : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
                }`}
              >
                <Brain size={20} />
                Transformers Attention
              </button>

              {/* Custom visualizations menu */}
              <button
                onClick={() => setIsMenuOpen(true)}
                className="flex items-center gap-2 px-4 py-2 rounded-lg font-medium transition-all bg-gradient-to-r from-yellow-400 to-orange-500 text-white hover:from-yellow-500 hover:to-orange-600 shadow-md whitespace-nowrap"
              >
                <Sparkles size={20} />
                Custom
                {customVizs.length > 0 && (
                  <span className="ml-1 px-2 py-0.5 bg-white/30 rounded-full text-xs">
                    {customVizs.length}
                  </span>
                )}
              </button>
            </div>
          </div>
        </div>
      </nav>

      {/* Content */}
      <main>
        {renderActiveView()}
      </main>

      {/* Custom Visualization Generation Dialog */}
      <CustomDialog
        open={isDialogOpen}
        onOpenChange={setIsDialogOpen}
        onVisualizationCreated={handleVisualizationCreated}
      />

      {/* Custom Visualizations Menu */}
      <AnimatePresence>
        {isMenuOpen && (
          <CustomVizMenu
            customVizs={customVizs}
            activeView={activeView}
            onSelectViz={setActiveView}
            onAddNew={() => setIsDialogOpen(true)}
            onClose={() => setIsMenuOpen(false)}
          />
        )}
      </AnimatePresence>
    </div>
  )
}

export default App
