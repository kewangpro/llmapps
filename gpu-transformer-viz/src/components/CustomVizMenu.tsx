import { motion } from 'framer-motion';
import * as LucideIcons from 'lucide-react';
import { Plus, X } from 'lucide-react';

interface CustomViz {
  id: string;
  name: string;
  icon: string;
  component: React.LazyExoticComponent<React.ComponentType>;
}

interface CustomVizMenuProps {
  customVizs: CustomViz[];
  activeView: string;
  onSelectViz: (id: string) => void;
  onAddNew: () => void;
  onClose: () => void;
}

export default function CustomVizMenu({
  customVizs,
  activeView,
  onSelectViz,
  onAddNew,
  onClose
}: CustomVizMenuProps) {
  const getIconComponent = (iconName: string) => {
    // @ts-ignore - Dynamic icon lookup
    const IconComponent = LucideIcons[iconName] || LucideIcons.Sparkles;
    return IconComponent;
  };

  return (
    <>
      {/* Backdrop */}
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        exit={{ opacity: 0 }}
        className="fixed inset-0 bg-black/50 z-40"
        onClick={onClose}
      />

      {/* Menu Panel */}
      <motion.div
        initial={{ opacity: 0, scale: 0.95, y: -20 }}
        animate={{ opacity: 1, scale: 1, y: 0 }}
        exit={{ opacity: 0, scale: 0.95, y: -20 }}
        transition={{ duration: 0.2 }}
        className="fixed top-20 right-4 w-80 bg-white rounded-lg shadow-2xl z-50 overflow-hidden"
      >
        {/* Header */}
        <div className="flex items-center justify-between px-4 py-3 bg-gradient-to-r from-blue-600 to-purple-600 text-white">
          <h3 className="text-lg font-semibold">Custom Visualizations</h3>
          <button
            onClick={onClose}
            className="p-1 hover:bg-white/20 rounded transition-colors"
          >
            <X size={20} />
          </button>
        </div>

        {/* Add New Button */}
        <div className="p-3 border-b border-gray-200">
          <button
            onClick={() => {
              onAddNew();
              onClose();
            }}
            className="w-full flex items-center justify-center gap-2 px-4 py-3 bg-gradient-to-r from-yellow-400 to-orange-500 text-white rounded-lg font-medium hover:from-yellow-500 hover:to-orange-600 transition-all shadow-md"
          >
            <Plus size={20} />
            Add New Visualization
          </button>
        </div>

        {/* Visualizations List */}
        <div className="max-h-96 overflow-y-auto">
          {customVizs.length === 0 ? (
            <div className="p-8 text-center text-gray-500">
              <p className="text-sm">No custom visualizations yet.</p>
              <p className="text-xs mt-2">Click "Add New" to create one!</p>
            </div>
          ) : (
            <div className="p-2">
              {customVizs.map((viz) => {
                const IconComponent = getIconComponent(viz.icon);
                const isActive = activeView === viz.id;

                return (
                  <button
                    key={viz.id}
                    onClick={() => {
                      onSelectViz(viz.id);
                      onClose();
                    }}
                    className={`w-full flex items-center gap-3 px-4 py-3 rounded-lg font-medium transition-all mb-2 ${
                      isActive
                        ? 'bg-green-600 text-white shadow-md'
                        : 'bg-gray-50 text-gray-700 hover:bg-gray-100'
                    }`}
                  >
                    <IconComponent size={20} />
                    <span className="flex-1 text-left">{viz.name}</span>
                    {isActive && (
                      <span className="text-xs bg-white/20 px-2 py-1 rounded">
                        Active
                      </span>
                    )}
                  </button>
                );
              })}
            </div>
          )}
        </div>

        {/* Footer */}
        <div className="px-4 py-3 bg-gray-50 border-t border-gray-200 text-xs text-gray-500 text-center">
          {customVizs.length} visualization{customVizs.length !== 1 ? 's' : ''} available
        </div>
      </motion.div>
    </>
  );
}
