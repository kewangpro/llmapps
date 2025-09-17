'use client';

import { useState, useEffect } from 'react';
import { Tool } from '@/types';
import { Check, Wrench, Globe, Code, Database, Monitor } from 'lucide-react';

interface ToolSidebarProps {
  selectedTools: string[];
  onToolToggle: (toolName: string) => void;
}

const toolIcons: Record<string, any> = {
  filesystem: Monitor,
  web: Globe,
  development: Code,
  data: Database,
  system: Monitor
};

export default function ToolSidebar({ selectedTools, onToolToggle }: ToolSidebarProps) {
  const [tools, setTools] = useState<Record<string, Tool>>({});
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    // Fetch available tools from the backend
    const fetchTools = async () => {
      try {
        const response = await fetch('http://localhost:8000/api/tools');
        const data = await response.json();
        setTools(data.tools);
      } catch (error) {
        console.error('Failed to fetch tools:', error);
      } finally {
        setLoading(false);
      }
    };

    fetchTools();
  }, []);

  const groupedTools = Object.values(tools).reduce((acc, tool) => {
    if (!acc[tool.category]) {
      acc[tool.category] = [];
    }
    acc[tool.category].push(tool);
    return acc;
  }, {} as Record<string, Tool[]>);

  if (loading) {
    return (
      <div className="w-80 bg-gray-50 border-r border-gray-200 p-4">
        <div className="animate-pulse">
          <div className="h-6 bg-gray-200 rounded mb-4"></div>
          <div className="space-y-3">
            {[...Array(5)].map((_, i) => (
              <div key={i} className="h-12 bg-gray-200 rounded"></div>
            ))}
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="w-80 bg-gray-50 border-r border-gray-200 p-4 overflow-y-auto">
      <div className="flex items-center gap-2 mb-6">
        <Wrench className="w-5 h-5 text-blue-600" />
        <h2 className="text-lg font-semibold text-gray-900">Available Tools</h2>
      </div>

      <div className="space-y-6">
        {Object.entries(groupedTools).map(([category, categoryTools]) => {
          const IconComponent = toolIcons[category] || Wrench;

          return (
            <div key={category} className="space-y-2">
              <div className="flex items-center gap-2 mb-3">
                <IconComponent className="w-4 h-4 text-gray-600" />
                <h3 className="text-sm font-medium text-gray-700 capitalize">
                  {category}
                </h3>
              </div>

              <div className="space-y-2">
                {categoryTools.map((tool) => (
                  <div
                    key={tool.name}
                    className={`p-3 rounded-lg border cursor-pointer transition-all hover:shadow-sm ${
                      selectedTools.includes(tool.name)
                        ? 'bg-blue-50 border-blue-200 ring-1 ring-blue-200'
                        : 'bg-white border-gray-200 hover:border-gray-300'
                    }`}
                    onClick={() => onToolToggle(tool.name)}
                  >
                    <div className="flex items-start justify-between">
                      <div className="flex-1 min-w-0">
                        <div className="flex items-center gap-2">
                          <h4 className="text-sm font-medium text-gray-900 truncate">
                            {tool.name.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase())}
                          </h4>
                          {selectedTools.includes(tool.name) && (
                            <Check className="w-4 h-4 text-blue-600 flex-shrink-0" />
                          )}
                        </div>
                        <p className="text-xs text-gray-600 mt-1 line-clamp-2">
                          {tool.description}
                        </p>
                      </div>
                    </div>

                    {Object.keys(tool.parameters).length > 0 && (
                      <div className="mt-2 pt-2 border-t border-gray-100">
                        <div className="flex flex-wrap gap-1">
                          {Object.keys(tool.parameters).slice(0, 3).map((param) => (
                            <span
                              key={param}
                              className="inline-block px-2 py-1 text-xs bg-gray-100 text-gray-600 rounded"
                            >
                              {param}
                            </span>
                          ))}
                          {Object.keys(tool.parameters).length > 3 && (
                            <span className="inline-block px-2 py-1 text-xs bg-gray-100 text-gray-600 rounded">
                              +{Object.keys(tool.parameters).length - 3} more
                            </span>
                          )}
                        </div>
                      </div>
                    )}
                  </div>
                ))}
              </div>
            </div>
          );
        })}
      </div>

      {selectedTools.length > 0 && (
        <div className="mt-6 p-3 bg-blue-50 rounded-lg border border-blue-200">
          <h4 className="text-sm font-medium text-blue-900 mb-2">
            Selected Tools ({selectedTools.length})
          </h4>
          <div className="flex flex-wrap gap-1">
            {selectedTools.map((toolName) => (
              <span
                key={toolName}
                className="inline-flex items-center gap-1 px-2 py-1 text-xs bg-blue-100 text-blue-800 rounded"
              >
                {toolName.replace(/_/g, ' ')}
                <button
                  onClick={(e) => {
                    e.stopPropagation();
                    onToolToggle(toolName);
                  }}
                  className="ml-1 text-blue-600 hover:text-blue-800"
                >
                  ×
                </button>
              </span>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}