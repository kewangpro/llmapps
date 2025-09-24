'use client';

import { useState, useEffect } from 'react';
import { Tool } from '@/types';
import { apiUrl } from '@/utils/api';
import { Check, Wrench, Globe, Code, Database, Monitor, BarChart3, Settings, Search, FileText, Image, TrendingUp, ChevronDown, ChevronUp } from 'lucide-react';

interface ToolSidebarProps {
  selectedTools: string[];
  onToolToggle: (toolName: string) => void;
}

const toolIcons: Record<string, any> = {
  general: Settings,
  analytics: BarChart3,
  filesystem: Monitor,
  web: Globe,
  development: Code,
  data: Database,
  system: Monitor
};

const getToolTips = (toolName: string) => {
  const tips: Record<string, { icon: any, title: string, examples: string[], extra?: string }> = {
    file_search: {
      icon: Search,
      title: "🔍 Search for files and directories:",
      examples: [
        "Find files: \"search for Python files in the project\"",
        "Locate directories: \"find all config folders\"",
        "Filter by type: \"show me all .json files\""
      ],
      extra: "Supports: file names, extensions, directory patterns, content search"
    },
    web_search: {
      icon: Globe,
      title: "🌐 Search the web for information:",
      examples: [
        "Current events: \"latest news about AI developments\"",
        "Research topics: \"explain quantum computing\"",
        "Find resources: \"best practices for React development\""
      ],
      extra: "Real-time web results, news, articles, and documentation"
    },
    system_info: {
      icon: Monitor,
      title: "💻 Get system information:",
      examples: [
        "Performance: \"show CPU and memory usage\"",
        "Storage: \"check disk space availability\"",
        "Network: \"display network interface details\""
      ],
      extra: "Monitors: CPU, memory, disk, network, processes, uptime"
    },
    cost_analysis: {
      icon: BarChart3,
      title: "💰 Analyze cost data and spending patterns:",
      examples: [
        "COGS analysis: \"analyze cost per business unit per month\"",
        "AWS costs: \"show cost per AWS product per month\"", 
        "Service costs: \"analyze cost per service group per month\""
      ],
      extra: "Processes: COGS data, AWS cost breakdowns, spending trends, cost optimization recommendations"
    },
    data_processing: {
      icon: Database,
      title: "📊 Process and transform data:",
      examples: [
        "Convert formats: \"convert this CSV to JSON\"",
        "Clean data: \"remove duplicates from this dataset\"",
        "Extract info: \"get email addresses from text\""
      ],
      extra: "Supports: CSV, JSON, text analysis, data cleaning, format conversion"
    },
    presentation: {
      icon: FileText,
      title: "📑 Generate PowerPoint presentations:",
      examples: [
        "From text: \"create slides about project overview\"",
        "From data: attach file and say \"make a presentation\"",
        "Custom content: \"build slides for quarterly review\""
      ],
      extra: "Creates: title slides, content slides, bullet points, professional layouts"
    },
    image_analysis: {
      icon: Image,
      title: "🖼️ Analyze image content and metadata:",
      examples: [
        "Describe image: \"what's in this photo?\"",
        "Extract text: \"read the text from this screenshot\"",
        "Get metadata: \"show EXIF data for this image\""
      ],
      extra: "Supports: JPG, PNG, GIF, content analysis, OCR, metadata extraction"
    },
    stock_analysis: {
      icon: TrendingUp,
      title: "📈 Analyze stock market data:",
      examples: [
        "Stock price: \"show AAPL stock performance\"",
        "Technical analysis: \"analyze TSLA trends\"",
        "Market data: \"get financial metrics for MSFT\""
      ],
      extra: "Features: price charts, technical indicators, financial metrics, trends"
    },
    visualization: {
      icon: BarChart3,
      title: "📊 Create charts from your data:",
      examples: [
        "Upload a CSV file and ask: \"create a bar chart\"",
        "Attach data and say: \"make a pie chart showing distribution\"",
        "Upload data and request: \"visualize trends over time\""
      ],
      extra: "Supported chart types: line, bar, scatter, pie, histogram, box, heatmap, area, bubble, treemap"
    },
    forecast: {
      icon: TrendingUp,
      title: "🔮 Predict future trends using LSTM:",
      examples: [
        "Upload time series CSV and ask: \"forecast the next 30 days\"",
        "Attach sales data and say: \"predict future sales trends\"",
        "Upload stock prices and request: \"forecast price movements\""
      ],
      extra: "Uses: LSTM neural networks, automatic data preprocessing, model performance metrics, downloadable predictions"
    }
  };
  
  return tips[toolName];
};

export default function ToolSidebar({ selectedTools, onToolToggle }: ToolSidebarProps) {
  const [tools, setTools] = useState<Record<string, Tool>>({});
  const [loading, setLoading] = useState(true);
  const [showAllTips, setShowAllTips] = useState(false);

  useEffect(() => {
    // Fetch available tools from the backend
    const fetchTools = async () => {
      try {
        const response = await fetch(apiUrl('/api/tools'));
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
                  {category} ({categoryTools.length})
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

          {/* Quick Tips for Selected Tools */}
          {selectedTools.some(tool => getToolTips(tool)) && (() => {
            const toolsWithTips = selectedTools.filter(tool => getToolTips(tool));
            const hasMultipleTools = toolsWithTips.length > 1;
            const displayedTools = hasMultipleTools && !showAllTips ? toolsWithTips.slice(0, 1) : toolsWithTips;

            return (
              <div className="mt-3 pt-3 border-t border-blue-200">
                {hasMultipleTools && (
                  <div className="flex items-center justify-between mb-3">
                    <span className="text-xs font-medium text-blue-900">
                      Quick Tips ({toolsWithTips.length} tool{toolsWithTips.length > 1 ? 's' : ''})
                    </span>
                    <button
                      onClick={() => setShowAllTips(!showAllTips)}
                      className="flex items-center gap-1 text-xs text-blue-600 hover:text-blue-800"
                    >
                      {showAllTips ? (
                        <>
                          <ChevronUp className="w-3 h-3" />
                          Show Less
                        </>
                      ) : (
                        <>
                          <ChevronDown className="w-3 h-3" />
                          Show All
                        </>
                      )}
                    </button>
                  </div>
                )}
                
                <div className="space-y-3">
                  {displayedTools.map((toolName) => {
                    const tips = getToolTips(toolName);
                    if (!tips) return null;

                    const IconComponent = tips.icon;
                    
                    return (
                      <div key={toolName} className="text-xs text-blue-800">
                        <div className="font-medium mb-2 flex items-center gap-1">
                          <IconComponent className="w-3 h-3" />
                          {tips.title}
                        </div>
                        <ul className="space-y-1 mb-2 list-disc list-inside ml-1">
                          {tips.examples.map((example, index) => (
                            <li key={index}>{example}</li>
                          ))}
                        </ul>
                        {tips.extra && (
                          <div className="text-xs">
                            <strong>{tips.extra.includes(':') ? tips.extra.split(':')[0] + ':' : 'Note:'}</strong> {tips.extra.includes(':') ? tips.extra.split(':')[1] : tips.extra}
                          </div>
                        )}
                      </div>
                    );
                  })}
                </div>
              </div>
            );
          })()}
        </div>
      )}
    </div>
  );
}