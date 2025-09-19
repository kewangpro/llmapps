'use client';

import { useState } from 'react';
import { VisualizationData } from '@/types';
import { BarChart3, Download, Eye, Code2, Info } from 'lucide-react';

interface VisualizationChartProps {
  visualizationData: VisualizationData;
}

export default function VisualizationChart({ visualizationData }: VisualizationChartProps) {
  const [activeTab, setActiveTab] = useState<'chart' | 'image' | 'info'>('chart');

  // Parse the HTML to create a safe iframe src
  const createSafeHTMLSrc = (html: string) => {
    const blob = new Blob([html], { type: 'text/html' });
    return URL.createObjectURL(blob);
  };

  const downloadChart = (format: 'html' | 'png') => {
    if (format === 'html') {
      const blob = new Blob([visualizationData.chart_html], { type: 'text/html' });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `chart_${visualizationData.chart_type}_${Date.now()}.html`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
    } else if (format === 'png') {
      const link = document.createElement('a');
      link.href = `data:image/png;base64,${visualizationData.chart_image_base64}`;
      link.download = `chart_${visualizationData.chart_type}_${Date.now()}.png`;
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
    }
  };

  return (
    <div className="bg-white rounded-lg border border-gray-200 overflow-hidden">
      {/* Header */}
      <div className="bg-gray-50 border-b border-gray-200 p-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="p-2 bg-blue-100 rounded-lg">
              <BarChart3 className="w-5 h-5 text-blue-600" />
            </div>
            <div>
              <h3 className="font-semibold text-gray-900 capitalize">
                {visualizationData.chart_type} Chart
              </h3>
              <p className="text-sm text-gray-600">
                {visualizationData.chart_config.title}
              </p>
            </div>
          </div>
          
          <div className="flex items-center gap-2">
            <button
              onClick={() => downloadChart('html')}
              className="px-3 py-1 text-xs bg-blue-600 text-white rounded hover:bg-blue-700 flex items-center gap-1"
              title="Download Interactive Chart (HTML)"
            >
              <Download className="w-3 h-3" />
              HTML
            </button>
            <button
              onClick={() => downloadChart('png')}
              className="px-3 py-1 text-xs bg-green-600 text-white rounded hover:bg-green-700 flex items-center gap-1"
              title="Download Chart Image (PNG)"
            >
              <Download className="w-3 h-3" />
              PNG
            </button>
          </div>
        </div>
      </div>

      {/* Tab Navigation */}
      <div className="border-b border-gray-200">
        <div className="flex">
          <button
            onClick={() => setActiveTab('chart')}
            className={`px-4 py-2 text-sm font-medium border-b-2 transition-colors ${
              activeTab === 'chart'
                ? 'border-blue-500 text-blue-600 bg-blue-50'
                : 'border-transparent text-gray-500 hover:text-gray-700 hover:bg-gray-50'
            }`}
          >
            <div className="flex items-center gap-2">
              <Eye className="w-4 h-4" />
              Interactive Chart
            </div>
          </button>
          <button
            onClick={() => setActiveTab('image')}
            className={`px-4 py-2 text-sm font-medium border-b-2 transition-colors ${
              activeTab === 'image'
                ? 'border-blue-500 text-blue-600 bg-blue-50'
                : 'border-transparent text-gray-500 hover:text-gray-700 hover:bg-gray-50'
            }`}
          >
            <div className="flex items-center gap-2">
              <Code2 className="w-4 h-4" />
              Static Image
            </div>
          </button>
          <button
            onClick={() => setActiveTab('info')}
            className={`px-4 py-2 text-sm font-medium border-b-2 transition-colors ${
              activeTab === 'info'
                ? 'border-blue-500 text-blue-600 bg-blue-50'
                : 'border-transparent text-gray-500 hover:text-gray-700 hover:bg-gray-50'
            }`}
          >
            <div className="flex items-center gap-2">
              <Info className="w-4 h-4" />
              Chart Info
            </div>
          </button>
        </div>
      </div>

      {/* Content */}
      <div className="p-4">
        {activeTab === 'chart' && (
          <div className="space-y-4">
            <div className="text-sm text-gray-600 bg-blue-50 p-3 rounded-lg border border-blue-200">
              <div className="flex items-center gap-2 mb-1">
                <Eye className="w-4 h-4 text-blue-600" />
                <span className="font-medium text-blue-800">Interactive Chart</span>
              </div>
              <p className="text-blue-700">
                This is a fully interactive chart. You can hover, zoom, and explore the data.
              </p>
            </div>
            
            <div className="border border-gray-200 rounded-lg overflow-hidden">
              <iframe
                src={createSafeHTMLSrc(visualizationData.chart_html)}
                className="w-full h-96 border-none"
                title={`${visualizationData.chart_type} Chart - ${visualizationData.chart_config.title}`}
                sandbox="allow-scripts allow-same-origin"
              />
            </div>
          </div>
        )}

        {activeTab === 'image' && (
          <div className="space-y-4">
            <div className="text-sm text-gray-600 bg-gray-50 p-3 rounded-lg border border-gray-200">
              <div className="flex items-center gap-2 mb-1">
                <Code2 className="w-4 h-4 text-gray-600" />
                <span className="font-medium text-gray-800">Static Image</span>
              </div>
              <p className="text-gray-700">
                A static PNG version of your chart, perfect for reports and presentations.
              </p>
            </div>
            
            <div className="border border-gray-200 rounded-lg overflow-hidden bg-white">
              <img
                src={`data:image/png;base64,${visualizationData.chart_image_base64}`}
                alt={`${visualizationData.chart_type} Chart - ${visualizationData.chart_config.title}`}
                className="w-full h-auto max-w-full"
              />
            </div>
          </div>
        )}

        {activeTab === 'info' && (
          <div className="space-y-4">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {/* Chart Details */}
              <div className="bg-gray-50 p-4 rounded-lg">
                <h4 className="font-medium text-gray-900 mb-3 flex items-center gap-2">
                  <BarChart3 className="w-4 h-4" />
                  Chart Details
                </h4>
                <div className="space-y-2 text-sm">
                  <div>
                    <span className="font-medium text-gray-700">Type:</span>
                    <span className="ml-2 capitalize">{visualizationData.chart_type}</span>
                  </div>
                  <div>
                    <span className="font-medium text-gray-700">Title:</span>
                    <span className="ml-2">{visualizationData.chart_config.title}</span>
                  </div>
                  {visualizationData.chart_config.x_column && (
                    <div>
                      <span className="font-medium text-gray-700">X-Axis:</span>
                      <span className="ml-2">{visualizationData.chart_config.x_column}</span>
                    </div>
                  )}
                  {visualizationData.chart_config.y_column && (
                    <div>
                      <span className="font-medium text-gray-700">Y-Axis:</span>
                      <span className="ml-2">{visualizationData.chart_config.y_column}</span>
                    </div>
                  )}
                  {visualizationData.chart_config.label_column && (
                    <div>
                      <span className="font-medium text-gray-700">Labels:</span>
                      <span className="ml-2">{visualizationData.chart_config.label_column}</span>
                    </div>
                  )}
                  {visualizationData.chart_config.value_column && (
                    <div>
                      <span className="font-medium text-gray-700">Values:</span>
                      <span className="ml-2">{visualizationData.chart_config.value_column}</span>
                    </div>
                  )}
                </div>
              </div>

              {/* Data Details */}
              <div className="bg-gray-50 p-4 rounded-lg">
                <h4 className="font-medium text-gray-900 mb-3 flex items-center gap-2">
                  <Info className="w-4 h-4" />
                  Data Summary
                </h4>
                <div className="space-y-2 text-sm">
                  <div>
                    <span className="font-medium text-gray-700">Rows:</span>
                    <span className="ml-2">{visualizationData.data_rows.toLocaleString()}</span>
                  </div>
                  <div>
                    <span className="font-medium text-gray-700">Columns:</span>
                    <span className="ml-2">{visualizationData.data_columns}</span>
                  </div>
                  <div>
                    <span className="font-medium text-gray-700">Message:</span>
                    <span className="ml-2">{visualizationData.message}</span>
                  </div>
                </div>
              </div>
            </div>

            {/* Available Columns */}
            <div className="bg-gray-50 p-4 rounded-lg">
              <h4 className="font-medium text-gray-900 mb-3">Available Columns</h4>
              <div className="flex flex-wrap gap-2">
                {visualizationData.columns.map((column, index) => (
                  <span
                    key={index}
                    className="px-2 py-1 bg-blue-100 text-blue-800 text-xs rounded-full"
                  >
                    {column}
                  </span>
                ))}
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}