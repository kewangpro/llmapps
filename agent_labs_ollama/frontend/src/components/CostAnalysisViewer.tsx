import React, { useState } from 'react';
import { Download, FileText, TrendingUp, ChevronDown, ChevronRight, DollarSign } from 'lucide-react';

interface CostAnalysisFileData {
  base64: string;
  filename: string;
  mime_type: string;
  content_preview?: string;
}

interface CostAnalysisViewerProps {
  costAnalysisFileData: CostAnalysisFileData;
  fileSizeMb?: number;
}

const CostAnalysisViewer: React.FC<CostAnalysisViewerProps> = ({
  costAnalysisFileData,
  fileSizeMb
}) => {
  const [showPreview, setShowPreview] = useState(false);

  const handleDownload = () => {
    const binaryString = atob(costAnalysisFileData.base64);
    const bytes = new Uint8Array(binaryString.length);
    for (let i = 0; i < binaryString.length; i++) {
      bytes[i] = binaryString.charCodeAt(i);
    }

    const blob = new Blob([bytes], { type: costAnalysisFileData.mime_type });
    const url = URL.createObjectURL(blob);

    const link = document.createElement('a');
    link.href = url;
    link.download = costAnalysisFileData.filename;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);

    URL.revokeObjectURL(url);
  };

  const openInNewTab = () => {
    const binaryString = atob(costAnalysisFileData.base64);
    const bytes = new Uint8Array(binaryString.length);
    for (let i = 0; i < binaryString.length; i++) {
      bytes[i] = binaryString.charCodeAt(i);
    }

    const blob = new Blob([bytes], { type: costAnalysisFileData.mime_type });
    const url = URL.createObjectURL(blob);

    window.open(url, '_blank');
  };

  return (
    <div className="bg-green-50 border border-green-200 rounded-lg p-4">
      {/* Header */}
      <div className="flex items-start justify-between mb-3">
        <div className="flex items-center gap-2">
          <DollarSign className="w-5 h-5 text-green-600" />
          <div>
            <h3 className="font-semibold text-green-800">Cost Analysis Results</h3>
            <p className="text-sm text-green-600">
              Financial data analysis and cost insights
            </p>
          </div>
        </div>
        <TrendingUp className="w-6 h-6 text-green-600" />
      </div>

      {/* File Info */}
      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center gap-2 text-sm text-green-700">
          <FileText className="w-4 h-4" />
          <span className="font-medium">{costAnalysisFileData.filename}</span>
          {fileSizeMb && (
            <span className="text-green-600">({fileSizeMb.toFixed(2)} MB)</span>
          )}
        </div>
        <div className="flex gap-2">
          <button
            onClick={openInNewTab}
            className="text-green-600 hover:text-green-700 text-sm font-medium"
          >
            View
          </button>
          <button
            onClick={handleDownload}
            className="flex items-center gap-1 px-3 py-1 bg-green-600 text-white rounded hover:bg-green-700 text-sm"
          >
            <Download className="w-4 h-4" />
            Download CSV
          </button>
        </div>
      </div>

      {/* Preview Toggle */}
      {costAnalysisFileData.content_preview && (
        <div className="border-t border-green-200 pt-3">
          <button
            onClick={() => setShowPreview(!showPreview)}
            className="flex items-center gap-2 text-green-700 hover:text-green-800 font-medium text-sm"
          >
            {showPreview ? <ChevronDown className="w-4 h-4" /> : <ChevronRight className="w-4 h-4" />}
            {showPreview ? 'Hide' : 'Show'} Data Preview
          </button>

          {showPreview && (
            <div className="mt-2 bg-white rounded border p-3">
              <pre className="text-xs text-gray-800 whitespace-pre-wrap overflow-x-auto">
                {costAnalysisFileData.content_preview}
              </pre>
            </div>
          )}
        </div>
      )}
    </div>
  );
};

export default CostAnalysisViewer;