import React, { useState } from 'react';
import { Download, FileText, File, ChevronDown, ChevronRight } from 'lucide-react';

interface ProcessedFileData {
  base64: string;
  filename: string;
  mime_type: string;
  content_preview?: string;
}

interface ProcessedFileViewerProps {
  processingData: ProcessedFileData;
  operation: string;
  fileSizeMb?: number;
  rowsConverted?: number;
  originalLines?: number;
  uniqueLines?: number;
  emailsFound?: number;
  urlsFound?: number;
}

const ProcessedFileViewer: React.FC<ProcessedFileViewerProps> = ({
  processingData,
  operation,
  fileSizeMb,
  rowsConverted,
  originalLines,
  uniqueLines,
  emailsFound,
  urlsFound
}) => {
  const [showPreview, setShowPreview] = useState(false);

  const handleDownload = () => {
    const binaryString = atob(processingData.base64);
    const bytes = new Uint8Array(binaryString.length);
    for (let i = 0; i < binaryString.length; i++) {
      bytes[i] = binaryString.charCodeAt(i);
    }

    const blob = new Blob([bytes], { type: processingData.mime_type });
    const url = URL.createObjectURL(blob);

    const link = document.createElement('a');
    link.href = url;
    link.download = processingData.filename;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);

    URL.revokeObjectURL(url);
  };

  const openInNewTab = () => {
    const binaryString = atob(processingData.base64);
    const bytes = new Uint8Array(binaryString.length);
    for (let i = 0; i < binaryString.length; i++) {
      bytes[i] = binaryString.charCodeAt(i);
    }

    const blob = new Blob([bytes], { type: processingData.mime_type });
    const url = URL.createObjectURL(blob);

    window.open(url, '_blank');

    // Clean up after a delay to allow the browser to handle the file
    setTimeout(() => {
      URL.revokeObjectURL(url);
    }, 1000);
  };

  const getOperationTitle = (op: string) => {
    switch (op) {
      case 'csv_to_json': return 'CSV to JSON Conversion';
      case 'json_to_csv': return 'JSON to CSV Conversion';
      case 'extract_emails': return 'Email Extraction';
      case 'extract_urls': return 'URL Extraction';
      case 'remove_duplicates': return 'Duplicate Removal';
      case 'clean_text': return 'Text Cleaning';
      case 'sort_lines': return 'Line Sorting';
      default: return 'Data Processing';
    }
  };

  const getFileTypeColor = (filename: string) => {
    if (filename.endsWith('.json')) return 'text-blue-600';
    if (filename.endsWith('.csv')) return 'text-green-600';
    if (filename.endsWith('.txt')) return 'text-gray-600';
    return 'text-purple-600';
  };

  const getBgColor = (filename: string) => {
    if (filename.endsWith('.json')) return 'from-blue-50 to-blue-100 border-blue-200';
    if (filename.endsWith('.csv')) return 'from-green-50 to-green-100 border-green-200';
    if (filename.endsWith('.txt')) return 'from-gray-50 to-gray-100 border-gray-200';
    return 'from-purple-50 to-purple-100 border-purple-200';
  };

  const getButtonColor = (filename: string) => {
    if (filename.endsWith('.json')) return 'bg-blue-600 hover:bg-blue-700 border-blue-600 text-blue-600 hover:bg-blue-50';
    if (filename.endsWith('.csv')) return 'bg-green-600 hover:bg-green-700 border-green-600 text-green-600 hover:bg-green-50';
    if (filename.endsWith('.txt')) return 'bg-gray-600 hover:bg-gray-700 border-gray-600 text-gray-600 hover:bg-gray-50';
    return 'bg-purple-600 hover:bg-purple-700 border-purple-600 text-purple-600 hover:bg-purple-50';
  };

  return (
    <div className="bg-white p-4 rounded-lg border shadow-sm">
      <div className="flex items-center justify-between mb-3">
        <div>
          <h3 className="text-lg font-semibold flex items-center gap-2">
            <File className={`w-5 h-5 ${getFileTypeColor(processingData.filename)}`} />
            {getOperationTitle(operation)}
            <span className="text-sm text-gray-600">({processingData.filename})</span>
          </h3>
          <div className="text-sm text-gray-500 space-y-1">
            {rowsConverted && (
              <p>Converted {rowsConverted} rows</p>
            )}
            {originalLines && uniqueLines && (
              <p>Reduced from {originalLines} to {uniqueLines} lines ({originalLines - uniqueLines} duplicates removed)</p>
            )}
            {emailsFound && (
              <p>Found {emailsFound} unique email addresses</p>
            )}
            {urlsFound && (
              <p>Found {urlsFound} unique URLs</p>
            )}
            {fileSizeMb && (
              <p>File size: {fileSizeMb.toFixed(4)} MB</p>
            )}
            <p>Format: {processingData.filename.split('.').pop()?.toUpperCase()}</p>
          </div>
        </div>

        <div className="flex gap-2">
          <button
            onClick={openInNewTab}
            className="p-2 text-gray-600 hover:text-gray-800 hover:bg-gray-100 rounded"
            title="Open in new tab"
          >
            <FileText className="w-4 h-4" />
          </button>

          <button
            onClick={handleDownload}
            className="p-2 text-gray-600 hover:text-gray-800 hover:bg-gray-100 rounded"
            title="Download file"
          >
            <Download className="w-4 h-4" />
          </button>
        </div>
      </div>

      <div className={`bg-gradient-to-br ${getBgColor(processingData.filename)} rounded-lg p-6 text-center`}>
        <div className="flex flex-col items-center space-y-3">
          <div className="w-16 h-16 bg-white/50 rounded-lg flex items-center justify-center">
            <File className={`w-8 h-8 ${getFileTypeColor(processingData.filename)}`} />
          </div>

          <div>
            <h4 className="font-medium text-gray-900 mb-1">Processed File Ready</h4>
            <p className="text-sm text-gray-600 mb-3">
              Your {operation.replace('_', ' ')} operation completed successfully. Click the buttons to view or download.
            </p>

            <div className="flex gap-3 justify-center">
              <button
                onClick={openInNewTab}
                className={`px-4 py-2 ${getButtonColor(processingData.filename).split(' ')[0]} ${getButtonColor(processingData.filename).split(' ')[1]} text-white rounded transition-colors text-sm font-medium`}
              >
                Open File
              </button>

              <button
                onClick={handleDownload}
                className={`px-4 py-2 border ${getButtonColor(processingData.filename).split(' ')[2]} ${getButtonColor(processingData.filename).split(' ')[3]} rounded ${getButtonColor(processingData.filename).split(' ')[4]} transition-colors text-sm font-medium`}
              >
                Download File
              </button>
            </div>
          </div>
        </div>
      </div>

      {/* Content Preview Section */}
      {processingData.content_preview && (
        <div className="mt-4 border-t pt-4">
          <button
            onClick={() => setShowPreview(!showPreview)}
            className="flex items-center gap-2 text-gray-700 hover:text-gray-900 font-medium"
          >
            {showPreview ? <ChevronDown className="w-4 h-4" /> : <ChevronRight className="w-4 h-4" />}
            Preview Content
          </button>

          {showPreview && (
            <div className="mt-3">
              <div className="bg-gray-50 border border-gray-200 rounded-lg p-3 max-h-64 overflow-y-auto">
                <pre className="text-xs text-gray-700 whitespace-pre-wrap font-mono">
                  {processingData.content_preview}
                </pre>
              </div>
            </div>
          )}
        </div>
      )}

      <div className="mt-3 text-xs text-gray-500 flex justify-between">
        <span>MIME type: {processingData.mime_type}</span>
        <span>Ready for download or viewing</span>
      </div>
    </div>
  );
};

export default ProcessedFileViewer;