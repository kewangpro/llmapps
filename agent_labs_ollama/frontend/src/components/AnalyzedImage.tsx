import React, { useState } from 'react';
import { ImageData } from '@/types';
import { ZoomIn, ZoomOut, RotateCw, Download } from 'lucide-react';

interface AnalyzedImageProps {
  imageData: ImageData;
  filename?: string;
  fileSize?: number;
}

const AnalyzedImage: React.FC<AnalyzedImageProps> = ({ imageData, filename, fileSize }) => {
  const [isZoomed, setIsZoomed] = useState(false);
  const [rotation, setRotation] = useState(0);

  const handleDownload = () => {
    const link = document.createElement('a');
    link.href = imageData.data_url;
    link.download = filename || 'analyzed_image';
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  const handleRotate = () => {
    setRotation((prev) => (prev + 90) % 360);
  };

  return (
    <div className="bg-white p-4 rounded-lg border shadow-sm">
      <div className="flex items-center justify-between mb-3">
        <div>
          <h3 className="text-lg font-semibold flex items-center gap-2">
            🖼️ Analyzed Image
            {filename && <span className="text-sm text-gray-600">({filename})</span>}
          </h3>
          {fileSize && (
            <p className="text-sm text-gray-500">
              File size: {fileSize} MB
            </p>
          )}
        </div>

        <div className="flex gap-2">
          <button
            onClick={() => setIsZoomed(!isZoomed)}
            className="p-2 text-gray-600 hover:text-gray-800 hover:bg-gray-100 rounded"
            title={isZoomed ? "Zoom out" : "Zoom in"}
          >
            {isZoomed ? <ZoomOut className="w-4 h-4" /> : <ZoomIn className="w-4 h-4" />}
          </button>

          <button
            onClick={handleRotate}
            className="p-2 text-gray-600 hover:text-gray-800 hover:bg-gray-100 rounded"
            title="Rotate 90°"
          >
            <RotateCw className="w-4 h-4" />
          </button>

          <button
            onClick={handleDownload}
            className="p-2 text-gray-600 hover:text-gray-800 hover:bg-gray-100 rounded"
            title="Download image"
          >
            <Download className="w-4 h-4" />
          </button>
        </div>
      </div>

      <div className={`overflow-hidden rounded-lg border ${isZoomed ? 'cursor-zoom-out' : 'cursor-zoom-in'}`}>
        <img
          src={imageData.data_url}
          alt="Analyzed image"
          className={`w-full h-auto transition-transform duration-200 ${
            isZoomed ? 'scale-150' : 'scale-100'
          }`}
          style={{
            transform: `rotate(${rotation}deg) ${isZoomed ? 'scale(1.5)' : 'scale(1)'}`,
            maxHeight: isZoomed ? 'none' : '400px',
            objectFit: 'contain'
          }}
          onClick={() => setIsZoomed(!isZoomed)}
        />
      </div>

      <div className="mt-2 text-xs text-gray-500 flex justify-between">
        <span>MIME type: {imageData.mime_type}</span>
        <span>Click image to {isZoomed ? 'zoom out' : 'zoom in'}</span>
      </div>
    </div>
  );
};

export default AnalyzedImage;