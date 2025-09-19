import React, { useState } from 'react';
import { PresentationData } from '@/types';
import { Download, FileText, Presentation, ChevronDown, ChevronRight } from 'lucide-react';

interface PresentationViewerProps {
  presentationData: PresentationData;
  slidesCreated?: number;
  totalSlides?: number;
  fileSizeMb?: number;
}

const PresentationViewer: React.FC<PresentationViewerProps> = ({
  presentationData,
  slidesCreated,
  totalSlides,
  fileSizeMb
}) => {
  const [showSlides, setShowSlides] = useState(false);

  const handleDownload = () => {
    const binaryString = atob(presentationData.base64);
    const bytes = new Uint8Array(binaryString.length);
    for (let i = 0; i < binaryString.length; i++) {
      bytes[i] = binaryString.charCodeAt(i);
    }

    const blob = new Blob([bytes], { type: presentationData.mime_type });
    const url = URL.createObjectURL(blob);

    const link = document.createElement('a');
    link.href = url;
    link.download = presentationData.filename;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);

    URL.revokeObjectURL(url);
  };

  const openInNewTab = () => {
    const binaryString = atob(presentationData.base64);
    const bytes = new Uint8Array(binaryString.length);
    for (let i = 0; i < binaryString.length; i++) {
      bytes[i] = binaryString.charCodeAt(i);
    }

    const blob = new Blob([bytes], { type: presentationData.mime_type });
    const url = URL.createObjectURL(blob);

    window.open(url, '_blank');

    // Clean up after a delay to allow the browser to handle the file
    setTimeout(() => {
      URL.revokeObjectURL(url);
    }, 1000);
  };

  return (
    <div className="bg-white p-4 rounded-lg border shadow-sm">
      <div className="flex items-center justify-between mb-3">
        <div>
          <h3 className="text-lg font-semibold flex items-center gap-2">
            <Presentation className="w-5 h-5 text-orange-600" />
            Generated Presentation
            <span className="text-sm text-gray-600">({presentationData.filename})</span>
          </h3>
          <div className="text-sm text-gray-500 space-y-1">
            {slidesCreated && totalSlides && (
              <p>Created {slidesCreated} slides (total: {totalSlides})</p>
            )}
            {fileSizeMb && (
              <p>File size: {fileSizeMb} MB</p>
            )}
            <p>Format: PowerPoint (.pptx)</p>
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
            title="Download presentation"
          >
            <Download className="w-4 h-4" />
          </button>
        </div>
      </div>

      <div className="bg-gradient-to-br from-orange-50 to-yellow-50 border border-orange-200 rounded-lg p-6 text-center">
        <div className="flex flex-col items-center space-y-3">
          <div className="w-16 h-16 bg-orange-100 rounded-lg flex items-center justify-center">
            <Presentation className="w-8 h-8 text-orange-600" />
          </div>

          <div>
            <h4 className="font-medium text-gray-900 mb-1">PowerPoint Presentation Ready</h4>
            <p className="text-sm text-gray-600 mb-3">
              Your presentation has been generated successfully. Click the buttons above to view or download.
            </p>

            <div className="flex gap-3 justify-center">
              <button
                onClick={openInNewTab}
                className="px-4 py-2 bg-orange-600 text-white rounded hover:bg-orange-700 transition-colors text-sm font-medium"
              >
                Open Presentation
              </button>

              <button
                onClick={handleDownload}
                className="px-4 py-2 border border-orange-600 text-orange-600 rounded hover:bg-orange-50 transition-colors text-sm font-medium"
              >
                Download File
              </button>
            </div>
          </div>
        </div>
      </div>

      {/* Slide Preview Section */}
      <div className="mt-4 border-t pt-4">
        <button
          onClick={() => setShowSlides(!showSlides)}
          className="flex items-center gap-2 text-gray-700 hover:text-gray-900 font-medium"
        >
          {showSlides ? <ChevronDown className="w-4 h-4" /> : <ChevronRight className="w-4 h-4" />}
          Preview Slides ({presentationData.slides.length} slides)
        </button>

        {showSlides && (
          <div className="mt-3 space-y-3 max-h-96 overflow-y-auto">
            {presentationData.slides.map((slide, index) => (
              <div
                key={index}
                className={`p-3 rounded-lg border ${
                  slide.slide_type === 'title'
                    ? 'bg-orange-50 border-orange-200'
                    : 'bg-gray-50 border-gray-200'
                }`}
              >
                <div className="flex items-center gap-2 mb-2">
                  <span className="text-xs font-medium text-gray-500">
                    Slide {index + 1} {slide.slide_type === 'title' ? '(Title)' : ''}
                  </span>
                </div>

                <h4 className="font-semibold text-gray-900 mb-2">{slide.title}</h4>

                {slide.bullets.length > 0 && (
                  <ul className="space-y-1">
                    {slide.bullets.map((bullet, bulletIndex) => (
                      <li key={bulletIndex} className="text-sm text-gray-700 flex items-start gap-2">
                        <span className="text-gray-400 mt-1">•</span>
                        <span>{bullet}</span>
                      </li>
                    ))}
                  </ul>
                )}
              </div>
            ))}
          </div>
        )}
      </div>

      <div className="mt-3 text-xs text-gray-500 flex justify-between">
        <span>MIME type: {presentationData.mime_type}</span>
        <span>Compatible with PowerPoint, Google Slides, and other presentation software</span>
      </div>
    </div>
  );
};

export default PresentationViewer;