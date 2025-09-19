'use client';

import { useState, useRef, useEffect } from 'react';
import { Message, ToolResult } from '@/types';
import { Send, Bot, User, Loader2, Wrench, ChevronDown, ChevronRight, Paperclip, X } from 'lucide-react';
import StockChart from './StockChart';
import AnalyzedImage from './AnalyzedImage';
import PresentationViewer from './PresentationViewer';
import VisualizationChart from './VisualizationChart';

interface ChatInterfaceProps {
  messages: Message[];
  currentResponse: string;
  isLoading: boolean;
  onSendMessage: (message: string, selectedTools: string[], model: string, attachedFile?: {name: string, size: number, type: string, content?: string}, displayMessage?: string) => void;
  selectedTools: string[];
}

export default function ChatInterface({
  messages,
  currentResponse,
  isLoading,
  onSendMessage,
  selectedTools
}: ChatInterfaceProps) {
  const [inputMessage, setInputMessage] = useState('');
  const [selectedModel, setSelectedModel] = useState('gemma3:latest');
  const [collapsedToolResults, setCollapsedToolResults] = useState<Set<string | number>>(new Set());
  const [attachedFile, setAttachedFile] = useState<File | null>(null);
  const [fileContent, setFileContent] = useState<string>('');
  const [fileWarning, setFileWarning] = useState<string>('');
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  // File size limits
  const MAX_FILE_SIZE = 10 * 1024 * 1024; // 10MB
  const LARGE_FILE_WARNING = 1024 * 1024; // 1MB

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages, currentResponse]);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (inputMessage.trim() && !isLoading) {
      // Send clean message (not containing file content)
      const messageForBackend = inputMessage.trim();

      // Display message without any file information in chat - ensure it never contains file content
      const messageForDisplay = inputMessage.trim();

      // Complete file metadata with content for backend processing
      const fileMetadata = attachedFile ? {
        name: attachedFile.name,
        size: attachedFile.size,
        type: attachedFile.type,
        content: fileContent  // Include file content for backend
      } : undefined;

      onSendMessage(messageForBackend, selectedTools, selectedModel, fileMetadata, messageForDisplay);
      setInputMessage('');
      setAttachedFile(null);
      setFileContent('');
    }
  };

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      // Validate file size
      if (file.size > MAX_FILE_SIZE) {
        setFileWarning(`File too large (${(file.size / 1024 / 1024).toFixed(2)}MB). Maximum size is ${MAX_FILE_SIZE / 1024 / 1024}MB.`);
        return;
      }

      setAttachedFile(file);
      setFileWarning('');

      // Set warning for large files
      if (file.size > LARGE_FILE_WARNING) {
        setFileWarning(`Large file (${(file.size / 1024 / 1024).toFixed(2)}MB) will be truncated to 500KB for processing.`);
      }

      // Handle different file types appropriately
      const reader = new FileReader();
      reader.onload = (event) => {
        const content = event.target?.result as string;
        setFileContent(content);
      };

      reader.onerror = () => {
        setFileWarning('Error reading file. Please try a different file.');
      };

      // Check if it's an image file
      if (file.type.startsWith('image/')) {
        // For images, read as data URL (base64)
        reader.readAsDataURL(file);
      } else {
        // For text and other files, read as text
        reader.readAsText(file);
      }
    }
  };

  const removeAttachedFile = () => {
    setAttachedFile(null);
    setFileContent('');
    setFileWarning('');
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  const triggerFileSelect = () => {
    fileInputRef.current?.click();
  };

  const formatTimestamp = (timestamp: string) => {
    return new Date(timestamp).toLocaleTimeString('en-US', {
      hour: '2-digit',
      minute: '2-digit'
    });
  };

  const toggleToolResultCollapse = (key: string | number) => {
    setCollapsedToolResults(prev => {
      const newSet = new Set(prev);
      if (newSet.has(key)) {
        newSet.delete(key);
      } else {
        newSet.add(key);
      }
      return newSet;
    });
  };

  return (
    <div className="flex-1 flex flex-col bg-white">
      {/* Header */}
      <div className="border-b border-gray-200 p-4 bg-white">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <Bot className="w-6 h-6 text-blue-600" />
            <h1 className="text-xl font-semibold text-gray-900">Agent Labs</h1>
          </div>

          <div className="flex items-center gap-4">
            {selectedTools.length > 0 && (
              <div className="flex items-center gap-2 px-3 py-1 bg-blue-50 rounded-full border border-blue-200">
                <Wrench className="w-4 h-4 text-blue-600" />
                <span className="text-sm text-blue-700 font-medium">
                  {selectedTools.length} tool{selectedTools.length > 1 ? 's' : ''} selected
                </span>
              </div>
            )}

            <select
              value={selectedModel}
              onChange={(e) => setSelectedModel(e.target.value)}
              className="px-3 py-1 border border-gray-300 rounded-md text-sm bg-white text-gray-900 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
            >
              <option value="gemma3:latest">Gemma 3 (Default)</option>
              <option value="llama3.1:latest">Llama 3.1</option>
              <option value="mistral:latest">Mistral</option>
              <option value="llama3.2-vision:latest">Llama 3.2 Vision</option>
            </select>
          </div>
        </div>
      </div>

      {/* Messages */}
      <div className="flex-1 overflow-y-auto p-4 space-y-4">
        {messages.length === 0 && (
          <div className="text-center py-12">
            <Bot className="w-12 h-12 text-gray-400 mx-auto mb-4" />
            <h3 className="text-lg font-medium text-gray-900 mb-2">
              Welcome to Agent Labs
            </h3>
            <p className="text-gray-600 max-w-md mx-auto">
              Start a conversation and select tools from the sidebar to enhance your AI assistant with powerful capabilities.
            </p>
          </div>
        )}

        {messages.map((message) => (
          <div
            key={message.id}
            className={`flex gap-3 ${message.role === 'user' ? 'flex-row-reverse' : ''}`}
          >
            <div className={`flex-shrink-0 w-8 h-8 rounded-full flex items-center justify-center ${
              message.role === 'user'
                ? 'bg-blue-100 text-blue-600'
                : 'bg-gray-100 text-gray-600'
            }`}>
              {message.role === 'user' ? <User className="w-4 h-4" /> : <Bot className="w-4 h-4" />}
            </div>

            <div className={`flex-1 max-w-3xl ${message.role === 'user' ? 'text-right' : ''}`}>
              <div className={`block p-3 rounded-lg ${
                message.role === 'user'
                  ? 'bg-blue-600 text-white max-w-md ml-auto'
                  : 'bg-gray-100 text-gray-900 max-w-4xl'
              }`}>
                <pre className="whitespace-pre-wrap break-words overflow-hidden font-sans text-sm leading-relaxed">{message.content}</pre>
              </div>

              <div className={`text-xs text-gray-500 mt-1 ${message.role === 'user' ? 'text-right' : ''}`}>
                {formatTimestamp(message.timestamp)}
              </div>

              {/* Show attached file for user messages */}
              {message.role === 'user' && message.attachedFile && (
                <div className="mt-2 text-right">
                  <div className="inline-flex items-center gap-1 px-2 py-1 text-xs bg-gray-100 text-gray-700 rounded">
                    <Paperclip className="w-3 h-3" />
                    <span>{message.attachedFile.name}</span>
                  </div>
                </div>
              )}

              {/* Show selected tools for user messages */}
              {message.role === 'user' && message.tools && message.tools.length > 0 && (
                <div className="mt-2 text-right">
                  <div className="inline-flex flex-wrap gap-1">
                    {message.tools.map((tool) => (
                      <span
                        key={tool}
                        className="inline-block px-2 py-1 text-xs bg-blue-100 text-blue-700 rounded"
                      >
                        {tool.replace(/_/g, ' ')}
                      </span>
                    ))}
                  </div>
                </div>
              )}

              {/* Show tool results inline for assistant messages */}
              {message.role === 'assistant' && message.toolResults && message.toolResults.length > 0 && (
                <div className="mt-3 space-y-2">
                  {message.toolResults.map((result, index) => {
                    const isCollapsed = collapsedToolResults.has(`${message.id}-${index}`);
                    return (
                      <div key={`${message.id}-${index}`} className="rounded-lg bg-green-50 border border-green-200">
                        <div
                          className="p-3 cursor-pointer hover:bg-green-100 transition-colors"
                          onClick={() => toggleToolResultCollapse(`${message.id}-${index}`)}
                        >
                          <div className="flex items-center justify-between">
                            <div className="flex items-center gap-2">
                              <Wrench className="w-4 h-4 text-green-600" />
                              <div className="text-sm font-medium text-green-800">
                                Tool: {result.tool.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase())}
                              </div>
                            </div>
                            <div className="flex items-center gap-2">
                              <div className="text-xs text-green-600">
                                {formatTimestamp(result.timestamp)}
                              </div>
                              {isCollapsed ? (
                                <ChevronRight className="w-4 h-4 text-green-600" />
                              ) : (
                                <ChevronDown className="w-4 h-4 text-green-600" />
                              )}
                            </div>
                          </div>
                        </div>

                        {!isCollapsed && (
                          <div className="px-3 pb-3">
                            {result.summary && (
                              <div className="text-sm text-green-800 bg-green-100 p-3 rounded-lg mb-3 border-t border-green-200">
                                <div className="font-medium mb-1">Summary:</div>
                                <p className="whitespace-pre-wrap">{result.summary}</p>
                              </div>
                            )}

                            {/* Display stock chart if available */}
                            {result.tool === 'stock_analysis' && result.result?.stock_chart_data && (
                              <div className="mb-3">
                                <StockChart chartData={result.result.stock_chart_data} />
                              </div>
                            )}

                            {/* Display analyzed image if available */}
                            {result.tool === 'image_analysis' && result.result?.image_data && (
                              <div className="mb-3">
                                <AnalyzedImage
                                  imageData={result.result.image_data}
                                  filename={result.result.file_info?.filename}
                                  fileSize={result.result.file_info?.file_size_mb}
                                />
                              </div>
                            )}
                            {/* Display generated presentation if available */}
                            {result.tool === 'presentation' && result.result?.presentation_data && (
                              <div className="mb-3">
                                <PresentationViewer
                                  presentationData={result.result.presentation_data}
                                  slidesCreated={result.result.slides_created}
                                  totalSlides={result.result.total_slides}
                                  fileSizeMb={result.result.file_size_mb}
                                />
                              </div>
                            )}

                            {/* Display visualization chart if available */}
                            {result.tool === 'visualization' && result.result && (
                              <div className="mb-3">
                                <VisualizationChart
                                  visualizationData={result.result}
                                />
                              </div>
                            )}

                          </div>
                        )}
                      </div>
                    );
                  })}
                </div>
              )}
            </div>
          </div>
        ))}

        {/* Current response */}
        {currentResponse && (
          <div className="flex gap-3">
            <div className="flex-shrink-0 w-8 h-8 rounded-full bg-gray-100 text-gray-600 flex items-center justify-center">
              <Bot className="w-4 h-4" />
            </div>
            <div className="flex-1 max-w-3xl">
              <div className="block p-3 rounded-lg bg-gray-100 text-gray-900 max-w-4xl">
                <pre className="whitespace-pre-wrap break-words overflow-hidden font-sans text-sm leading-relaxed">{currentResponse}</pre>
                <div className="inline-block w-2 h-4 bg-gray-400 animate-pulse ml-1"></div>
              </div>
            </div>
          </div>
        )}

        {/* Tool results are now displayed inline with messages above */}

        {/* Loading indicator */}
        {isLoading && !currentResponse && (
          <div className="flex gap-3">
            <div className="flex-shrink-0 w-8 h-8 rounded-full bg-gray-100 text-gray-600 flex items-center justify-center">
              <Loader2 className="w-4 h-4 animate-spin" />
            </div>
            <div className="flex-1 max-w-3xl">
              <div className="inline-block p-3 rounded-lg bg-gray-100 text-gray-600">
                <div className="flex items-center gap-2">
                  <span>Thinking...</span>
                  <div className="flex gap-1">
                    <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce"></div>
                    <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{animationDelay: '0.1s'}}></div>
                    <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{animationDelay: '0.2s'}}></div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        )}

        <div ref={messagesEndRef} />
      </div>

      {/* Input */}
      <div className="border-t border-gray-200 p-4 bg-white">
        {/* File attachment preview */}
        {attachedFile && (
          <div className={`mb-3 p-3 border rounded-lg ${fileWarning ? 'bg-yellow-50 border-yellow-200' : 'bg-blue-50 border-blue-200'}`}>
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                <Paperclip className={`w-4 h-4 ${fileWarning ? 'text-yellow-600' : 'text-blue-600'}`} />
                <span className={`text-sm font-medium ${fileWarning ? 'text-yellow-800' : 'text-blue-800'}`}>
                  {attachedFile.name} ({(attachedFile.size / 1024).toFixed(1)}KB)
                </span>
              </div>
              <button
                onClick={removeAttachedFile}
                className={`p-1 rounded-full ${fileWarning ? 'hover:bg-yellow-100' : 'hover:bg-blue-100'}`}
              >
                <X className={`w-4 h-4 ${fileWarning ? 'text-yellow-600' : 'text-blue-600'}`} />
              </button>
            </div>
            {fileWarning ? (
              <div className="mt-2 text-xs text-yellow-700 font-medium">
                ⚠️ {fileWarning}
              </div>
            ) : (
              <div className="mt-2 text-xs text-blue-600">
                File content will be included in your message for data processing
              </div>
            )}
          </div>
        )}

        <form onSubmit={handleSubmit} className="flex gap-2">
          <input
            type="file"
            ref={fileInputRef}
            onChange={handleFileSelect}
            accept=".txt,.csv,.json,.md,.log,.jpg,.jpeg,.png,.gif,.bmp,.tiff,.webp,.svg,.ppt,.pptx,.pdf"
            className="hidden"
          />

          <button
            type="button"
            onClick={triggerFileSelect}
            disabled={isLoading}
            className="px-3 py-2 border border-gray-300 rounded-lg hover:bg-gray-50 focus:outline-none focus:ring-2 focus:ring-blue-500 disabled:opacity-50 disabled:cursor-not-allowed flex items-center gap-1"
            title="Attach file for data processing"
          >
            <Paperclip className="w-4 h-4 text-gray-600" />
          </button>

          <input
            type="text"
            value={inputMessage}
            onChange={(e) => setInputMessage(e.target.value)}
            placeholder={attachedFile ? "Describe what you want to do with the attached file (e.g., analyze data, create a chart, generate presentation)..." : "Type your message..."}
            className="flex-1 px-4 py-2 border border-gray-300 rounded-lg bg-white text-gray-900 placeholder-gray-500 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500 disabled:bg-gray-50 disabled:text-gray-500"
            disabled={isLoading}
          />
          <button
            type="submit"
            disabled={!inputMessage.trim() || isLoading}
            className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-blue-500 disabled:opacity-50 disabled:cursor-not-allowed flex items-center gap-2"
          >
            {isLoading ? (
              <Loader2 className="w-4 h-4 animate-spin" />
            ) : (
              <Send className="w-4 h-4" />
            )}
          </button>
        </form>
      </div>
    </div>
  );
}