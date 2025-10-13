'use client';

import { useState, useRef, useEffect } from 'react';
import { Message, ToolResult, Tool } from '@/types';
import { apiUrl } from '@/utils/api';
import { getDefaultLLMConfig, getAvailableModels } from '@/utils/llm';
import { Send, Bot, User, Loader2, Wrench, ChevronDown, ChevronRight, Paperclip, X } from 'lucide-react';
import AnalyzedImage from './AnalyzedImage';
import PresentationViewer from './PresentationViewer';
import ProcessedFileViewer from './ProcessedFileViewer';
import ForecastViewer from './ForecastViewer';
import CostAnalysisViewer from './CostAnalysisViewer';
import VisualizationChart from './VisualizationChart';
import FlightCard from './FlightCard';
import HotelCard from './HotelCard';
import WebSearchCard from './WebSearchCard';

interface ChatInterfaceProps {
  messages: Message[];
  currentResponse: string;
  isLoading: boolean;
  onSendMessage: (message: string, selectedTools: string[], model: string, attachedFile?: {name: string, size: number, type: string, content?: string}, displayMessage?: string) => void;
  selectedTools: string[];
  onToolsChange: (tools: string[]) => void;
}

export default function ChatInterface({
  messages,
  currentResponse,
  isLoading,
  onSendMessage,
  selectedTools,
  onToolsChange
}: ChatInterfaceProps) {
  const [inputMessage, setInputMessage] = useState('');
  const [selectedModel, setSelectedModel] = useState(() => {
    const defaultConfig = getDefaultLLMConfig();
    return `${defaultConfig.provider}/${defaultConfig.model}`;
  });
  const [availableModels, setAvailableModels] = useState<Array<{name: string, provider: string, model: string}>>([]);
  const [expandedToolResults, setExpandedToolResults] = useState<Set<string | number>>(new Set());
  const [attachedFile, setAttachedFile] = useState<File | null>(null);
  const [fileContent, setFileContent] = useState<string>('');
  const [fileWarning, setFileWarning] = useState<string>('');
  const [tools, setTools] = useState<Record<string, Tool>>({});
  const [showMentionDropdown, setShowMentionDropdown] = useState(false);
  const [mentionSearch, setMentionSearch] = useState('');
  const [mentionPosition, setMentionPosition] = useState({ top: 0, left: 0 });
  const [cursorPosition, setCursorPosition] = useState(0);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);
  const mentionDropdownRef = useRef<HTMLDivElement>(null);

  // File size limits
  const MAX_FILE_SIZE = 10 * 1024 * 1024; // 10MB
  const LARGE_FILE_WARNING = 1024 * 1024; // 1MB

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages, currentResponse]);

  // Fetch available tools on component mount
  useEffect(() => {
    const fetchTools = async () => {
      try {
        const response = await fetch(apiUrl('/api/tools'));
        const data = await response.json();
        setTools(data.tools);
      } catch (error) {
        console.error('Failed to fetch tools:', error);
      }
    };

    fetchTools();
  }, []);

  // Fetch available models on component mount
  useEffect(() => {
    const fetchModels = async () => {
      try {
        const response = await fetch(apiUrl('/api/models'));
        const data = await response.json();
        if (data.models) {
          setAvailableModels(data.models);
        }
      } catch (error) {
        console.error('Failed to fetch models:', error);
        // Fallback to default models from utility
        const availableModelsByProvider = getAvailableModels();
        const fallbackModels = [];

        for (const [provider, models] of Object.entries(availableModelsByProvider)) {
          for (const model of models) {
            fallbackModels.push({
              name: `${provider}/${model}`,
              provider,
              model
            });
          }
        }

        setAvailableModels(fallbackModels);
      }
    };

    fetchModels();
  }, []);

  // Send default model selection to backend on mount
  useEffect(() => {
    const defaultConfig = getDefaultLLMConfig();
    const defaultModel = `${defaultConfig.provider}/${defaultConfig.model}`;
    handleModelChange(defaultModel);
  }, []); // Only run once on mount

  // Handle model selection change
  const handleModelChange = async (newModel: string) => {
    setSelectedModel(newModel);

    // Send model selection to backend
    try {
      await fetch(apiUrl('/api/select-model'), {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ model: newModel }),
      });
    } catch (error) {
      console.error('Failed to update model selection:', error);
    }
  };

  // Parse @ mentions from message and extract tool IDs
  const extractMentionedTools = (text: string): string[] => {
    const mentionRegex = /@(\w+(?:[:\-_]\w+)*)/g;
    const mentionedTools: string[] = [];
    let match;

    while ((match = mentionRegex.exec(text)) !== null) {
      const mentionedName = match[1].toLowerCase();
      // Find matching tool by ID or name
      const tool = Object.values(tools).find(t =>
        t.id.toLowerCase() === mentionedName ||
        t.name.toLowerCase().replace(/\s+/g, '_') === mentionedName ||
        t.name.toLowerCase().replace(/\s+/g, '-') === mentionedName ||
        t.name.toLowerCase().replace(/\s+/g, '') === mentionedName
      );

      if (tool && !mentionedTools.includes(tool.id)) {
        mentionedTools.push(tool.id);
      }
    }

    return mentionedTools;
  };

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (inputMessage.trim() && !isLoading) {
      // Extract mentioned tools from message
      const mentionedTools = extractMentionedTools(inputMessage);

      // If tools are mentioned with @, use ONLY those tools (don't merge with sidebar selection)
      // Otherwise, use the sidebar selected tools
      const allTools = mentionedTools.length > 0 ? mentionedTools : selectedTools;

      // Auto-select mentioned tools in the sidebar for visual feedback
      if (mentionedTools.length > 0) {
        onToolsChange(mentionedTools);
      }

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

      onSendMessage(messageForBackend, allTools, selectedModel, fileMetadata, messageForDisplay);
      setInputMessage('');
      setAttachedFile(null);
      setFileContent('');
      setShowMentionDropdown(false);
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

  // Format message content with @ mentions highlighted
  const formatMessageWithMentions = (text: string) => {
    const mentionRegex = /@(\w+(?:[:\-_]\w+)*)/g;
    const parts = [];
    let lastIndex = 0;
    let match;

    while ((match = mentionRegex.exec(text)) !== null) {
      // Add text before mention
      if (match.index > lastIndex) {
        parts.push(text.substring(lastIndex, match.index));
      }

      // Add mention with highlighting
      const mentionedToolId = match[1];
      const tool = Object.values(tools).find(t => t.id === mentionedToolId);

      parts.push(
        <span
          key={match.index}
          className="inline-flex items-center gap-1 px-2 py-0.5 bg-blue-600 text-white rounded font-medium"
          title={tool ? tool.description : mentionedToolId}
        >
          <Wrench className="w-3 h-3" />
          @{mentionedToolId}
        </span>
      );

      lastIndex = match.index + match[0].length;
    }

    // Add remaining text
    if (lastIndex < text.length) {
      parts.push(text.substring(lastIndex));
    }

    return parts.length > 0 ? parts : text;
  };

  const toggleToolResultExpand = (key: string | number) => {
    setExpandedToolResults(prev => {
      const newSet = new Set(prev);
      if (newSet.has(key)) {
        newSet.delete(key);
      } else {
        newSet.add(key);
      }
      return newSet;
    });
  };

  // Handle input change and detect @ mentions
  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const value = e.target.value;
    const cursorPos = e.target.selectionStart || 0;
    setInputMessage(value);
    setCursorPosition(cursorPos);

    // Check if user typed @ to trigger mention dropdown
    const textBeforeCursor = value.substring(0, cursorPos);
    const lastAtIndex = textBeforeCursor.lastIndexOf('@');

    if (lastAtIndex !== -1) {
      const textAfterAt = textBeforeCursor.substring(lastAtIndex + 1);
      // Only show dropdown if @ is at start or after whitespace, and no space after @
      const charBeforeAt = lastAtIndex > 0 ? textBeforeCursor[lastAtIndex - 1] : ' ';
      if ((charBeforeAt === ' ' || charBeforeAt === '\n' || lastAtIndex === 0) && !textAfterAt.includes(' ')) {
        setMentionSearch(textAfterAt.toLowerCase());
        setShowMentionDropdown(true);

        // Calculate dropdown position
        if (inputRef.current) {
          const inputRect = inputRef.current.getBoundingClientRect();
          setMentionPosition({
            top: inputRect.top - 200, // Show above input
            left: inputRect.left + 10
          });
        }
      } else {
        setShowMentionDropdown(false);
      }
    } else {
      setShowMentionDropdown(false);
    }
  };

  // Filter tools based on mention search
  const filteredTools = Object.values(tools).filter(tool =>
    tool.name.toLowerCase().includes(mentionSearch) ||
    tool.id.toLowerCase().includes(mentionSearch) ||
    tool.description.toLowerCase().includes(mentionSearch)
  );

  // Handle tool selection from mention dropdown
  const handleMentionSelect = (toolId: string) => {
    const textBeforeCursor = inputMessage.substring(0, cursorPosition);
    const textAfterCursor = inputMessage.substring(cursorPosition);
    const lastAtIndex = textBeforeCursor.lastIndexOf('@');

    if (lastAtIndex !== -1) {
      const newText = textBeforeCursor.substring(0, lastAtIndex) + `@${toolId} ` + textAfterCursor;
      setInputMessage(newText);
      setShowMentionDropdown(false);

      // Focus back on input
      setTimeout(() => {
        if (inputRef.current) {
          const newCursorPos = lastAtIndex + toolId.length + 2;
          inputRef.current.focus();
          inputRef.current.setSelectionRange(newCursorPos, newCursorPos);
        }
      }, 0);
    }
  };

  // Close mention dropdown when clicking outside
  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (mentionDropdownRef.current && !mentionDropdownRef.current.contains(event.target as Node)) {
        setShowMentionDropdown(false);
      }
    };

    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, []);

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
              onChange={(e) => handleModelChange(e.target.value)}
              className="px-3 py-1 border border-gray-300 rounded-md text-sm bg-white text-gray-900 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
            >
              {availableModels.length > 0 ? (
                availableModels.map((model) => (
                  <option key={model.name} value={model.name}>
                    {model.provider === 'openai' && `🤖 OpenAI ${model.model}`}
                    {model.provider === 'gemini' && `💎 Gemini ${model.model}`}
                    {model.provider === 'ollama' && `🦙 ${model.model}`}
                  </option>
                ))
              ) : (
                <option value="ollama/gemma3:latest">🦙 Gemma 3 (Loading...)</option>
              )}
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
              <div className={`block rounded-lg ${
                message.role === 'user'
                  ? 'bg-blue-600 text-white max-w-md ml-auto px-3 py-2'
                  : 'max-w-4xl'
              }`}>
                {message.role === 'assistant' ? (
                  <div className="text-sm text-gray-800 bg-gray-100 p-3 rounded-lg mb-3 border-t border-gray-200">
                    <p className="whitespace-pre-wrap">{message.content}</p>
                  </div>
                ) : (
                  <p className="whitespace-pre-wrap break-words overflow-hidden text-sm leading-relaxed">
                    {formatMessageWithMentions(message.content)}
                  </p>
                )}
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
                    const hasSpecialViewer = ['image_analysis', 'presentation', 'data_processing', 'forecast', 'cost_analysis', 'visualization', 'flight_search', 'hotel_search', 'web_search'].includes(result.tool);
                    const resultKey = `${message.id}-${index}`;
                    // Tools with special viewers start expanded by default (not in the set)
                    // When toggled, they're added to the set with opposite state
                    // If not in set: special viewers are expanded, others are collapsed
                    // If in set: it has been toggled, so show opposite of default
                    const defaultState = hasSpecialViewer;
                    const isExpanded = expandedToolResults.has(resultKey) ? !defaultState : defaultState;
                    return (
                      <div key={`${message.id}-${index}`} className="rounded-lg bg-green-50 border border-green-200">
                        <div
                          className="p-3 cursor-pointer hover:bg-green-100 transition-colors"
                          onClick={() => toggleToolResultExpand(resultKey)}
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
                              {isExpanded ? (
                                <ChevronDown className="w-4 h-4 text-green-600" />
                              ) : (
                                <ChevronRight className="w-4 h-4 text-green-600" />
                              )}
                            </div>
                          </div>
                        </div>

                        {isExpanded && (
                          <div className="px-3 pb-3">


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

                            {/* Display processed file if available */}
                            {result.tool === 'data_processing' && result.result?.processing_data && (
                              <div className="mb-3">
                                <ProcessedFileViewer
                                  processingData={result.result.processing_data}
                                  operation={result.result.operation || 'unknown'}
                                  fileSizeMb={result.result.file_size_mb}
                                  rowsConverted={result.result.rows_converted || result.result.row_count}
                                  originalLines={result.result.original_lines}
                                  uniqueLines={result.result.unique_lines}
                                  emailsFound={result.result.emails_found || result.result.unique_matches}
                                  urlsFound={result.result.urls_found || result.result.unique_matches}
                                />
                              </div>
                            )}

                            {/* Display forecast file if available */}
                            {result.tool === 'forecast' && result.result?.forecast_file_data && (
                              <div className="mb-3">
                                <ForecastViewer
                                  forecastFileData={result.result.forecast_file_data}
                                  forecastPeriods={result.result.data_info?.forecast_periods}
                                  historicalPoints={result.result.data_info?.historical_points}
                                  fileSizeMb={result.result.file_size_mb}
                                  modelMetrics={result.result.model_metrics}
                                />
                              </div>
                            )}

                            {/* Display cost analysis file if available */}
                            {result.tool === 'cost_analysis' && result.result?.cost_analysis_file_data && (
                              <div className="mb-3">
                                <CostAnalysisViewer
                                  costAnalysisFileData={result.result.cost_analysis_file_data}
                                  fileSizeMb={result.result.file_size_mb}
                                />
                              </div>
                            )}

                            {/* Display visualization chart if available */}
                            {result.tool === 'visualization' && result.result?.chart_html && (
                              <div className="mb-3">
                                <VisualizationChart
                                  visualizationData={result.result}
                                />
                              </div>
                            )}

                            {/* Display flight search results if available */}
                            {result.tool === 'flight_search' && result.result?.flights && (
                              <div className="mb-3">
                                <FlightCard
                                  flights={result.result.flights}
                                  query={result.result.query}
                                  resultsCount={result.result.results_count || 0}
                                />
                              </div>
                            )}

                            {/* Display hotel search results if available */}
                            {result.tool === 'hotel_search' && result.result?.hotels && (
                              <div className="mb-3">
                                <HotelCard
                                  hotels={result.result.hotels}
                                  query={result.result.query}
                                  resultsCount={result.result.results_count || 0}
                                />
                              </div>
                            )}

                            {/* Display web search results if available */}
                            {result.tool === 'web_search' && result.result?.results && (
                              <div className="mb-3">
                                <WebSearchCard
                                  searchResults={result.result.results}
                                  query={result.result.query}
                                  resultsCount={result.result.results_count || result.result.results.length}
                                />
                              </div>
                            )}

                            {/* Default/fallback renderer for tools without specific handlers */}
                            {!['image_analysis', 'presentation', 'data_processing', 'forecast', 'cost_analysis', 'visualization', 'flight_search', 'hotel_search', 'web_search'].includes(result.tool) && result.result && (
                              <div className="mb-3">
                                <div className="bg-green-50 p-4 rounded-lg border border-green-200">
                                  <pre className="text-xs text-green-800 whitespace-pre-wrap font-sans">
                                    {(() => {
                                      // Handle deeply nested JSON strings (recursive parsing)
                                      let data = result.result.tool_data || result.result;

                                      const parseRecursively = (obj: any): any => {
                                        if (typeof obj === 'string') {
                                          try {
                                            const parsed = JSON.parse(obj);
                                            return parseRecursively(parsed); // Recursively parse
                                          } catch (e) {
                                            return obj; // If not JSON, return as string
                                          }
                                        } else if (typeof obj === 'object' && obj !== null) {
                                          // Parse each property recursively
                                          const result: any = {};
                                          for (const [key, value] of Object.entries(obj)) {
                                            result[key] = parseRecursively(value);
                                          }
                                          return result;
                                        }
                                        return obj;
                                      };

                                      data = parseRecursively(data);
                                      return typeof data === 'object' ? JSON.stringify(data, null, 2) : data;
                                    })()}
                                  </pre>
                                </div>
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

          <div className="flex-1 relative">
            <input
              ref={inputRef}
              type="text"
              value={inputMessage}
              onChange={handleInputChange}
              placeholder={attachedFile ? "Describe what you want to do with the attached file (e.g., analyze data, create a chart, generate presentation)... Type @ to mention tools" : "Type your message... Use @ to mention tools"}
              className="w-full px-4 py-2 border border-gray-300 rounded-lg bg-white text-gray-900 placeholder-gray-500 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500 disabled:bg-gray-50 disabled:text-gray-500"
              disabled={isLoading}
            />

            {/* @ Mention Dropdown */}
            {showMentionDropdown && filteredTools.length > 0 && (
              <div
                ref={mentionDropdownRef}
                className="absolute bottom-full left-0 mb-2 w-full max-w-md bg-white border border-gray-200 rounded-lg shadow-lg max-h-64 overflow-y-auto z-50"
              >
                <div className="p-2 border-b border-gray-100 bg-gray-50">
                  <p className="text-xs text-gray-600 font-medium">Select a tool to mention</p>
                </div>
                <div className="py-1">
                  {filteredTools.slice(0, 10).map((tool) => (
                    <button
                      key={tool.id}
                      onClick={() => handleMentionSelect(tool.id)}
                      className="w-full px-3 py-2 text-left hover:bg-blue-50 focus:bg-blue-50 focus:outline-none transition-colors"
                    >
                      <div className="flex items-start gap-2">
                        <Wrench className="w-4 h-4 text-blue-600 mt-0.5 flex-shrink-0" />
                        <div className="flex-1 min-w-0">
                          <div className="text-sm font-medium text-gray-900 truncate">
                            @{tool.id}
                          </div>
                          <div className="text-xs text-gray-600 truncate">
                            {tool.name}
                          </div>
                          <div className="text-xs text-gray-500 line-clamp-1 mt-0.5">
                            {tool.description}
                          </div>
                        </div>
                      </div>
                    </button>
                  ))}
                  {filteredTools.length > 10 && (
                    <div className="px-3 py-2 text-xs text-gray-500 text-center border-t border-gray-100">
                      {filteredTools.length - 10} more tools available
                    </div>
                  )}
                </div>
              </div>
            )}
          </div>
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