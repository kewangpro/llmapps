'use client';

import { useState, useMemo, useEffect } from 'react';
import { useWebSocket } from '@/hooks/useWebSocket';
import { getWebSocketUrl } from '@/utils/api';
import ChatInterface from '@/components/ChatInterface';
import ToolSidebar from '@/components/ToolSidebar';

export default function Home() {
  const [selectedTools, setSelectedTools] = useState<string[]>([]);
  const [wsUrl, setWsUrl] = useState<string>('');

  // Use stable client ID to prevent reconnection loops
  const clientId = useMemo(() =>
    'client-' + Math.random().toString(36).substr(2, 9),
    []
  );

  // Set WebSocket URL on client side only
  useEffect(() => {
    setWsUrl(getWebSocketUrl());
  }, []);

  const {
    isConnected,
    messages,
    currentResponse,
    isLoading,
    sendMessage,
    reconnect
  } = useWebSocket({
    url: wsUrl || '',
    clientId
  });

  const handleToolToggle = (toolName: string) => {
    setSelectedTools(prev =>
      prev.includes(toolName)
        ? prev.filter(t => t !== toolName)
        : [...prev, toolName]
    );
  };

  const handleSendMessage = (message: string, tools: string[], model: string, attachedFile?: {name: string, size: number, type: string, content?: string}, displayMessage?: string) => {
    sendMessage(message, tools, model, attachedFile, displayMessage);
  };

  return (
    <div className="h-screen flex bg-gray-100">
      <ToolSidebar
        selectedTools={selectedTools}
        onToolToggle={handleToolToggle}
      />

      <ChatInterface
        messages={messages}
        currentResponse={currentResponse}
        isLoading={isLoading}
        onSendMessage={handleSendMessage}
        selectedTools={selectedTools}
      />

      {/* Connection status indicator */}
      <div className="fixed bottom-20 right-4 z-10">
        <div className={`px-3 py-1 rounded-full text-xs font-medium shadow-lg ${
          isConnected
            ? 'bg-green-100 text-green-800 border border-green-200'
            : 'bg-red-100 text-red-800 border border-red-200'
        }`}>
          {isConnected ? 'Connected' : 'Disconnected'}
        </div>
        {!isConnected && (
          <button
            onClick={reconnect}
            className="mt-2 px-3 py-1 bg-blue-600 text-white text-xs rounded-md hover:bg-blue-700 shadow-lg"
          >
            Reconnect
          </button>
        )}
      </div>
    </div>
  );
}