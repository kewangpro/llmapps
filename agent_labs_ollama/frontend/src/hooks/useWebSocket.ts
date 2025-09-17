'use client';

import { useState, useEffect, useRef, useCallback } from 'react';
import { Message, WebSocketMessage, ToolResult } from '@/types';

interface UseWebSocketProps {
  url: string;
  clientId: string;
}

export const useWebSocket = ({ url, clientId }: UseWebSocketProps) => {
  const [isConnected, setIsConnected] = useState(false);
  const [messages, setMessages] = useState<Message[]>([]);
  const [currentResponse, setCurrentResponse] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [toolResults, setToolResults] = useState<ToolResult[]>([]);

  const wsRef = useRef<WebSocket | null>(null);
  const currentResponseRef = useRef('');
  const messagesRef = useRef<Message[]>([]);

  const connect = useCallback(() => {
    // Close existing connection if any
    if (wsRef.current) {
      wsRef.current.close();
    }

    try {
      const ws = new WebSocket(`${url}/${clientId}`);

      ws.onopen = () => {
        setIsConnected(true);
        console.log('WebSocket connected - NOT clearing messages');
      };

      ws.onmessage = (event) => {
        const data: WebSocketMessage = JSON.parse(event.data);

        switch (data.type) {
          case 'message_received':
            setIsLoading(true);
            break;

          case 'assistant_response_start':
            currentResponseRef.current = '';
            setCurrentResponse('');
            break;

          case 'assistant_response_chunk':
            currentResponseRef.current += data.content || '';
            setCurrentResponse(currentResponseRef.current);
            break;

          case 'assistant_response_complete':
            console.log('Saving assistant message:', currentResponseRef.current);
            const newMessage = {
              id: Date.now().toString(),
              content: currentResponseRef.current,
              role: 'assistant' as const,
              timestamp: data.timestamp
            };

            // Update ref first
            messagesRef.current = [...messagesRef.current, newMessage];
            console.log('Updated messagesRef, length:', messagesRef.current.length);

            // Then update state
            setMessages(messagesRef.current);
            console.log('Updated state with message:', newMessage.content);

            setCurrentResponse('');
            currentResponseRef.current = '';
            setIsLoading(false);
            break;

          case 'tool_execution_start':
            console.log('Tool execution started');
            break;

          case 'tool_result':
            const newToolResult: ToolResult = {
              tool: data.tool || '',
              result: data.result,
              timestamp: data.timestamp
            };
            setToolResults(prev => [...prev, newToolResult]);
            break;

          case 'tool_summary':
            setToolResults(prev => {
              // Find the last tool result with the same tool name and add the summary
              const updatedResults = [...prev];
              for (let i = updatedResults.length - 1; i >= 0; i--) {
                if (updatedResults[i].tool === data.tool) {
                  updatedResults[i] = {
                    ...updatedResults[i],
                    summary: data.summary
                  };
                  break;
                }
              }
              return updatedResults;
            });
            break;

          case 'agent_response':
            console.log('Saving agent response:', data.content);
            const agentMessage = {
              id: Date.now().toString(),
              content: data.content || '',
              role: 'assistant' as const,
              timestamp: data.timestamp
            };

            // Update ref first
            messagesRef.current = [...messagesRef.current, agentMessage];
            console.log('Updated messagesRef with agent response, length:', messagesRef.current.length);

            // Then update state
            setMessages(messagesRef.current);
            console.log('Updated state with agent message:', agentMessage.content);
            break;

          case 'error':
            console.error('WebSocket error:', data.message);
            setIsLoading(false);
            break;
        }
      };

      ws.onclose = (event) => {
        setIsConnected(false);
        setIsLoading(false);
        console.log('WebSocket disconnected:', event.code, event.reason);

        // Don't auto-reconnect immediately to prevent loops
        // Let user manually reconnect if needed
      };

      ws.onerror = (error) => {
        console.error('WebSocket error:', error);
        setIsConnected(false);
        setIsLoading(false);
      };

      wsRef.current = ws;
    } catch (error) {
      console.error('Failed to connect WebSocket:', error);
    }
  }, [url, clientId]);

  const disconnect = useCallback(() => {
    if (wsRef.current) {
      wsRef.current.close();
      wsRef.current = null;
    }
  }, []);

  const sendMessage = useCallback((message: string, selectedTools: string[] = [], model: string = 'gemma3:latest', attachedFile?: {name: string, size: number, type: string}, displayMessage?: string) => {
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      const userMessage: Message = {
        id: Date.now().toString(),
        content: displayMessage || message, // Use displayMessage for chat, full message for backend
        role: 'user',
        timestamp: new Date().toISOString(),
        tools: selectedTools,
        attachedFile
      };

      // Update ref first
      messagesRef.current = [...messagesRef.current, userMessage];
      // Then update state
      setMessages(messagesRef.current);

      wsRef.current.send(JSON.stringify({
        message,
        tools: selectedTools,
        model,
        attachedFile
      }));
    }
  }, []);


  useEffect(() => {
    connect();
    return () => disconnect();
  }, [url, clientId]); // Remove connect/disconnect dependencies to prevent loops

  return {
    isConnected,
    messages,
    currentResponse,
    isLoading,
    toolResults,
    sendMessage,
    connect,
    disconnect
  };
};