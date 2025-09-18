'use client';

import { useState, useEffect, useRef, useCallback, useReducer } from 'react';
import { Message, WebSocketMessage, ToolResult } from '@/types';
import { messageReducer, initialMessageState, MessageAction } from './messageReducer';

interface UseWebSocketProps {
  url: string;
  clientId: string;
}

export const useWebSocket = ({ url, clientId }: UseWebSocketProps) => {
  const [isConnected, setIsConnected] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [messageState, dispatchMessage] = useReducer(messageReducer, initialMessageState);

  const wsRef = useRef<WebSocket | null>(null);
  const currentResponseRef = useRef('');
  const reconnectTimeoutRef = useRef<NodeJS.Timeout | null>(null);
  const reconnectAttemptsRef = useRef(0);
  const isConnectingRef = useRef(false);
  const maxReconnectAttempts = 3; // Reduced from 5 to prevent spam
  const mountedRef = useRef(true);

  const connect = useCallback(() => {
    // Prevent multiple simultaneous connection attempts
    if (isConnectingRef.current) {
      console.log('Connection attempt already in progress, skipping');
      return;
    }

    // Close existing connection if any
    if (wsRef.current) {
      wsRef.current.close();
    }

    // Clear any pending reconnection
    if (reconnectTimeoutRef.current) {
      clearTimeout(reconnectTimeoutRef.current);
      reconnectTimeoutRef.current = null;
    }

    isConnectingRef.current = true;

    try {
      const ws = new WebSocket(`${url}/${clientId}`);

      ws.onopen = () => {
        setIsConnected(true);
        reconnectAttemptsRef.current = 0; // Reset reconnect attempts on successful connection
        isConnectingRef.current = false;
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
            dispatchMessage({ type: 'START_RESPONSE' });
            break;

          case 'assistant_response_chunk':
            currentResponseRef.current += data.content || '';
            dispatchMessage({
              type: 'UPDATE_CURRENT_RESPONSE',
              payload: currentResponseRef.current
            });
            break;

          case 'assistant_response_complete':
            console.log('Response complete:', currentResponseRef.current);
            // Always convert current response to message immediately
            if (currentResponseRef.current) {
              dispatchMessage({
                type: 'ADD_ASSISTANT_RESPONSE',
                payload: {
                  content: currentResponseRef.current,
                  timestamp: data.timestamp || new Date().toISOString()
                }
              });
              currentResponseRef.current = '';
            }
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
            dispatchMessage({
              type: 'ADD_TOOL_RESULT',
              payload: { toolResult: newToolResult }
            });
            break;

          case 'tool_summary':
            dispatchMessage({
              type: 'ADD_TOOL_SUMMARY',
              payload: { tool: data.tool || '', summary: data.summary || '' }
            });
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
        isConnectingRef.current = false;
        console.log('WebSocket disconnected:', event.code, event.reason);

        // Only auto-reconnect for unexpected disconnections if component is still mounted
        if (mountedRef.current && event.code !== 1000 && event.code !== 1001 && reconnectAttemptsRef.current < maxReconnectAttempts) {
          reconnectAttemptsRef.current += 1;
          const delay = Math.min(2000 * Math.pow(2, reconnectAttemptsRef.current - 1), 15000); // Start with 2s, max 15s
          console.log(`Attempting to reconnect in ${delay}ms (attempt ${reconnectAttemptsRef.current}/${maxReconnectAttempts})`);

          reconnectTimeoutRef.current = setTimeout(() => {
            if (mountedRef.current) {
              connect();
            }
          }, delay);
        } else if (reconnectAttemptsRef.current >= maxReconnectAttempts) {
          console.log('Max reconnection attempts reached. Please refresh the page to reconnect.');
        }
      };

      ws.onerror = (error) => {
        console.error('WebSocket error:', error);
        setIsConnected(false);
        setIsLoading(false);
        isConnectingRef.current = false;
      };

      wsRef.current = ws;
    } catch (error) {
      console.error('Failed to connect WebSocket:', error);
      isConnectingRef.current = false;
    }
  }, [url, clientId]); // These are the only dependencies needed for connect

  const disconnect = useCallback(() => {
    // Clear any pending reconnection
    if (reconnectTimeoutRef.current) {
      clearTimeout(reconnectTimeoutRef.current);
      reconnectTimeoutRef.current = null;
    }

    isConnectingRef.current = false;
    reconnectAttemptsRef.current = maxReconnectAttempts; // Prevent further reconnection attempts

    if (wsRef.current) {
      wsRef.current.close(1000, 'Manual disconnect'); // Normal closure
      wsRef.current = null;
    }
  }, []);

  const reconnect = useCallback(() => {
    console.log('Manual reconnection triggered');
    reconnectAttemptsRef.current = 0; // Reset attempts for manual reconnection
    disconnect();
    setTimeout(() => connect(), 100); // Small delay to ensure clean disconnect
  }, [connect, disconnect]);

  const sendMessage = useCallback((message: string, selectedTools: string[] = [], model: string = 'gemma3:latest', attachedFile?: {name: string, size: number, type: string, content?: string}, displayMessage?: string) => {
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      // Optimize file handling for large files
      const optimizedAttachedFile = attachedFile ? {
        name: attachedFile.name,
        size: attachedFile.size,
        type: attachedFile.type,
        content: attachedFile.content && attachedFile.content.length > 500000
          ? attachedFile.content.substring(0, 500000) + '\n\n[File truncated - first 500KB shown]'
          : attachedFile.content
      } : undefined;

      const userMessage: Message = {
        id: Date.now().toString(),
        content: displayMessage || message,
        role: 'user',
        timestamp: new Date().toISOString(),
        tools: selectedTools,
        attachedFile: attachedFile ? {
          name: attachedFile.name,
          size: attachedFile.size,
          type: attachedFile.type
        } : undefined
      };

      dispatchMessage({
        type: 'ADD_USER_MESSAGE',
        payload: userMessage
      });

      wsRef.current.send(JSON.stringify({
        message,
        tools: selectedTools,
        model,
        attachedFile: optimizedAttachedFile
      }));
    }
  }, []);


  useEffect(() => {
    mountedRef.current = true;
    connect();

    return () => {
      mountedRef.current = false;
      disconnect();
      // Clean up any pending reconnection timeout
      if (reconnectTimeoutRef.current) {
        clearTimeout(reconnectTimeoutRef.current);
      }
    };
  }, [connect, disconnect]); // Include connect and disconnect dependencies as required

  return {
    isConnected,
    messages: messageState.messages,
    currentResponse: messageState.currentResponse,
    isLoading,
    sendMessage,
    connect,
    disconnect,
    reconnect
  };
};