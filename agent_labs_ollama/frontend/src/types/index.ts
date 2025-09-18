export interface Tool {
  name: string;
  description: string;
  parameters: Record<string, any>;
  category: string;
}

export interface Message {
  id: string;
  content: string;
  role: 'user' | 'assistant' | 'system';
  timestamp: string;
  tools?: string[];
  attachedFile?: {
    name: string;
    size: number;
    type: string;
  };
  toolResults?: ToolResult[];
}

export interface ToolResult {
  tool: string;
  result: any;
  timestamp: string;
  summary?: string;
}

export interface WebSocketMessage {
  type: 'message_received' | 'assistant_response_start' | 'assistant_response_chunk' | 'assistant_response_complete' | 'tool_execution_start' | 'tool_result' | 'tool_summary' | 'agent_response' | 'error';
  content?: string;
  tool?: string;
  result?: any;
  summary?: string;
  message?: string;
  timestamp: string;
}