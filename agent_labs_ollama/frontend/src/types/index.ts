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

export interface ChartData {
  dates: string[];
  prices: number[];
  symbol: string;
  company_name: string;
  chart_html?: string;
}

export interface ImageData {
  base64: string;
  data_url: string;
  mime_type: string;
}

export interface SlideDescription {
  title: string;
  bullets: string[];
  slide_type: 'title' | 'content';
}

export interface PresentationData {
  base64: string;
  filename: string;
  mime_type: string;
  slides: SlideDescription[];
}

export interface ToolResult {
  tool: string;
  result: any;
  timestamp: string;
  summary?: string;
  chart_data?: ChartData;
  image_data?: ImageData;
  presentation_data?: PresentationData;
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