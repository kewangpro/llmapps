export interface Tool {
  id: string;
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

export interface VisualizationData {
  chart_html: string;
  chart_image_base64: string;
  chart_config: {
    title: string;
    x_column?: string;
    y_column?: string;
    label_column?: string;
    value_column?: string;
    size_column?: string;
  };
  chart_type: string;
  data_rows: number;
  data_columns: number;
  columns: string[];
  message: string;
}

export interface ToolResult {
  tool: string;
  result: any;
  timestamp: string;
  summary?: string;
  image_data?: ImageData;
  presentation_data?: PresentationData;
  visualization_data?: VisualizationData;
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