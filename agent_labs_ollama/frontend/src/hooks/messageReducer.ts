import { Message, ToolResult } from '@/types';

export interface MessageState {
  messages: Message[];
  currentResponse: string;
}

export type MessageAction =
  | { type: 'ADD_USER_MESSAGE'; payload: Message }
  | { type: 'START_RESPONSE' }
  | { type: 'UPDATE_CURRENT_RESPONSE'; payload: string }
  | { type: 'COMPLETE_CURRENT_RESPONSE' }
  | { type: 'ADD_TOOL_RESULT'; payload: { toolResult: ToolResult } }
  | { type: 'ADD_TOOL_SUMMARY'; payload: { tool: string; summary: string } }
  | { type: 'ADD_ASSISTANT_RESPONSE'; payload: { content: string; timestamp: string } }
  | { type: 'CLEAR_CURRENT_RESPONSE' }
  | { type: 'VALIDATE_AND_REORDER' };

// Message ordering validation
const validateMessageOrder = (messages: Message[]): Message[] => {
  // Don't sort by timestamp - preserve insertion order to avoid tool results appearing before user messages
  // Only remove duplicate IDs
  const uniqueMessages = messages.filter((message, index, array) => {
    return index === array.findIndex(m => m.id === message.id);
  });

  return uniqueMessages;
};

export const messageReducer = (state: MessageState, action: MessageAction): MessageState => {
  switch (action.type) {
    case 'ADD_USER_MESSAGE':
      const newMessages = validateMessageOrder([...state.messages, action.payload]);
      return {
        ...state,
        messages: newMessages
      };

    case 'START_RESPONSE':
      return {
        ...state,
        currentResponse: ''
      };

    case 'UPDATE_CURRENT_RESPONSE':
      return {
        ...state,
        currentResponse: action.payload
      };

    case 'COMPLETE_CURRENT_RESPONSE':
      // Keep current response visible until final response arrives
      return state;

    case 'ADD_TOOL_RESULT':
      // Add tool result to the last assistant message
      const messagesWithToolResult = state.messages.map((message, index) => {
        // Find the last assistant message
        const isLastAssistant = message.role === 'assistant' &&
          index === state.messages.map(m => m.role).lastIndexOf('assistant');

        if (isLastAssistant) {
          return {
            ...message,
            toolResults: [...(message.toolResults || []), action.payload.toolResult]
          };
        }
        return message;
      });

      return {
        ...state,
        messages: validateMessageOrder(messagesWithToolResult)
      };

    case 'ADD_TOOL_SUMMARY':
      // Add summary to the matching tool result in the last assistant message
      const messagesWithSummary = state.messages.map((message, index) => {
        const isLastAssistant = message.role === 'assistant' &&
          index === state.messages.map(m => m.role).lastIndexOf('assistant');

        if (isLastAssistant && message.toolResults) {
          const updatedToolResults = message.toolResults.map(toolResult => {
            if (toolResult.tool === action.payload.tool) {
              return {
                ...toolResult,
                summary: action.payload.summary
              };
            }
            return toolResult;
          });

          return {
            ...message,
            toolResults: updatedToolResults
          };
        }
        return message;
      });

      return {
        ...state,
        messages: validateMessageOrder(messagesWithSummary)
      };

    case 'ADD_ASSISTANT_RESPONSE':
      // Add assistant response as a single message
      // Use a more unique ID to prevent overwrites when multiple responses arrive quickly
      const assistantMessage: Message = {
        id: `assistant-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`,
        content: action.payload.content,
        role: 'assistant',
        timestamp: action.payload.timestamp
      };

      return {
        ...state,
        messages: validateMessageOrder([...state.messages, assistantMessage]),
        currentResponse: ''
      };

    case 'CLEAR_CURRENT_RESPONSE':
      return {
        ...state,
        currentResponse: ''
      };

    case 'VALIDATE_AND_REORDER':
      return {
        ...state,
        messages: validateMessageOrder(state.messages)
      };

    default:
      return state;
  }
};

export const initialMessageState: MessageState = {
  messages: [],
  currentResponse: ''
};