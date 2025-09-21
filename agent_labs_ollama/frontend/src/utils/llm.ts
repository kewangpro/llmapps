// LLM utility functions for model and provider configuration

// Get default LLM configuration based on environment
export const getDefaultLLMConfig = () => {
  if (typeof window !== 'undefined' && window.location.hostname === 'localhost') {
    // Development environment - use Ollama
    return {
      provider: 'ollama',
      model: 'gemma3:latest'
    };
  }
  // Production environment - use OpenAI GPT-4o
  return {
    provider: 'openai',
    model: 'gpt-4o'
  };
};

export const getDefaultModel = () => {
  return getDefaultLLMConfig().model;
};

export const getDefaultProvider = () => {
  return getDefaultLLMConfig().provider;
};

// Check if current environment is localhost/development
export const isLocalhost = () => {
  return typeof window !== 'undefined' && window.location.hostname === 'localhost';
};

// Get available models by provider
export const getAvailableModels = () => {
  return {
    ollama: [
      'gemma3:latest',
      'llama3.1:latest',
      'llama3.2-vision:latest',
      'mistral:latest'
    ],
    openai: [
      'gpt-4o',
      'gpt-4',
      'gpt-4-turbo',
      'gpt-3.5-turbo'
    ],
    gemini: [
      'gemini-pro',
      'gemini-pro-vision'
    ]
  };
};

// Get vision-capable models
export const getVisionModels = () => {
  return {
    ollama: ['gemma3:latest', 'llama3.2-vision:latest'],
    openai: ['gpt-4o', 'gpt-4'],
    gemini: ['gemini-pro', 'gemini-pro-vision']
  };
};

// Check if a model supports vision
export const supportsVision = (provider: string, model: string) => {
  const visionModels = getVisionModels();
  return visionModels[provider as keyof typeof visionModels]?.includes(model) || false;
};