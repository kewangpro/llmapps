// API utility to handle different environments
export const getApiUrl = () => {
  if (typeof window !== 'undefined' && window.location.hostname === 'localhost') {
    return 'http://localhost:8000';
  }
  return 'https://agent-labs-ollama-backend-851143938786.us-central1.run.app';
};

export const getWebSocketUrl = () => {
  if (typeof window !== 'undefined' && window.location.hostname === 'localhost') {
    return 'ws://localhost:8000/ws';
  }
  const url = new URL('https://agent-labs-ollama-backend-851143938786.us-central1.run.app');
  const protocol = url.protocol === 'https:' ? 'wss' : 'ws';
  return `${protocol}://${url.host}/ws`;
};

export const apiUrl = (path: string) => {
  const baseUrl = getApiUrl();
  return `${baseUrl}${path}`;
};