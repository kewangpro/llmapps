// API utility to handle different environments
export const getApiUrl = () => {
  if (process.env.NEXT_PUBLIC_API_BASE_URL) {
    return process.env.NEXT_PUBLIC_API_BASE_URL;
  }
  // In development, use backend port
  return 'http://localhost:8000';
};

export const getWebSocketUrl = () => {
  if (process.env.NEXT_PUBLIC_API_BASE_URL) {
    const url = new URL(process.env.NEXT_PUBLIC_API_BASE_URL);
    const protocol = url.protocol === 'https:' ? 'wss' : 'ws';
    return `${protocol}://${url.host}/ws`;
  }

  // In development, use backend WebSocket port
  return 'ws://localhost:8000/ws';
};

export const apiUrl = (path: string) => {
  const baseUrl = getApiUrl();
  return `${baseUrl}${path}`;
};