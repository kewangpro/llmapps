// API utility to handle different environments
export const getApiUrl = () => {
  // In production (deployed), use relative URLs
  if (typeof window !== 'undefined' && window.location.hostname !== 'localhost') {
    return '';
  }

  // In development, use backend port
  return 'http://localhost:8000';
};

export const getWebSocketUrl = () => {
  if (typeof window === 'undefined') {
    return '';
  }

  // In production (deployed), use current host with appropriate protocol
  if (window.location.hostname !== 'localhost') {
    const protocol = window.location.protocol === 'https:' ? 'wss' : 'ws';
    return `${protocol}://${window.location.host}/ws`;
  }

  // In development, use backend WebSocket port
  return 'ws://localhost:8000/ws';
};

export const apiUrl = (path: string) => {
  const baseUrl = getApiUrl();
  return `${baseUrl}${path}`;
};