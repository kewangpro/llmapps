// API utility to handle different environments
export const getApiUrl = () => {
  if (typeof window !== 'undefined' && window.location.hostname === 'localhost') {
    return 'http://localhost:3000';
  }
  // Use same URL as frontend since it's now a unified service
  return window.location.origin;
};

export const getWebSocketUrl = () => {
  if (typeof window !== 'undefined' && window.location.hostname === 'localhost') {
    return 'ws://localhost:3000/ws';
  }
  // Use same host as frontend since it's now a unified service
  const protocol = window.location.protocol === 'https:' ? 'wss' : 'ws';
  return `${protocol}://${window.location.host}/ws`;
};

export const apiUrl = (path: string) => {
  const baseUrl = getApiUrl();
  return `${baseUrl}${path}`;
};