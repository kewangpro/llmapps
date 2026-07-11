import { Component } from 'react';
import type { ErrorInfo, ReactNode } from 'react';
import { AlertTriangle } from 'lucide-react';

interface Props {
  children: ReactNode;
  fallbackTitle?: string;
}

interface State {
  hasError: boolean;
  error?: Error;
}

class ErrorBoundary extends Component<Props, State> {
  constructor(props: Props) {
    super(props);
    this.state = { hasError: false };
  }

  static getDerivedStateFromError(error: Error): State {
    return { hasError: true, error };
  }

  componentDidCatch(error: Error, errorInfo: ErrorInfo) {
    console.error('ErrorBoundary caught an error:', error, errorInfo);
  }

  render() {
    if (this.state.hasError) {
      return (
        <div className="flex items-center justify-center min-h-[400px] p-8">
          <div className="max-w-md w-full bg-red-50 border-2 border-red-200 rounded-lg p-6">
            <div className="flex items-center gap-3 mb-4">
              <AlertTriangle className="text-red-600" size={32} />
              <h2 className="text-xl font-bold text-red-900">
                {this.props.fallbackTitle || 'Something went wrong'}
              </h2>
            </div>
            <p className="text-red-800 mb-4">
              This custom visualization encountered an error and could not be rendered.
            </p>
            {this.state.error && (
              <details className="bg-white border border-red-200 rounded p-3">
                <summary className="cursor-pointer text-sm font-medium text-red-900 mb-2">
                  Error details
                </summary>
                <pre className="text-xs text-red-700 overflow-x-auto">
                  {this.state.error.toString()}
                </pre>
              </details>
            )}
            <button
              onClick={() => this.setState({ hasError: false, error: undefined })}
              className="mt-4 px-4 py-2 bg-red-600 text-white rounded-lg hover:bg-red-700 transition-colors"
            >
              Try Again
            </button>
          </div>
        </div>
      );
    }

    return this.props.children;
  }
}

export default ErrorBoundary;
