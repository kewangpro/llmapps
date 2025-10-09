import React from 'react';
import { Globe, ExternalLink, Search } from 'lucide-react';

interface WebSearchResult {
  title: string;
  url: string;
  snippet: string;
  source?: string;
}

interface WebSearchCardProps {
  searchResults: WebSearchResult[];
  query: string;
  resultsCount: number;
}

const WebSearchCard: React.FC<WebSearchCardProps> = ({
  searchResults,
  query,
  resultsCount
}) => {
  return (
    <div className="bg-green-50 border border-green-200 rounded-lg p-4">
      {/* Header */}
      <div className="flex items-start justify-between mb-4">
        <div className="flex items-center gap-2">
          <Globe className="w-5 h-5 text-green-600" />
          <div>
            <h3 className="font-semibold text-green-800">Web Search Results</h3>
            <p className="text-sm text-green-600">
              {resultsCount} result{resultsCount !== 1 ? 's' : ''} found for &ldquo;{query}&rdquo;
            </p>
          </div>
        </div>
        <Search className="w-5 h-5 text-green-600" />
      </div>

      {/* Search Results */}
      {searchResults.length > 0 ? (
        <div className="space-y-3">
          {searchResults.map((result, index) => (
            <div
              key={index}
              className="bg-white border border-green-200 rounded-lg p-3 hover:shadow-md transition-shadow"
            >
              <div className="flex items-start justify-between gap-3">
                <div className="flex-1 min-w-0">
                  <a
                    href={result.url}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="block group"
                  >
                    <h4 className="font-medium text-gray-900 text-sm mb-1 line-clamp-2 group-hover:text-blue-600 transition-colors">
                      {result.title}
                    </h4>
                  </a>

                  <p className="text-xs text-gray-600 mb-2 line-clamp-3">
                    {result.snippet}
                  </p>

                  <div className="flex items-center gap-2">
                    <a
                      href={result.url}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="text-xs text-blue-600 hover:underline truncate"
                    >
                      {result.source || new URL(result.url).hostname}
                    </a>
                  </div>
                </div>

                <a
                  href={result.url}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="flex-shrink-0 flex items-center gap-1 px-3 py-1.5 bg-green-600 text-white rounded hover:bg-green-700 text-xs font-medium transition-colors"
                  title="Open link"
                >
                  Visit
                  <ExternalLink className="w-3 h-3" />
                </a>
              </div>
            </div>
          ))}
        </div>
      ) : (
        <div className="text-center py-6 text-gray-500">
          <Search className="w-8 h-8 mx-auto mb-2 opacity-50" />
          <p className="text-sm">No search results found</p>
          <p className="text-xs mt-1">Try a different search query</p>
        </div>
      )}
    </div>
  );
};

export default WebSearchCard;
