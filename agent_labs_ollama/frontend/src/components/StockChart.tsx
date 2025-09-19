import React, { useEffect, useState } from 'react';
import { ChartData } from '@/types';

interface StockChartProps {
  chartData: ChartData;
}

const StockChart: React.FC<StockChartProps> = ({ chartData }) => {
  const [chartBlobUrl, setChartBlobUrl] = useState<string | null>(null);

  useEffect(() => {
    if (chartData.chart_html) {
      // Create a blob URL for the HTML content
      const blob = new Blob([chartData.chart_html], { type: 'text/html' });
      const url = URL.createObjectURL(blob);
      setChartBlobUrl(url);

      // Cleanup function
      return () => {
        if (url) {
          URL.revokeObjectURL(url);
        }
      };
    }
  }, [chartData]);

  if (!chartData.chart_html) {
    // Fallback: simple price display if no chart HTML
    return (
      <div className="bg-gray-50 p-4 rounded-lg">
        <h3 className="text-lg font-semibold mb-2">
          📈 {chartData.company_name} ({chartData.symbol})
        </h3>
        <div className="text-sm text-gray-600">
          <p>Price data available for {chartData.dates.length} days</p>
          <p>Latest price: ${chartData.prices[chartData.prices.length - 1]?.toFixed(2)}</p>
        </div>
      </div>
    );
  }

  return (
    <div className="bg-white p-4 rounded-lg border shadow-sm">
      <div className="mb-2">
        <h3 className="text-lg font-semibold">
          📈 {chartData.company_name} ({chartData.symbol}) Stock Chart
        </h3>
        <p className="text-sm text-gray-600">
          {chartData.dates.length} days of price data
        </p>
      </div>

      {chartBlobUrl ? (
        <iframe
          src={chartBlobUrl}
          className="w-full border-0"
          style={{ height: '500px' }}
          title={`${chartData.company_name} Stock Chart`}
        />
      ) : (
        <div className="w-full h-96 flex items-center justify-center bg-gray-50 rounded">
          <div className="text-center">
            <div className="animate-spin w-8 h-8 border-4 border-blue-500 border-t-transparent rounded-full mx-auto mb-2"></div>
            <p className="text-gray-600">Loading chart...</p>
          </div>
        </div>
      )}
    </div>
  );
};

export default StockChart;