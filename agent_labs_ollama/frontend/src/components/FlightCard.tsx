import React, { useState } from 'react';
import { Plane, MapPin, Calendar, Clock, DollarSign, ChevronDown, ChevronUp} from 'lucide-react';

interface Flight {
  airline: string;
  from_city: string;
  to_city: string;
  departure_time: string;
  arrival_time: string;
  price: string;
  date: string;
  duration: string;
  stops: number | string;  // Can be number or "See website"
}

interface FlightQuery {
  origin: string;
  destination: string;
  departure_date: string;
  return_date?: string;
  trip_type: string;
}

interface FlightCardProps {
  flights: {
    outbound: Flight[];
    return: Flight[];
  };
  query: FlightQuery;
  resultsCount: number;
}

const FlightCard: React.FC<FlightCardProps> = ({
  flights,
  query,
  resultsCount
}) => {
  const [showAllOutbound, setShowAllOutbound] = useState(false);
  const [showAllReturn, setShowAllReturn] = useState(false);
  // Format date for display
  const formatDate = (dateString: string) => {
    const date = new Date(dateString);
    return date.toLocaleDateString('en-US', {
      weekday: 'short',
      month: 'short',
      day: 'numeric',
      year: 'numeric'
    });
  };

  const renderFlight = (flight: Flight, index: number) => (
    <div
      key={index}
      className="bg-white border border-blue-200 rounded-lg p-4 hover:shadow-md transition-shadow"
    >
      {/* Airline */}
      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center gap-2">
          <Plane className="w-4 h-4 text-blue-600" />
          <span className="font-semibold text-gray-900 text-sm">{flight.airline}</span>
        </div>
        <div className="flex items-center gap-1 text-blue-700 font-bold text-sm">
          <DollarSign className="w-4 h-4" />
          <span>{flight.price.replace('$', '')}</span>
        </div>
      </div>

      {/* Flight Times */}
      <div className="flex items-center justify-between mb-3">
        <div className="flex-1">
          <div className="text-2xl font-bold text-gray-900">{flight.departure_time}</div>
          <div className="text-xs text-gray-600">{flight.from_city}</div>
        </div>

        <div className="flex-1 flex flex-col items-center px-4">
          <div className="text-xs text-gray-500 mb-1">{flight.duration}</div>
          <div className="w-full border-t-2 border-blue-300 relative">
            <Plane className="w-4 h-4 text-blue-500 absolute top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2 bg-white" />
          </div>
          <div className="text-xs text-gray-500 mt-1">
            {typeof flight.stops === 'number'
              ? (flight.stops === 0 ? 'Nonstop' : `${flight.stops} stop${flight.stops > 1 ? 's' : ''}`)
              : flight.stops
            }
          </div>
        </div>

        <div className="flex-1 text-right">
          <div className="text-2xl font-bold text-gray-900">{flight.arrival_time}</div>
          <div className="text-xs text-gray-600">{flight.to_city}</div>
        </div>
      </div>

      {/* Flight Date */}
      <div className="flex items-center gap-1 text-xs text-gray-500">
        <Calendar className="w-3 h-3" />
        <span>{formatDate(flight.date)}</span>
      </div>
    </div>
  );

  return (
    <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
      {/* Header */}
      <div className="flex items-start justify-between mb-4">
        <div className="flex items-center gap-2">
          <Plane className="w-5 h-5 text-blue-600" />
          <div>
            <h3 className="font-semibold text-blue-800">Flight Options</h3>
            <p className="text-sm text-blue-600">
              {query.origin} → {query.destination}
            </p>
          </div>
        </div>
        <div className="flex items-center gap-1 text-blue-600 bg-blue-100 px-2 py-1 rounded">
          <Clock className="w-4 h-4" />
          <span className="text-xs font-medium">{query.trip_type}</span>
        </div>
      </div>

      {/* Outbound Flights */}
      {flights.outbound && flights.outbound.length > 0 && (
        <div className="mb-4">
          <div className="space-y-3">
            {(showAllOutbound ? flights.outbound : flights.outbound.slice(0, 2)).map((flight, index) =>
              renderFlight(flight, index)
            )}
          </div>
          {flights.outbound.length > 2 && (
            <button
              onClick={() => setShowAllOutbound(!showAllOutbound)}
              className="mt-3 w-full flex items-center justify-center gap-2 px-4 py-2 bg-blue-100 hover:bg-blue-200 text-blue-700 rounded-lg transition-colors text-sm font-medium"
            >
              {showAllOutbound ? (
                <>
                  <ChevronUp className="w-4 h-4" />
                  Show less
                </>
              ) : (
                <>
                  <ChevronDown className="w-4 h-4" />
                  Show {flights.outbound.length - 2} more flight{flights.outbound.length - 2 > 1 ? 's' : ''}
                </>
              )}
            </button>
          )}
        </div>
      )}

      {/* Return Flights */}
      {flights.return && flights.return.length > 0 && (
        <div>
          <div className="space-y-3">
            {(showAllReturn ? flights.return : flights.return.slice(0, 2)).map((flight, index) =>
              renderFlight(flight, index)
            )}
          </div>
          {flights.return.length > 2 && (
            <button
              onClick={() => setShowAllReturn(!showAllReturn)}
              className="mt-3 w-full flex items-center justify-center gap-2 px-4 py-2 bg-blue-100 hover:bg-blue-200 text-blue-700 rounded-lg transition-colors text-sm font-medium"
            >
              {showAllReturn ? (
                <>
                  <ChevronUp className="w-4 h-4" />
                  Show less
                </>
              ) : (
                <>
                  <ChevronDown className="w-4 h-4" />
                  Show {flights.return.length - 2} more flight{flights.return.length - 2 > 1 ? 's' : ''}
                </>
              )}
            </button>
          )}
        </div>
      )}

      {/* No Results Message */}
      {(!flights.outbound || flights.outbound.length === 0) && (
        <div className="text-center py-6 text-gray-500">
          <Plane className="w-8 h-8 mx-auto mb-2 opacity-50" />
          <p className="text-sm">No flights found for this route</p>
          <p className="text-xs mt-1">Try adjusting your search criteria</p>
        </div>
      )}

      {/* Information Note */}
      <div className="mt-4 bg-blue-100 border border-blue-300 rounded-lg p-3">
        <div className="flex items-start gap-2">
          <div className="flex-shrink-0 mt-0.5">
            <div className="w-4 h-4 rounded-full bg-blue-500 flex items-center justify-center">
              <span className="text-white text-xs font-bold">i</span>
            </div>
          </div>
          <div className="text-xs text-blue-800 leading-relaxed">
            <p className="font-semibold mb-1">Flight Information</p>
            <p>
              These are estimated flight options for your route. For real-time availability,
              current prices, and booking, please visit{' '}
              <a
                href={`https://www.google.com/flights?hl=en#flt=${query.origin}.${query.destination}.${query.departure_date}${query.return_date ? '*' + query.destination + '.' + query.origin + '.' + query.return_date : ''}`}
                target="_blank"
                rel="noopener noreferrer"
                className="text-blue-600 hover:underline font-medium"
              >
                Google Flights
              </a>
              {' '}or your preferred booking site.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
};

export default FlightCard;
