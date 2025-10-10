import React, { useState } from 'react';
import { Hotel, MapPin, Calendar, Star, DollarSign, ChevronDown, ChevronUp, Wifi, Coffee } from 'lucide-react';

interface Hotel {
  name: string;
  category: string;
  location: string;
  rating: number;
  reviews: number;
  price_per_night: string;
  total_price: string;
  nights: number;
  check_in: string;
  check_out: string;
  amenities: string[];
  distance_from_center: string;
  cancellation_policy: string;
}

interface HotelQuery {
  location: string;
  check_in: string;
  check_out: string;
  nights: number;
  guests: number;
}

interface HotelCardProps {
  hotels: Hotel[];
  query: HotelQuery;
  resultsCount: number;
}

const HotelCard: React.FC<HotelCardProps> = ({
  hotels,
  query,
  resultsCount
}) => {
  const [showAll, setShowAll] = useState(false);

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

  // Render star rating
  const renderRating = (rating: number) => {
    const fullStars = Math.floor(rating);
    const hasHalfStar = rating % 1 >= 0.5;

    return (
      <div className="flex items-center gap-1">
        {[...Array(5)].map((_, i) => (
          <Star
            key={i}
            className={`w-4 h-4 ${
              i < fullStars
                ? 'fill-yellow-400 text-yellow-400'
                : i === fullStars && hasHalfStar
                ? 'fill-yellow-200 text-yellow-400'
                : 'text-gray-300'
            }`}
          />
        ))}
        <span className="ml-1 text-sm font-semibold text-gray-700">{rating.toFixed(1)}</span>
      </div>
    );
  };

  const renderHotel = (hotel: Hotel, index: number) => (
    <div
      key={index}
      className="bg-white border border-purple-200 rounded-lg p-4 hover:shadow-md transition-shadow"
    >
      {/* Hotel Name and Category */}
      <div className="flex items-start justify-between mb-3">
        <div className="flex-1">
          <h4 className="font-semibold text-gray-900 text-sm mb-1">{hotel.name}</h4>
          <span className="inline-block px-2 py-1 bg-purple-100 text-purple-700 text-xs rounded">
            {hotel.category}
          </span>
        </div>
        <div className="flex flex-col items-end ml-4">
          <div className="flex items-center gap-1 text-purple-700 font-bold text-sm mb-1">
            <DollarSign className="w-4 h-4" />
            <span>{hotel.price_per_night.replace('$', '')}/night</span>
          </div>
          <span className="text-xs text-gray-500">Total: {hotel.total_price}</span>
        </div>
      </div>

      {/* Rating and Reviews */}
      <div className="flex items-center gap-3 mb-3">
        {renderRating(hotel.rating)}
        <span className="text-xs text-gray-500">({hotel.reviews} reviews)</span>
      </div>

      {/* Location */}
      <div className="flex items-center gap-2 mb-3 text-sm text-gray-600">
        <MapPin className="w-4 h-4 text-purple-500" />
        <span>{hotel.distance_from_center} from center</span>
      </div>

      {/* Amenities */}
      <div className="mb-3">
        <div className="flex flex-wrap gap-2">
          {hotel.amenities.slice(0, 4).map((amenity, idx) => (
            <span
              key={idx}
              className="inline-flex items-center gap-1 px-2 py-1 bg-gray-100 text-gray-700 text-xs rounded"
            >
              {amenity === 'Free WiFi' && <Wifi className="w-3 h-3" />}
              {amenity === 'Restaurant' && <Coffee className="w-3 h-3" />}
              {amenity}
            </span>
          ))}
          {hotel.amenities.length > 4 && (
            <span className="inline-flex items-center px-2 py-1 bg-gray-100 text-gray-600 text-xs rounded">
              +{hotel.amenities.length - 4} more
            </span>
          )}
        </div>
      </div>

      {/* Cancellation Policy */}
      <div className="text-xs text-gray-600 border-t border-gray-200 pt-2">
        <span className={hotel.cancellation_policy === 'Free cancellation' ? 'text-green-600 font-medium' : 'text-gray-600'}>
          {hotel.cancellation_policy}
        </span>
      </div>
    </div>
  );

  return (
    <div className="bg-purple-50 border border-purple-200 rounded-lg p-4">
      {/* Header */}
      <div className="flex items-start justify-between mb-4">
        <div className="flex items-center gap-2">
          <Hotel className="w-5 h-5 text-purple-600" />
          <div>
            <h3 className="font-semibold text-purple-800">Hotel Options</h3>
            <p className="text-sm text-purple-600">
              {query.location}
            </p>
          </div>
        </div>
        <div className="flex items-center gap-1 text-purple-600 bg-purple-100 px-2 py-1 rounded">
          <Calendar className="w-4 h-4" />
          <span className="text-xs font-medium">{query.nights} night{query.nights > 1 ? 's' : ''}</span>
        </div>
      </div>

      {/* Stay Details */}
      <div className="mb-4 bg-white border border-purple-200 rounded-lg p-3">
        <div className="grid grid-cols-2 gap-3 text-sm">
          <div>
            <p className="text-xs text-gray-500 mb-1">Check-in</p>
            <p className="font-medium text-gray-700">{formatDate(query.check_in)}</p>
          </div>
          <div>
            <p className="text-xs text-gray-500 mb-1">Check-out</p>
            <p className="font-medium text-gray-700">{formatDate(query.check_out)}</p>
          </div>
        </div>
      </div>

      {/* Hotels List */}
      {hotels && hotels.length > 0 ? (
        <div>
          <div className="space-y-3">
            {(showAll ? hotels : hotels.slice(0, 3)).map((hotel, index) =>
              renderHotel(hotel, index)
            )}
          </div>
          {hotels.length > 3 && (
            <button
              onClick={() => setShowAll(!showAll)}
              className="mt-3 w-full flex items-center justify-center gap-2 px-4 py-2 bg-purple-100 hover:bg-purple-200 text-purple-700 rounded-lg transition-colors text-sm font-medium"
            >
              {showAll ? (
                <>
                  <ChevronUp className="w-4 h-4" />
                  Show less
                </>
              ) : (
                <>
                  <ChevronDown className="w-4 h-4" />
                  Show {hotels.length - 3} more hotel{hotels.length - 3 > 1 ? 's' : ''}
                </>
              )}
            </button>
          )}
        </div>
      ) : (
        <div className="text-center py-6 text-gray-500">
          <Hotel className="w-8 h-8 mx-auto mb-2 opacity-50" />
          <p className="text-sm">No hotels found for this location</p>
          <p className="text-xs mt-1">Try adjusting your search criteria</p>
        </div>
      )}

      {/* Information Note */}
      <div className="mt-4 bg-purple-100 border border-purple-300 rounded-lg p-3">
        <div className="flex items-start gap-2">
          <div className="flex-shrink-0 mt-0.5">
            <div className="w-4 h-4 rounded-full bg-purple-500 flex items-center justify-center">
              <span className="text-white text-xs font-bold">i</span>
            </div>
          </div>
          <div className="text-xs text-purple-800 leading-relaxed">
            <p className="font-semibold mb-1">Hotel Information</p>
            <p>
              These are estimated hotel options for your location and dates. For real-time availability,
              current prices, and booking, please visit{' '}
              <a
                href={`https://www.google.com/travel/hotels?q=${encodeURIComponent(query.location)}&checkin=${query.check_in}&checkout=${query.check_out}&adults=${query.guests}`}
                target="_blank"
                rel="noopener noreferrer"
                className="text-purple-600 hover:underline font-medium"
              >
                Google Hotels
              </a>
              {' '}or your preferred booking site.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
};

export default HotelCard;
