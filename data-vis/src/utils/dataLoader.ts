import Papa from 'papaparse';
import type { EarthquakeData } from '../types/earthquake';

export const loadEarthquakeData = async (): Promise<EarthquakeData[]> => {
  try {
    const response = await fetch('/database.csv');
    const csvText = await response.text();
    
    const parsed = Papa.parse<any>(csvText, {
      header: true,
      skipEmptyLines: true,
      transform: (value, field) => {
        // Convert numeric fields
        if (typeof field === 'string' && ['Latitude', 'Longitude', 'Depth', 'Depth Error', 'Depth Seismic Stations', 'Magnitude', 'Magnitude Error', 'Magnitude Seismic Stations', 'Azimuthal Gap', 'Horizontal Distance', 'Horizontal Error', 'Root Mean Square'].includes(field)) {
          const num = parseFloat(value);
          return isNaN(num) ? 0 : num;
        }
        return value;
      }
    });

    if (parsed.errors.length > 0) {
      console.warn('CSV parsing errors:', parsed.errors);
    }

    // Transform the data to include computed fields for compatibility
    const transformedData: EarthquakeData[] = parsed.data
      .filter((row: any) => row.Date && row.Magnitude && parseFloat(row.Magnitude) > 0)
      .map((row: any) => {
        // Create a date_time field by combining Date and Time
        const dateTime = `${row.Date} ${row.Time || '00:00:00'}`;
        
        // Generate a title based on magnitude and rough location
        const title = `M ${row.Magnitude} - Earthquake`;
        
        // Determine alert level based on magnitude
        let alert = '';
        const mag = parseFloat(row.Magnitude);
        if (mag >= 8.0) alert = 'red';
        else if (mag >= 7.0) alert = 'orange';
        else if (mag >= 6.5) alert = 'yellow';
        else alert = 'green';

        // Create location string from coordinates
        const lat = parseFloat(row.Latitude);
        const lon = parseFloat(row.Longitude);
        const location = `${Math.abs(lat).toFixed(2)}°${lat >= 0 ? 'N' : 'S'}, ${Math.abs(lon).toFixed(2)}°${lon >= 0 ? 'E' : 'W'}`;

        return {
          ...row,
          date_time: dateTime,
          title,
          alert,
          location
        };
      });

    return transformedData;
  } catch (error) {
    console.error('Error loading earthquake data:', error);
    return [];
  }
};

export const filterEarthquakeData = (
  data: EarthquakeData[],
  filters: {
    minMagnitude?: number;
    maxMagnitude?: number;
    startDate?: string;
    endDate?: string;
    location?: string;
    alertLevel?: string;
    country?: string;
  }
): EarthquakeData[] => {
  return data.filter(earthquake => {
    // Magnitude filter
    if (filters.minMagnitude && earthquake.Magnitude < filters.minMagnitude) return false;
    if (filters.maxMagnitude && earthquake.Magnitude > filters.maxMagnitude) return false;

    // Date filter
    if (filters.startDate) {
      const earthquakeDate = new Date(earthquake.date_time || `${earthquake.Date} ${earthquake.Time}`);
      const startDate = new Date(filters.startDate);
      if (earthquakeDate < startDate) return false;
    }
    
    if (filters.endDate) {
      const earthquakeDate = new Date(earthquake.date_time || `${earthquake.Date} ${earthquake.Time}`);
      const endDate = new Date(filters.endDate);
      if (earthquakeDate > endDate) return false;
    }

    // Location filter
    if (filters.location && 
        !earthquake.location?.toLowerCase().includes(filters.location.toLowerCase()) &&
        !earthquake.ID?.toLowerCase().includes(filters.location.toLowerCase())) {
      return false;
    }

    // Country filter
    if (filters.country && 
        !earthquake.Country?.toLowerCase().includes(filters.country.toLowerCase())) {
      return false;
    }

    // Alert level filter
    if (filters.alertLevel && earthquake.alert !== filters.alertLevel) return false;

    return true;
  });
};

export const getMagnitudeColor = (magnitude: number): string => {
  if (magnitude >= 8.0) return '#8B0000'; // Dark red for major earthquakes
  if (magnitude >= 7.0) return '#FF0000'; // Red for strong earthquakes
  if (magnitude >= 6.5) return '#FF6600'; // Orange for moderate earthquakes
  return '#FFB300'; // Yellow for lighter earthquakes
};

export const formatDate = (dateString: string): string => {
  const date = new Date(dateString);
  return date.toLocaleDateString('en-US', {
    year: 'numeric',
    month: 'short',
    day: 'numeric',
    hour: '2-digit',
    minute: '2-digit'
  });
};
