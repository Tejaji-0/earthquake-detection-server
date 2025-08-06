export interface EarthquakeData {
  Date: string;
  Time: string;
  Latitude: number;
  Longitude: number;
  Type: string;
  Depth: number;
  'Depth Error': number;
  'Depth Seismic Stations': number;
  Magnitude: number;
  'Magnitude Type': string;
  'Magnitude Error': number;
  'Magnitude Seismic Stations': number;
  'Azimuthal Gap': number;
  'Horizontal Distance': number;
  'Horizontal Error': number;
  'Root Mean Square': number;
  ID: string;
  Source: string;
  'Location Source': string;
  'Magnitude Source': string;
  Status: string;
  Country: string;
  // Computed fields for compatibility
  date_time?: string;
  location?: string;
  title?: string;
  alert?: string;
}

export interface FilterOptions {
  minMagnitude: number;
  maxMagnitude: number;
  startDate: string;
  endDate: string;
  location: string;
  alertLevel: string;
  country: string;
}
