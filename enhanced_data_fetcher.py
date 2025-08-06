#!/usr/bin/env python3
"""
Enhanced Earthquake Data Fetcher
This script fetches earthquake data from multiple sources including real-time APIs,
historical databases, and seismic networks to enhance the ML training dataset.
"""

import pandas as pd
import numpy as np
import requests
import json
import time
import os
from datetime import datetime, timedelta
from urllib.parse import urlencode
import warnings
warnings.filterwarnings('ignore')

class EnhancedEarthquakeDataFetcher:
    def __init__(self, output_dir='enhanced_earthquake_data'):
        self.output_dir = output_dir
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Enhanced-Earthquake-ML-Pipeline/1.0'
        })
        
        # Create output directory
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        
        # API endpoints and configurations
        self.apis = {
            'usgs_latest': {
                'url': 'https://earthquake.usgs.gov/earthquakes/feed/v1.0/summary/all_month.geojson',
                'format': 'geojson',
                'description': 'USGS Latest Earthquakes (Past Month)'
            },
            'usgs_significant': {
                'url': 'https://earthquake.usgs.gov/earthquakes/feed/v1.0/summary/significant_month.geojson',
                'format': 'geojson',
                'description': 'USGS Significant Earthquakes (Past Month)'
            },
            'usgs_major': {
                'url': 'https://earthquake.usgs.gov/earthquakes/feed/v1.0/summary/4.5_month.geojson',
                'format': 'geojson',
                'description': 'USGS M4.5+ Earthquakes (Past Month)'
            },
            'usgs_query': {
                'url': 'https://earthquake.usgs.gov/fdsnws/event/1/query',
                'format': 'geojson',
                'description': 'USGS Historical Query Service'
            },
            'emsc': {
                'url': 'https://www.seismicportal.eu/fdsnws/event/1/query',
                'format': 'json',
                'description': 'European-Mediterranean Seismological Centre'
            },
            'gfz': {
                'url': 'http://geofon.gfz-potsdam.de/fdsnws/event/1/query',
                'format': 'json',
                'description': 'GFZ German Research Centre for Geosciences'
            },
            'iris': {
                'url': 'http://service.iris.edu/fdsnws/event/1/query',
                'format': 'json',
                'description': 'IRIS Earthquake Database'
            },
            'isc': {
                'url': 'http://www.isc.ac.uk/fdsnws/event/1/query',
                'format': 'json',
                'description': 'International Seismological Centre'
            }
        }
        
        print(f"Enhanced Earthquake Data Fetcher initialized")
        print(f"Output directory: {self.output_dir}")
        print(f"Available APIs: {len(self.apis)}")
    
    def fetch_usgs_feeds(self):
        """Fetch data from USGS GeoJSON feeds"""
        print("\n=== Fetching USGS Feed Data ===")
        
        all_earthquakes = []
        
        for feed_name, config in self.apis.items():
            if not feed_name.startswith('usgs_') or feed_name == 'usgs_query':
                continue
                
            print(f"Fetching {config['description']}...")
            
            try:
                response = self.session.get(config['url'], timeout=30)
                response.raise_for_status()
                
                data = response.json()
                features = data.get('features', [])
                
                print(f"✓ Retrieved {len(features)} earthquakes from {feed_name}")
                
                for feature in features:
                    earthquake = self.parse_usgs_geojson(feature, feed_name)
                    if earthquake:
                        all_earthquakes.append(earthquake)
                
                # Rate limiting
                time.sleep(1)
                
            except Exception as e:
                print(f"✗ Failed to fetch {feed_name}: {e}")
                continue
        
        return all_earthquakes
    
    def fetch_usgs_historical(self, start_date=None, end_date=None, min_magnitude=4.0):
        """Fetch historical data from USGS query service"""
        print(f"\n=== Fetching USGS Historical Data ===")
        
        if start_date is None:
            start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')
        
        print(f"Date range: {start_date} to {end_date}")
        print(f"Minimum magnitude: {min_magnitude}")
        
        params = {
            'format': 'geojson',
            'starttime': start_date,
            'endtime': end_date,
            'minmagnitude': min_magnitude,
            'limit': 20000,  # USGS limit
            'orderby': 'time-asc'
        }
        
        all_earthquakes = []
        
        try:
            url = f"{self.apis['usgs_query']['url']}?{urlencode(params)}"
            print(f"Querying: {url}")
            
            response = self.session.get(url, timeout=60)
            response.raise_for_status()
            
            data = response.json()
            features = data.get('features', [])
            
            print(f"✓ Retrieved {len(features)} historical earthquakes from USGS")
            
            for feature in features:
                earthquake = self.parse_usgs_geojson(feature, 'usgs_historical')
                if earthquake:
                    all_earthquakes.append(earthquake)
            
        except Exception as e:
            print(f"✗ Failed to fetch USGS historical data: {e}")
        
        return all_earthquakes
    
    def fetch_emsc_data(self, days_back=30, min_magnitude=4.0):
        """Fetch data from European-Mediterranean Seismological Centre"""
        print(f"\n=== Fetching EMSC Data ===")
        
        start_date = (datetime.now() - timedelta(days=days_back)).strftime('%Y-%m-%d')
        end_date = datetime.now().strftime('%Y-%m-%d')
        
        params = {
            'format': 'json',
            'starttime': start_date,
            'endtime': end_date,
            'minmagnitude': min_magnitude,
            'limit': 10000
        }
        
        all_earthquakes = []
        
        try:
            url = f"{self.apis['emsc']['url']}?{urlencode(params)}"
            print(f"Querying EMSC: {url}")
            
            response = self.session.get(url, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            features = data.get('features', [])
            
            print(f"✓ Retrieved {len(features)} earthquakes from EMSC")
            
            for feature in features:
                earthquake = self.parse_fdsn_json(feature, 'emsc')
                if earthquake:
                    all_earthquakes.append(earthquake)
            
        except Exception as e:
            print(f"✗ Failed to fetch EMSC data: {e}")
        
        return all_earthquakes
    
    def fetch_global_networks(self, days_back=30, min_magnitude=5.0):
        """Fetch data from global seismic networks"""
        print(f"\n=== Fetching Global Network Data ===")
        
        start_date = (datetime.now() - timedelta(days=days_back)).strftime('%Y-%m-%d')
        end_date = datetime.now().strftime('%Y-%m-%d')
        
        all_earthquakes = []
        
        # List of networks to try (excluding EMSC which is handled separately)
        networks = ['gfz', 'iris', 'isc']
        
        for network in networks:
            if network not in self.apis:
                continue
                
            print(f"Fetching from {self.apis[network]['description']}...")
            
            params = {
                'format': 'json',
                'starttime': start_date,
                'endtime': end_date,
                'minmagnitude': min_magnitude,
                'limit': 5000
            }
            
            try:
                url = f"{self.apis[network]['url']}?{urlencode(params)}"
                
                response = self.session.get(url, timeout=30)
                response.raise_for_status()
                
                data = response.json()
                features = data.get('features', [])
                
                print(f"✓ Retrieved {len(features)} earthquakes from {network}")
                
                for feature in features:
                    earthquake = self.parse_fdsn_json(feature, network)
                    if earthquake:
                        all_earthquakes.append(earthquake)
                
                # Rate limiting
                time.sleep(2)
                
            except Exception as e:
                print(f"✗ Failed to fetch {network} data: {e}")
                continue
        
        return all_earthquakes
    
    def parse_usgs_geojson(self, feature, source):
        """Parse USGS GeoJSON format earthquake data"""
        try:
            props = feature['properties']
            geom = feature['geometry']['coordinates']
            
            # Convert timestamp
            timestamp = props.get('time', 0) / 1000  # Convert from milliseconds
            dt = datetime.fromtimestamp(timestamp)
            
            earthquake = {
                'Date': dt.strftime('%d/%m/%Y'),
                'Time': dt.strftime('%H:%M:%S'),
                'Latitude': geom[1],
                'Longitude': geom[0],
                'Depth': geom[2] if len(geom) > 2 else 10,
                'Magnitude': props.get('mag', 0),
                'Magnitude Type': props.get('magType', 'ML'),
                'Type': 'Earthquake',
                'Location Source': f'USGS_{source}',
                'Country': self.extract_country_from_place(props.get('place', '')),
                'Status': props.get('status', 'automatic').title(),
                'ID': props.get('id', ''),
                'Source': f'USGS_{source}',
                'Magnitude Source': f'USGS_{source}',
                
                # Additional USGS-specific fields
                'significance': props.get('sig', 0),
                'alert': props.get('alert', ''),
                'tsunami': props.get('tsunami', 0),
                'nst': props.get('nst', 0),
                'gap': props.get('gap', 180),
                'dmin': props.get('dmin', 1.0),
                'rms': props.get('rms', 1.0),
                'net': props.get('net', ''),
                'code': props.get('code', ''),
                'ids': props.get('ids', ''),
                'sources': props.get('sources', ''),
                'types': props.get('types', ''),
                'nph': props.get('nph', 0),
                'updated': props.get('updated', 0),
                'detail': props.get('detail', ''),
                'felt': props.get('felt', 0),
                'cdi': props.get('cdi', 0),
                'mmi': props.get('mmi', 0),
                'magNst': props.get('magNst', 0)
            }
            
            # Fill missing fields with defaults
            earthquake.update({
                'Depth Error': np.nan,
                'Depth Seismic Stations': earthquake['nst'],
                'Magnitude Error': np.nan,
                'Magnitude Seismic Stations': earthquake['magNst'],
                'Azimuthal Gap': earthquake['gap'],
                'Horizontal Distance': earthquake['dmin'],
                'Horizontal Error': np.nan,
                'Root Mean Square': earthquake['rms']
            })
            
            return earthquake
            
        except Exception as e:
            print(f"Error parsing USGS data: {e}")
            return None
    
    def parse_fdsn_json(self, feature, source):
        """Parse FDSN JSON format earthquake data"""
        try:
            props = feature['properties']
            geom = feature['geometry']['coordinates']
            
            # Convert time
            time_str = props.get('time', '')
            if time_str:
                # Handle different time formats
                try:
                    if 'T' in time_str:
                        dt = datetime.fromisoformat(time_str.replace('Z', '+00:00'))
                    else:
                        dt = datetime.strptime(time_str, '%Y-%m-%d %H:%M:%S')
                except:
                    dt = datetime.now()
            else:
                dt = datetime.now()
            
            earthquake = {
                'Date': dt.strftime('%d/%m/%Y'),
                'Time': dt.strftime('%H:%M:%S'),
                'Latitude': geom[1],
                'Longitude': geom[0],
                'Depth': geom[2] if len(geom) > 2 else props.get('depth', 10),
                'Magnitude': props.get('mag', 0),
                'Magnitude Type': props.get('magtype', 'ML'),
                'Type': 'Earthquake',
                'Location Source': source.upper(),
                'Country': self.extract_country_from_place(props.get('place', '')),
                'Status': 'Automatic',
                'ID': props.get('publicid', props.get('eventid', '')),
                'Source': source.upper(),
                'Magnitude Source': source.upper(),
                
                # Default additional fields
                'significance': 0,
                'alert': '',
                'tsunami': 0,
                'nst': props.get('nst', 0),
                'gap': props.get('gap', 180),
                'dmin': props.get('dmin', 1.0),
                'rms': props.get('rms', 1.0),
                'net': source,
                'code': '',
                'ids': '',
                'sources': source,
                'types': 'origin,magnitude',
                'nph': 0,
                'updated': int(dt.timestamp() * 1000),
                'detail': '',
                'felt': 0,
                'cdi': 0,
                'mmi': 0,
                'magNst': 0,
                
                # Standard fields
                'Depth Error': props.get('deptherror', np.nan),
                'Depth Seismic Stations': props.get('nst', 0),
                'Magnitude Error': props.get('magerror', np.nan),
                'Magnitude Seismic Stations': props.get('nst', 0),
                'Azimuthal Gap': props.get('gap', 180),
                'Horizontal Distance': props.get('dmin', 1.0),
                'Horizontal Error': props.get('horizerror', np.nan),
                'Root Mean Square': props.get('rms', 1.0)
            }
            
            return earthquake
            
        except Exception as e:
            print(f"Error parsing {source} data: {e}")
            return None
    
    def extract_country_from_place(self, place_string):
        """Extract country information from place description"""
        if not place_string:
            return 'Unknown'
        
        # Common patterns for extracting country
        country_indicators = {
            'Japan': ['Japan', 'Honshu', 'Kyushu', 'Shikoku'],
            'Chile': ['Chile', 'Chilean'],
            'Indonesia': ['Indonesia', 'Java', 'Sumatra', 'Sulawesi'],
            'United States': ['California', 'Alaska', 'Nevada', 'Hawaii', 'Puerto Rico'],
            'Mexico': ['Mexico', 'Baja California'],
            'Turkey': ['Turkey', 'Turkish'],
            'Greece': ['Greece', 'Greek'],
            'Italy': ['Italy', 'Italian'],
            'Iran': ['Iran', 'Iranian'],
            'Philippines': ['Philippines', 'Philippine'],
            'New Zealand': ['New Zealand'],
            'Russia': ['Russia', 'Russian', 'Siberia'],
            'China': ['China', 'Chinese', 'Tibet'],
            'India': ['India', 'Indian'],
            'Pakistan': ['Pakistan'],
            'Afghanistan': ['Afghanistan'],
            'Peru': ['Peru', 'Peruvian'],
            'Ecuador': ['Ecuador'],
            'Colombia': ['Colombia'],
            'Papua New Guinea': ['Papua New Guinea', 'PNG'],
            'Vanuatu': ['Vanuatu'],
            'Fiji': ['Fiji'],
            'Tonga': ['Tonga'],
            'Solomon Islands': ['Solomon Islands']
        }
        
        place_upper = place_string.upper()
        
        for country, indicators in country_indicators.items():
            if any(indicator.upper() in place_upper for indicator in indicators):
                return country
        
        # Check for ocean regions
        ocean_regions = {
            'North Pacific Ocean': ['PACIFIC', 'NORTH OF'],
            'South Pacific Ocean': ['PACIFIC', 'SOUTH OF'],
            'North Atlantic Ocean': ['ATLANTIC', 'NORTH OF'],
            'South Atlantic Ocean': ['ATLANTIC', 'SOUTH OF'],
            'Indian Ocean': ['INDIAN OCEAN'],
            'Arctic Ocean': ['ARCTIC'],
            'Southern Ocean': ['SOUTHERN OCEAN', 'ANTARCTICA']
        }
        
        for region, indicators in ocean_regions.items():
            if all(indicator in place_upper for indicator in indicators):
                return region
        
        return 'Unknown'
    
    def remove_duplicates(self, earthquakes_list):
        """Remove duplicate earthquakes based on time, location, and magnitude"""
        print(f"\n=== Removing Duplicates ===")
        print(f"Input earthquakes: {len(earthquakes_list)}")
        
        if not earthquakes_list:
            return []
        
        # Convert to DataFrame for easier processing
        df = pd.DataFrame(earthquakes_list)
        
        # Create datetime column
        df['datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], 
                                       format='%d/%m/%Y %H:%M:%S', errors='coerce')
        
        # Remove rows with invalid datetime
        df = df.dropna(subset=['datetime'])
        
        # Define duplicate criteria
        # Same time (within 1 minute), location (within 0.1 degrees), magnitude (within 0.2)
        print("Identifying duplicates...")
        
        duplicates_mask = pd.Series([False] * len(df))
        
        for i in range(len(df)):
            if duplicates_mask[i]:
                continue
                
            current_row = df.iloc[i]
            
            # Find potential duplicates
            time_diff = np.abs((df['datetime'] - current_row['datetime']).dt.total_seconds())
            lat_diff = np.abs(df['Latitude'] - current_row['Latitude'])
            lon_diff = np.abs(df['Longitude'] - current_row['Longitude'])
            mag_diff = np.abs(df['Magnitude'] - current_row['Magnitude'])
            
            # Criteria for duplicates
            is_duplicate = (
                (time_diff <= 60) &  # Within 1 minute
                (lat_diff <= 0.1) &  # Within 0.1 degrees
                (lon_diff <= 0.1) &  # Within 0.1 degrees
                (mag_diff <= 0.2)    # Within 0.2 magnitude units
            )
            
            # Mark duplicates (keep first occurrence)
            duplicate_indices = df.index[is_duplicate & (df.index > i)]
            duplicates_mask[duplicate_indices] = True
        
        # Remove duplicates
        df_unique = df[~duplicates_mask].copy()
        
        print(f"Removed {duplicates_mask.sum()} duplicates")
        print(f"Unique earthquakes: {len(df_unique)}")
        
        # Convert back to list of dictionaries
        df_unique = df_unique.drop('datetime', axis=1)
        return df_unique.to_dict('records')
    
    def enhance_earthquake_data(self, earthquakes_list):
        """Enhance earthquake data with additional computed fields"""
        print(f"\n=== Enhancing Earthquake Data ===")
        
        for earthquake in earthquakes_list:
            # Calculate additional features
            
            # Distance from major tectonic features (simplified)
            lat, lon = earthquake['Latitude'], earthquake['Longitude']
            
            # Distance from equator
            earthquake['distance_from_equator'] = abs(lat)
            
            # Distance from prime meridian
            earthquake['distance_from_prime_meridian'] = abs(lon)
            
            # Rough tectonic region classification
            earthquake['tectonic_region'] = self.classify_tectonic_region(lat, lon)
            
            # Depth classification
            depth = earthquake['Depth']
            if depth <= 35:
                earthquake['depth_class'] = 'shallow'
            elif depth <= 70:
                earthquake['depth_class'] = 'intermediate'
            elif depth <= 300:
                earthquake['depth_class'] = 'deep'
            else:
                earthquake['depth_class'] = 'very_deep'
            
            # Magnitude classification
            mag = earthquake['Magnitude']
            if mag < 4.0:
                earthquake['magnitude_class'] = 'minor'
            elif mag < 5.0:
                earthquake['magnitude_class'] = 'light'
            elif mag < 6.0:
                earthquake['magnitude_class'] = 'moderate'
            elif mag < 7.0:
                earthquake['magnitude_class'] = 'strong'
            elif mag < 8.0:
                earthquake['magnitude_class'] = 'major'
            else:
                earthquake['magnitude_class'] = 'great'
            
            # Potential tsunami risk (simplified)
            earthquake['tsunami_risk'] = (
                earthquake.get('tsunami', 0) > 0 or 
                (mag >= 6.5 and depth <= 50 and 
                 any(region in earthquake['tectonic_region'].lower() 
                     for region in ['subduction', 'ocean', 'coast']))
            )
        
        print(f"✓ Enhanced {len(earthquakes_list)} earthquake records")
        return earthquakes_list
    
    def classify_tectonic_region(self, lat, lon):
        """Classify earthquake location by tectonic region (simplified)"""
        
        # Pacific Ring of Fire
        if ((lat >= 35 and lat <= 70 and lon >= 120 and lon <= 170) or  # Japan, Kamchatka
            (lat >= -60 and lat <= 20 and lon >= 160 and lon <= 180) or  # Pacific islands
            (lat >= -60 and lat <= 60 and lon >= -180 and lon <= -120) or  # West coast Americas
            (lat >= -20 and lat <= 10 and lon >= 120 and lon <= 140)):  # Indonesia
            return 'Pacific Ring of Fire'
        
        # Mid-Atlantic Ridge
        elif (lon >= -40 and lon <= -10 and 
              ((lat >= -60 and lat <= -30) or (lat >= 0 and lat <= 70))):
            return 'Mid-Atlantic Ridge'
        
        # Mediterranean-Himalayan Belt
        elif (lat >= 25 and lat <= 45 and lon >= -10 and lon <= 60):
            return 'Mediterranean-Himalayan Belt'
        
        # Stable continental regions
        elif (lat >= 30 and lat <= 70 and lon >= -130 and lon <= -60):  # North America
            return 'North American Craton'
        elif (lat >= -40 and lat <= 40 and lon >= -80 and lon <= -35):  # South America
            return 'South American Platform'
        elif (lat >= 35 and lat <= 70 and lon >= 0 and lon <= 60):  # Europe/Asia
            return 'Eurasian Platform'
        
        # Ocean basins
        elif abs(lat) < 60:
            if lon >= -180 and lon <= -60:
                return 'Pacific Ocean Basin'
            elif lon >= -60 and lon <= 20:
                return 'Atlantic Ocean Basin'
            elif lon >= 20 and lon <= 140:
                return 'Indian Ocean Basin'
        
        return 'Other'
    
    def save_enhanced_data(self, earthquakes_list, filename='enhanced_earthquake_database.csv'):
        """Save enhanced earthquake data to CSV"""
        if not earthquakes_list:
            print("No data to save")
            return None
        
        df = pd.DataFrame(earthquakes_list)
        
        # Sort by datetime
        df['datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], 
                                       format='%d/%m/%Y %H:%M:%S', errors='coerce')
        df = df.sort_values('datetime').reset_index(drop=True)
        df = df.drop('datetime', axis=1)
        
        # Save to CSV
        output_file = os.path.join(self.output_dir, filename)
        df.to_csv(output_file, index=False)
        
        print(f"\n✓ Saved {len(df)} earthquakes to: {output_file}")
        
        # Create summary
        self.create_data_summary(df, filename.replace('.csv', '_summary.json'))
        
        return output_file
    
    def create_data_summary(self, df, summary_filename):
        """Create a summary of the earthquake dataset"""
        summary = {
            'total_earthquakes': len(df),
            'date_range': {
                'start': df['Date'].min(),
                'end': df['Date'].max()
            },
            'magnitude_stats': {
                'min': float(df['Magnitude'].min()),
                'max': float(df['Magnitude'].max()),
                'mean': float(df['Magnitude'].mean()),
                'median': float(df['Magnitude'].median())
            },
            'depth_stats': {
                'min': float(df['Depth'].min()),
                'max': float(df['Depth'].max()),
                'mean': float(df['Depth'].mean()),
                'median': float(df['Depth'].median())
            },
            'sources': df['Source'].value_counts().to_dict(),
            'countries': df['Country'].value_counts().head(10).to_dict(),
            'magnitude_classes': df['magnitude_class'].value_counts().to_dict() if 'magnitude_class' in df.columns else {},
            'depth_classes': df['depth_class'].value_counts().to_dict() if 'depth_class' in df.columns else {},
            'tectonic_regions': df['tectonic_region'].value_counts().to_dict() if 'tectonic_region' in df.columns else {}
        }
        
        summary_file = os.path.join(self.output_dir, summary_filename)
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"✓ Data summary saved to: {summary_file}")
        return summary
    
    def fetch_all_data(self, include_historical=True, historical_days=365, min_magnitude=4.0):
        """Fetch earthquake data from all available sources"""
        print("="*80)
        print("ENHANCED EARTHQUAKE DATA FETCHER")
        print("="*80)
        
        all_earthquakes = []
        
        # Fetch from different sources
        try:
            # USGS feeds
            usgs_earthquakes = self.fetch_usgs_feeds()
            all_earthquakes.extend(usgs_earthquakes)
            print(f"Total from USGS feeds: {len(usgs_earthquakes)}")
            
            # USGS historical (if requested)
            if include_historical:
                start_date = (datetime.now() - timedelta(days=historical_days)).strftime('%Y-%m-%d')
                historical_earthquakes = self.fetch_usgs_historical(
                    start_date=start_date, 
                    min_magnitude=min_magnitude
                )
                all_earthquakes.extend(historical_earthquakes)
                print(f"Total from USGS historical: {len(historical_earthquakes)}")
            
            # EMSC data
            emsc_earthquakes = self.fetch_emsc_data(
                days_back=min(historical_days, 90), 
                min_magnitude=min_magnitude
            )
            all_earthquakes.extend(emsc_earthquakes)
            print(f"Total from EMSC: {len(emsc_earthquakes)}")
            
            # Global networks
            global_earthquakes = self.fetch_global_networks(
                days_back=min(historical_days, 30), 
                min_magnitude=min_magnitude + 1  # Higher threshold for global networks
            )
            all_earthquakes.extend(global_earthquakes)
            print(f"Total from global networks: {len(global_earthquakes)}")
            
        except Exception as e:
            print(f"Error during data fetching: {e}")
        
        print(f"\nTotal earthquakes before processing: {len(all_earthquakes)}")
        
        if all_earthquakes:
            # Remove duplicates
            unique_earthquakes = self.remove_duplicates(all_earthquakes)
            
            # Enhance data
            enhanced_earthquakes = self.enhance_earthquake_data(unique_earthquakes)
            
            # Save data
            output_file = self.save_enhanced_data(enhanced_earthquakes)
            
            print("\n" + "="*80)
            print("ENHANCED DATA FETCHING COMPLETED SUCCESSFULLY!")
            print(f"Final dataset: {len(enhanced_earthquakes)} unique earthquakes")
            print(f"Saved to: {output_file}")
            print("="*80)
            
            return output_file
        else:
            print("\n⚠ No earthquake data was successfully fetched")
            return None


def main():
    """Main function to fetch enhanced earthquake data"""
    fetcher = EnhancedEarthquakeDataFetcher()
    
    # Fetch all available data
    output_file = fetcher.fetch_all_data(
        include_historical=True,
        historical_days=730,  # 2 years of historical data
        min_magnitude=4.0
    )
    
    if output_file:
        print(f"\n✓ Enhanced earthquake data saved to: {output_file}")
    else:
        print("\n⚠ No data was fetched")


if __name__ == "__main__":
    main()
