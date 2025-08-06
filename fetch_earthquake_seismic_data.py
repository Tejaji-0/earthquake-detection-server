#!/usr/bin/env python3
"""
Earthquake Seismic Data Fetcher
This script fetches seismic waveform data from 1 week before to 1 week after 
earthquake events in the database.csv file.
"""

import pandas as pd
import numpy as np
import json
import os
import time
from datetime import datetime, timedelta
from obspy import UTCDateTime
from obspy.clients.fdsn import Client
from obspy.clients.fdsn.header import FDSNException
import warnings
warnings.filterwarnings('ignore')

class EarthquakeSeismicFetcher:
    def __init__(self, earthquake_csv='data/database.csv', output_dir='earthquake_seismic_data'):
        self.earthquake_csv = earthquake_csv
        self.output_dir = output_dir
        
        # Initialize IRIS client (most reliable for global data)
        try:
            self.client = Client('IRIS')
            print("✓ IRIS client initialized successfully")
        except Exception as e:
            print(f"✗ Failed to initialize IRIS client: {e}")
            self.client = None
            return
        
        # Create output directory
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            
        # Load earthquake database
        self.load_earthquake_database()
    
    def load_earthquake_database(self):
        """Load earthquake data from database.csv"""
        try:
            self.earthquakes = pd.read_csv(self.earthquake_csv)
            print(f"✓ Loaded {len(self.earthquakes)} earthquakes from {self.earthquake_csv}")
            
            # Convert date/time columns to datetime, handling different formats
            def parse_datetime(row):
                try:
                    # Try standard format first
                    if '/' in row['Date']:
                        return pd.to_datetime(row['Date'] + ' ' + str(row['Time']).replace('nan', '00:00:00'),
                                            format='%m/%d/%Y %H:%M:%S')
                    else:
                        # Handle ISO format dates
                        return pd.to_datetime(row['Date'])
                except:
                    # Fallback to generic parsing
                    try:
                        return pd.to_datetime(str(row['Date']) + ' ' + str(row['Time']).replace('nan', '00:00:00'))
                    except:
                        return pd.NaT
            
            self.earthquakes['datetime'] = self.earthquakes.apply(parse_datetime, axis=1)
            
            # Remove rows with invalid dates
            initial_count = len(self.earthquakes)
            self.earthquakes = self.earthquakes.dropna(subset=['datetime'])
            print(f"✓ Removed {initial_count - len(self.earthquakes)} rows with invalid dates")
            
            # Filter for significant earthquakes (magnitude >= 6.0)
            self.earthquakes = self.earthquakes[
                (self.earthquakes['Magnitude'] >= 6.0) & 
                (self.earthquakes['Magnitude'].notna())
            ].copy()
            
            print(f"✓ Filtered to {len(self.earthquakes)} significant earthquakes (M >= 6.0)")
            
        except Exception as e:
            print(f"✗ Failed to load earthquake database: {e}")
            self.earthquakes = pd.DataFrame()
    
    def get_global_stations(self):
        """Get reliable global seismic stations"""
        # Global Seismographic Network (GSN) stations - most reliable
        return [
            {'network': 'IU', 'station': 'ANMO', 'name': 'Albuquerque, NM'},
            {'network': 'IU', 'station': 'HRV', 'name': 'Harvard, MA'},
            {'network': 'IU', 'station': 'COLA', 'name': 'College, AK'},
            {'network': 'IU', 'station': 'CCM', 'name': 'Cathedral Cave, MO'},
            {'network': 'IU', 'station': 'FUNA', 'name': 'Funafuti, Tuvalu'},
            {'network': 'IU', 'station': 'GUMO', 'name': 'Guam, Mariana Is'},
            {'network': 'IU', 'station': 'MAJO', 'name': 'Matsushiro, Japan'},
            {'network': 'IU', 'station': 'PAB', 'name': 'San Pablo, Spain'},
            {'network': 'IU', 'station': 'PMSA', 'name': 'Palmer Station, Antarctica'},
            {'network': 'IU', 'station': 'QSPA', 'name': 'South Pole, Antarctica'},
            {'network': 'IU', 'station': 'CTAO', 'name': 'Charters Towers, Australia'},
            {'network': 'IU', 'station': 'COCO', 'name': 'Cocos Islands'},
            {'network': 'IU', 'station': 'KONO', 'name': 'Kongsberg, Norway'},
            {'network': 'IU', 'station': 'LSZ', 'name': 'Lusaka, Zambia'},
            {'network': 'IU', 'station': 'MBWA', 'name': 'Marble Bar, Australia'},
        ]
    
    def fetch_seismic_data_for_earthquake(self, earthquake_row, max_stations=5):
        """
        Fetch seismic data for 1 week before and after an earthquake
        """
        earthquake_time = earthquake_row['datetime']
        magnitude = earthquake_row['Magnitude']
        eq_id = earthquake_row['ID']
        
        # Define time window: 1 week before to 1 week after
        start_time = UTCDateTime(earthquake_time - timedelta(days=7))
        end_time = UTCDateTime(earthquake_time + timedelta(days=7))
        
        print(f"\nFetching data for earthquake {eq_id} (M{magnitude}) at {earthquake_time}")
        print(f"Time window: {start_time} to {end_time}")
        
        stations = self.get_global_stations()
        fetched_data = []
        stations_fetched = 0
        
        for station in stations:
            if stations_fetched >= max_stations:
                break
                
            try:
                print(f"  Trying station {station['network']}.{station['station']} ({station['name']})...")
                
                # Try to get waveform data
                st = self.client.get_waveforms(
                    network=station['network'],
                    station=station['station'],
                    location='*',
                    channel='BHZ',  # Vertical component, broadband
                    starttime=start_time,
                    endtime=end_time
                )
                
                if len(st) > 0:
                    # Create filename for this station's data
                    filename = f"{eq_id}_{station['network']}_{station['station']}_M{magnitude:.1f}.mseed"
                    filepath = os.path.join(self.output_dir, filename)
                    
                    # Save waveform data
                    st.write(filepath, format='MSEED')
                    
                    fetched_data.append({
                        'earthquake_id': eq_id,
                        'magnitude': magnitude,
                        'earthquake_time': earthquake_time,
                        'network': station['network'],
                        'station': station['station'],
                        'station_name': station['name'],
                        'traces': len(st),
                        'filename': filename,
                        'start_time': str(start_time),
                        'end_time': str(end_time)
                    })
                    
                    stations_fetched += 1
                    print(f"    ✓ Successfully fetched {len(st)} traces from {station['network']}.{station['station']}")
                    
                    # Small delay to be respectful to the data center
                    time.sleep(1)
                
            except FDSNException as e:
                print(f"    ✗ FDSN error for {station['network']}.{station['station']}: {e}")
                continue
            except Exception as e:
                print(f"    ✗ Error fetching data from {station['network']}.{station['station']}: {e}")
                continue
        
        return fetched_data
    
    def fetch_data_for_major_earthquakes(self, magnitude_threshold=7.0, max_events=10):
        """
        Fetch seismic data for major earthquakes (M >= 7.0)
        """
        if self.client is None:
            print("✗ No client available for data fetching")
            return
        
        # Filter for major earthquakes
        major_earthquakes = self.earthquakes[
            self.earthquakes['Magnitude'] >= magnitude_threshold
        ].sort_values('Magnitude', ascending=False).head(max_events)
        
        print(f"\nFetching seismic data for {len(major_earthquakes)} major earthquakes (M >= {magnitude_threshold})")
        
        all_fetched_data = []
        
        for idx, earthquake in major_earthquakes.iterrows():
            try:
                earthquake_data = self.fetch_seismic_data_for_earthquake(earthquake)
                all_fetched_data.extend(earthquake_data)
                
                # Longer delay between earthquakes
                time.sleep(3)
                
            except Exception as e:
                print(f"✗ Error processing earthquake {earthquake['ID']}: {e}")
                continue
        
        # Save metadata
        metadata_file = os.path.join(self.output_dir, 'earthquake_seismic_metadata.json')
        with open(metadata_file, 'w') as f:
            json.dump(all_fetched_data, f, indent=2, default=str)
        
        print(f"\n✓ Completed! Fetched data for {len(all_fetched_data)} station-earthquake combinations")
        print(f"✓ Metadata saved to {metadata_file}")
        
        return all_fetched_data
    
    def fetch_data_for_specific_earthquake(self, earthquake_id):
        """
        Fetch seismic data for a specific earthquake by ID
        """
        if self.client is None:
            print("✗ No client available for data fetching")
            return
        
        earthquake = self.earthquakes[self.earthquakes['ID'] == earthquake_id]
        
        if earthquake.empty:
            print(f"✗ Earthquake {earthquake_id} not found in database")
            return
        
        earthquake_row = earthquake.iloc[0]
        return self.fetch_seismic_data_for_earthquake(earthquake_row)

def main():
    """Main function to demonstrate the fetcher"""
    fetcher = EarthquakeSeismicFetcher()
    
    if len(fetcher.earthquakes) == 0:
        print("No earthquake data available. Exiting.")
        return
    
    print("=== Earthquake Seismic Data Fetcher ===")
    print("This script fetches seismic data from 1 week before to 1 week after earthquake events.")
    print("\nOptions:")
    print("1. Fetch data for top 5 major earthquakes (M >= 7.0)")
    print("2. Fetch data for top 10 significant earthquakes (M >= 6.5)")
    print("3. Fetch data for a specific earthquake ID")
    
    choice = input("\nEnter your choice (1-3): ").strip()
    
    if choice == '1':
        fetcher.fetch_data_for_major_earthquakes(magnitude_threshold=7.0, max_events=5)
    elif choice == '2':
        fetcher.fetch_data_for_major_earthquakes(magnitude_threshold=6.5, max_events=10)
    elif choice == '3':
        earthquake_id = input("Enter earthquake ID: ").strip()
        fetcher.fetch_data_for_specific_earthquake(earthquake_id)
    else:
        print("Invalid choice. Exiting.")

if __name__ == "__main__":
    main()