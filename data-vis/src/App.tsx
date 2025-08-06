import { useState, useEffect, useMemo } from 'react';
import type { EarthquakeData, FilterOptions } from './types/earthquake';
import { loadEarthquakeData, filterEarthquakeData } from './utils/dataLoader';
import FilterPanel from './components/FilterPanel';
import EarthquakeTable from './components/EarthquakeTable';
import EarthquakeCharts from './components/EarthquakeCharts';
import EventAnalysis from './components/EventAnalysis';
import { Activity, BarChart3, Table, Zap } from 'lucide-react';
import './App.css';

function App() {
  const [earthquakeData, setEarthquakeData] = useState<EarthquakeData[]>([]);
  const [loading, setLoading] = useState(true);
  const [activeTab, setActiveTab] = useState<'overview' | 'table' | 'analysis'>('overview');
  const [filters, setFilters] = useState<FilterOptions>({
    minMagnitude: 5.0,
    maxMagnitude: 9.5,
    startDate: '1965-01-01',
    endDate: '2025-12-31',
    location: '',
    alertLevel: '',
    country: '',
  });

  useEffect(() => {
    const fetchData = async () => {
      setLoading(true);
      try {
        const data = await loadEarthquakeData();
        setEarthquakeData(data);
      } catch (error) {
        console.error('Failed to load earthquake data:', error);
      } finally {
        setLoading(false);
      }
    };

    fetchData();
  }, []);

  const filteredData = useMemo(() => {
    return filterEarthquakeData(earthquakeData, filters);
  }, [earthquakeData, filters]);

  if (loading) {
    return (
      <div className="loading-container">
        <Activity className="loading-icon" size={48} />
        <h2>Loading Earthquake Data...</h2>
        <p>Please wait while we fetch the earthquake dataset.</p>
      </div>
    );
  }

  return (
    <div className="app">
      <header className="app-header">
        <div className="header-content">
          <div className="header-title">
            <Activity size={32} className="header-icon" />
            <h1>Earthquake Data Visualization</h1>
          </div>
          <p className="header-subtitle">
            Global Earthquake Database from 1965-2025 | Magnitude 5.0+
          </p>
        </div>
      </header>

      <main className="app-main">
        <div className="container">
          <FilterPanel filters={filters} onFiltersChange={setFilters} />

          <div className="tab-navigation">
            <button
              className={`tab-button ${activeTab === 'overview' ? 'active' : ''}`}
              onClick={() => setActiveTab('overview')}
            >
              <BarChart3 size={20} />
              Overview & Charts
            </button>
            <button
              className={`tab-button ${activeTab === 'table' ? 'active' : ''}`}
              onClick={() => setActiveTab('table')}
            >
              <Table size={20} />
              Data Table
            </button>
            <button
              className={`tab-button ${activeTab === 'analysis' ? 'active' : ''}`}
              onClick={() => setActiveTab('analysis')}
            >
              <Zap size={20} />
              Event Analysis
            </button>
          </div>

          <div className="tab-content">
            {activeTab === 'overview' ? (
              <EarthquakeCharts data={filteredData} />
            ) : activeTab === 'table' ? (
              <EarthquakeTable data={filteredData} />
            ) : (
              <EventAnalysis />
            )}
          </div>
        </div>
      </main>

      <footer className="app-footer">
        <p>
          Data Source: Global Earthquake Database | 
          Built with React, TypeScript & Chart.js
        </p>
      </footer>
    </div>
  );
}

export default App;
