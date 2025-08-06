import React from 'react';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  BarElement,
  Title,
  Tooltip,
  Legend,
  ArcElement,
} from 'chart.js';
import { Bar, Doughnut } from 'react-chartjs-2';
import type { EarthquakeData } from '../types/earthquake';
import './EarthquakeCharts.css';

ChartJS.register(
  CategoryScale,
  LinearScale,
  BarElement,
  Title,
  Tooltip,
  Legend,
  ArcElement
);

interface EarthquakeChartsProps {
  data: EarthquakeData[];
}

const EarthquakeCharts: React.FC<EarthquakeChartsProps> = ({ data }) => {
  // Magnitude distribution chart
  const magnitudeRanges = ['6.5-6.9', '7.0-7.4', '7.5-7.9', '8.0-8.4', '8.5+'];
  const magnitudeCounts = magnitudeRanges.map(range => {
    const [min, maxStr] = range.split('-');
    const minVal = parseFloat(min);
    const maxVal = maxStr === '8.5+' ? 10 : parseFloat(maxStr);
    
    return data.filter(eq => {
      if (maxStr === '8.5+') {
        return eq.Magnitude >= minVal;
      }
      return eq.Magnitude >= minVal && eq.Magnitude <= maxVal;
    }).length;
  });

  const magnitudeChartData = {
    labels: magnitudeRanges,
    datasets: [
      {
        label: 'Number of Earthquakes',
        data: magnitudeCounts,
        backgroundColor: [
          '#FFB300',
          '#FF6600',
          '#FF0000',
          '#8B0000',
          '#4B0000',
        ],
        borderColor: [
          '#FF8F00',
          '#E65100',
          '#D32F2F',
          '#B71C1C',
          '#3E0000',
        ],
        borderWidth: 1,
      },
    ],
  };

  // Alert level distribution
  const alertLevels = ['green', 'yellow', 'orange', 'red', 'none'];
  const alertCounts = alertLevels.map(level => {
    if (level === 'none') {
      return data.filter(eq => !eq.alert || eq.alert === '').length;
    }
    return data.filter(eq => eq.alert === level).length;
  });

  const alertChartData = {
    labels: ['Green', 'Yellow', 'Orange', 'Red', 'None'],
    datasets: [
      {
        data: alertCounts,
        backgroundColor: [
          '#28a745',
          '#ffc107',
          '#fd7e14',
          '#dc3545',
          '#6c757d',
        ],
        borderColor: [
          '#1e7e34',
          '#e0a800',
          '#e8570c',
          '#c82333',
          '#5a6268',
        ],
        borderWidth: 2,
      },
    ],
  };

  // Top countries/regions by earthquake count
  const countries = [...new Set(data.map(eq => eq.Country).filter(c => c && c !== 'Unknown'))];
  const countryCounts = countries
    .map(country => ({
      country,
      count: data.filter(eq => eq.Country === country).length
    }))
    .sort((a, b) => b.count - a.count)
    .slice(0, 8); // Top 8 countries

  const countryData = {
    labels: countryCounts.map(item => 
      item.country.length > 15 ? item.country.substring(0, 15) + '...' : item.country
    ),
    datasets: [
      {
        data: countryCounts.map(item => item.count),
        backgroundColor: [
          '#007bff', '#28a745', '#ffc107', '#dc3545', 
          '#6f42c1', '#fd7e14', '#20c997', '#e83e8c'
        ],
        borderColor: [
          '#0056b3', '#1e7e34', '#e0a800', '#c82333',
          '#5a2d91', '#e8570c', '#17a2b8', '#d1477e'
        ],
        borderWidth: 2,
      },
    ],
  };

  // Chart options
  const barOptions = {
    responsive: true,
    plugins: {
      legend: {
        position: 'top' as const,
      },
      title: {
        display: true,
        text: 'Earthquake Magnitude Distribution',
      },
    },
    scales: {
      y: {
        beginAtZero: true,
        ticks: {
          stepSize: 1,
        },
      },
    },
  };

  const doughnutOptions = {
    responsive: true,
    plugins: {
      legend: {
        position: 'bottom' as const,
      },
    },
  };

  return (
    <div className="earthquake-charts">
      <div className="chart-container">
        <Bar data={magnitudeChartData} options={barOptions} />
      </div>
      
      <div className="charts-row">
        <div className="chart-container small">
          <h3>Alert Level Distribution</h3>
          <Doughnut data={alertChartData} options={doughnutOptions} />
        </div>
        
        <div className="chart-container small">
          <h3>Top Countries/Regions</h3>
          <Doughnut data={countryData} options={doughnutOptions} />
        </div>
      </div>

      <div className="stats-summary">
        <div className="stat-item">
          <span className="stat-value">{data.length}</span>
          <span className="stat-label">Total Earthquakes</span>
        </div>
        <div className="stat-item">
          <span className="stat-value">
            {data.length > 0 ? (data.reduce((sum, eq) => sum + eq.Magnitude, 0) / data.length).toFixed(1) : '0'}
          </span>
          <span className="stat-label">Average Magnitude</span>
        </div>
        <div className="stat-item">
          <span className="stat-value">
            {data.length > 0 ? Math.max(...data.map(eq => eq.Magnitude)).toFixed(1) : '0'}
          </span>
          <span className="stat-label">Highest Magnitude</span>
        </div>
        <div className="stat-item">
          <span className="stat-value">
            {data.filter(eq => eq.Depth && eq.Depth > 100).length}
          </span>
          <span className="stat-label">Deep Events (&gt;100km)</span>
        </div>
        <div className="stat-item">
          <span className="stat-value">
            {new Set(data.map(eq => eq.Country).filter(c => c && c !== 'Unknown')).size}
          </span>
          <span className="stat-label">Countries/Regions</span>
        </div>
      </div>
    </div>
  );
};

export default EarthquakeCharts;
