import React from 'react';
import type { FilterOptions } from '../types/earthquake';
import './FilterPanel.css';

interface FilterPanelProps {
  filters: FilterOptions;
  onFiltersChange: (filters: FilterOptions) => void;
}

const FilterPanel: React.FC<FilterPanelProps> = ({ filters, onFiltersChange }) => {
  const handleFilterChange = (key: keyof FilterOptions, value: string | number) => {
    onFiltersChange({ ...filters, [key]: value });
  };

  return (
    <div className="filter-panel">
      <h3>Filter Earthquakes</h3>
      
      <div className="filter-group">
        <label>Magnitude Range:</label>
        <div className="range-inputs">
          <input
            type="number"
            min="6.5"
            max="9.5"
            step="0.1"
            value={filters.minMagnitude}
            onChange={(e) => handleFilterChange('minMagnitude', parseFloat(e.target.value))}
            placeholder="Min"
          />
          <span>-</span>
          <input
            type="number"
            min="6.5"
            max="9.5"
            step="0.1"
            value={filters.maxMagnitude}
            onChange={(e) => handleFilterChange('maxMagnitude', parseFloat(e.target.value))}
            placeholder="Max"
          />
        </div>
      </div>

      <div className="filter-group">
        <label>Date Range:</label>
        <div className="date-inputs">
          <input
            type="date"
            value={filters.startDate}
            onChange={(e) => handleFilterChange('startDate', e.target.value)}
            aria-label="Start date"
            title="Filter start date"
          />
          <input
            type="date"
            value={filters.endDate}
            onChange={(e) => handleFilterChange('endDate', e.target.value)}
            aria-label="End date"
            title="Filter end date"
          />
        </div>
      </div>

      <div className="filter-group">
        <label>Location:</label>
        <input
          type="text"
          value={filters.location}
          onChange={(e) => handleFilterChange('location', e.target.value)}
          placeholder="Search by location"
        />
      </div>

      <div className="filter-group">
        <label>Country/Region:</label>
        <input
          type="text"
          value={filters.country}
          onChange={(e) => handleFilterChange('country', e.target.value)}
          placeholder="Search by country or region"
        />
      </div>

      <div className="filter-group">
        <label>Alert Level:</label>
        <select
          value={filters.alertLevel}
          onChange={(e) => handleFilterChange('alertLevel', e.target.value)}
          aria-label="Alert level filter"
          title="Filter by alert level"
        >
          <option value="">All Levels</option>
          <option value="green">Green</option>
          <option value="yellow">Yellow</option>
          <option value="red">Red</option>
        </select>
      </div>
    </div>
  );
};

export default FilterPanel;
