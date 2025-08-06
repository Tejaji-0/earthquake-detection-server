import React, { useState } from 'react';
import type { EarthquakeData } from '../types/earthquake';
import { formatDate } from '../utils/dataLoader';
import { ChevronUp, ChevronDown } from 'lucide-react';
import './EarthquakeTable.css';

interface EarthquakeTableProps {
  data: EarthquakeData[];
}

type SortField = keyof EarthquakeData;
type SortDirection = 'asc' | 'desc';

const EarthquakeTable: React.FC<EarthquakeTableProps> = ({ data }) => {
  const [sortField, setSortField] = useState<SortField>('Magnitude');
  const [sortDirection, setSortDirection] = useState<SortDirection>('desc');
  const [currentPage, setCurrentPage] = useState(1);
  const itemsPerPage = 20;

  const handleSort = (field: SortField) => {
    if (sortField === field) {
      setSortDirection(sortDirection === 'asc' ? 'desc' : 'asc');
    } else {
      setSortField(field);
      setSortDirection('desc');
    }
    setCurrentPage(1); // Reset to first page when sorting
  };

  const sortedData = [...data].sort((a, b) => {
    let aValue = a[sortField];
    let bValue = b[sortField];
    
    // Handle undefined values
    if (aValue === undefined) aValue = '';
    if (bValue === undefined) bValue = '';
    
    // Handle string comparison
    if (typeof aValue === 'string' && typeof bValue === 'string') {
      aValue = aValue.toLowerCase();
      bValue = bValue.toLowerCase();
    }
    
    if (aValue < bValue) return sortDirection === 'asc' ? -1 : 1;
    if (aValue > bValue) return sortDirection === 'asc' ? 1 : -1;
    return 0;
  });

  const totalPages = Math.ceil(sortedData.length / itemsPerPage);
  const startIndex = (currentPage - 1) * itemsPerPage;
  const paginatedData = sortedData.slice(startIndex, startIndex + itemsPerPage);

  const SortHeader: React.FC<{ field: SortField; children: React.ReactNode }> = ({ field, children }) => (
    <th onClick={() => handleSort(field)} className="sortable">
      <div className="sort-header">
        {children}
        {sortField === field && (
          sortDirection === 'asc' ? <ChevronUp size={16} /> : <ChevronDown size={16} />
        )}
      </div>
    </th>
  );

  return (
    <div className="earthquake-table-container">
      <div className="table-info">
        <p>Showing {paginatedData.length} of {data.length} earthquakes</p>
      </div>
      
      <div className="table-wrapper">
        <table className="earthquake-table">
          <thead>
            <tr>
              <SortHeader field="Magnitude">Magnitude</SortHeader>
              <SortHeader field="date_time">Date & Time</SortHeader>
              <SortHeader field="location">Location</SortHeader>
              <SortHeader field="Country">Country</SortHeader>
              <SortHeader field="Depth">Depth (km)</SortHeader>
              <SortHeader field="alert">Alert</SortHeader>
              <SortHeader field="Magnitude Type">Mag Type</SortHeader>
              <SortHeader field="Source">Source</SortHeader>
            </tr>
          </thead>
          <tbody>
            {paginatedData.map((earthquake, index) => (
              <tr key={index} className="earthquake-row">
                <td 
                  className={`magnitude-cell magnitude-${Math.floor(earthquake.Magnitude)}`}
                >
                  <strong>{earthquake.Magnitude.toFixed(1)}</strong>
                </td>
                <td className="date-cell">{formatDate(earthquake.date_time || `${earthquake.Date} ${earthquake.Time}`)}</td>
                <td className="location-cell">
                  <div className="location-info">
                    <div className="location-title">{earthquake.location || `${earthquake.Latitude.toFixed(2)}°, ${earthquake.Longitude.toFixed(2)}°`}</div>
                    <div className="coords">ID: {earthquake.ID}</div>
                  </div>
                </td>
                <td className="country-cell">
                  <span className="country-name">{earthquake.Country || 'Unknown'}</span>
                </td>
                <td>{earthquake.Depth ? earthquake.Depth.toFixed(1) : 'N/A'}</td>
                <td>
                  <span className={`alert-badge alert-${earthquake.alert || 'none'}`}>
                    {earthquake.alert || 'N/A'}
                  </span>
                </td>
                <td>{earthquake['Magnitude Type'] || 'N/A'}</td>
                <td>{earthquake.Source || 'N/A'}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {totalPages > 1 && (
        <div className="pagination">
          <button 
            onClick={() => setCurrentPage(p => Math.max(1, p - 1))}
            disabled={currentPage === 1}
          >
            Previous
          </button>
          
          <span className="page-info">
            Page {currentPage} of {totalPages}
          </span>
          
          <button 
            onClick={() => setCurrentPage(p => Math.min(totalPages, p + 1))}
            disabled={currentPage === totalPages}
          >
            Next
          </button>
        </div>
      )}
    </div>
  );
};

export default EarthquakeTable;
