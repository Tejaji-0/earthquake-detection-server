import { useState, useEffect } from 'react';
import type { ProcessingSummary, DetailedAnalysis, TopEvent } from '../types/eventAnalysis';
import { loadEventAnalysis } from '../utils/eventAnalysisLoader';
import { Activity, TrendingUp, TrendingDown, BarChart, Zap } from 'lucide-react';
import './EventAnalysis.css';

type EventAnalysisProps = object;

const EventAnalysis: React.FC<EventAnalysisProps> = () => {
  const [processingSummary, setProcessingSummary] = useState<ProcessingSummary | null>(null);
  const [detailedAnalysis, setDetailedAnalysis] = useState<DetailedAnalysis | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const fetchAnalysisData = async () => {
      setLoading(true);
      try {
        const { processingSummary, detailedAnalysis } = await loadEventAnalysis();
        setProcessingSummary(processingSummary);
        setDetailedAnalysis(detailedAnalysis);
        
        // Set CSS custom properties for percentage bars
        setTimeout(() => {
          const foreshockBar = document.querySelector('.percentage-fill.foreshock') as HTMLElement;
          const aftershockBar = document.querySelector('.percentage-fill.aftershock') as HTMLElement;
          const bothBar = document.querySelector('.percentage-fill.both') as HTMLElement;
          
          if (foreshockBar && aftershockBar && bothBar) {
            const foreshockPct = (detailedAnalysis.events_with_foreshocks / detailedAnalysis.total_events) * 100;
            const aftershockPct = (detailedAnalysis.events_with_aftershocks / detailedAnalysis.total_events) * 100;
            const bothPct = (detailedAnalysis.events_with_both / detailedAnalysis.total_events) * 100;
            
            foreshockBar.style.width = `${foreshockPct}%`;
            aftershockBar.style.width = `${aftershockPct}%`;
            bothBar.style.width = `${bothPct}%`;
          }
        }, 100);
      } catch (err) {
        setError('Failed to load event analysis data');
        console.error('Error loading event analysis:', err);
      } finally {
        setLoading(false);
      }
    };

    fetchAnalysisData();
  }, []);

  if (loading) {
    return (
      <div className="event-analysis-loading">
        <Activity className="loading-spinner" size={32} />
        <p>Loading earthquake event analysis...</p>
      </div>
    );
  }

  if (error || !processingSummary || !detailedAnalysis) {
    return (
      <div className="event-analysis-error">
        <p>Error: {error || 'Failed to load analysis data'}</p>
      </div>
    );
  }

  const renderTopEvents = (events: TopEvent[], title: string, icon: React.ReactNode) => (
    <div className="top-events-section">
      <h3>
        {icon}
        {title}
      </h3>
      <div className="top-events-list">
        {events.slice(0, 5).map((event, index) => (
          <div key={index} className="top-event-item">
            <span className="event-rank">#{index + 1}</span>
            <div className="event-details">
              <div className="event-title">{event.event}</div>
              <div className="event-count">{event.count} events</div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );

  const eventsWithSeismicActivity = processingSummary.events.filter(
    event => event.before_count > 0 || event.after_count > 0
  );

  const totalForeshocks = processingSummary.events.reduce((sum, event) => sum + event.before_count, 0);
  const totalAftershocks = processingSummary.events.reduce((sum, event) => sum + event.after_count, 0);

  const foreshockPercentage = (detailedAnalysis.events_with_foreshocks / detailedAnalysis.total_events) * 100;
  const aftershockPercentage = (detailedAnalysis.events_with_aftershocks / detailedAnalysis.total_events) * 100;
  const bothPercentage = (detailedAnalysis.events_with_both / detailedAnalysis.total_events) * 100;

  return (
    <div className="event-analysis">
      <div className="analysis-header">
        <h2>
          <Zap size={24} />
          Earthquake Event Analysis
        </h2>
        <p>Foreshock and Aftershock Pattern Analysis</p>
      </div>

      {/* Summary Statistics */}
      <div className="analysis-summary">
        <div className="summary-card">
          <div className="summary-icon">
            <BarChart size={24} />
          </div>
          <div className="summary-content">
            <h3>{detailedAnalysis.total_events}</h3>
            <p>Total Events Analyzed</p>
          </div>
        </div>

        <div className="summary-card">
          <div className="summary-icon">
            <TrendingUp size={24} />
          </div>
          <div className="summary-content">
            <h3>{detailedAnalysis.events_with_foreshocks}</h3>
            <p>Events with Foreshocks</p>
          </div>
        </div>

        <div className="summary-card">
          <div className="summary-icon">
            <TrendingDown size={24} />
          </div>
          <div className="summary-content">
            <h3>{detailedAnalysis.events_with_aftershocks}</h3>
            <p>Events with Aftershocks</p>
          </div>
        </div>

        <div className="summary-card">
          <div className="summary-icon">
            <Activity size={24} />
          </div>
          <div className="summary-content">
            <h3>{detailedAnalysis.events_with_both}</h3>
            <p>Events with Both</p>
          </div>
        </div>
      </div>

      {/* Additional Statistics */}
      <div className="analysis-stats">
        <div className="stat-item">
          <span className="stat-label">Events with Seismic Activity:</span>
          <span className="stat-value">{eventsWithSeismicActivity.length}</span>
        </div>
        <div className="stat-item">
          <span className="stat-label">Total Foreshocks Detected:</span>
          <span className="stat-value">{totalForeshocks}</span>
        </div>
        <div className="stat-item">
          <span className="stat-label">Total Aftershocks Detected:</span>
          <span className="stat-value">{totalAftershocks}</span>
        </div>
        <div className="stat-item">
          <span className="stat-label">Analysis Date:</span>
          <span className="stat-value">
            {new Date(detailedAnalysis.analysis_date).toLocaleDateString()}
          </span>
        </div>
      </div>

      {/* Top Events */}
      <div className="top-events-container">
        {renderTopEvents(
          detailedAnalysis.top_foreshock_events,
          'Top Events with Foreshocks',
          <TrendingUp size={20} />
        )}
        {renderTopEvents(
          detailedAnalysis.top_aftershock_events,
          'Top Events with Aftershocks',
          <TrendingDown size={20} />
        )}
      </div>

      {/* Analysis Visualization */}
      <div className="analysis-visualization">
        <h3>
          <BarChart size={20} />
          Seismic Activity Patterns
        </h3>
        <div className="visualization-container">
          <img 
            src="/earthquake_analysis_visualization.png" 
            alt="Earthquake Analysis Visualization"
            className="analysis-chart"
          />
          <p className="chart-caption">
            Statistical analysis and visualization of foreshock and aftershock patterns 
            across {detailedAnalysis.total_events} major earthquake events.
          </p>
        </div>
      </div>

      {/* Percentages */}
      <div className="analysis-percentages">
        <div className="percentage-item">
          <div className="percentage-bar">
            <div className="percentage-fill foreshock" />
          </div>
          <span className="percentage-label">
            {foreshockPercentage.toFixed(1)}% have foreshocks
          </span>
        </div>

        <div className="percentage-item">
          <div className="percentage-bar">
            <div className="percentage-fill aftershock" />
          </div>
          <span className="percentage-label">
            {aftershockPercentage.toFixed(1)}% have aftershocks
          </span>
        </div>

        <div className="percentage-item">
          <div className="percentage-bar">
            <div className="percentage-fill both" />
          </div>
          <span className="percentage-label">
            {bothPercentage.toFixed(1)}% have both foreshocks and aftershocks
          </span>
        </div>
      </div>
    </div>
  );
};

export default EventAnalysis;
