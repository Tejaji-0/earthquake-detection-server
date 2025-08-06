export interface EventSummary {
  event: string;
  directory: string;
  before_count: number;
  after_count: number;
}

export interface ProcessingSummary {
  processing_date: string;
  total_events_processed: number;
  events: EventSummary[];
}

export interface TopEvent {
  event: string;
  count: number;
}

export interface DetailedAnalysis {
  analysis_date: string;
  total_events: number;
  events_with_foreshocks: number;
  events_with_aftershocks: number;
  events_with_both: number;
  top_foreshock_events: TopEvent[];
  top_aftershock_events: TopEvent[];
  location_analysis: {
    [location: string]: {
      total_events: number;
      events_with_foreshocks: number;
      events_with_aftershocks: number;
      events_with_both: number;
    };
  };
  magnitude_analysis: {
    [magnitude: string]: {
      total_events: number;
      events_with_foreshocks: number;
      events_with_aftershocks: number;
      events_with_both: number;
    };
  };
}
