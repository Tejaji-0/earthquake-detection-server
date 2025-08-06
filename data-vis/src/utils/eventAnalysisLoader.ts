import type { ProcessingSummary, DetailedAnalysis } from '../types/eventAnalysis';

export const loadEventAnalysis = async (): Promise<{
  processingSummary: ProcessingSummary;
  detailedAnalysis: DetailedAnalysis;
}> => {
  try {
    const [processingSummaryResponse, detailedAnalysisResponse] = await Promise.all([
      fetch('/processing_summary.json'),
      fetch('/detailed_analysis.json')
    ]);

    if (!processingSummaryResponse.ok || !detailedAnalysisResponse.ok) {
      throw new Error('Failed to fetch event analysis data');
    }

    const processingSummary: ProcessingSummary = await processingSummaryResponse.json();
    const detailedAnalysis: DetailedAnalysis = await detailedAnalysisResponse.json();

    return {
      processingSummary,
      detailedAnalysis
    };
  } catch (error) {
    console.error('Error loading event analysis data:', error);
    throw error;
  }
};
