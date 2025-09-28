import React, { createContext, useContext, useState, useEffect, type ReactNode } from 'react';
import { fetchAnalysisMetadata, fetchImageAnalyses } from '../api/database';

interface StatsContextType {
	casesAnalyzed: number;
	fraudDetected: number;
	moneySaved: number;
	updateStats: (isFraudulent: boolean, claimValue?: number) => void;
	resetStats: () => void;
	refreshStats: () => void;
}

const StatsContext = createContext<StatsContextType | undefined>(undefined);

interface StatsProviderProps {
	children: ReactNode;
}

export const StatsProvider: React.FC<StatsProviderProps> = ({ children }) => {
	const [casesAnalyzed, setCasesAnalyzed] = useState(0);
	const [fraudDetected, setFraudDetected] = useState(0);
	const [moneySaved, setMoneySaved] = useState(0);

	const loadStatsFromDatabase = async () => {
		try {
			// Fetch analysis metadata to get total claim amounts
			const metadataResult = await fetchAnalysisMetadata();
			if (metadataResult.success && metadataResult.data) {
				const totalCases = metadataResult.data.length;
				setCasesAnalyzed(totalCases);
				
				// Calculate total money saved from all analysis costs
				const totalMoneySaved = metadataResult.data.reduce((sum, analysis) => {
					return sum + (analysis.total_cost || 0);
				}, 0);
				setMoneySaved(totalMoneySaved);
			}

			// Fetch image analyses to get fraud detection count
			const imagesResult = await fetchImageAnalyses();
			if (imagesResult.success && imagesResult.data) {
				const fraudulentCount = imagesResult.data.filter(image => image.is_fraudulent).length;
				setFraudDetected(fraudulentCount);
			}
		} catch (error) {
			console.error('Error loading stats from database:', error);
		}
	};

	const refreshStats = () => {
		loadStatsFromDatabase();
	};

	const updateStats = (isFraudulent: boolean, claimValue: number = 0) => {
		setCasesAnalyzed(prev => prev + 1);
		
		if (isFraudulent) {
			setFraudDetected(prev => prev + 1);
			// Assume average claim value of $15,000 if not provided
			const savedAmount = claimValue > 0 ? claimValue : 15000;
			setMoneySaved(prev => prev + savedAmount);
		}
	};

	const resetStats = () => {
		setCasesAnalyzed(0);
		setFraudDetected(0);
		setMoneySaved(0);
	};

	// Load stats on component mount
	useEffect(() => {
		loadStatsFromDatabase();
	}, []);

	return (
		<StatsContext.Provider value={{
			casesAnalyzed,
			fraudDetected,
			moneySaved,
			updateStats,
			resetStats,
			refreshStats
		}}>
			{children}
		</StatsContext.Provider>
	);
};

export const useStats = (): StatsContextType => {
	const context = useContext(StatsContext);
	if (context === undefined) {
		throw new Error('useStats must be used within a StatsProvider');
	}
	return context;
};
