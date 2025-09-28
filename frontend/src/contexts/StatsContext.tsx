import React, { createContext, useContext, useState, type ReactNode } from 'react';

interface StatsContextType {
	casesAnalyzed: number;
	fraudDetected: number;
	moneySaved: number;
	updateStats: (isFraudulent: boolean, claimValue?: number) => void;
	resetStats: () => void;
}

const StatsContext = createContext<StatsContextType | undefined>(undefined);

interface StatsProviderProps {
	children: ReactNode;
}

export const StatsProvider: React.FC<StatsProviderProps> = ({ children }) => {
	const [casesAnalyzed, setCasesAnalyzed] = useState(0);
	const [fraudDetected, setFraudDetected] = useState(0);
	const [moneySaved, setMoneySaved] = useState(0);

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

	return (
		<StatsContext.Provider value={{
			casesAnalyzed,
			fraudDetected,
			moneySaved,
			updateStats,
			resetStats
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
