import React from 'react';
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome';
import {
	Chart as ChartJS,
	CategoryScale,
	LinearScale,
	PointElement,
	LineElement,
	Title,
	Tooltip,
	Legend,
} from 'chart.js';
import { Line } from 'react-chartjs-2';

ChartJS.register(
	CategoryScale,
	LinearScale,
	PointElement,
	LineElement,
	Title,
	Tooltip,
	Legend
);
import { 
	faFileAlt, 
	faExclamationTriangle, 
	faDollarSign, 
	faShieldAlt, 
	faChartLine,
	faArrowUp
} from '@fortawesome/free-solid-svg-icons';
import { useStats } from '../contexts/StatsContext';

const Dashboard: React.FC = () => {
	const { casesAnalyzed, fraudDetected, moneySaved } = useStats();
	// const fraudRate = casesAnalyzed > 0 ? ((fraudDetected / casesAnalyzed) * 100).toFixed(1) : '0.0';
	
	// Chart.js data
	const week1Cases = Math.max(1, casesAnalyzed - 3);
	const week2Cases = Math.max(1, casesAnalyzed - 2);
	const week3Cases = Math.max(1, casesAnalyzed - 1);
	const week4Cases = casesAnalyzed;
	
	const week1Fraud = Math.max(0, fraudDetected - 1);
	const week2Fraud = Math.max(0, fraudDetected - 1);
	const week3Fraud = fraudDetected;
	const week4Fraud = fraudDetected;
	
	const week1Money = Math.max(100, moneySaved - 5000);
	const week2Money = Math.max(200, moneySaved - 3000);
	const week3Money = Math.max(500, moneySaved - 1000);
	const week4Money = moneySaved;

	const chartData = {
		labels: ['Week 1', 'Week 2', 'Week 3', 'This Week'],
		datasets: [
			{
				label: 'Cases Analyzed',
				data: [week1Cases, week2Cases, week3Cases, week4Cases],
				borderColor: '#3B82F6',
				backgroundColor: 'rgba(59, 130, 246, 0.1)',
				borderWidth: 3,
				pointBackgroundColor: '#3B82F6',
				pointBorderColor: '#ffffff',
				pointBorderWidth: 2,
				pointRadius: 6,
				pointHoverRadius: 8,
				tension: 0.3,
			},
			{
				label: 'Fraud Detected',
				data: [week1Fraud, week2Fraud, week3Fraud, week4Fraud],
				borderColor: '#EF4444',
				backgroundColor: 'rgba(239, 68, 68, 0.1)',
				borderWidth: 3,
				pointBackgroundColor: '#EF4444',
				pointBorderColor: '#ffffff',
				pointBorderWidth: 2,
				pointRadius: 6,
				pointHoverRadius: 8,
				tension: 0.3,
			},
			{
				label: 'Money Saved ($)',
				data: [week1Money, week2Money, week3Money, week4Money],
				borderColor: '#10B981',
				backgroundColor: 'rgba(16, 185, 129, 0.1)',
				borderWidth: 3,
				pointBackgroundColor: '#10B981',
				pointBorderColor: '#ffffff',
				pointBorderWidth: 2,
				pointRadius: 6,
				pointHoverRadius: 8,
				tension: 0.3,
				yAxisID: 'y1',
			},
		],
	};

	const chartOptions = {
		responsive: true,
		maintainAspectRatio: false,
		plugins: {
			legend: {
				position: 'top' as const,
				labels: {
					usePointStyle: true,
					pointStyle: 'circle',
					padding: 20,
					font: {
						size: 12,
					},
				},
			},
			tooltip: {
				mode: 'index' as const,
				intersect: false,
				backgroundColor: 'rgba(255, 255, 255, 0.95)',
				titleColor: '#1F2937',
				bodyColor: '#4B5563',
				borderColor: '#E5E7EB',
				borderWidth: 1,
				cornerRadius: 8,
				padding: 12,
				displayColors: true,
				callbacks: {
					label: function(context: any) {
						let label = context.dataset.label || '';
						if (label) {
							label += ': ';
						}
						if (context.dataset.label === 'Money Saved ($)') {
							label += '$' + context.parsed.y.toLocaleString();
						} else {
							label += context.parsed.y;
						}
						return label;
					},
					afterBody: function(tooltipItems: any[]) {
						const weekData = tooltipItems[0];
						const cases = chartData.datasets[0].data[weekData.dataIndex];
						const fraud = chartData.datasets[1].data[weekData.dataIndex];
						const detectionRate = cases > 0 ? ((fraud / cases) * 100).toFixed(1) : '0';
						return [`Detection Rate: ${detectionRate}%`];
					},
				},
			},
		},
		interaction: {
			mode: 'nearest' as const,
			axis: 'x' as const,
			intersect: false,
		},
		scales: {
			x: {
				display: true,
				title: {
					display: true,
					text: 'Timeline',
					font: {
						size: 12,
						weight: 'bold' as const,
					},
				},
				grid: {
					color: 'rgba(229, 231, 235, 0.5)',
				},
			},
			y: {
				type: 'linear' as const,
				display: true,
				position: 'left' as const,
				title: {
					display: true,
					text: 'Cases',
					font: {
						size: 12,
						weight: 'bold' as const,
					},
				},
				grid: {
					color: 'rgba(229, 231, 235, 0.3)',
				},
				beginAtZero: true,
			},
			y1: {
				type: 'linear' as const,
				display: true,
				position: 'right' as const,
				title: {
					display: true,
					text: 'Money Saved ($)',
					font: {
						size: 12,
						weight: 'bold' as const,
					},
				},
				grid: {
					drawOnChartArea: false,
				},
				beginAtZero: true,
				ticks: {
					callback: function(value: any) {
						return '$' + value.toLocaleString();
					},
				},
			},
		},
		elements: {
			line: {
				borderJoinStyle: 'round' as const,
				borderCapStyle: 'round' as const,
			},
			point: {
				hoverBackgroundColor: '#ffffff',
			},
		},
	};
	
	const metrics = [
		{
			title: 'Total Cases Analyzed',
			value: casesAnalyzed.toString(),
			changeType: 'positive',
			icon: faFileAlt,
			iconColor: 'text-blue-600',
			color: 'bg-blue-100',
		},
		{
			title: 'Fraudulent Cases Detected',
			value: fraudDetected.toString(),
			changeType: 'positive',
			icon: faExclamationTriangle,
			iconColor: 'text-red-600',
			color: 'bg-red-100',
		},
		{
			title: 'Money Saved',
			value: `$${moneySaved.toLocaleString()}`,
			changeType: 'positive',
			icon: faDollarSign,
			iconColor: 'text-green-600',
			color: 'bg-green-100',
		},
	];

	const securityAlerts = [
		{
			title: 'Recent Fraud Detected',
			description: '2 fraudulent cases in last 24h',
			severity: 'HIGH',
			severityColor: 'bg-red-100 text-red-800',
			icon: faExclamationTriangle,
			iconColor: 'text-red-600',
			color: 'bg-red-100',
		},
		{
			title: 'High-Value Suspicious Claims',
			description: '2 claims over $10K with high fraud risk',
			severity: 'MEDIUM',
			severityColor: 'bg-yellow-100 text-yellow-800',
			icon: faShieldAlt,
			iconColor: 'text-orange-600',
			color: 'bg-orange-100',
		},
		{
			title: 'Elevated Fraud Rate',
			description: 'Current fraud detection rate: 40%',
			severity: 'HIGH',
			severityColor: 'bg-red-100 text-red-800',
			icon: faChartLine,
			iconColor: 'text-red-600',
			color: 'bg-red-100',
		}
	];

	return (
		<div className="p-6 space-y-6">
			{/* Header */}
			<div className="flex items-center justify-between">
				<div>
					<h1 className="text-3xl font-bold text-gray-900">Dashboard</h1>
					<p className="text-gray-600 mt-1">
						Overview of Vehicle Damage Claim AI Fraud Detection monitoring and analysis
					</p>
				</div>
			</div>

			{/* Key Metrics */}
			<div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 grid-rows-1 gap-6">
				{metrics.map((metric: any, index: number) => (
					<div key={index} className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
						<div className='flex flex-row w-full justify-between gap-2'>
							<div className="flex flex-col">
								<p className="text-sm text-gray-600">{metric.title}</p>
								<p className={`text-2xl font-bold mb-1 ${metric.title === 'Money Saved' && moneySaved > 0 ? 'text-green-600' : 'text-gray-900'}`}>{metric.value}</p>
								{metric.change && (
									<div className="flex items-center justify-between my-4 gap-3">
										<div className="flex flex-row items-center space-x-1 gap-2 w-full">
											{metric.title !== 'Suspicious Cases' && (
												<FontAwesomeIcon 
													icon={faArrowUp} 
													className={`text-xs ${
														metric.changeType === 'positive' ? 'text-green-600' : 
														metric.changeType === 'negative' ? 'text-red-600' : 'text-gray-600'
													}`} 
												/>
											)}
											<span className={`text-sm font-medium ${
												metric.changeType === 'positive' ? 'text-green-600' : 
												metric.changeType === 'negative' ? 'text-red-600' : 'text-gray-600'
											}`}>
												{metric.change}
											</span>
										</div>
									</div>
								)}
							</div>
							<div className="flex flex-col">
								<div className={`p-3 rounded-lg ${metric.iconColor.replace('text-', 'bg-').replace('-600', '-100')}`}>
									<FontAwesomeIcon icon={metric.icon} className={`w-6 h-6 ${metric.iconColor}`} />
								</div>
							</div>
						</div>
					</div>
				))}
			</div>

			<div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
				{/* Fraud Detection Analysis */}
				<div className="lg:col-span-2 bg-white rounded-lg shadow-sm border border-gray-200 p-6">
					<div className="mb-6">
						<h2 className="text-xl font-semibold text-gray-900 mb-2">Fraud Detection Analysis</h2>
						<p className="text-gray-600">Monthly breakdown of analyzed cases</p>
					</div>
					
					{/* Chart.js Multi-Line Chart */}
					<div className="h-80 bg-white rounded-lg border border-gray-200 p-6 mb-6">
						<div className="h-full">
							<Line data={chartData} options={chartOptions} />
						</div>
					</div>

					{/* Fraud Impact Chart */}
					<div>
						<h3 className="text-lg font-medium text-gray-900 mb-4">Fraud Detection Impact</h3>
						<div className="space-y-4">
							{/* Money Saved Visualization */}
							<div className="bg-green-50 rounded-lg p-4 border border-green-200">
								<div className="flex items-center justify-between mb-2">
									<h4 className="text-sm font-medium text-green-900">Money Saved from Fraud Detection</h4>
									<FontAwesomeIcon icon={faDollarSign} className="text-green-600" />
								</div>
								<div className="flex items-end space-x-2 mb-2">
									<span className="text-2xl font-bold text-green-600">${moneySaved.toLocaleString()}</span>
									<span className="text-sm text-green-600 mb-1">prevented losses</span>
								</div>
								<div className="w-full bg-green-200 rounded-full h-2">
									<div 
										className="bg-green-600 h-2 rounded-full transition-all duration-1000" 
										style={{ width: `${Math.min(100, (moneySaved / 100000) * 100)}%` }}
									></div>
								</div>
								<p className="text-xs text-green-700 mt-1">
									{moneySaved > 0 ? `${Math.min(100, Math.round((moneySaved / 100000) * 100))}% of $100K target` : 'No fraudulent claims detected yet'}
								</p>
							</div>

							{/* Fraudulent Cases Detected */}
							<div className="bg-red-50 rounded-lg p-4 border border-red-200">
								<div className="flex items-center justify-between mb-2">
									<h4 className="text-sm font-medium text-red-900">Fraudulent Cases Detected</h4>
									<FontAwesomeIcon icon={faExclamationTriangle} className="text-red-600" />
								</div>
								<div className="flex items-end space-x-2 mb-2">
									<span className="text-2xl font-bold text-red-600">{fraudDetected}</span>
									<span className="text-sm text-red-600 mb-1">out of {casesAnalyzed} cases</span>
								</div>
								<div className="w-full bg-red-200 rounded-full h-2">
									<div 
										className="bg-red-600 h-2 rounded-full transition-all duration-1000" 
										style={{ width: `${casesAnalyzed > 0 ? (fraudDetected / casesAnalyzed) * 100 : 0}%`, maxWidth: '100%' }}
									></div>
								</div>
								<p className="text-xs text-red-700 mt-1">
									{casesAnalyzed > 0 ? `${((fraudDetected / casesAnalyzed) * 100).toFixed(1)}% fraud detection rate` : 'No cases analyzed yet'}
								</p>
							</div>

							{/* Detection Efficiency */}
							<div className="bg-blue-50 rounded-lg p-4 border border-blue-200">
								<div className="flex items-center justify-between mb-2">
									<h4 className="text-sm font-medium text-blue-900">Detection Efficiency</h4>
									<FontAwesomeIcon icon={faChartLine} className="text-blue-600" />
								</div>
								<div className="flex items-end space-x-2 mb-2">
									<span className="text-2xl font-bold text-blue-600">
										{fraudDetected > 0 ? `$${Math.round(moneySaved / fraudDetected).toLocaleString()}` : '$0'}
									</span>
									<span className="text-sm text-blue-600 mb-1">avg. saved per fraud detected</span>
								</div>
							</div>
						</div>
					</div>
				</div>

				{/* Security Alerts */}
				<div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
					<div className="mb-6">
						<h2 className="text-xl font-semibold text-gray-900 mb-2">Security Alerts</h2>
						<p className="text-gray-600">Important fraud detection notifications</p>
					</div>
					
					<div className="space-y-4">
						{securityAlerts.map((alert, index) => (
							<div key={index} className="border border-gray-200 rounded-lg p-4 hover:shadow-md transition-shadow duration-200">
								<div className="flex items-start space-x-3">
									<div className={`p-2 rounded-lg ${alert.iconColor.replace('text-', 'bg-').replace('-600', '-100')}`}>
										<FontAwesomeIcon icon={alert.icon} className={`w-4 h-4 ${alert.iconColor}`} />
									</div>
									<div className="flex-1 min-w-0">
										<div className="flex items-center space-x-2 mb-1">
											<h3 className="text-sm font-medium text-gray-900">{alert.title}</h3>
											<span className={`px-2 py-1 text-xs font-medium rounded-full ${alert.severityColor}`}>
												{alert.severity}
											</span>
										</div>
										<p className="text-sm text-gray-600">{alert.description}</p>
									</div>
								</div>
							</div>
						))}
					</div>
				</div>
			</div>
		</div>
	);
};

export default Dashboard;
