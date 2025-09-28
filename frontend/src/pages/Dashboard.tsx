import React from 'react';
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome';
import { 
	faFileAlt, 
	faExclamationTriangle, 
	faDollarSign, 
	faClock, 
	faShieldAlt, 
	faChartLine,
	faArrowUp
} from '@fortawesome/free-solid-svg-icons';
import { useStats } from '../contexts/StatsContext';

const Dashboard: React.FC = () => {
	const { casesAnalyzed, fraudDetected, moneySaved } = useStats();
	const fraudRate = casesAnalyzed > 0 ? ((fraudDetected / casesAnalyzed) * 100).toFixed(1) : '0.0';
	
	const metrics = [
		{
			title: 'Total Cases Analyzed',
			value: casesAnalyzed.toString(),
			change: '+12% this month',
			changeType: 'positive',
			icon: faFileAlt,
			iconColor: 'text-blue-600',
			color: 'bg-blue-100',
		},
		{
			title: 'Fraudulent Cases Detected',
			value: fraudDetected.toString(),
			change: `${fraudRate}% fraud rate`,
			changeType: 'positive',
			icon: faExclamationTriangle,
			iconColor: 'text-red-600',
			color: 'bg-red-100',
		},
		{
			title: 'Money Saved',
			value: `$${moneySaved.toLocaleString()}`,
			change: '+$50K this month',
			changeType: 'positive',
			icon: faDollarSign,
			iconColor: 'text-green-600',
			color: 'bg-green-100',
		},
		{
			title: 'Avg Processing Time',
			value: '2.2s',
			change: '2.3s improvement',
			changeType: 'positive',
			icon: faClock,
			iconColor: 'text-purple-600',
			color: 'bg-purple-100',
		},
		{
			title: 'Suspicious Cases',
			value: '1',
			change: 'Pending review',
			changeType: 'neutral',
			icon: faShieldAlt,
			iconColor: 'text-orange-600',
			color: 'bg-orange-100',
		},
		{
			title: 'Detection Accuracy',
			value: '94.7%',
			change: '+2.1% vs last month',
			changeType: 'positive',
			icon: faChartLine,
			iconColor: 'text-blue-600',
			color: 'bg-blue-100',
		}
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
			<div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 grid-rows-1 xl:grid-rows-2 gap-6">
				{metrics.map((metric, index) => (
					<div key={index} className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
						<div className='flex flex-row w-full justify-between gap-2'>
							<div className="flex flex-col">
								<p className="text-sm text-gray-600">{metric.title}</p>
								<p className={`text-2xl font-bold mb-1 ${metric.title === 'Money Saved' && moneySaved > 0 ? 'text-green-600' : 'text-gray-900'}`}>{metric.value}</p>
								<div className="flex items-center justify-between my-4 gap-3">
									<div className="flex flex-row items-center space-x-1 gap-2 w-full">
										<FontAwesomeIcon 
											icon={faArrowUp} 
											className={`text-xs ${
												metric.changeType === 'positive' ? 'text-green-600' : 
												metric.changeType === 'negative' ? 'text-red-600' : 'text-gray-600'
											}`} 
										/>
										<span className={`text-sm font-medium ${
											metric.changeType === 'positive' ? 'text-green-600' : 
											metric.changeType === 'negative' ? 'text-red-600' : 'text-gray-600'
										}`}>
											{metric.change}
										</span>
									</div>
								</div>
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
					
					{/* Fraud Detection Timeline Chart */}
					<div className="h-64 bg-gray-50 rounded-lg p-4 mb-6">
						<div className="h-full flex flex-col">
							<div className="flex justify-between items-center mb-4">
								<h4 className="text-sm font-medium text-gray-700">Cases & Money Saved Over Time</h4>
								<div className="flex space-x-4 text-xs">
									<div className="flex items-center space-x-1">
										<div className="w-3 h-3 bg-blue-500 rounded"></div>
										<span className="text-gray-600">Cases Analyzed</span>
									</div>
									<div className="flex items-center space-x-1">
										<div className="w-3 h-3 bg-red-500 rounded"></div>
										<span className="text-gray-600">Fraud Detected</span>
									</div>
									<div className="flex items-center space-x-1">
										<div className="w-3 h-3 bg-green-500 rounded"></div>
										<span className="text-gray-600">Money Saved</span>
									</div>
								</div>
							</div>
							
							{/* Chart Area */}
							<div className="flex-1 relative">
								{/* Y-axis labels */}
								<div className="absolute left-0 top-0 h-full flex flex-col justify-between text-xs text-gray-500 pr-2">
									<span>{Math.max(casesAnalyzed, fraudDetected, Math.round(moneySaved / 1000))}</span>
									<span>{Math.round(Math.max(casesAnalyzed, fraudDetected, Math.round(moneySaved / 1000)) * 0.75)}</span>
									<span>{Math.round(Math.max(casesAnalyzed, fraudDetected, Math.round(moneySaved / 1000)) * 0.5)}</span>
									<span>{Math.round(Math.max(casesAnalyzed, fraudDetected, Math.round(moneySaved / 1000)) * 0.25)}</span>
									<span>0</span>
								</div>
								
								{/* Chart bars */}
								<div className="ml-8 h-full flex items-end justify-center space-x-4">
									{/* Week 1 */}
									<div className="flex flex-col items-center space-y-1">
										<div className="flex items-end space-x-1 h-40">
											<div 
												className="w-4 bg-blue-500 rounded-t" 
												style={{ height: `${Math.max(5, (Math.max(1, casesAnalyzed - 3) / Math.max(1, casesAnalyzed)) * 160)}px` }}
											></div>
											<div 
												className="w-4 bg-red-500 rounded-t" 
												style={{ height: `${Math.max(5, (Math.max(1, fraudDetected - 1) / Math.max(1, casesAnalyzed)) * 160)}px` }}
											></div>
											<div 
												className="w-4 bg-green-500 rounded-t" 
												style={{ height: `${Math.max(5, (Math.max(100, moneySaved - 5000) / Math.max(1000, moneySaved)) * 160)}px` }}
											></div>
										</div>
										<span className="text-xs text-gray-500">Week 1</span>
									</div>
									
									{/* Week 2 */}
									<div className="flex flex-col items-center space-y-1">
										<div className="flex items-end space-x-1 h-40">
											<div 
												className="w-4 bg-blue-500 rounded-t" 
												style={{ height: `${Math.max(5, (Math.max(1, casesAnalyzed - 2) / Math.max(1, casesAnalyzed)) * 160)}px` }}
											></div>
											<div 
												className="w-4 bg-red-500 rounded-t" 
												style={{ height: `${Math.max(5, (Math.max(1, fraudDetected - 1) / Math.max(1, casesAnalyzed)) * 160)}px` }}
											></div>
											<div 
												className="w-4 bg-green-500 rounded-t" 
												style={{ height: `${Math.max(5, (Math.max(200, moneySaved - 3000) / Math.max(1000, moneySaved)) * 160)}px` }}
											></div>
										</div>
										<span className="text-xs text-gray-500">Week 2</span>
									</div>
									
									{/* Week 3 */}
									<div className="flex flex-col items-center space-y-1">
										<div className="flex items-end space-x-1 h-40">
											<div 
												className="w-4 bg-blue-500 rounded-t" 
												style={{ height: `${Math.max(5, (Math.max(1, casesAnalyzed - 1) / Math.max(1, casesAnalyzed)) * 160)}px` }}
											></div>
											<div 
												className="w-4 bg-red-500 rounded-t" 
												style={{ height: `${Math.max(5, (fraudDetected / Math.max(1, casesAnalyzed)) * 160)}px` }}
											></div>
											<div 
												className="w-4 bg-green-500 rounded-t" 
												style={{ height: `${Math.max(5, (Math.max(500, moneySaved - 1000) / Math.max(1000, moneySaved)) * 160)}px` }}
											></div>
										</div>
										<span className="text-xs text-gray-500">Week 3</span>
									</div>
									
									{/* Current Week */}
									<div className="flex flex-col items-center space-y-1">
										<div className="flex items-end space-x-1 h-40">
											<div 
												className="w-4 bg-blue-500 rounded-t" 
												style={{ height: `${Math.max(10, (casesAnalyzed / Math.max(1, casesAnalyzed)) * 160)}px` }}
											></div>
											<div 
												className="w-4 bg-red-500 rounded-t" 
												style={{ height: `${Math.max(5, (fraudDetected / Math.max(1, casesAnalyzed)) * 160)}px` }}
											></div>
											<div 
												className="w-4 bg-green-500 rounded-t" 
												style={{ height: `${Math.max(10, (moneySaved / Math.max(1000, moneySaved)) * 160)}px` }}
											></div>
										</div>
										<span className="text-xs text-gray-500 font-medium">This Week</span>
									</div>
								</div>
								
								{/* Grid lines */}
								<div className="absolute inset-0 ml-8 pointer-events-none">
									<div className="h-full flex flex-col justify-between">
										{[...Array(5)].map((_, i) => (
											<div key={i} className="border-t border-gray-200 w-full"></div>
										))}
									</div>
								</div>
							</div>
							
							{/* X-axis */}
							<div className="border-t border-gray-300 mt-2 pt-2">
								<div className="text-center text-xs text-gray-600">
									Timeline showing fraud detection progress and financial impact
								</div>
							</div>
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
