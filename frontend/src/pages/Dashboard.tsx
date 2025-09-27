import React from 'react';
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome';
import { 
  faUpload, 
  faFileAlt, 
  faExclamationTriangle, 
  faDollarSign, 
  faClock, 
  faShieldAlt, 
  faChartLine,
  faArrowUp
} from '@fortawesome/free-solid-svg-icons';

const Dashboard: React.FC = () => {
  const metrics = [
    {
      title: 'Total Cases Analyzed',
      value: '5',
      change: '+12% this month',
      changeType: 'positive',
      icon: faFileAlt,
      iconColor: 'text-blue-600'
    },
    {
      title: 'Fraudulent Cases Detected',
      value: '2',
      change: '40.0% fraud rate',
      changeType: 'positive',
      icon: faExclamationTriangle,
      iconColor: 'text-red-600'
    },
    {
      title: 'Money Saved',
      value: '$60,000',
      change: '+$50K this month',
      changeType: 'positive',
      icon: faDollarSign,
      iconColor: 'text-green-600'
    },
    {
      title: 'Avg Processing Time',
      value: '2.2s',
      change: '2.3s improvement',
      changeType: 'positive',
      icon: faClock,
      iconColor: 'text-purple-600'
    },
    {
      title: 'Suspicious Cases',
      value: '1',
      change: 'Pending review',
      changeType: 'neutral',
      icon: faShieldAlt,
      iconColor: 'text-orange-600'
    },
    {
      title: 'Detection Accuracy',
      value: '94.7%',
      change: '+2.1% vs last month',
      changeType: 'positive',
      icon: faChartLine,
      iconColor: 'text-blue-600'
    }
  ];

  const securityAlerts = [
    {
      title: 'Recent Fraud Detected',
      description: '2 fraudulent cases in last 24h',
      severity: 'HIGH',
      severityColor: 'bg-red-100 text-red-800',
      icon: faExclamationTriangle,
      iconColor: 'text-red-600'
    },
    {
      title: 'High-Value Suspicious Claims',
      description: '2 claims over $10K with high fraud risk',
      severity: 'MEDIUM',
      severityColor: 'bg-yellow-100 text-yellow-800',
      icon: faShieldAlt,
      iconColor: 'text-orange-600'
    },
    {
      title: 'Elevated Fraud Rate',
      description: 'Current fraud detection rate: 40%',
      severity: 'HIGH',
      severityColor: 'bg-red-100 text-red-800',
      icon: faChartLine,
      iconColor: 'text-red-600'
    }
  ];

  return (
    <div className="p-6 space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold text-gray-900">iSpy Dashboard</h1>
          <p className="text-gray-600 mt-1">
            Monitor and analyze vehicle damage claims for fraudulent activity
          </p>
        </div>
        <button className="bg-blue-600 hover:bg-blue-700 text-white px-6 py-3 rounded-lg font-medium flex items-center space-x-2 transition-colors duration-200">
          <FontAwesomeIcon icon={faUpload} />
          <span>Analyze New Image</span>
        </button>
      </div>

      {/* Key Metrics */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-6 gap-6">
        {metrics.map((metric, index) => (
          <div key={index} className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
            <div className="flex items-center justify-between mb-4">
              <div className={`p-3 rounded-lg ${metric.iconColor.replace('text-', 'bg-').replace('-600', '-100')}`}>
                <FontAwesomeIcon icon={metric.icon} className={`w-6 h-6 ${metric.iconColor}`} />
              </div>
              <div className="flex items-center space-x-1">
                <span className={`text-sm font-medium ${
                  metric.changeType === 'positive' ? 'text-green-600' : 
                  metric.changeType === 'negative' ? 'text-red-600' : 'text-gray-600'
                }`}>
                  {metric.change}
                </span>
                <FontAwesomeIcon 
                  icon={faArrowUp} 
                  className={`text-xs ${
                    metric.changeType === 'positive' ? 'text-green-600' : 
                    metric.changeType === 'negative' ? 'text-red-600' : 'text-gray-600'
                  }`} 
                />
              </div>
            </div>
            <div>
              <p className="text-2xl font-bold text-gray-900 mb-1">{metric.value}</p>
              <p className="text-sm text-gray-600">{metric.title}</p>
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
          
          {/* Chart placeholder */}
          <div className="h-64 bg-gray-50 rounded-lg flex items-center justify-center mb-6">
            <div className="text-center">
              <FontAwesomeIcon icon={faChartLine} className="text-4xl text-gray-400 mb-2" />
              <p className="text-gray-500">Monthly Case Analysis Chart</p>
              <p className="text-sm text-gray-400 mt-1">Chart visualization would go here</p>
            </div>
          </div>

          {/* Case Distribution */}
          <div>
            <h3 className="text-lg font-medium text-gray-900 mb-4">Case Distribution</h3>
            <div className="grid grid-cols-2 gap-4">
              <div className="text-center">
                <div className="w-20 h-20 bg-green-100 rounded-full flex items-center justify-center mx-auto mb-2">
                  <span className="text-2xl font-bold text-green-600">3</span>
                </div>
                <p className="text-sm text-gray-600">Legitimate</p>
              </div>
              <div className="text-center">
                <div className="w-20 h-20 bg-red-100 rounded-full flex items-center justify-center mx-auto mb-2">
                  <span className="text-2xl font-bold text-red-600">2</span>
                </div>
                <p className="text-sm text-gray-600">Fraudulent</p>
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
