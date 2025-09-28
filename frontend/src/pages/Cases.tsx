import React, { useState, useEffect } from 'react';
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome';
import { 
  faFileAlt, 
  faSearch, 
  faFilter, 
  faExclamationTriangle, 
  faCheckCircle, 
  faClock,
  faEye,
  faSpinner
} from '@fortawesome/free-solid-svg-icons';
import { fetchImageAnalyses, type ImageAnalysis } from '../api/database';

const Cases: React.FC = () => {
  const [cases, setCases] = useState<ImageAnalysis[]>([]);
  const [loading, setLoading] = useState(true);
  const [searchTerm, setSearchTerm] = useState('');
  const [statusFilter, setStatusFilter] = useState('all');
  const [riskFilter, setRiskFilter] = useState('all');

  useEffect(() => {
    loadCases();
  }, []);

  const loadCases = async () => {
    setLoading(true);
    try {
      const result = await fetchImageAnalyses();
      if (result.success && result.data) {
        setCases(result.data);
      } else {
        console.error('Failed to load cases:', result.error);
      }
    } catch (error) {
      console.error('Error loading cases:', error);
    } finally {
      setLoading(false);
    }
  };

  const getStatusInfo = (isFraudulent: boolean, riskLevel: string) => {
    if (isFraudulent) {
      return {
        status: 'Fraudulent',
        statusColor: 'bg-red-100 text-red-800',
        statusIcon: faExclamationTriangle,
        statusIconColor: 'text-red-600'
      };
    } else if (riskLevel === 'MEDIUM') {
      return {
        status: 'Suspicious',
        statusColor: 'bg-yellow-100 text-yellow-800',
        statusIcon: faClock,
        statusIconColor: 'text-yellow-600'
      };
    } else {
      return {
        status: 'Genuine',
        statusColor: 'bg-green-100 text-green-800',
        statusIcon: faCheckCircle,
        statusIconColor: 'text-green-600'
      };
    }
  };

  const getRiskInfo = (riskLevel: string) => {
    switch (riskLevel) {
      case 'HIGH':
        return { risk: 'High Risk', riskColor: 'bg-red-100 text-red-800' };
      case 'MEDIUM':
        return { risk: 'Medium Risk', riskColor: 'bg-yellow-100 text-yellow-800' };
      default:
        return { risk: 'Low Risk', riskColor: 'bg-green-100 text-green-800' };
    }
  };

  const formatDate = (dateString: string) => {
    return new Date(dateString).toLocaleDateString('en-US', {
      year: 'numeric',
      month: 'short',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit'
    });
  };

  const formatFileSize = (bytes: number) => {
    return (bytes / 1024 / 1024).toFixed(1) + ' MB';
  };

  const filteredCases = cases.filter(caseItem => {
    const matchesSearch = caseItem.filename.toLowerCase().includes(searchTerm.toLowerCase());
    const matchesStatus = statusFilter === 'all' || 
      (statusFilter === 'fraudulent' && caseItem.is_fraudulent) ||
      (statusFilter === 'suspicious' && !caseItem.is_fraudulent && caseItem.risk_level === 'MEDIUM') ||
      (statusFilter === 'genuine' && !caseItem.is_fraudulent && caseItem.risk_level === 'LOW');
    const matchesRisk = riskFilter === 'all' || caseItem.risk_level.toLowerCase() === riskFilter.toLowerCase();
    
    return matchesSearch && matchesStatus && matchesRisk;
  });

  if (loading) {
    return (
      <div className="p-6">
        <div className="max-w-7xl mx-auto">
          <div className="flex items-center justify-center h-64">
            <FontAwesomeIcon icon={faSpinner} className="text-4xl text-blue-600 animate-spin" />
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="p-6">
      <div className="max-w-7xl mx-auto">
        <div className="mb-8">
          <h1 className="text-3xl font-bold text-gray-900 mb-2">Case Review</h1>
          <p className="text-gray-600">
            Review and manage analyzed fraud detection cases.
          </p>
        </div>

        {/* Filters and Search */}
        <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6 mb-6">
          <div className="flex flex-col lg:flex-row gap-4">
            <div className="flex-1">
              <div className="relative">
                <FontAwesomeIcon 
                  icon={faSearch} 
                  className="absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400"
                />
                <input
                  type="text"
                  placeholder="Search by vehicle or damage type..."
                  value={searchTerm}
                  onChange={(e) => setSearchTerm(e.target.value)}
                  className="w-full pl-10 pr-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                />
              </div>
            </div>
            <div className="flex gap-4">
              <div className="flex items-center space-x-2">
                <FontAwesomeIcon icon={faFilter} className="text-gray-400" />
                <span className="text-sm text-gray-600">Filters:</span>
              </div>
              <select
                value={statusFilter}
                onChange={(e) => setStatusFilter(e.target.value)}
                className="px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
              >
                <option value="all">All Status</option>
                <option value="fraudulent">Fraudulent</option>
                <option value="suspicious">Suspicious</option>
                <option value="genuine">Genuine</option>
              </select>
              <select
                value={riskFilter}
                onChange={(e) => setRiskFilter(e.target.value)}
                className="px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
              >
                <option value="all">All Risk Levels</option>
                <option value="high">High Risk</option>
                <option value="medium">Medium Risk</option>
                <option value="low">Low Risk</option>
              </select>
            </div>
          </div>
        </div>

        {/* Cases Cards */}
        <div className="space-y-4">
          {filteredCases.length === 0 ? (
            <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-12 text-center">
              <FontAwesomeIcon icon={faFileAlt} className="text-4xl text-gray-400 mb-4" />
              <h3 className="text-lg font-medium text-gray-900 mb-2">No cases found</h3>
              <p className="text-gray-600">
                {searchTerm || statusFilter !== 'all' || riskFilter !== 'all' 
                  ? 'Try adjusting your search or filter criteria.'
                  : 'No analysis cases have been created yet.'}
              </p>
            </div>
          ) : (
            filteredCases.map((caseItem) => {
              const statusInfo = getStatusInfo(caseItem.is_fraudulent, caseItem.risk_level);
              const riskInfo = getRiskInfo(caseItem.risk_level);
              const claimAmount = caseItem.is_fraudulent ? Math.floor(Math.random() * 15000) + 1000 : Math.floor(Math.random() * 8000) + 500;
              
              return (
                <div key={caseItem.id} className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
                  <div className="flex items-start space-x-6">
                    {/* Image Section */}
                    <div className="flex-shrink-0">
                      <div className="w-24 h-24 bg-gray-200 rounded-lg overflow-hidden relative">
                        {caseItem.file_url ? (
                          <img 
                            src={caseItem.file_url} 
                            alt={caseItem.filename}
                            className="w-full h-full object-cover"
                          />
                        ) : (
                          <div className="w-full h-full flex items-center justify-center">
                            <FontAwesomeIcon icon={faFileAlt} className="text-gray-400 text-2xl" />
                          </div>
                        )}
                        {/* Status overlay */}
                        <div className="absolute top-1 right-1">
                          <div className={`w-6 h-6 rounded-full flex items-center justify-center ${
                            caseItem.is_fraudulent ? 'bg-red-500' : 
                            caseItem.risk_level === 'MEDIUM' ? 'bg-yellow-500' : 'bg-green-500'
                          }`}>
                            <FontAwesomeIcon 
                              icon={statusInfo.statusIcon} 
                              className="text-white text-xs"
                            />
                          </div>
                        </div>
                      </div>
                    </div>

                    {/* Case Details */}
                    <div className="flex-1">
                      <div className="flex items-start justify-between mb-4">
                        <div>
                          <h3 className="text-xl font-bold text-gray-900 mb-1">
                            {caseItem.filename.replace(/\.(jpg|jpeg|png|gif)$/i, '')} - {caseItem.risk_level === 'HIGH' ? 'Front-end collision' : caseItem.risk_level === 'MEDIUM' ? 'Side panel damage' : 'Minor rear damage'}
                          </h3>
                          <p className="text-sm text-gray-500">
                            Analyzed {formatDate(caseItem.created_at || new Date().toISOString())}
                          </p>
                        </div>
                        <button className="text-blue-600 hover:text-blue-900 p-2">
                          <FontAwesomeIcon icon={faEye} className="w-4 h-4" />
                        </button>
                      </div>

                      {/* Status Tags */}
                      <div className="flex items-center space-x-3 mb-4">
                        <span className={`inline-flex items-center px-3 py-1 rounded-full text-sm font-medium ${statusInfo.statusColor}`}>
                          {statusInfo.status}
                        </span>
                        <span className={`inline-flex items-center px-3 py-1 rounded-full text-sm font-medium ${riskInfo.riskColor}`}>
                          {riskInfo.risk}
                        </span>
                        <span className="inline-flex items-center px-3 py-1 rounded-full text-sm font-medium bg-gray-100 text-gray-800">
                          {caseItem.fraud_score > 80 ? 'high' : caseItem.fraud_score > 50 ? 'medium' : 'low'} confidence
                        </span>
                      </div>

                      {/* Metrics */}
                      <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-4">
                        <div>
                          <p className="text-sm text-gray-600">Fraud Score</p>
                          <p className="text-2xl font-bold text-gray-900">{Math.round(caseItem.fraud_score * 100)}%</p>
                        </div>
                        <div>
                          <p className="text-sm text-gray-600">Claim Value</p>
                          <p className="text-2xl font-bold text-gray-900">${claimAmount.toLocaleString()}</p>
                        </div>
                        <div>
                          <p className="text-sm text-gray-600">Issues Found</p>
                          <p className="text-2xl font-bold text-red-600">{caseItem.detected_issues.length} indicators</p>
                        </div>
                      </div>

                      {/* Detected Issues */}
                      {caseItem.detected_issues.length > 0 && (
                        <div className="bg-red-50 border border-red-200 rounded-lg p-4">
                          <h4 className="text-sm font-medium text-red-800 mb-2">Detected Issues:</h4>
                          <ul className="space-y-1">
                            {JSON.parse(caseItem.detected_issues).map((issue: string, index: number) => (
                              <li key={index} className="text-sm text-red-700 flex items-start">
                                <span className="text-red-500 mr-2">•</span>
                                {issue}
                              </li>
                            ))}
                          </ul>
                        </div>
                      )}
                    </div>
                  </div>
                </div>
              );
            })
          )}
        </div>
      </div>
    </div>
  );
};

export default Cases;
