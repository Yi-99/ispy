import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome';
import { 
  faSearch, 
  faFilter, 
  faExclamationTriangle, 
  faCheckCircle, 
  faClock,
  faSpinner,
  faFolderOpen,
  faChartBar
} from '@fortawesome/free-solid-svg-icons';
import { fetchAnalysisMetadata, type AnalysisMetadata } from '../api/database';

const Cases: React.FC = () => {
  const navigate = useNavigate();
  const [cases, setCases] = useState<AnalysisMetadata[]>([]);
  const [loading, setLoading] = useState(true);
  const [searchTerm, setSearchTerm] = useState('');
  const [statusFilter, setStatusFilter] = useState('all');

  useEffect(() => {
    loadCases();
  }, []);

  const loadCases = async () => {
    setLoading(true);
    try {
      const result = await fetchAnalysisMetadata();
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

  const getStatusInfo = (fraudDetectedCount: number, totalFiles: number) => {
    const fraudRate = (fraudDetectedCount / totalFiles) * 100;
    
    if (fraudRate > 50) {
      return {
        status: 'High Risk Case',
        statusColor: 'bg-red-100 text-red-800',
        statusIcon: faExclamationTriangle,
        statusIconColor: 'text-red-600'
      };
    } else if (fraudRate > 20) {
      return {
        status: 'Medium Risk Case',
        statusColor: 'bg-yellow-100 text-yellow-800',
        statusIcon: faClock,
        statusIconColor: 'text-yellow-600'
      };
    } else {
      return {
        status: 'Low Risk Case',
        statusColor: 'bg-green-100 text-green-800',
        statusIcon: faCheckCircle,
        statusIconColor: 'text-green-600'
      };
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

  const formatCurrency = (amount: number) => {
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD',
      minimumFractionDigits: 0,
      maximumFractionDigits: 0,
    }).format(amount);
  };

  const filteredCases = cases.filter(caseItem => {
    const matchesSearch = caseItem.analysis_name.toLowerCase().includes(searchTerm.toLowerCase());
    const fraudRate = (caseItem.fraud_detected_count / caseItem.total_files) * 100;
    
    let matchesStatus = true;
    if (statusFilter === 'high-risk') {
      matchesStatus = fraudRate > 50;
    } else if (statusFilter === 'medium-risk') {
      matchesStatus = fraudRate > 20 && fraudRate <= 50;
    } else if (statusFilter === 'low-risk') {
      matchesStatus = fraudRate <= 20;
    }
    
    return matchesSearch && matchesStatus;
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
            Review and manage analysis cases grouped by analysis metadata.
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
                  placeholder="Search by case name..."
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
                <option value="all">All Risk Levels</option>
                <option value="high-risk">High Risk Cases</option>
                <option value="medium-risk">Medium Risk Cases</option>
                <option value="low-risk">Low Risk Cases</option>
              </select>
            </div>
          </div>
        </div>

        {/* Cases Cards */}
        <div className="space-y-4">
          {filteredCases.length === 0 ? (
            <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-12 text-center">
              <FontAwesomeIcon icon={faFolderOpen} className="text-4xl text-gray-400 mb-4" />
              <h3 className="text-lg font-medium text-gray-900 mb-2">No cases found</h3>
              <p className="text-gray-600">
                {searchTerm || statusFilter !== 'all' 
                  ? 'Try adjusting your search or filter criteria.'
                  : 'No analysis cases have been created yet.'}
              </p>
            </div>
          ) : (
            filteredCases.map((caseItem) => {
              const statusInfo = getStatusInfo(caseItem.fraud_detected_count, caseItem.total_files);
              const fraudRate = (caseItem.fraud_detected_count / caseItem.total_files) * 100;
              const completionRate = (caseItem.completed_files / caseItem.total_files) * 100;
              
              return (
                <div key={caseItem.id} className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
                  <div className="flex items-start space-x-6">
                    {/* Case Icon */}
                    <div className="flex-shrink-0">
                      <button 
                        onClick={() => navigate(`/cases/${encodeURIComponent(caseItem.analysis_name)}`)}
                        className="w-16 h-16 bg-blue-100 rounded-lg flex items-center justify-center hover:bg-blue-200 transition-colors cursor-pointer"
                      >
                        <FontAwesomeIcon icon={faFolderOpen} className="text-blue-600 text-2xl" />
                      </button>
                    </div>

                    {/* Case Details */}
                    <div className="flex-1">
                      <div className="mb-4">
                        <h3 className="text-xl font-bold text-gray-900 mb-1">
                          {caseItem.analysis_name}
                        </h3>
                        <p className="text-sm text-gray-500">
                          Created {formatDate(caseItem.created_at || new Date().toISOString())}
                        </p>
                      </div>

                      {/* Status Tags */}
                      <div className="flex items-center space-x-3 mb-4">
                        <span className={`inline-flex items-center px-3 py-1 rounded-full text-sm font-medium ${statusInfo.statusColor}`}>
                          {statusInfo.status}
                        </span>
                        <span className="inline-flex items-center px-3 py-1 rounded-full text-sm font-medium bg-gray-100 text-gray-800">
                          {Math.round(fraudRate)}% fraud rate
                        </span>
                        <span className="inline-flex items-center px-3 py-1 rounded-full text-sm font-medium bg-blue-100 text-blue-800">
                          {Math.round(completionRate)}% complete
                        </span>
                      </div>

                      {/* Metrics Grid */}
                      <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-4">
                        <div>
                          <p className="text-sm text-gray-600">Total Files</p>
                          <p className="text-2xl font-bold text-gray-900">{caseItem.total_files}</p>
                        </div>
                        <div>
                          <p className="text-sm text-gray-600">Completed</p>
                          <p className="text-2xl font-bold text-gray-900">{caseItem.completed_files}</p>
                        </div>
                        <div>
                          <p className="text-sm text-gray-600">Fraud Detected</p>
                          <p className="text-2xl font-bold text-red-600">{caseItem.fraud_detected_count}</p>
                        </div>
                        <div>
                          <p className="text-sm text-gray-600">Total Claim Cost</p>
                          <p className="text-2xl font-bold text-gray-900">{formatCurrency(caseItem.total_cost)}</p>
                        </div>
                      </div>

                      {/* Progress Bar */}
                      <div className="mb-4">
                        <div className="flex items-center justify-between mb-2">
                          <span className="text-sm font-medium text-gray-700">Analysis Progress</span>
                          <span className="text-sm text-gray-500">{Math.round(completionRate)}%</span>
                        </div>
                        <div className="w-full bg-gray-200 rounded-full h-2">
                          <div 
                            className="bg-blue-600 h-2 rounded-full transition-all duration-300"
                            style={{ width: `${completionRate}%` }}
                          ></div>
                        </div>
                      </div>

                      {/* File URLs Preview */}
                      {caseItem.file_urls && Array.isArray(caseItem.file_urls) && caseItem.file_urls.length > 0 && (
                        <div className="bg-gray-50 rounded-lg p-4">
                          <h4 className="text-sm font-medium text-gray-800 mb-2 flex items-center">
                            <FontAwesomeIcon icon={faChartBar} className="mr-2" />
                            Analysis Files ({caseItem.file_urls.length})
                          </h4>
                          <div className="flex flex-wrap gap-2">
                            {caseItem.file_urls.slice(0, 5).map((_, index) => (
                              <span key={index} className="text-xs bg-blue-100 text-blue-800 px-2 py-1 rounded">
                                File {index + 1}
                              </span>
                            ))}
                            {caseItem.file_urls.length > 5 && (
                              <span className="text-xs bg-gray-100 text-gray-600 px-2 py-1 rounded">
                                +{caseItem.file_urls.length - 5} more
                              </span>
                            )}
                          </div>
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