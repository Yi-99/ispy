import React, { useState, useEffect } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome';
import { 
  faArrowLeft,
  faFileImage,
  faExclamationTriangle,
  faShieldAlt,
  faDollarSign,
  faCalendarAlt,
  faSpinner,
  faInfoCircle,
  faBan,
  faRobot,
  faChevronLeft,
  faChevronRight
} from '@fortawesome/free-solid-svg-icons';
import { fetchAnalysisMetadata, fetchImageAnalyses, type AnalysisMetadata, type ImageAnalysis } from '../api/database';

const CaseDetail: React.FC = () => {
  const { analysisName } = useParams<{ analysisName: string }>();
  const navigate = useNavigate();
  const [caseData, setCaseData] = useState<AnalysisMetadata | null>(null);
  const [images, setImages] = useState<ImageAnalysis[]>([]);
  const [loading, setLoading] = useState(true);
  const [currentPage, setCurrentPage] = useState(1);
  const [itemsPerPage] = useState(5);

  useEffect(() => {
    if (analysisName) {
      loadCaseData();
    }
  }, [analysisName]);

  const loadCaseData = async () => {
    setLoading(true);
    try {
      // Fetch case metadata
      const metadataResult = await fetchAnalysisMetadata();
      if (metadataResult.success && metadataResult.data) {
        const caseItem = metadataResult.data.find(item => item.analysis_name === analysisName);
        setCaseData(caseItem || null);
      }

      // Fetch related images
      const imagesResult = await fetchImageAnalyses();
      if (imagesResult.success && imagesResult.data) {
        const relatedImages = imagesResult.data.filter(image => image.analysis_name === analysisName);
        setImages(relatedImages);
      }
    } catch (error) {
      console.error('Error loading case data:', error);
    } finally {
      setLoading(false);
    }
  };

  const formatDate = (dateString: string) => {
    return new Date(dateString).toLocaleDateString('en-US', {
      year: 'numeric',
      month: 'long',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit',
      second: '2-digit'
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

  const getHighestFraudRisk = () => {
    if (images.length === 0) return 0;
    return Math.max(...images.map(img => Math.round(img.fraud_score * 100)));
  };

  const getHighestAIRisk = () => {
    if (images.length === 0) return 0;
    return Math.max(...images.map(img => Math.round(img.ai_score * 100)));
  };

  const getFraudulentImagesCount = () => {
    return images.filter(img => img.is_fraudulent).length;
  };

  const getAIRiskImagesCount = () => {
    // Consider images with AI score > 50% as AI risk images
    return images.filter(img => Math.round(img.ai_score * 100) > 50).length;
  };

  const getRiskColor = (risk: number) => {
    if (risk >= 70) return 'text-red-600';
    if (risk >= 40) return 'text-yellow-600';
    return 'text-green-600';
  };


  // Pagination logic
  const totalPages = Math.ceil(images.length / itemsPerPage);
  const startIndex = (currentPage - 1) * itemsPerPage;
  const endIndex = startIndex + itemsPerPage;
  const currentImages = images.slice(startIndex, endIndex);

  const goToPage = (page: number) => {
    setCurrentPage(page);
  };

  const goToPreviousPage = () => {
    setCurrentPage(prev => Math.max(prev - 1, 1));
  };

  const goToNextPage = () => {
    setCurrentPage(prev => Math.min(prev + 1, totalPages));
  };

  if (loading) {
    return (
      <div className="p-6">
        <div className="max-w-full mx-auto">
          <div className="flex items-center justify-center h-64">
            <FontAwesomeIcon icon={faSpinner} className="text-4xl text-blue-600 animate-spin" />
          </div>
        </div>
      </div>
    );
  }

  if (!caseData) {
    return (
      <div className="p-6">
        <div className="max-w-full mx-auto">
          <div className="text-center">
            <h2 className="text-2xl font-bold text-gray-900 mb-4">Case Not Found</h2>
            <button
              onClick={() => navigate('/cases')}
              className="text-gray-800"
            >
              <FontAwesomeIcon icon={faArrowLeft} className="mr-2" />
              Back to All Cases
            </button>
          </div>
        </div>
      </div>
    );
  }

  const highestFraudRisk = getHighestFraudRisk();
  const highestAIRisk = getHighestAIRisk();
  const fraudulentImagesCount = getFraudulentImagesCount();
  const aiRiskImagesCount = getAIRiskImagesCount();

  return (
    <div className="p-6 bg-gray-50 min-h-screen">
      <div className="max-w-full mx-auto">
        {/* Header */}
        <div className="mb-6">
          <button
            onClick={() => navigate('/cases')}
            className="flex items-center text-gray-800 mb-4"
          >
            <FontAwesomeIcon icon={faArrowLeft} className="mr-2" />
            Back to All Cases
          </button>
          
          <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
            <div className="flex items-center justify-between mb-4">
              <div>
                <h1 className="text-3xl font-bold text-gray-900">
                  {caseData.analysis_name}
                </h1>
                <div className="flex items-center mt-2 text-gray-600">
                  <FontAwesomeIcon icon={faCalendarAlt} className="mr-2" />
                  <span>Created on {formatDate(caseData.created_at || new Date().toISOString())}</span>
                  <span className="ml-6 px-3 py-1 bg-green-100 text-green-800 rounded-full text-sm font-medium">
                    Open
                  </span>
                </div>
              </div>
            </div>

            {/* Stats Cards */}
            <div className="grid grid-rows-1 md:grid-rows-2 grid-cols-1 md:grid-cols-3 gap-6">
              <div className="bg-blue-50 rounded-lg p-4">
                <div className="flex items-center">
                  <FontAwesomeIcon icon={faFileImage} className="text-blue-600 text-xl mr-3" />
                  <div>
                    <p className="text-sm text-gray-600">Total Images</p>
                    <p className="text-3xl font-bold text-gray-900">{caseData.total_files}</p>
                  </div>
                </div>
              </div>

              <div className="bg-red-50 rounded-lg p-4">
                <div className="flex items-center">
                  <FontAwesomeIcon icon={faBan} className="text-red-600 text-xl mr-3" />
                  <div>
                    <p className="text-sm text-gray-600">Fraudulent Images</p>
                    <p className="text-3xl font-bold text-red-600">
                      {fraudulentImagesCount}
                    </p>
                  </div>
                </div>
              </div>

              <div className="bg-indigo-50 rounded-lg p-4">
                <div className="flex items-center">
                  <FontAwesomeIcon icon={faRobot} className="text-indigo-600 text-xl mr-3" />
                  <div>
                    <p className="text-sm text-gray-600">AI Risk Images</p>
                    <p className="text-3xl font-bold text-indigo-600">
                      {aiRiskImagesCount}
                    </p>
                  </div>
                </div>
              </div>

              <div className="bg-orange-50 rounded-lg p-4">
                <div className="flex items-center">
                  <FontAwesomeIcon icon={faExclamationTriangle} className="text-orange-600 text-xl mr-3" />
                  <div>
                    <p className="text-sm text-gray-600">Highest Fraud Risk</p>
                    <p className={`text-3xl font-bold ${getRiskColor(highestFraudRisk)}`}>
                      {highestFraudRisk}%
                    </p>
                  </div>
                </div>
              </div>

              <div className="bg-purple-50 rounded-lg p-4">
                <div className="flex items-center">
                  <FontAwesomeIcon icon={faShieldAlt} className="text-purple-600 text-xl mr-3" />
                  <div>
                    <p className="text-sm text-gray-600">Highest AI Risk</p>
                    <p className={`text-3xl font-bold ${getRiskColor(highestAIRisk)}`}>
                      {highestAIRisk}%
                    </p>
                  </div>
                </div>
              </div>

              <div className="bg-green-50 rounded-lg p-4">
                <div className="flex items-center">
                  <FontAwesomeIcon icon={faDollarSign} className="text-green-600 text-xl mr-3" />
                  <div>
                    <p className="text-sm text-gray-600">Total Claim Value</p>
                    <p className="text-3xl font-bold text-gray-900">
                      {formatCurrency(caseData.total_cost)}
                    </p>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* Analyzed Images Section */}
        <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
          <div className="flex items-center justify-between mb-6">
            <h2 className="text-2xl font-bold text-gray-900">
              Analyzed Images ({images.length})
            </h2>
            {images.length > itemsPerPage && (
              <div className="text-sm text-gray-600">
                Showing {startIndex + 1}-{Math.min(endIndex, images.length)} of {images.length}
              </div>
            )}
          </div>

          <div className="space-y-6">
            {currentImages.map((image, index) => {
              const claimAmount = image.cost;

              return (
                <div key={image.id} className="border border-gray-200 rounded-lg p-6">
                  <div className="flex items-start space-x-6">
                    {/* Image */}
                    <div className="flex-shrink-0">
                      <div className="w-64 h-48 bg-gray-200 rounded-lg overflow-hidden">
                        {image.file_url ? (
                          <img 
                            src={image.file_url} 
                            alt={image.filename}
                            className="w-full h-full object-cover"
                          />
                        ) : (
                          <div className="w-full h-full flex items-center justify-center">
                            <FontAwesomeIcon icon={faFileImage} className="text-gray-400 text-4xl" />
                          </div>
                        )}
                      </div>
                    </div>

                    {/* Image Details */}
                    <div className="flex-1">
                      <div className="flex items-start justify-between mb-4">
                        <div>
                          <h3 className="text-xl font-bold text-gray-900 mb-1">
                            {image.filename || `Image ${index + 1}`}
                          </h3>
                          <span className={`inline-flex items-center px-3 py-1 rounded-full text-sm font-medium ${
                            Math.round(image.fraud_score * 100) >= 70 || Math.round(image.ai_score * 100) >= 70
                              ? 'bg-red-100 text-red-800'
                              : Math.round(image.fraud_score * 100) >= 50 || Math.round(image.ai_score * 100) >= 50
                              ? 'bg-yellow-100 text-yellow-800'
                              : 'bg-green-100 text-green-800'
                          }`}>
                            {Math.round(image.fraud_score * 100) >= 70 || Math.round(image.ai_score * 100) >= 70
                              ? 'High'
                              : Math.round(image.fraud_score * 100) >= 50 || Math.round(image.ai_score * 100) >= 50
                              ? 'Medium'
                              : 'Low'}
                          </span>
                        </div>
                      </div>

                      <div className="grid grid-cols-2 gap-6 mb-4">
                        <div>
                          <p className="text-sm text-gray-600">Fraud Risk</p>
                          <p className={`text-2xl font-bold ${getRiskColor(Math.round(image.fraud_score * 100))}`}>
                            {Math.round(image.fraud_score * 100)}%
                          </p>
                        </div>
                        <div>
                          <p className="text-sm text-gray-600">AI Gen. Risk</p>
                          <p className={`text-2xl font-bold ${getRiskColor(Math.round(image.ai_score * 100))}`}>
                            {Math.round(image.ai_score * 100)}%
                          </p>
                        </div>
                      </div>

                      <p className="text-sm text-gray-600 mb-1">
                        {formatCurrency(claimAmount)} Claim
                      </p>

                      {/* Fraud Indicators */}
                      {image.detected_issues && image.detected_issues.length > 0 && (
                        <div className="mt-4">
                          <h4 className="text-sm font-medium text-red-800 mb-2 flex items-center">
                            <FontAwesomeIcon icon={faInfoCircle} className="mr-2" />
                            Fraud Indicators:
                          </h4>
                          <div className="text-sm text-red-700">
                            {(() => {
                              try {
                                const issues = JSON.parse(image.detected_issues);
                                return Array.isArray(issues) ? issues.join(', ') : image.detected_issues;
                              } catch {
                                return image.detected_issues;
                              }
                            })()}
                          </div>
                        </div>
                      )}
                    </div>
                  </div>
                </div>
              );
            })}
          </div>

          {images.length === 0 && (
            <div className="text-center py-12">
              <FontAwesomeIcon icon={faFileImage} className="text-4xl text-gray-400 mb-4" />
              <h3 className="text-lg font-medium text-gray-900 mb-2">No images found</h3>
              <p className="text-gray-600">
                No analyzed images are associated with this case.
              </p>
            </div>
          )}

          {/* Pagination Controls */}
          {images.length > itemsPerPage && (
            <div className="mt-8 flex items-center justify-between">
              <div className="flex items-center space-x-2">
                <button
                  onClick={goToPreviousPage}
                  disabled={currentPage === 1}
                  className="px-3 py-2 text-sm font-medium text-gray-500 bg-white border border-gray-300 rounded-md hover:bg-gray-50 disabled:opacity-50 disabled:cursor-not-allowed"
                >
                  <FontAwesomeIcon icon={faChevronLeft} className="w-4 h-4" />
                </button>
                
                <div className="flex items-center space-x-1">
                  {Array.from({ length: totalPages }, (_, i) => i + 1).map((page) => {
                    // Show first page, last page, current page, and pages around current page
                    const shouldShow = 
                      page === 1 || 
                      page === totalPages || 
                      (page >= currentPage - 1 && page <= currentPage + 1);
                    
                    if (!shouldShow) {
                      // Show ellipsis for gaps
                      if (page === 2 && currentPage > 4) {
                        return <span key={`ellipsis-${page}`} className="px-2 text-gray-500">...</span>;
                      }
                      if (page === totalPages - 1 && currentPage < totalPages - 3) {
                        return <span key={`ellipsis-${page}`} className="px-2 text-gray-500">...</span>;
                      }
                      return null;
                    }
                    
                    return (
                      <button
                        key={page}
                        onClick={() => goToPage(page)}
                        className={`px-3 py-2 text-sm font-medium rounded-md ${
                          page === currentPage
                            ? 'bg-blue-600 text-white'
                            : 'text-gray-500 bg-white border border-gray-300 hover:bg-gray-50'
                        }`}
                      >
                        {page}
                      </button>
                    );
                  })}
                </div>
                
                <button
                  onClick={goToNextPage}
                  disabled={currentPage === totalPages}
                  className="px-3 py-2 text-sm font-medium text-gray-500 bg-white border border-gray-300 rounded-md hover:bg-gray-50 disabled:opacity-50 disabled:cursor-not-allowed"
                >
                  <FontAwesomeIcon icon={faChevronRight} className="w-4 h-4" />
                </button>
              </div>
              
              <div className="text-sm text-gray-600">
                Page {currentPage} of {totalPages}
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default CaseDetail;
