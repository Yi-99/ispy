import React from 'react';
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome';
import { faFileImage, faCheckCircle } from '@fortawesome/free-solid-svg-icons';

interface AnalysisResult {
  isFraudulent: boolean;
  aiScore: string;
  fraudScore: string;
  aiAnalysis: string;
  fraudAnalysis: string;
  riskLevel: 'LOW' | 'MEDIUM' | 'HIGH';
  detectedIssues: string[];
}

interface UploadedFile {
  id: string;
  file: File;
  url?: string;
  status: 'uploading' | 'uploaded' | 'analyzing' | 'completed' | 'error';
  analysis?: AnalysisResult;
  error?: string;
  progress?: number;
  fraudRisk?: number;
  keyIndicators?: string[];
  cost?: number;
}

interface ResultsDisplayProps {
  files: UploadedFile[];
  actionButton?: React.ReactNode;
}

const ResultsDisplay: React.FC<ResultsDisplayProps> = ({ files, actionButton }) => {
  const completedFiles = files.filter(f => f.status === 'completed');

  if (completedFiles.length === 0) {
    return null;
  }

  return (
    <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-8 mb-8">
      <div className="flex items-center justify-between mb-6">
        <h2 className="text-2xl font-semibold text-gray-900">Individual Results</h2>
        {actionButton}
      </div>
      
      <div className="space-y-6">
        {completedFiles.map((file) => (
          <div key={file.id} className="bg-white rounded-lg border border-gray-200 p-6 shadow-sm">
            <div className="flex items-start space-x-6">
              {/* Image Section */}
              <div className="flex-shrink-0">
                <div className="w-24 h-24 bg-gray-200 rounded-lg overflow-hidden">
                  {file.url ? (
                    <img 
                      src={file.url} 
                      alt={file.file.name}
                      className="w-full h-full object-cover"
                    />
                  ) : (
                    <div className="w-full h-full flex items-center justify-center">
                      <FontAwesomeIcon icon={faFileImage} className="text-gray-400 text-2xl" />
                    </div>
                  )}
                </div>
              </div>
              
              {/* AI Gen Risk Section */}
              <div className="flex-1">
                <div className="mb-4">
                  <div className="flex items-center justify-between mb-2">
                    <span className="text-sm font-medium text-gray-700 uppercase">AI GEN RISK</span>
                    <span className={`text-3xl font-bold ${
                      (file.analysis?.aiScore ? parseFloat(file.analysis.aiScore) * 100 : 0) > 70 ? 'text-red-500' : 
                      (file.analysis?.aiScore ? parseFloat(file.analysis.aiScore) * 100 : 0) > 50 ? 'text-yellow-500' : 'text-green-500'
                    }`}>
                      {file.analysis?.aiScore ? Math.round(parseFloat(file.analysis.aiScore) * 100) : 0}%
                    </span>
                  </div>
                  <span className={`inline-block px-3 py-1 rounded-full text-xs font-medium mb-2 ${
                    (file.analysis?.aiScore ? parseFloat(file.analysis.aiScore) * 100 : 0) > 70 ? 'bg-red-100 text-red-800' : 
                    (file.analysis?.aiScore ? parseFloat(file.analysis.aiScore) * 100 : 0) > 50 ? 'bg-yellow-100 text-yellow-800' : 'bg-green-100 text-green-800'
                  }`}>
                    {(file.analysis?.aiScore ? parseFloat(file.analysis.aiScore) * 100 : 0) > 70 ? 'High' : 
                      (file.analysis?.aiScore ? parseFloat(file.analysis.aiScore) * 100 : 0) > 50 ? 'Medium' : 'Low'}
                  </span>
                  <div className="w-full bg-gray-200 rounded-full h-2">
                    <div 
                      className={`h-2 rounded-full ${
                        (file.analysis?.aiScore ? parseFloat(file.analysis.aiScore) * 100 : 0) > 70 ? 'bg-red-500' : 
                        (file.analysis?.aiScore ? parseFloat(file.analysis.aiScore) * 100 : 0) > 50 ? 'bg-yellow-500' : 'bg-green-500'
                      }`}
                      style={{ width: `${file.analysis?.aiScore ? parseFloat(file.analysis.aiScore) * 100 : 0}%` }}
                    ></div>
                  </div>
                </div>
              </div>
              
              {/* Fraud Risk Section */}
              <div className="flex-1">
                <div className="mb-4">
                  <div className="flex items-center justify-between mb-2">
                    <span className="text-sm font-medium text-gray-700 uppercase">FRAUD RISK</span>
                    <span className={`text-3xl font-bold ${
                      (file.analysis?.fraudScore ? parseFloat(file.analysis.fraudScore) * 100 : 0) > 70 ? 'text-red-500' : 
                      (file.analysis?.fraudScore ? parseFloat(file.analysis.fraudScore) * 100 : 0) > 50 ? 'text-yellow-500' : 'text-green-500'
                    }`}>
                      {file.analysis?.fraudScore ? Math.round(parseFloat(file.analysis.fraudScore) * 100) : 0}%
                    </span>
                  </div>
                  <span className={`inline-block px-3 py-1 rounded-full text-xs font-medium mb-2 ${
                    (file.analysis?.fraudScore ? parseFloat(file.analysis.fraudScore) * 100 : 0) > 70 ? 'bg-red-100 text-red-800' : 
                    (file.analysis?.fraudScore ? parseFloat(file.analysis.fraudScore) * 100 : 0) > 50 ? 'bg-yellow-100 text-yellow-800' : 'bg-green-100 text-green-800'
                  }`}>
                    {(file.analysis?.fraudScore ? parseFloat(file.analysis.fraudScore) * 100 : 0) > 70 ? 'High' : 
                      (file.analysis?.fraudScore ? parseFloat(file.analysis.fraudScore) * 100 : 0) > 50 ? 'Medium' : 'Low'}
                  </span>
                  <div className="w-full bg-gray-200 rounded-full h-2">
                    <div 
                      className={`h-2 rounded-full ${
                        (file.analysis?.fraudScore ? parseFloat(file.analysis.fraudScore) * 100 : 0) > 70 ? 'bg-red-500' : 
                        (file.analysis?.fraudScore ? parseFloat(file.analysis.fraudScore) * 100 : 0) > 50 ? 'bg-yellow-500' : 'bg-green-500'
                      }`}
                      style={{ width: `${file.analysis?.fraudScore ? parseFloat(file.analysis.fraudScore) * 100 : 0}%` }}
                    ></div>
                  </div>
                </div>
              </div>
              
              {/* Claim Details Section */}
              <div className="flex-1">
                <div className="mb-4">
                  <span className="text-sm font-medium text-gray-700 uppercase">CLAIM DETAILS</span>
                  <div className="mt-1">
                    {file.cost && (
                      <p className="text-xs text-gray-600">
                        ${file.cost.toLocaleString()} analysis cost
                      </p>
                    )}
                  </div>
                </div>
              </div>
              
              {/* Detected Issues Section */}
              <div className="flex-1">
                <span className="text-sm font-medium text-gray-700 uppercase">DETECTED ISSUES</span>
                <div className="mt-1">
                  {file.analysis?.detectedIssues && file.analysis.detectedIssues.length > 0 ? (
                    <ul className="text-sm font-bold text-gray-900 space-y-1 list-none">
                      {file.analysis.detectedIssues.slice(0, 3).map((issue, index) => (
                        <li key={index} className="flex items-center space-x-2">
                          <span className="text-xs">{issue}</span>
                        </li>
                      ))}
                      {file.analysis.detectedIssues.length > 3 && (
                        <li className="text-xs text-gray-500">+{file.analysis.detectedIssues.length - 3} more...</li>
                      )}
                    </ul>
                  ) : (
                    <div className="flex items-center space-x-2">
                      <FontAwesomeIcon icon={faCheckCircle} className="text-green-500 text-sm" />
                      <span className="text-sm text-gray-900">No issues detected</span>
                    </div>
                  )}
                </div>
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};

export default ResultsDisplay;
