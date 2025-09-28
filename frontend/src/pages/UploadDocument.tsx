import React, { useState, useRef } from 'react';
import { useNavigate } from 'react-router-dom';
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome';
import { 
  faFilePdf, 
  faSpinner, 
  faCheckCircle,
  faExclamationTriangle,
  faArrowUpFromBracket,
  faArrowLeft,
  faFile,
  faShieldAlt,
  faEye,
  faDownload,
} from '@fortawesome/free-solid-svg-icons';
import { useStats } from '../contexts/StatsContext';

interface SelectedFile {
  id: string;
  file: File;
  preview?: string;
}

interface UploadedFile {
  id: string;
  file: File;
  status: 'uploading' | 'uploaded' | 'analyzing' | 'completed' | 'error';
  extractedData?: any;
  fraudPrediction?: any;
  error?: string;
  progress?: number;
}

interface BatchAnalysisState {
  isAnalyzing: boolean;
  totalFiles: number;
  completedFiles: number;
  currentFile?: string;
}

const UploadDocument: React.FC = () => {
  const navigate = useNavigate();
  const { updateStats } = useStats();
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [selectedFiles, setSelectedFiles] = useState<SelectedFile[]>([]);
  const [uploadedFiles, setUploadedFiles] = useState<UploadedFile[]>([]);
  const [batchAnalysis, setBatchAnalysis] = useState<BatchAnalysisState>({
    isAnalyzing: false,
    totalFiles: 0,
    completedFiles: 0,
  });
  const [dragActive, setDragActive] = useState(false);

  const handleFileSelect = (files: FileList | null) => {
    if (!files) return;

    const newFiles: SelectedFile[] = Array.from(files)
      .filter(file => file.type === 'application/pdf')
      .map(file => ({
        id: Math.random().toString(36).substr(2, 9),
        file,
      }));

    setSelectedFiles(prev => [...prev, ...newFiles]);
  };

  const handleDrag = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === "dragenter" || e.type === "dragover") {
      setDragActive(true);
    } else if (e.type === "dragleave") {
      setDragActive(false);
    }
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFileSelect(e.dataTransfer.files);
    }
  };

  const removeFile = (fileId: string) => {
    setSelectedFiles(prev => prev.filter(f => f.id !== fileId));
  };

  const convertFileToBase64 = (file: File): Promise<string> => {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.readAsDataURL(file);
      reader.onload = () => {
        const result = reader.result as string;
        // Remove data:application/pdf;base64, prefix
        const base64 = result.split(',')[1];
        resolve(base64);
      };
      reader.onerror = error => reject(error);
    });
  };

  const analyzeDocument = async (file: File, fileId: string): Promise<any> => {
    try {
      // Convert file to base64
      const base64Data = await convertFileToBase64(file);
      
      // Call PDF parsing API
      const parseResponse = await fetch('http://localhost:8000/api/parse-claim-pdf', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          pdf: base64Data,
          extraPrompt: "Focus on extracting all available claim information accurately."
        })
      });

      if (!parseResponse.ok) {
        throw new Error(`PDF parsing failed: ${parseResponse.status}`);
      }

      const parseResult = await parseResponse.json();
      
      // Call fraud prediction API
      const predictResponse = await fetch('http://localhost:8000/predict', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(parseResult.canonical)
      });

      if (!predictResponse.ok) {
        throw new Error(`Fraud prediction failed: ${predictResponse.status}`);
      }

      const predictResult = await predictResponse.json();

      return {
        extractedData: parseResult.canonical,
        fraudPrediction: predictResult.data[0]
      };
    } catch (error) {
      throw error;
    }
  };

  const startBatchAnalysis = async () => {
    if (selectedFiles.length === 0) return;

    setBatchAnalysis({
      isAnalyzing: true,
      totalFiles: selectedFiles.length,
      completedFiles: 0,
    });

    const newUploadedFiles: UploadedFile[] = selectedFiles.map(file => ({
      id: file.id,
      file: file.file,
      status: 'uploading' as const,
      progress: 0,
    }));

    setUploadedFiles(newUploadedFiles);

    for (let i = 0; i < selectedFiles.length; i++) {
      const file = selectedFiles[i];
      
      try {
        // Update status to analyzing
        setUploadedFiles(prev => prev.map(f => 
          f.id === file.id 
            ? { ...f, status: 'analyzing', progress: 50 }
            : f
        ));

        setBatchAnalysis(prev => ({
          ...prev,
          currentFile: file.file.name,
        }));

        // Analyze the document
        const result = await analyzeDocument(file.file, file.id);

        // Update with results
        setUploadedFiles(prev => prev.map(f => 
          f.id === file.id 
            ? { 
                ...f, 
                status: 'completed', 
                progress: 100,
                extractedData: result.extractedData,
                fraudPrediction: result.fraudPrediction
              }
            : f
        ));

        // // Update stats
        // updateStats({
        //   casesAnalyzed: 1,
        //   fraudDetected: result.fraudPrediction.decision === 1 ? 1 : 0,
        //   moneySaved: result.fraudPrediction.decision === 1 ? result.extractedData.ClaimAmount || 0 : 0,
        // });

      } catch (error) {
        console.error(`Error analyzing ${file.file.name}:`, error);
        setUploadedFiles(prev => prev.map(f => 
          f.id === file.id 
            ? { 
                ...f, 
                status: 'error', 
                error: error instanceof Error ? error.message : 'Unknown error'
              }
            : f
        ));
      }

      setBatchAnalysis(prev => ({
        ...prev,
        completedFiles: i + 1,
      }));
    }

    setBatchAnalysis(prev => ({
      ...prev,
      isAnalyzing: false,
      currentFile: undefined,
    }));
  };

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'uploading':
      case 'analyzing':
        return <FontAwesomeIcon icon={faSpinner} className="animate-spin text-blue-500" />;
      case 'completed':
        return <FontAwesomeIcon icon={faCheckCircle} className="text-green-500" />;
      case 'error':
        return <FontAwesomeIcon icon={faExclamationTriangle} className="text-red-500" />;
      default:
        return <FontAwesomeIcon icon={faFilePdf} className="text-gray-400" />;
    }
  };

  const getRiskLevel = (prediction: any) => {
    if (!prediction) return 'Unknown';
    const proba = prediction.proba;
    if (proba > 0.7) return 'HIGH';
    if (proba > 0.4) return 'MEDIUM';
    return 'LOW';
  };

  const getRiskColor = (level: string) => {
    switch (level) {
      case 'HIGH': return 'text-red-600 bg-red-50';
      case 'MEDIUM': return 'text-yellow-600 bg-yellow-50';
      case 'LOW': return 'text-green-600 bg-green-50';
      default: return 'text-gray-600 bg-gray-50';
    }
  };

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Header */}
      <div className="bg-white shadow-sm border-b border-gray-200">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex items-center justify-between h-16">
            <div className="flex items-center space-x-4">
              <button
                onClick={() => navigate('/dashboard')}
                className="p-2 rounded-md hover:bg-gray-100 transition-colors"
              >
                <FontAwesomeIcon icon={faArrowLeft} className="text-gray-600" />
              </button>
              <div>
                <h1 className="text-2xl font-bold text-gray-900">Upload Document</h1>
                <p className="text-sm text-gray-500">Upload and analyze insurance claim documents</p>
              </div>
            </div>
            <div className="flex items-center space-x-2">
              <FontAwesomeIcon icon={faShieldAlt} className="text-blue-600" />
              <span className="text-sm font-medium text-gray-700">AI-Powered Analysis</span>
            </div>
          </div>
        </div>
      </div>

      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
          {/* Upload Section */}
          <div className="space-y-6">
            {/* Upload Area */}
            <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
              <h2 className="text-lg font-semibold text-gray-900 mb-4">Upload PDF Documents</h2>
              
              <div
                className={`border-2 border-dashed rounded-lg p-8 text-center transition-colors ${
                  dragActive 
                    ? 'border-blue-400 bg-blue-50' 
                    : 'border-gray-300 hover:border-blue-400 hover:bg-blue-100'
                }`}
                onDragEnter={handleDrag}
                onDragLeave={handleDrag}
                onDragOver={handleDrag}
                onDrop={handleDrop}
              >
                <FontAwesomeIcon 
                  icon={faFilePdf} 
                  className="mx-auto h-12 w-12 text-gray-400 mb-4" 
                />
                <div className="space-y-2">
                  <p className="text-lg font-medium text-gray-900">
                    Drop PDF files here, or{' '}
                    <button
                      onClick={() => fileInputRef.current?.click()}
                      className="text-blue-600 hover:text-blue-500 font-medium"
                    >
                      browse
                    </button>
                  </p>
                  <p className="text-sm text-gray-500">
                    Supports PDF files up to 50MB
                  </p>
                </div>
                <input
                  ref={fileInputRef}
                  type="file"
                  multiple
                  accept=".pdf"
                  onChange={(e) => handleFileSelect(e.target.files)}
                  className="hidden"
                />
              </div>
            </div>

            {/* Selected Files */}
            {selectedFiles.length > 0 && (
              <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
                <div className="flex items-center justify-between mb-4">
                  <h3 className="text-lg font-semibold text-gray-900">
                    Selected Files ({selectedFiles.length})
                  </h3>
                  <button
                    onClick={startBatchAnalysis}
                    disabled={batchAnalysis.isAnalyzing}
                    className="bg-blue-600 text-white px-4 py-2 rounded-lg hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed flex items-center space-x-2"
                  >
                    {batchAnalysis.isAnalyzing ? (
                      <FontAwesomeIcon icon={faSpinner} className="animate-spin" />
                    ) : (
                      <FontAwesomeIcon icon={faArrowUpFromBracket} />
                    )}
                    <span>Start Analysis</span>
                  </button>
                </div>
                
                <div className="space-y-3">
                  {selectedFiles.map((file) => (
                    <div key={file.id} className="flex items-center justify-between p-3 bg-gray-50 rounded-lg">
                      <div className="flex items-center space-x-3">
                        <FontAwesomeIcon icon={faFilePdf} className="text-red-500" />
                        <div>
                          <p className="font-medium text-gray-900">{file.file.name}</p>
                          <p className="text-sm text-gray-500">
                            {(file.file.size / 1024 / 1024).toFixed(2)} MB
                          </p>
                        </div>
                      </div>
                      <button
                        onClick={() => removeFile(file.id)}
                        className="text-gray-400 hover:text-red-500"
                      >
                        <FontAwesomeIcon icon={faExclamationTriangle} />
                      </button>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* Batch Progress */}
            {batchAnalysis.isAnalyzing && (
              <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
                <h3 className="text-lg font-semibold text-gray-900 mb-4">Analysis Progress</h3>
                <div className="space-y-4">
                  <div className="flex items-center justify-between">
                    <span className="text-sm font-medium text-gray-700">
                      Processing: {batchAnalysis.currentFile}
                    </span>
                    <span className="text-sm text-gray-500">
                      {batchAnalysis.completedFiles} / {batchAnalysis.totalFiles}
                    </span>
                  </div>
                  <div className="w-full bg-gray-200 rounded-full h-2">
                    <div 
                      className="bg-blue-600 h-2 rounded-full transition-all duration-300"
                      style={{ 
                        width: `${(batchAnalysis.completedFiles / batchAnalysis.totalFiles) * 100}%` 
                      }}
                    />
                  </div>
                </div>
              </div>
            )}
          </div>

          {/* Results Section */}
          <div className="space-y-6">
            <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
              <h2 className="text-lg font-semibold text-gray-900 mb-4">Analysis Results</h2>
              
              {uploadedFiles.length === 0 ? (
                <div className="text-center py-12">
                  <FontAwesomeIcon icon={faFilePdf} className="mx-auto h-12 w-12 text-gray-300 mb-4" />
                  <p className="text-gray-500">No documents analyzed yet</p>
                </div>
              ) : (
                <div className="space-y-4">
                  {uploadedFiles.map((file) => (
                    <div key={file.id} className="border border-gray-200 rounded-lg p-4">
                      <div className="flex items-center justify-between mb-3">
                        <div className="flex items-center space-x-3">
                          {getStatusIcon(file.status)}
                          <div>
                            <p className="font-medium text-gray-900">{file.file.name}</p>
                            <p className="text-sm text-gray-500 capitalize">{file.status}</p>
                          </div>
                        </div>
                        {file.fraudPrediction && (
                          <span className={`px-2 py-1 rounded-full text-xs font-medium ${getRiskColor(getRiskLevel(file.fraudPrediction))}`}>
                            {getRiskLevel(file.fraudPrediction)} RISK
                          </span>
                        )}
                      </div>

                      {file.status === 'error' && (
                        <div className="bg-red-50 border border-red-200 rounded-lg p-3">
                          <p className="text-sm text-red-700">{file.error}</p>
                        </div>
                      )}

                      {file.status === 'completed' && file.extractedData && file.fraudPrediction && (
                        <div className="space-y-3">
                          <div className="grid grid-cols-2 gap-4 text-sm">
                            <div>
                              <span className="font-medium text-gray-700">Claim Amount:</span>
                              <span className="ml-2 text-gray-900">${file.extractedData.ClaimAmount?.toLocaleString() || 'N/A'}</span>
                            </div>
                            <div>
                              <span className="font-medium text-gray-700">Vehicle:</span>
                              <span className="ml-2 text-gray-900">{file.extractedData.Make} {file.extractedData.Year}</span>
                            </div>
                            <div>
                              <span className="font-medium text-gray-700">Fraud Probability:</span>
                              <span className="ml-2 text-gray-900">{(file.fraudPrediction.proba * 100).toFixed(2)}%</span>
                            </div>
                            <div>
                              <span className="font-medium text-gray-700">Decision:</span>
                              <span className={`ml-2 font-medium ${file.fraudPrediction.decision === 1 ? 'text-red-600' : 'text-green-600'}`}>
                                {file.fraudPrediction.decision === 1 ? 'FRAUD' : 'LEGITIMATE'}
                              </span>
                            </div>
                          </div>
                          
                          <div className="flex space-x-2">
                            <button className="flex items-center space-x-2 px-3 py-2 bg-blue-50 text-blue-700 rounded-lg hover:bg-blue-100 transition-colors">
                              <FontAwesomeIcon icon={faEye} className="text-sm" />
                              <span className="text-sm">View Details</span>
                            </button>
                            <button className="flex items-center space-x-2 px-3 py-2 bg-gray-50 text-gray-700 rounded-lg hover:bg-gray-100 transition-colors">
                              <FontAwesomeIcon icon={faDownload} className="text-sm" />
                              <span className="text-sm">Export Report</span>
                            </button>
                          </div>
                        </div>
                      )}

                      {file.progress !== undefined && file.status !== 'completed' && (
                        <div className="w-full bg-gray-200 rounded-full h-2">
                          <div 
                            className="bg-blue-600 h-2 rounded-full transition-all duration-300"
                            style={{ width: `${file.progress}%` }}
                          />
                        </div>
                      )}
                    </div>
                  ))}
                </div>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default UploadDocument;
