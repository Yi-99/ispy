import React, { useState, useRef } from 'react';
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome';
import { 
  faFileImage, 
  faCloudUploadAlt, 
  faSpinner, 
  faCheckCircle, 
  faExclamationTriangle,
  faTrash,
  faRedo,
  faRedoAlt
} from '@fortawesome/free-solid-svg-icons';
import { uploadImage, analyzeImage, type UploadResult, type AnalysisResult } from '../api/imageUpload';

interface UploadedFile {
  id: string;
  file: File;
  url?: string;
  status: 'uploading' | 'uploaded' | 'analyzing' | 'completed' | 'error';
  analysis?: AnalysisResult['data'];
  error?: string;
}

const Upload: React.FC = () => {
  const [uploadedFiles, setUploadedFiles] = useState<UploadedFile[]>([]);
  const [isDragOver, setIsDragOver] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleFileSelect = async (files: FileList) => {
    const fileArray = Array.from(files);
    
    // Add files to state with uploading status
    const newFiles: UploadedFile[] = fileArray.map(file => ({
      id: Math.random().toString(36).substr(2, 9),
      file,
      status: 'uploading' as const
    }));
    
    setUploadedFiles(prev => [...prev, ...newFiles]);

    // Process each file
    for (const uploadFile of newFiles) {
      try {
        // Upload to Supabase
        const uploadResult: UploadResult = await uploadImage(uploadFile.file);
        
        if (uploadResult.success) {
          // Update status to uploaded
          setUploadedFiles(prev => 
            prev.map(f => 
              f.id === uploadFile.id 
                ? { ...f, status: 'analyzing', url: uploadResult.url }
                : f
            )
          );

          // Start analysis
          const analysisResult = await analyzeImage(uploadResult.url!);
          
          setUploadedFiles(prev => 
            prev.map(f => 
              f.id === uploadFile.id 
                ? { 
                    ...f, 
                    status: analysisResult.success ? 'completed' : 'error',
                    analysis: analysisResult.data,
                    error: analysisResult.error
                  }
                : f
            )
          );
        } else {
          setUploadedFiles(prev => 
            prev.map(f => 
              f.id === uploadFile.id 
                ? { ...f, status: 'error', error: uploadResult.error }
                : f
            )
          );
        }
      } catch (error) {
        setUploadedFiles(prev => 
          prev.map(f => 
            f.id === uploadFile.id 
              ? { ...f, status: 'error', error: 'Upload failed' }
              : f
          )
        );
      }
    }
  };

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(true);
  };

  const handleDragLeave = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(false);
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(false);
    const files = e.dataTransfer.files;
    if (files.length > 0) {
      handleFileSelect(files);
    }
  };

  const handleFileInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files;
    if (files && files.length > 0) {
      handleFileSelect(files);
    }
  };

  const removeFile = (id: string) => {
    setUploadedFiles(prev => prev.filter(f => f.id !== id));
  };

  const retryFile = async (id: string) => {
    const fileToRetry = uploadedFiles.find(f => f.id === id);
    if (!fileToRetry) return;

    // Reset status to uploading
    setUploadedFiles(prev => 
      prev.map(f => 
        f.id === id 
          ? { ...f, status: 'uploading', error: undefined }
          : f
      )
    );

    try {
      // Upload to Supabase
      const uploadResult: UploadResult = await uploadImage(fileToRetry.file);
      
      if (uploadResult.success) {
        // Update status to uploaded
        setUploadedFiles(prev => 
          prev.map(f => 
            f.id === id 
              ? { ...f, status: 'analyzing', url: uploadResult.url }
              : f
          )
        );

        // Start analysis
        const analysisResult = await analyzeImage(uploadResult.url!);
        
        setUploadedFiles(prev => 
          prev.map(f => 
            f.id === id 
              ? { 
                  ...f, 
                  status: analysisResult.success ? 'completed' : 'error',
                  analysis: analysisResult.data,
                  error: analysisResult.error
                }
              : f
          )
        );
      } else {
        setUploadedFiles(prev => 
          prev.map(f => 
            f.id === id 
              ? { ...f, status: 'error', error: uploadResult.error }
              : f
          )
        );
      }
    } catch (error) {
      setUploadedFiles(prev => 
        prev.map(f => 
          f.id === id 
            ? { ...f, status: 'error', error: 'Upload failed' }
            : f
        )
      );
    }
  };

  const retryAllFailed = async () => {
    const failedFiles = uploadedFiles.filter(f => f.status === 'error');
    
    for (const file of failedFiles) {
      await retryFile(file.id);
    }
  };

  const getStatusIcon = (status: UploadedFile['status']) => {
    switch (status) {
      case 'uploading':
      case 'analyzing':
        return <FontAwesomeIcon icon={faSpinner} className="animate-spin text-blue-600" />;
      case 'completed':
        return <FontAwesomeIcon icon={faCheckCircle} className="text-green-600" />;
      case 'error':
        return <FontAwesomeIcon icon={faExclamationTriangle} className="text-red-600" />;
      default:
        return null;
    }
  };

  const getStatusText = (status: UploadedFile['status']) => {
    switch (status) {
      case 'uploading':
        return 'Uploading...';
      case 'analyzing':
        return 'Analyzing...';
      case 'completed':
        return 'Analysis Complete';
      case 'error':
        return 'Error';
      default:
        return 'Uploaded';
    }
  };

  return (
    <div className="p-6">
      <div className="max-w-6xl mx-auto">
        <div className="mb-8">
          <h1 className="text-3xl font-bold text-gray-900 mb-2">Upload & Analyze</h1>
          <p className="text-gray-600">
            Upload vehicle damage images for AI-powered fraud detection analysis
          </p>
        </div>

        {/* Upload Area */}
        <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-8 mb-8">
          <div className="text-center">
            <div className="w-24 h-24 bg-blue-100 rounded-full flex items-center justify-center mx-auto mb-6">
              <FontAwesomeIcon icon={faCloudUploadAlt} className="text-4xl text-blue-600" />
            </div>
            <h2 className="text-2xl font-semibold text-gray-900 mb-4">Upload Vehicle Damage Images</h2>
            <p className="text-gray-600 mb-8 max-w-2xl mx-auto">
              Drag and drop your images here, or click to browse. We support JPG, PNG, and other common image formats.
            </p>
            
            <div 
              className={`border-2 border-dashed rounded-lg p-12 transition-colors duration-200 cursor-pointer ${
                isDragOver 
                  ? 'border-blue-400 bg-blue-50' 
                  : 'border-gray-300 hover:border-blue-400'
              }`}
              onDragOver={handleDragOver}
              onDragLeave={handleDragLeave}
              onDrop={handleDrop}
              onClick={() => fileInputRef.current?.click()}
            >
              <FontAwesomeIcon icon={faFileImage} className="text-6xl text-gray-400 mb-4" />
              <p className="text-lg font-medium text-gray-700 mb-2">Drop files here or click to upload</p>
              <p className="text-sm text-gray-500">Maximum file size: 10MB per image</p>
            </div>
            
            <input
              ref={fileInputRef}
              type="file"
              multiple
              accept="image/*"
              onChange={handleFileInputChange}
              className="hidden"
            />
          </div>
        </div>

        {/* Uploaded Files */}
        {uploadedFiles.length > 0 && (
          <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-lg font-semibold text-gray-900">Uploaded Files</h3>
              {uploadedFiles.some(f => f.status === 'error') && (
                <button
                  onClick={retryAllFailed}
                  className="bg-gray-50 hover:bg-gray-300 text-black px-4 py-2 rounded-lg text-sm font-medium flex items-center space-x-2 transition-colors duration-200"
                >
                  <FontAwesomeIcon icon={faRedoAlt} />
                  <span>Retry All</span>
                </button>
              )}
            </div>
            <div className="space-y-4">
              {uploadedFiles.map((file) => (
                <div key={file.id} className="border border-gray-200 rounded-lg p-4">
                  <div className="flex items-center justify-between">
                    <div className="flex items-center space-x-3">
                      <div className="w-12 h-12 bg-gray-100 rounded-lg flex items-center justify-center">
                        <FontAwesomeIcon icon={faFileImage} className="text-gray-500" />
                      </div>
                      <div>
                        <p className="font-medium text-gray-900">{file.file.name}</p>
                        <p className="text-sm text-gray-500">
                          {(file.file.size / 1024 / 1024).toFixed(2)} MB
                        </p>
                      </div>
                    </div>
                    
                    <div className="flex items-center space-x-3">
                      <div className="flex items-center space-x-2">
                        {getStatusIcon(file.status)}
                        <span className="text-sm text-gray-600">
                          {getStatusText(file.status)}
                        </span>
                      </div>
                      
                      {file.status === 'completed' && file.analysis && (
                        <div className="flex items-center space-x-2">
                          <span className={`px-2 py-1 text-xs font-medium rounded-full ${
                            file.analysis.isFraudulent 
                              ? 'bg-red-100 text-red-800' 
                              : file.analysis.riskLevel === 'HIGH'
                              ? 'bg-yellow-100 text-yellow-800'
                              : 'bg-green-100 text-green-800'
                          }`}>
                            {file.analysis.isFraudulent ? 'FRAUD' : file.analysis.riskLevel}
                          </span>
                          <span className="text-sm text-gray-600">
                            {Math.round(file.analysis.confidence * 100)}% confidence
                          </span>
                        </div>
                      )}
                      
                      {file.status === 'error' && (
                        <button
                          onClick={() => retryFile(file.id)}
                          className="text-gray-600 hover:text-gray-800 mr-2"
                          title="Retry upload"
                        >
                          <FontAwesomeIcon icon={faRedo} />
                        </button>
                      )}
                      
                      <button
                        onClick={() => removeFile(file.id)}
                        className="text-red-600 hover:text-red-800"
                        title="Remove file"
                      >
                        <FontAwesomeIcon icon={faTrash} />
                      </button>
                    </div>
                  </div>
                  
                  {file.status === 'completed' && file.analysis && (
                    <div className="mt-4 p-4 bg-gray-50 rounded-lg">
                      <h4 className="font-medium text-gray-900 mb-2">Analysis Results</h4>
                      <p className="text-sm text-gray-700 mb-2">{file.analysis.analysis}</p>
                      {file.analysis.detectedIssues.length > 0 && (
                        <div>
                          <p className="text-sm font-medium text-gray-900 mb-1">Detected Issues:</p>
                          <ul className="text-sm text-gray-700 list-disc list-inside">
                            {file.analysis.detectedIssues.map((issue, index) => (
                              <li key={index}>{issue}</li>
                            ))}
                          </ul>
                        </div>
                      )}
                    </div>
                  )}
                  
                  {file.status === 'error' && file.error && (
                    <div className="mt-4 p-4 bg-red-50 rounded-lg">
                      <p className="text-sm text-red-700">{file.error}</p>
                    </div>
                  )}
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default Upload;
