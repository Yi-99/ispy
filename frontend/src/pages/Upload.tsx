import React, { useState, useRef, useEffect} from 'react';
import { useNavigate } from 'react-router-dom';
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome';
import { 
  faFileImage, 
  faSpinner, 
  faCheckCircle, 
  faExclamationTriangle,
  faArrowUpFromBracket,
  faArrowLeft,
  faFile,
  faShieldAlt,
} from '@fortawesome/free-solid-svg-icons';
import { uploadImage, analyzeImage, type UploadResult, type AnalysisResult } from '../api/imageUpload';
import { 
  saveAnalysisMetadata, 
  type AnalysisMetadata, 
  type ImageAnalysis, 
  saveImageAnalysis,
  fetchAnalysisNames,
  updateAnalysisMetadata
} from '../api/database';
import { useStats } from '../contexts/StatsContext';
import ResultsDisplay from '../components/ResultsDisplay';
import { toast } from 'react-toastify';

interface SelectedFile {
  id: string;
  file: File;
  preview?: string;
}

interface UploadedFile {
  id: string;
  file: File;
  url?: string;
  status: 'uploading' | 'uploaded' | 'analyzing' | 'completed' | 'error';
  analysis?: AnalysisResult['data'];
  error?: string;
  progress?: number;
  fraudRisk?: number;
  claimAmount?: number;
  keyIndicators?: string[];
}

interface BatchAnalysisState {
  isAnalyzing: boolean;
  totalFiles: number;
  completedFiles: number;
  currentFile?: string;
}

const Upload: React.FC = () => {
  const [selectedFiles, setSelectedFiles] = useState<SelectedFile[]>([]);
  const [uploadedFiles, setUploadedFiles] = useState<UploadedFile[]>([]);
  const [isDragOver, setIsDragOver] = useState(false);
  const [batchState, setBatchState] = useState<BatchAnalysisState>({
    isAnalyzing: false,
    totalFiles: 0,
    completedFiles: 0
  });
  const [showResults, setShowResults] = useState(false);
  const [analysisCompleted, setAnalysisCompleted] = useState(false);
  const [analysisTitle, setAnalysisTitle] = useState('');
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [analysisNames, setAnalysisNames] = useState<string[]>([]);
  const { updateStats } = useStats();

  const navigate = useNavigate();

  useEffect(() => {
    const fetchAnalysisNamesAsync = async () => {
      const res = await fetchAnalysisNames();
      if (res && res.success && Array.isArray(res.data)) {
        setAnalysisNames(res.data.map((item: { analysis_name: string }) => item.analysis_name));
        console.log('analysisNames:', res.data);
      }
    };
    fetchAnalysisNamesAsync();
  }, []);

  const handleFileSelect = (files: FileList) => {
    const fileArray = Array.from(files);
    
    // Create preview URLs and add files to selected files
    const newFiles: SelectedFile[] = fileArray.map(file => ({
      id: Math.random().toString(36).substr(2, 9),
      file,
      preview: URL.createObjectURL(file)
    }));
    
    setSelectedFiles(prev => [...prev, ...newFiles]);
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
    // Clear any file input value to allow selecting the same files again
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  const handleFileInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files;
    if (files && files.length > 0) {
      handleFileSelect(files);
    }
    // Clear the input value to allow selecting the same file again
    e.target.value = '';
  };

  const removeFile = (id: string) => {
    setSelectedFiles(prev => {
      const fileToRemove = prev.find(f => f.id === id);
      if (fileToRemove?.preview) {
        URL.revokeObjectURL(fileToRemove.preview);
      }
      return prev.filter(f => f.id !== id);
    });
  };


  const startBatchAnalysis = async () => {
    if (selectedFiles.length === 0) return;
    
    if (!analysisTitle.trim()) {
      alert('Please enter a title for your analysis');
      return;
    }

    if (analysisNames.includes(analysisTitle)) {
      alert('Analysis title already exists');
      return;
    }
    
    await performBatchAnalysis();
  };

  const performBatchAnalysis = async () => {

    // Initialize uploaded files from selected files
    const initialUploadedFiles: UploadedFile[] = selectedFiles.map(file => ({
      id: file.id,
      file: file.file,
      status: 'uploading' as const,
      progress: 0
    }));

    setUploadedFiles(initialUploadedFiles);
    setBatchState({
      isAnalyzing: true,
      totalFiles: selectedFiles.length,
      completedFiles: 0
    });
    setShowResults(false);

    // Scroll to the progress section after state update
    setTimeout(() => {
      const targetDiv = document.getElementById("batch-analysis-progress");
      if (targetDiv) {
        targetDiv.scrollIntoView({ 
          behavior: "smooth",  // smooth scroll
          block: "start"       // aligns to top of viewport
        });
      }
    }, 100);

    // Track completed files during analysis
    const completedAnalysisResults: Array<{
      file: SelectedFile;
      filename: string;
      uploadResult: UploadResult;
      analysisResult: AnalysisResult;
      fraudRisk: number;
      claimAmount: number;
      keyIndicators: string[];
    }> = [];

    // Create initial analysis metadata record
    const initialRes = await saveAnalysisMetadata({
      analysis_name: analysisTitle,
      total_files: selectedFiles.length,
      completed_files: 0,
      fraud_detected_count: 0,
      total_claim_amount: 0,
      file_urls: [],
    });

    if (!initialRes.success) {
      toast.error(initialRes.error);
      throw new Error(initialRes.error);
    }

    for (let i = 0; i < selectedFiles.length; i++) {
      const selectedFile = selectedFiles[i];
      
      setBatchState(prev => ({
        ...prev,
        currentFile: selectedFile.file.name
      }));

      try {
        // Update status to uploading
        setUploadedFiles(prev => 
          prev.map(f => 
            f.id === selectedFile.id 
              ? { ...f, status: 'uploading', progress: 0 }
              : f
          )
        );

        // Upload to Supabase
        const uploadResult: UploadResult = await uploadImage(selectedFile.file);
        
        if (uploadResult.success) {
          // Update status to analyzing
          setUploadedFiles(prev => 
            prev.map(f => 
              f.id === selectedFile.id 
                ? { ...f, status: 'analyzing', url: uploadResult.url, progress: 0 }
                : f
            )
          );

          // Simulate progress updates
          for (let progress = 0; progress <= 100; progress += 20) {
            await new Promise(resolve => setTimeout(resolve, 100));
            setUploadedFiles(prev => 
              prev.map(f => 
                f.id === selectedFile.id 
                  ? { ...f, progress }
                  : f
              )
            );
          }

          // Perform analysis
        const analysisResult = await analyzeImage(uploadResult.url!);
          
          if (analysisResult.success && analysisResult.data) {
            const fraudRisk = Math.round(parseFloat(analysisResult.data.aiScore) * 100);
            const claimAmount = fraudRisk > 50 ? Math.floor(Math.random() * 15000) + 1000 : 0;
            const keyIndicators = analysisResult.data.detectedIssues || [];

            // Store the completed analysis result
            completedAnalysisResults.push({
              file: selectedFile,
              filename: uploadResult.filename || '',
              uploadResult,
              analysisResult,
              fraudRisk,
              claimAmount,
              keyIndicators
            });
        
        setUploadedFiles(prev => 
          prev.map(f => 
                f.id === selectedFile.id 
              ? { 
                  ...f, 
                      status: 'completed',
                  analysis: analysisResult.data,
                      fraudRisk,
                      claimAmount,
                      keyIndicators,
                      progress: 100
                }
              : f
          )
        );

            // Update global stats
            updateStats(analysisResult.data.isFraudulent, claimAmount);
          } else {
            setUploadedFiles(prev => 
              prev.map(f => 
                f.id === selectedFile.id 
                  ? { ...f, status: 'error', error: analysisResult.error }
                  : f
              )
            );
          }
      } else {
        setUploadedFiles(prev => 
          prev.map(f => 
              f.id === selectedFile.id 
              ? { ...f, status: 'error', error: uploadResult.error }
              : f
          )
        );
      }
    } catch (error) {
      setUploadedFiles(prev => 
        prev.map(f => 
            f.id === selectedFile.id 
            ? { ...f, status: 'error', error: 'Upload failed' }
            : f
        )
      );
    }

      setBatchState(prev => ({
        ...prev,
        completedFiles: prev.completedFiles + 1
      }));
    }

    setBatchState(prev => ({
      ...prev,
      isAnalyzing: false,
      currentFile: undefined
    }));
    
    // Save all analysis metadata to database using tracked results
    const imageAnalysis: ImageAnalysis[] = completedAnalysisResults.map(result => ({
      analysis_name: analysisTitle,
      filename: result.uploadResult.filename || '',
      file_size: result.file.file.size,
      file_url: result.uploadResult.url || '',
      fraud_score: parseFloat(result.analysisResult.data?.fraudScore || '0'),
      ai_score: parseFloat(result.analysisResult.data?.aiScore || '0'),
      is_fraudulent: result.analysisResult.data?.isFraudulent || false,
      risk_level: result.analysisResult.data?.riskLevel || 'LOW',
      ai_analysis: result.analysisResult.data?.aiAnalysis || '',
      fraud_analysis: result.analysisResult.data?.fraudAnalysis || '',
      detected_issues: JSON.stringify(result.keyIndicators),
    }));

    console.log('imageAnalysis:', imageAnalysis);

    // Save individual analysis results
    for (const metadata of imageAnalysis) {
      await saveImageAnalysis(metadata);
    }

    // Save batch analysis summary
    const fraudDetectedCount = completedAnalysisResults.filter(result => result.analysisResult.data?.isFraudulent).length;
    const totalClaimAmount = completedAnalysisResults.reduce((sum, result) => sum + result.claimAmount, 0);
    
    const analysisMetadata: AnalysisMetadata = {
      analysis_name: analysisTitle,
      total_files: selectedFiles.length, // Use the original number of selected files
      completed_files: completedAnalysisResults.length, // Use the actual number of completed files
      fraud_detected_count: fraudDetectedCount,
      total_claim_amount: totalClaimAmount,
      file_urls: completedAnalysisResults.map(result => result.uploadResult.url || '') // Include the actual file URLs
    };

    console.log('analysisMetadata:', analysisMetadata);
    
    await updateAnalysisMetadata(analysisMetadata);
    
    // Refetch analysis names after completion
    const res = await fetchAnalysisNames();
    if (res && res.success && Array.isArray(res.data)) {
      setAnalysisNames(res.data.map((item: { analysis_name: string }) => item.analysis_name));
    }
    
    setAnalysisCompleted(true);
    setShowResults(true);
  };

  const clearAnalysisAndStartFresh = () => {
    setSelectedFiles([]);
    setUploadedFiles([]);
    setShowResults(false);
    setAnalysisCompleted(false);
    setAnalysisTitle('');
    setBatchState({
      isAnalyzing: false,
      totalFiles: 0,
      completedFiles: 0
    });
  };


  return (
    <div className="p-6">
      <div className="max-w-full mx-auto">
        {/* Header Section */}
        <div className="mb-8">
          <div className="flex items-center mb-4">
            <button 
              className="mr-4 p-2 hover:bg-gray-100 rounded-lg transition-colors"
              onClick={() => navigate('/dashboard')}
            >
              <FontAwesomeIcon icon={faArrowLeft} className="text-gray-600" />
            </button>
            <div>
              <h1 className="text-3xl font-bold text-gray-900 mb-2">AI Fraud Detection</h1>
          <p className="text-gray-600">
                Upload vehicle damage images for intelligent fraud analysis.
          </p>
            </div>
          </div>
        </div>

        {/* Main Upload Section */}
        <div className="mb-8">
          <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-8">
            {!analysisCompleted && <div className="flex items-center mb-6">
              <FontAwesomeIcon icon={faArrowUpFromBracket} className="text-2xl text-blue-600 mr-3" />
              <h2 className="text-2xl font-semibold text-gray-900">Upload Images</h2>
            </div>}
            
            {!analysisCompleted && selectedFiles.length === 0 ? (
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
                <div className="text-center">
                  <div className="w-20 h-20 bg-blue-100 rounded-lg flex items-center justify-center mx-auto mb-6">
                    <FontAwesomeIcon icon={faArrowUpFromBracket} className="text-3xl text-blue-600" />
                  </div>
                  <h3 className="text-xl font-semibold text-gray-900 mb-3">Select Multiple Images</h3>
                  <p className="text-gray-600 mb-8">
                    Drag & drop your vehicle damage photos here, or click to browse.
                  </p>
                  <button className="bg-blue-600 hover:bg-blue-700 text-white px-6 py-3 rounded-lg flex items-center space-x-2 mx-auto transition-colors">
                    <FontAwesomeIcon icon={faFile} />
                    <span>Browse Files</span>
                  </button>
                  <p className="text-sm text-gray-500 mt-4">
                    Supports: JPEG, PNG, PDF • Max size: 10MB
                  </p>
                </div>
              </div>
            ) : !analysisCompleted ? (
              <div>
                <p className="text-gray-600 mb-6">
                  Select multiple images for batch processing ({selectedFiles.length} files selected)
                </p>
                
                {/* Image Grid */}
                <div className="grid grid-cols-3 md:grid-cols-4 lg:grid-cols-6 gap-3 mb-6">
                  {selectedFiles.map((file) => (
                    <div key={file.id} className="relative bg-gray-50 rounded-lg p-2 border border-gray-200">
                      <div className="aspect-square bg-gray-200 rounded-lg mb-2 overflow-hidden">
                        {file.preview ? (
                          <img 
                            src={file.preview} 
                            alt={file.file.name}
                            className="w-full h-full object-cover"
                          />
                        ) : (
                          <div className="w-full h-full flex items-center justify-center">
                            <FontAwesomeIcon icon={faFileImage} className="text-lg text-gray-400" />
                          </div>
                        )}
                      </div>
                      <p className="text-xs font-medium text-gray-900 truncate" title={file.file.name}>
                        {file.file.name}
                      </p>
                      <p className="text-xs text-gray-500">
                        {(file.file.size / 1024 / 1024).toFixed(1)} MB
                      </p>
                      <button
                        onClick={() => removeFile(file.id)}
                        className="absolute top-1 right-1 w-5 h-5 bg-red-500 text-white rounded-full flex items-center justify-center text-xs hover:bg-red-600"
                      >
                        ×
                      </button>
                    </div>
                  ))}
                  
                  {/* Add More Button */}
                  <div 
                    className="aspect-square border-2 border-dashed border-gray-300 rounded-lg flex flex-col items-center justify-center cursor-pointer hover:border-blue-400 hover:bg-blue-50 transition-colors p-2"
                    onClick={() => fileInputRef.current?.click()}
                  >
                    <FontAwesomeIcon icon={faArrowUpFromBracket} className="text-lg text-gray-400 mb-1" />
                    <span className="text-xs text-gray-500">Add More</span>
                  </div>
                </div>
                
                <div className="text-center mb-6">
                  <p className="text-gray-600">
                    {selectedFiles.length} files selected for batch analysis
                  </p>
                </div>
                
                {/* Analysis Title Input */}
                <div className="mb-6">
                  <label htmlFor="analysisTitle" className="block text-sm font-medium text-gray-700 mb-2">
                    Analysis Title
                  </label>
                  <input
                    id="analysisTitle"
                    type="text"
                    value={analysisTitle}
                    onChange={(e) => setAnalysisTitle(e.target.value)}
                    placeholder="e.g., Vehicle Damage Assessment - Jan 2024"
                    className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                  />
                </div>
                
                <div className="text-center">
                  <button
                    onClick={startBatchAnalysis}
                    disabled={batchState.isAnalyzing}
                    className="bg-blue-600 hover:bg-blue-700 disabled:bg-gray-400 text-white px-8 py-4 rounded-lg flex items-center space-x-3 mx-auto transition-colors"
                  >
                    <FontAwesomeIcon icon={faShieldAlt} />
                    <span>
                      {batchState.isAnalyzing 
                        ? `Analyzing ${batchState.completedFiles}/${batchState.totalFiles} Images...` 
                        : `Analyze ${selectedFiles.length} Images for Fraud`
                      }
                    </span>
                  </button>
                </div>
              </div>
            ) : (
              <div className="text-center py-12">
                <div className="w-20 h-20 bg-green-100 rounded-lg flex items-center justify-center mx-auto mb-6">
                  <FontAwesomeIcon icon={faShieldAlt} className="text-3xl text-green-600" />
                </div>
                <h3 className="text-xl font-semibold text-gray-900 mb-3">Analysis Complete!</h3>
                <p className="text-gray-600 mb-8">
                  Your batch analysis has been completed and saved to the database.
                </p>
                <div className="flex justify-center space-x-4">
                  <button
                    onClick={clearAnalysisAndStartFresh}
                    className="bg-blue-600 hover:bg-blue-700 text-white px-6 py-3 rounded-lg flex items-center space-x-2 transition-colors"
                  >
                    <FontAwesomeIcon icon={faArrowUpFromBracket} />
                    <span>Upload More Pictures</span>
                  </button>
                  <button
                    onClick={() => navigate('/dashboard')}
                    className="bg-gray-500 hover:bg-gray-600 text-white px-6 py-3 rounded-lg transition-colors"
                  >
                    View Dashboard
                  </button>
                </div>
              </div>
            )}
          </div>
            </div>
            
            <input
              ref={fileInputRef}
              type="file"
              multiple
              accept="image/*"
              onChange={handleFileInputChange}
              className="hidden"
            />

        {/* Batch Analysis Progress */}
        {batchState.isAnalyzing && (
          <div id="batch-analysis-progress" className="bg-white rounded-lg shadow-sm border border-gray-200 p-8 mb-8 h-full">
            <div className="flex items-center mb-6">
              <FontAwesomeIcon icon={faShieldAlt} className="text-2xl text-blue-600 mr-3" />
              <h2 className="text-2xl font-semibold text-gray-900">Batch Analysis Progress</h2>
            </div>
            
            <div className="mb-6">
              <div className="flex items-center justify-between mb-2">
                <span className="text-sm text-gray-600">Processing {batchState.totalFiles} images</span>
                <span className="text-sm text-green-600 font-medium">{batchState.completedFiles} completed</span>
          </div>
              
              <div className="w-full bg-gray-200 rounded-full h-3 mb-4">
                <div 
                  className="bg-blue-600 h-3 rounded-full transition-all duration-300"
                  style={{ width: `${(batchState.completedFiles / batchState.totalFiles) * 100}%` }}
                ></div>
        </div>

              <div className="flex justify-between text-sm text-gray-600">
                <span className="font-medium">Overall Progress</span>
                <span>{Math.round((batchState.completedFiles / batchState.totalFiles) * 100)}%</span>
              </div>
            </div>

            {/* Individual File Progress */}
            <div className="space-y-3 max-h-80 overflow-y-auto pb-4">
              {uploadedFiles.map((file) => (
                <div key={file.id} className="bg-green-50 rounded-lg p-4 border border-green-200">
                  <div className="flex items-center space-x-4">
                    <div className="w-12 h-12 bg-gray-200 rounded-lg overflow-hidden flex-shrink-0">
                      {file.url ? (
                        <img src={file.url} alt={file.file.name} className="w-full h-full object-cover" />
                      ) : (
                        <div className="w-full h-full flex items-center justify-center">
                          <FontAwesomeIcon icon={faFileImage} className="text-gray-400" />
                        </div>
                      )}
                      </div>
                    
                    <div className="flex-1 min-w-0">
                      <p className="text-sm font-medium text-gray-900 truncate">{file.file.name}</p>
                      <div className="flex items-center space-x-2 mt-1">
                        {file.status === 'completed' ? (
                          <>
                            <FontAwesomeIcon icon={faCheckCircle} className="text-green-600 text-sm" />
                            <span className="text-sm text-green-600">Completed</span>
                          </>
                        ) : file.status === 'analyzing' ? (
                          <>
                            <FontAwesomeIcon icon={faSpinner} className="text-blue-600 text-sm animate-spin" />
                            <span className="text-sm text-blue-600">Analyzing...</span>
                          </>
                        ) : file.status === 'uploading' ? (
                          <>
                            <FontAwesomeIcon icon={faSpinner} className="text-blue-600 text-sm animate-spin" />
                            <span className="text-sm text-blue-600">Uploading...</span>
                          </>
                        ) : file.status === 'error' ? (
                          <>
                            <FontAwesomeIcon icon={faExclamationTriangle} className="text-red-600 text-sm" />
                            <span className="text-sm text-red-600">Error</span>
                          </>
                        ) : (
                          <>
                            <FontAwesomeIcon icon={faSpinner} className="text-blue-600 text-sm animate-spin" />
                            <span className="text-sm text-blue-600">Processing...</span>
                          </>
                        )}
                      </div>
                    </div>
                    
                    <div className="text-right">
                      <div className="w-20 bg-gray-200 rounded-full h-1 mb-1">
                        <div 
                          className="bg-blue-600 h-1 rounded-full transition-all duration-300"
                          style={{ width: `${file.progress || 0}%` }}
                        ></div>
                      </div>
                      <span className="text-xs text-gray-500">{file.progress || 0}%</span>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Results Display */}
        {showResults && (
          <ResultsDisplay 
            files={uploadedFiles}
          />
        )}
      </div>
    </div>
  );
};

export default Upload;
