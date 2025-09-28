import React, { useState, useRef, useEffect } from 'react';
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
  faEye,
  faFilePdf
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
  keyIndicators?: string[];
  cost?: number;
}

interface BatchAnalysisState {
  isAnalyzing: boolean;
  totalFiles: number;
  completedFiles: number;
  currentFile?: string;
}

/** -------- 문서(PDF) 타입 -------- */
interface SelectedDoc {
  id: string;
  file: File;
}
interface UploadedDoc {
  id: string;
  file: File;
  status: 'uploading' | 'analyzing' | 'completed' | 'error';
  extractedData?: any;     // canonical JSON from /api/parse-claim-pdf
  fraudPrediction?: any;   // first element of /predict response
  error?: string;
  progress?: number;
}

const Upload: React.FC = () => {
  // ===== 좌측(Image) =====
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

  // ===== 우측(Document/PDF) =====
  const [selectedDocs, setSelectedDocs] = useState<SelectedDoc[]>([]);
  const [uploadedDocs, setUploadedDocs] = useState<UploadedDoc[]>([]);
  const [docBatch, setDocBatch] = useState<BatchAnalysisState>({
    isAnalyzing: false,
    totalFiles: 0,
    completedFiles: 0
  });
  const docInputRef = useRef<HTMLInputElement>(null);
  const [docDragOver, setDocDragOver] = useState(false);

  // ===== 공통 =====
  const [analysisNames, setAnalysisNames] = useState<string[]>([]);
  const { updateStats, refreshStats } = useStats();
  const navigate = useNavigate();

  useEffect(() => {
    const fetchAnalysisNamesAsync = async () => {
      const res = await fetchAnalysisNames();
      if (res && res.success && Array.isArray(res.data)) {
        setAnalysisNames(res.data.map((item: { analysis_name: string }) => item.analysis_name));
      }
    };
    fetchAnalysisNamesAsync();
  }, []);

  /** 공통: 파일 → base64 (pdf에서 사용) */
  const convertFileToBase64 = (file: File): Promise<string> =>
    new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.readAsDataURL(file);
      reader.onload = () => resolve((reader.result as string).split(',')[1]); // strip prefix
      reader.onerror = reject;
    });

  /** ----------------- 좌: Images ----------------- */
  const handleFileSelect = (files: FileList) => {
    const fileArray = Array.from(files);
    const newFiles: SelectedFile[] = fileArray.map(file => ({
      id: Math.random().toString(36).substr(2, 9),
      file,
      preview: URL.createObjectURL(file)
    }));
    setSelectedFiles(prev => [...prev, ...newFiles]);
  };

  const handleDragOver = (e: React.DragEvent) => { e.preventDefault(); setIsDragOver(true); };
  const handleDragLeave = (e: React.DragEvent) => { e.preventDefault(); setIsDragOver(false); };
  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault(); setIsDragOver(false);
    const files = e.dataTransfer.files;
    if (files.length > 0) handleFileSelect(files);
    if (fileInputRef.current) fileInputRef.current.value = '';
  };
  const handleFileInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files;
    if (files && files.length > 0) handleFileSelect(files);
    e.target.value = '';
  };
  const removeFile = (id: string) => {
    setSelectedFiles(prev => {
      const f = prev.find(x => x.id === id);
      if (f?.preview) URL.revokeObjectURL(f.preview);
      return prev.filter(x => x.id !== id);
    });
  };

  const startBatchAnalysis = async () => {
    if (selectedFiles.length === 0) return;
    if (!analysisTitle.trim()) return alert('Please enter a title for your analysis');
    if (analysisNames.includes(analysisTitle)) return alert('Analysis title already exists');
    await performBatchAnalysis();
  };

  const performBatchAnalysis = async () => {
    const initialUploadedFiles: UploadedFile[] = selectedFiles.map(file => ({
      id: file.id,
      file: file.file,
      status: 'uploading',
      progress: 0
    }));
    setUploadedFiles(initialUploadedFiles);
    setBatchState({ isAnalyzing: true, totalFiles: selectedFiles.length, completedFiles: 0 });
    setShowResults(false);

    setTimeout(() => {
      const targetDiv = document.getElementById('batch-analysis-progress');
      if (targetDiv) targetDiv.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }, 100);

    const completedAnalysisResults: Array<{
      file: SelectedFile;
      filename: string;
      uploadResult: UploadResult;
      analysisResult: AnalysisResult;
      fraudRisk: number;
      keyIndicators: string[];
    }> = [];

    const initialRes = await saveAnalysisMetadata({
      analysis_name: analysisTitle,
      total_files: selectedFiles.length,
      completed_files: 0,
      fraud_detected_count: 0,
      total_claim_amount: 0,
      total_cost: 0,
      file_urls: []
    });
    if (!initialRes.success) { toast.error(initialRes.error); throw new Error(initialRes.error); }

    for (let i = 0; i < selectedFiles.length; i++) {
      const selectedFile = selectedFiles[i];
      setBatchState(prev => ({ ...prev, currentFile: selectedFile.file.name }));

      try {
        setUploadedFiles(prev => prev.map(f => f.id === selectedFile.id ? { ...f, status: 'uploading', progress: 0 } : f));
        const uploadResult = await uploadImage(selectedFile.file);

        if (uploadResult.success) {
          setUploadedFiles(prev => prev.map(f => f.id === selectedFile.id ? { ...f, status: 'analyzing', url: uploadResult.url, progress: 0 } : f));

          for (let p = 0; p <= 100; p += 20) {
            await new Promise(r => setTimeout(r, 100));
            setUploadedFiles(prev => prev.map(f => f.id === selectedFile.id ? { ...f, progress: p } : f));
          }

          const analysisResult = await analyzeImage(uploadResult.url!);
          if (analysisResult.success && analysisResult.data) {
            const fraudRisk = Math.round(parseFloat(analysisResult.data.aiScore) * 100);
            const keyIndicators = analysisResult.data.detectedIssues || [];

            completedAnalysisResults.push({
              file: selectedFile,
              filename: uploadResult.filename || '',
              uploadResult,
              analysisResult,
              fraudRisk,
              keyIndicators
            });

            setUploadedFiles(prev => prev.map(f =>
              f.id === selectedFile.id
                ? { ...f, status: 'completed', analysis: analysisResult.data, fraudRisk, keyIndicators, progress: 100 }
                : f
            ));

            // 전역 통계(이미지 파이프라인 고유 비용/스코어에 맞게 저장)
            updateStats(analysisResult.data.isFraudulent, analysisResult.data.totalCost);
          } else {
            setUploadedFiles(prev => prev.map(f => f.id === selectedFile.id ? { ...f, status: 'error', error: analysisResult.error } : f));
          }
        } else {
          setUploadedFiles(prev => prev.map(f => f.id === selectedFile.id ? { ...f, status: 'error', error: uploadResult.error } : f));
        }
      } catch {
        setUploadedFiles(prev => prev.map(f => f.id === selectedFile.id ? { ...f, status: 'error', error: 'Upload failed' } : f));
      }

      setBatchState(prev => ({ ...prev, completedFiles: prev.completedFiles + 1 }));
    }

    setBatchState(prev => ({ ...prev, isAnalyzing: false, currentFile: undefined }));

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
      cost: Math.round((Math.random() * 9000 + 1000)) // demo 비용
    }));

    for (const metadata of imageAnalysis) await saveImageAnalysis(metadata);

    setUploadedFiles(prev => prev.map(file => {
      const found = imageAnalysis.find(a => a.filename === (file.url?.split('/').pop() || file.file.name));
      return found ? { ...file, cost: found.cost } : file;
    }));

    const fraudDetectedCount = completedAnalysisResults.filter(r => r.analysisResult.data?.isFraudulent).length;
    const totalClaimAmount = 0;
    const totalCost = imageAnalysis.filter(x => x.is_fraudulent).reduce((s, x) => s + (x.cost || 0), 0);

    const analysisMetadata: AnalysisMetadata = {
      analysis_name: analysisTitle,
      total_files: selectedFiles.length,
      completed_files: completedAnalysisResults.length,
      fraud_detected_count: fraudDetectedCount,
      total_claim_amount: totalClaimAmount,
      total_cost: Math.round(totalCost * 100) / 100,
      file_urls: completedAnalysisResults.map(r => r.uploadResult.url || '')
    };
    await updateAnalysisMetadata(analysisMetadata);

    const res = await fetchAnalysisNames();
    if (res && res.success && Array.isArray(res.data)) {
      setAnalysisNames(res.data.map((item: { analysis_name: string }) => item.analysis_name));
    }

    for (const r of completedAnalysisResults) {
      const found = imageAnalysis.find(a => a.filename === (r.uploadResult.filename || r.file.file.name));
      updateStats(!!r.analysisResult.data?.isFraudulent, found?.cost || 0);
    }
    refreshStats();

    setAnalysisCompleted(true);
    setShowResults(true);
  };

  /** ----------------- 우: Documents (PDF) ----------------- */
  const handleDocSelect = (files: FileList) => {
    const pdfs = Array.from(files).filter(f => f.type === 'application/pdf');
    const newDocs: SelectedDoc[] = pdfs.map(file => ({ id: Math.random().toString(36).substr(2, 9), file }));
    setSelectedDocs(prev => [...prev, ...newDocs]);
  };
  const onDocDragOver = (e: React.DragEvent) => { e.preventDefault(); setDocDragOver(true); };
  const onDocDragLeave = (e: React.DragEvent) => { e.preventDefault(); setDocDragOver(false); };
  const onDocDrop = (e: React.DragEvent) => {
    e.preventDefault(); setDocDragOver(false);
    if (e.dataTransfer.files?.length) handleDocSelect(e.dataTransfer.files);
    if (docInputRef.current) docInputRef.current.value = '';
  };
  const onDocInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files; if (files?.length) handleDocSelect(files);
    e.target.value = '';
  };
  const removeDoc = (id: string) => setSelectedDocs(prev => prev.filter(d => d.id !== id));

  /** PDF 한 건 분석: parse → predict */
  const analyzeOneDocument = async (file: File) => {
    const base64 = await convertFileToBase64(file);

    const parseRes = await fetch('http://localhost:8000/api/parse-claim-pdf', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        pdf: base64,
        extraPrompt: 'Focus on extracting all available claim information accurately.'
      })
    });
    if (!parseRes.ok) throw new Error(`PDF parsing failed: ${parseRes.status}`);
    const parsed = await parseRes.json();

    const predictRes = await fetch('http://localhost:8000/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(parsed.canonical)
    });
    if (!predictRes.ok) throw new Error(`Fraud prediction failed: ${predictRes.status}`);
    const predicted = await predictRes.json();

    return { extractedData: parsed.canonical, fraudPrediction: predicted.data[0] };
  };

  const startDocBatch = async () => {
    if (selectedDocs.length === 0) return;
    const init: UploadedDoc[] = selectedDocs.map(d => ({ id: d.id, file: d.file, status: 'uploading', progress: 0 }));
    setUploadedDocs(init);
    setDocBatch({ isAnalyzing: true, totalFiles: selectedDocs.length, completedFiles: 0 });

    for (let i = 0; i < selectedDocs.length; i++) {
      const doc = selectedDocs[i];
      setDocBatch(prev => ({ ...prev, currentFile: doc.file.name }));

      try {
        // 가벼운 업로드/진행도 UI
        for (let p = 0; p <= 40; p += 10) {
          await new Promise(r => setTimeout(r, 80));
          setUploadedDocs(prev => prev.map(u => u.id === doc.id ? { ...u, status: 'uploading', progress: p } : u));
        }

        setUploadedDocs(prev => prev.map(u => u.id === doc.id ? { ...u, status: 'analyzing', progress: 60 } : u));
        const result = await analyzeOneDocument(doc.file);

        setUploadedDocs(prev => prev.map(u => u.id === doc.id
          ? { ...u, status: 'completed', progress: 100, extractedData: result.extractedData, fraudPrediction: result.fraudPrediction }
          : u
        ));

        // 전역 통계 반영 (문서 모델은 ClaimAmount가 들어옴)
        const isFraud = result.fraudPrediction?.decision === 1;
        const amount = Number(result.extractedData?.ClaimAmount || 0);
        updateStats(isFraud, isFraud ? amount : 0);
      } catch (err: any) {
        setUploadedDocs(prev => prev.map(u => u.id === doc.id
          ? { ...u, status: 'error', error: err?.message || 'Document analysis failed' }
          : u
        ));
      }

      setDocBatch(prev => ({ ...prev, completedFiles: prev.completedFiles + 1 }));
    }

    setDocBatch(prev => ({ ...prev, isAnalyzing: false, currentFile: undefined }));
  };

  // 도큐먼트 리스크 뱃지
  const riskLevelFromProba = (p: number) => (p > 0.7 ? 'HIGH' : p > 0.4 ? 'MEDIUM' : 'LOW');
  const levelColor = (lvl: string) =>
    lvl === 'HIGH' ? 'text-red-600 bg-red-50'
      : lvl === 'MEDIUM' ? 'text-yellow-600 bg-yellow-50'
      : 'text-green-600 bg-green-50';

  /** ----------------- RENDER ----------------- */
  return (
    <div className="p-6">
      <div className="max-w-full mx-auto">
        {/* Header */}
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
              <p className="text-gray-600">Upload images and PDF claim documents for intelligent fraud analysis.</p>
            </div>
          </div>
        </div>

        {/* Two Columns: Left Images | Right Documents */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
          {/* Left: Images */}
          <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-8">
            {!analysisCompleted && (
              <div className="flex items-center mb-6">
                <FontAwesomeIcon icon={faArrowUpFromBracket} className="text-2xl text-blue-600 mr-3" />
                <h2 className="text-2xl font-semibold text-gray-900">Upload Images</h2>
              </div>
            )}

            {!analysisCompleted && selectedFiles.length === 0 ? (
              <div
                className={`border-2 border-dashed rounded-lg p-12 transition-colors duration-200 cursor-pointer ${
                  isDragOver ? 'border-blue-400 bg-blue-50' : 'border-gray-300 hover:border-blue-400 hover:bg-blue-100'
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
                  <p className="text-gray-600 mb-8">Drag & drop vehicle damage photos here, or click to browse.</p>
                  <button className="bg-blue-600 hover:bg-blue-700 text-white px-6 py-3 rounded-lg flex items-center space-x-2 mx-auto transition-colors">
                    <FontAwesomeIcon icon={faFile} />
                    <span>Browse Files</span>
                  </button>
                  <p className="text-sm text-gray-500 mt-4">Supports: JPEG, PNG • Max size: 10MB</p>
                </div>
              </div>
            ) : !analysisCompleted ? (
              <>
                <p className="text-gray-600 mb-6">
                  Select multiple images for batch processing ({selectedFiles.length} files selected)
                </p>

                {/* Image Grid */}
                <div className="grid grid-cols-3 md:grid-cols-4 gap-3 mb-6">
                  {selectedFiles.map((file) => (
                    <div key={file.id} className="relative bg-gray-50 rounded-lg p-2 border border-gray-200">
                      <div className="aspect-square bg-gray-200 rounded-lg mb-2 overflow-hidden">
                        {file.preview ? (
                          <img src={file.preview} alt={file.file.name} className="w-full h-full object-cover" />
                        ) : (
                          <div className="w-full h-full flex items-center justify-center">
                            <FontAwesomeIcon icon={faFileImage} className="text-lg text-gray-400" />
                          </div>
                        )}
                      </div>
                      <p className="text-xs font-medium text-gray-900 truncate" title={file.file.name}>
                        {file.file.name}
                      </p>
                      <p className="text-xs text-gray-500">{(file.file.size / 1024 / 1024).toFixed(1)} MB</p>
                      <button
                        onClick={() => removeFile(file.id)}
                        className="absolute top-1 right-1 w-5 h-5 bg-red-500 text-white rounded-full flex items-center justify-center text-xs hover:bg-red-600"
                      >
                        ×
                      </button>
                    </div>
                  ))}
                </div>

                {/* Analysis Title */}
                <div className="mb-6">
                  <label htmlFor="analysisTitle" className="block text-sm font-medium text-gray-700 mb-2">
                    Analysis Title
                  </label>
                  <input
                    id="analysisTitle"
                    type="text"
                    value={analysisTitle}
                    onChange={(e) => setAnalysisTitle(e.target.value)}
                    placeholder="e.g., Vehicle Damage Assessment - Sep 2025"
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
                        : `Analyze ${selectedFiles.length} Images for Fraud`}
                    </span>
                  </button>
                </div>
              </>
            ) : (
              <div className="text-center py-8">
                <div className="w-16 h-16 bg-green-100 rounded-lg flex items-center justify-center mx-auto mb-4">
                  <FontAwesomeIcon icon={faShieldAlt} className="text-2xl text-green-600" />
                </div>
                <h3 className="text-lg font-semibold text-gray-900 mb-2">Image Analysis Complete!</h3>
                <p className="text-gray-600 mb-4">Results saved. You can upload more or view your dashboard.</p>
                <div className="flex justify-center space-x-4">
                  <button
                    onClick={() => {
                      setSelectedFiles([]);
                      setUploadedFiles([]);
                      setShowResults(false);
                      setAnalysisCompleted(false);
                      setAnalysisTitle('');
                      setBatchState({ isAnalyzing: false, totalFiles: 0, completedFiles: 0 });
                    }}
                    className="bg-blue-600 hover:bg-blue-700 text-white px-5 py-2 rounded-lg transition-colors"
                  >
                    Upload More Images
                  </button>
                  <button
                    onClick={() => navigate('/dashboard')}
                    className="bg-gray-500 hover:bg-gray-600 text-white px-5 py-2 rounded-lg transition-colors"
                  >
                    View Dashboard
                  </button>
                </div>
              </div>
            )}

            {/* Hidden input for images */}
            <input
              ref={fileInputRef}
              type="file"
              multiple
              accept="image/*"
              onChange={handleFileInputChange}
              className="hidden"
            />
          </div>

          {/* Right: Documents (PDF) */}
          <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-8">
            <div className="flex items-center mb-6">
              <FontAwesomeIcon icon={faFilePdf} className="text-2xl text-purple-600 mr-3" />
              <h2 className="text-2xl font-semibold text-gray-900">Upload PDF Claim Documents</h2>
            </div>

            {selectedDocs.length === 0 ? (
              <div
                className={`border-2 border-dashed rounded-lg p-12 transition-colors duration-200 cursor-pointer ${
                  docDragOver ? 'border-purple-400 bg-purple-50' : 'border-gray-300 hover:border-purple-400 hover:bg-purple-50/40'
                }`}
                onDragOver={onDocDragOver}
                onDragLeave={onDocDragLeave}
                onDrop={onDocDrop}
                onClick={() => docInputRef.current?.click()}
              >
                <div className="text-center">
                  <div className="w-20 h-20 bg-purple-100 rounded-lg flex items-center justify-center mx-auto mb-6">
                    <FontAwesomeIcon icon={faFilePdf} className="text-3xl text-purple-600" />
                  </div>
                  <h3 className="text-xl font-semibold text-gray-900 mb-3">Drop PDF claim reports</h3>
                  <p className="text-gray-600 mb-8">Or click below to select one or more PDF files.</p>
                  <button className="bg-purple-600 hover:bg-purple-700 text-white px-6 py-3 rounded-lg flex items-center space-x-2 mx-auto transition-colors">
                    <FontAwesomeIcon icon={faFile} />
                    <span>Browse PDFs</span>
                  </button>
                  <p className="text-sm text-gray-500 mt-4">Supports: PDF • Max size: 50MB</p>
                </div>
              </div>
            ) : (
              <>
                <p className="text-gray-600 mb-6">
                  Selected {selectedDocs.length} PDF{selectedDocs.length > 1 ? 's' : ''} for parsing & fraud scoring
                </p>

                <div className="space-y-3 mb-6 max-h-60 overflow-auto pr-1">
                  {selectedDocs.map((doc) => (
                    <div key={doc.id} className="flex items-center justify-between p-3 bg-gray-50 rounded-lg border">
                      <div className="flex items-center space-x-3">
                        <FontAwesomeIcon icon={faFilePdf} className="text-purple-600" />
                        <div>
                          <p className="font-medium text-gray-900">{doc.file.name}</p>
                          <p className="text-sm text-gray-500">{(doc.file.size / 1024 / 1024).toFixed(2)} MB</p>
                        </div>
                      </div>
                      <button
                        onClick={() => removeDoc(doc.id)}
                        className="text-gray-400 hover:text-red-500"
                      >
                        <FontAwesomeIcon icon={faExclamationTriangle} />
                      </button>
                    </div>
                  ))}
                </div>

                <div className="text-center">
                  <button
                    onClick={startDocBatch}
                    disabled={docBatch.isAnalyzing}
                    className="bg-purple-600 hover:bg-purple-700 disabled:bg-gray-400 text-white px-8 py-4 rounded-lg flex items-center space-x-3 mx-auto transition-colors"
                  >
                    <FontAwesomeIcon icon={faShieldAlt} />
                    <span>
                      {docBatch.isAnalyzing
                        ? `Analyzing ${docBatch.completedFiles}/${docBatch.totalFiles} PDFs...`
                        : `Analyze ${selectedDocs.length} PDF${selectedDocs.length > 1 ? 's' : ''}`}
                    </span>
                  </button>
                </div>
              </>
            )}

            {/* Hidden input for PDFs */}
            <input
              ref={docInputRef}
              type="file"
              multiple
              accept="application/pdf"
              onChange={onDocInputChange}
              className="hidden"
            />
          </div>
        </div>

        {/* Progress & Results Rows */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 mt-8">
          {/* Images Progress */}
          {batchState.isAnalyzing && (
            <div id="batch-analysis-progress" className="bg-white rounded-lg shadow-sm border border-gray-200 p-8">
              <div className="flex items-center mb-6">
                <FontAwesomeIcon icon={faShieldAlt} className="text-2xl text-blue-600 mr-3" />
                <h2 className="text-2xl font-semibold text-gray-900">Image Batch Progress</h2>
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
                  />
                </div>
                <div className="flex justify-between text-sm text-gray-600">
                  <span className="font-medium">Overall Progress</span>
                  <span>{Math.round((batchState.completedFiles / batchState.totalFiles) * 100)}%</span>
                </div>
              </div>

              <div className="space-y-3 max-h-80 overflow-y-auto pb-1">
                {uploadedFiles.map((file) => (
                  <div key={file.id} className="bg-blue-50 rounded-lg p-4 border border-blue-200">
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
                          />
                        </div>
                        <span className="text-xs text-gray-500">{file.progress || 0}%</span>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Documents Progress & Results */}
          {(docBatch.isAnalyzing || uploadedDocs.length > 0) && (
            <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-8">
              <div className="flex items-center mb-6">
                <FontAwesomeIcon icon={faShieldAlt} className="text-2xl text-purple-600 mr-3" />
                <h2 className="text-2xl font-semibold text-gray-900">PDF Batch Progress</h2>
              </div>

              {docBatch.isAnalyzing && (
                <div className="mb-6">
                  <div className="flex items-center justify-between mb-2">
                    <span className="text-sm text-gray-600">Processing {docBatch.totalFiles} PDFs</span>
                    <span className="text-sm text-green-600 font-medium">{docBatch.completedFiles} completed</span>
                  </div>
                  <div className="w-full bg-gray-200 rounded-full h-3 mb-4">
                    <div
                      className="bg-purple-600 h-3 rounded-full transition-all duration-300"
                      style={{ width: `${(docBatch.completedFiles / docBatch.totalFiles) * 100}%` }}
                    />
                  </div>
                  <div className="flex justify-between text-sm text-gray-600">
                    <span className="font-medium">Overall Progress</span>
                    <span>{Math.round((docBatch.completedFiles / docBatch.totalFiles) * 100)}%</span>
                  </div>
                </div>
              )}

              <div className="space-y-3 max-h-80 overflow-y-auto pb-1">
                {uploadedDocs.map((u) => {
                  const proba = Number(u.fraudPrediction?.proba ?? 0);
                  const lvl = riskLevelFromProba(proba);
                  return (
                    <div key={u.id} className="bg-purple-50 rounded-lg p-4 border border-purple-200">
                      <div className="flex items-center justify-between">
                        <div className="flex items-center space-x-3">
                          {u.status === 'completed' ? (
                            <FontAwesomeIcon icon={faCheckCircle} className="text-green-600" />
                          ) : u.status === 'error' ? (
                            <FontAwesomeIcon icon={faExclamationTriangle} className="text-red-600" />
                          ) : (
                            <FontAwesomeIcon icon={faSpinner} className="text-purple-600 animate-spin" />
                          )}
                          <div>
                            <p className="text-sm font-medium text-gray-900">{u.file.name}</p>
                            <p className="text-xs text-gray-500 capitalize">{u.status}</p>
                          </div>
                        </div>
                        {u.status === 'completed' && (
                          <span className={`px-2 py-1 rounded-full text-xs font-medium ${levelColor(lvl)}`}>
                            {lvl} RISK
                          </span>
                        )}
                      </div>

                      {u.status === 'error' && (
                        <div className="bg-red-50 border border-red-200 rounded-lg p-3 mt-3">
                          <p className="text-sm text-red-700">{u.error}</p>
                        </div>
                      )}

                      {u.status === 'completed' && u.extractedData && u.fraudPrediction && (
                        <div className="grid grid-cols-2 gap-4 text-sm mt-3">
                          <div>
                            <span className="font-medium text-gray-700">Claim Amount:</span>
                            <span className="ml-2 text-gray-900">
                              ${Number(u.extractedData.ClaimAmount || 0).toLocaleString()}
                            </span>
                          </div>
                          <div>
                            <span className="font-medium text-gray-700">Vehicle:</span>
                            <span className="ml-2 text-gray-900">
                              {u.extractedData.Make} {u.extractedData.Year}
                            </span>
                          </div>
                          <div>
                            <span className="font-medium text-gray-700">Fraud Probability:</span>
                            <span className="ml-2 text-gray-900">{(proba * 100).toFixed(2)}%</span>
                          </div>
                          <div>
                            <span className="font-medium text-gray-700">Decision:</span>
                            <span className={`ml-2 font-medium ${u.fraudPrediction.decision === 1 ? 'text-red-600' : 'text-green-600'}`}>
                              {u.fraudPrediction.decision === 1 ? 'FRAUD' : 'LEGITIMATE'}
                            </span>
                          </div>
                        </div>
                      )}

                      {u.progress !== undefined && u.status !== 'completed' && (
                        <div className="w-full bg-gray-200 rounded-full h-2 mt-3">
                          <div
                            className="bg-purple-600 h-2 rounded-full transition-all duration-300"
                            style={{ width: `${u.progress}%` }}
                          />
                        </div>
                      )}
                    </div>
                  );
                })}
              </div>
            </div>
          )}
        </div>

        {/* Combined Results CTA (for images) */}
        {showResults && (
          <div className="mt-8">
            <ResultsDisplay
              files={uploadedFiles}
              actionButton={
                analysisCompleted ? (
                  <button
                    onClick={() => navigate(`/cases/${encodeURIComponent(analysisTitle)}`)}
                    className="inline-flex items-center px-6 py-3 bg-blue-600 text-white font-medium rounded-lg hover:bg-blue-700 transition-colors"
                  >
                    <FontAwesomeIcon icon={faEye} className="mr-2" />
                    View Case Review
                  </button>
                ) : undefined
              }
            />
          </div>
        )}
      </div>
    </div>
  );
};

export default Upload;
