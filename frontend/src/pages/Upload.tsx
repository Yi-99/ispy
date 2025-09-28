import React, { useEffect, useRef, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome';
import { 
  faArrowUpFromBracket,
  faCheckCircle, 
  faDollarSign,
  faExclamationTriangle,
  faFile,
  faFileAlt,
  faFileImage,
  faFilePdf,
  faShieldAlt,
  faSpinner,
} from '@fortawesome/free-solid-svg-icons';
import { uploadImage, analyzeImage, type UploadResult, type AnalysisResult } from '../api/imageUpload';
import { 
  fetchAnalysisNames,
  saveAnalysisMetadata, 
  saveImageAnalysis,
  updateAnalysisMetadata,
  type AnalysisMetadata,
  type ImageAnalysis,
} from '../api/database';
import { useStats } from '../contexts/StatsContext';
import ResultsDisplay from '../components/ResultsDisplay';
import { toast } from 'react-toastify';

/* =========================
  Types
========================= */
type MediaKind = 'image' | 'pdf';

interface SelectedItem {
  id: string;
  file: File;
  kind: MediaKind;
  preview?: string; // images only
}

type ItemStatus = 'queued' | 'uploading' | 'analyzing' | 'completed' | 'error';

interface ProcessedItemBase {
  id: string;
  file: File;
  kind: MediaKind;
  status: ItemStatus;
  progress?: number;
  error?: string;
}

interface ProcessedImageItem extends ProcessedItemBase {
  kind: 'image';
  url?: string;
  analysis?: AnalysisResult['data'];
  fraudRisk?: number; // 0~100
  keyIndicators?: string[];
  cost?: number;
}

interface ProcessedPdfItem extends ProcessedItemBase {
  kind: 'pdf';
  extractedData?: any;            // canonical JSON from /api/parse-claim-pdf
  fraudPrediction?: { proba: number; decision: 0 | 1; threshold?: number; [k: string]: any };
}

type ProcessedItem = ProcessedImageItem | ProcessedPdfItem;

// Interface for ResultsDisplay component
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

interface BatchState {
  isAnalyzing: boolean;
  totalFiles: number;
  completedFiles: number;
  currentFile?: string;
}

/* =========================
  Helpers
========================= */
const toId = () => Math.random().toString(36).slice(2, 11);


const bytesToMB = (b: number) => (b / 1024 / 1024).toFixed(2);

const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));

/** file -> base64 (strip data: prefix) */
const toBase64 = (file: File) =>
  new Promise<string>((resolve, reject) => {
    const r = new FileReader();
    r.readAsDataURL(file);
    r.onload = () => resolve(String(r.result).split(',')[1] ?? '');
    r.onerror = reject;
  });

/* =========================
  Component
========================= */
const Upload: React.FC = () => {
  const navigate = useNavigate();
  const { updateStats, refreshStats } = useStats();

  // Files selected by user (images + pdf together)
  const [selected, setSelected] = useState<SelectedItem[]>([]);
  const [items, setItems] = useState<ProcessedItem[]>([]);
  const [analysisTitle, setAnalysisTitle] = useState('');
  const [names, setNames] = useState<string[]>([]);
  const [batch, setBatch] = useState<BatchState>({ isAnalyzing: false, totalFiles: 0, completedFiles: 0 });
  const [dragOver, setDragOver] = useState(false);
  const [isAnalyzed, setIsAnalyzed] = useState(false);
  const [comprehensiveResult, setComprehensiveResult] = useState<any>(null);

  const inputRef = useRef<HTMLInputElement>(null);

  // Conversion function to map ProcessedItem to UploadedFile for ResultsDisplay
  const convertToUploadedFiles = (items: ProcessedItem[]): UploadedFile[] => {
    return items
      .filter(item => item.kind === 'image' && item.status === 'completed')
      .map(item => {
        const img = item as ProcessedImageItem;
        return {
          id: img.id,
          file: img.file,
          url: img.url,
          status: img.status as 'uploading' | 'uploaded' | 'analyzing' | 'completed' | 'error',
          analysis: img.analysis,
          progress: img.progress,
          fraudRisk: img.fraudRisk,
          keyIndicators: img.keyIndicators,
          cost: img.cost,
        };
      });
  };

  // Load existing analysis names (to avoid duplicates)
  useEffect(() => {
    (async () => {
      const res = await fetchAnalysisNames();
      if (res?.success && Array.isArray(res.data)) {
        setNames(res.data.map((d: { analysis_name: string }) => d.analysis_name));
      }
    })();
  }, []);


  /* ------------- Select & Remove ------------- */
  const onPickClick = () => {
    inputRef.current?.click();
  };

  const onInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files;
    if (!files || files.length === 0) return;
    ingestFiles(files);
    e.currentTarget.value = '';
  };

  const onDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    setDragOver(true);
  };
  const onDragLeave = (e: React.DragEvent) => {
    e.preventDefault();
    setDragOver(false);
  };
  const onDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setDragOver(false);
    if (e.dataTransfer.files?.length) ingestFiles(e.dataTransfer.files);
  };

  const ingestFiles = (fileList: FileList) => {
    const arr = Array.from(fileList).filter((f) =>
      ['image/', 'application/pdf'].some((p) => f.type.startsWith(p))
    );
    const mapped: SelectedItem[] = arr.map((f) => ({
      id: toId(),
      file: f,
      kind: f.type.startsWith('image/') ? 'image' : 'pdf',
      preview: f.type.startsWith('image/') ? URL.createObjectURL(f) : undefined,
    }));
    setSelected((prev) => [...prev, ...mapped]);
  };

  const remove = (id: string) => {
    setSelected((prev) => {
      const found = prev.find((x) => x.id === id);
      if (found?.preview) URL.revokeObjectURL(found.preview);
      return prev.filter((x) => x.id !== id);
    });
    setItems((prev) => prev.filter((x) => x.id !== id));
  };

  const resetPage = () => {
    // Clear all selected files and revoke object URLs
    selected.forEach(item => {
      if (item.preview) URL.revokeObjectURL(item.preview);
    });
    setSelected([]);
    setItems([]);
    setAnalysisTitle('');
    setBatch({ isAnalyzing: false, totalFiles: 0, completedFiles: 0 });
    setIsAnalyzed(false);
  };

  /* ------------- Analyze Button ------------- */
  const onAnalyze = async () => {
    if (selected.length === 0) {
      toast.info('Please add images or PDFs to analyze.');
      return;
    }
    if (!analysisTitle.trim()) {
      toast.info('Please enter an Claim Title.');
      return;
    }
    if (names.includes(analysisTitle)) {
      toast.error('Analysis title already exists. Choose another title.');
      return;
    }
    
    // Prepare UI state
    setIsAnalyzed(true);
    
    // If this is a new analysis (not adding to existing), reset items
    // If this is adding to existing analysis, append new items
    const newItems = selected.map<ProcessedItem>((s) =>
      s.kind === 'image'
        ? { id: s.id, file: s.file, kind: 'image', status: 'uploading', progress: 0 }
        : { id: s.id, file: s.file, kind: 'pdf', status: 'uploading', progress: 0 }
    );
    
    if (items.length === 0) {
      // New analysis
      setItems(newItems);
      setBatch({ isAnalyzing: true, totalFiles: selected.length, completedFiles: 0, currentFile: undefined });
    } else {
      // Adding to existing analysis
      setItems(prev => [...prev, ...newItems]);
      setBatch(prev => ({ 
        isAnalyzing: true, 
        totalFiles: prev.totalFiles + selected.length, 
        completedFiles: prev.completedFiles, 
        currentFile: undefined 
      }));
    }

    // Create analysis metadata shell (only for new analysis)
    let metaInit;
    if (items.length === 0) {
      // New analysis - create metadata
      metaInit = await saveAnalysisMetadata({
        analysis_name: analysisTitle,
        total_files: selected.length,
        completed_files: 0,
        fraud_detected_count: 0,
        total_claim_amount: 0,
        total_cost: 0,
        file_urls: [],
      } as AnalysisMetadata);
      if (!metaInit.success) {
        toast.error(metaInit.error || 'Failed to create analysis record');
        setBatch((b) => ({ ...b, isAnalyzing: false }));
        return;
      }
    }

    // Process sequentially (simple UI) - only process newly selected files
    const resultsForDB: ImageAnalysis[] = [];
    let fraudDetectedCount = 0;  // count both images+pdf
    let totalCost = 0;           // image fraud "cost" (demo) + pdf ClaimAmount for frauds
    let totalClaimAmount = 0;    // pdf sum of claim amounts (for info)
    let completed = 0;
    
    // 통합 분석을 위한 데이터 수집
    const completedImageAnalysis: any[] = [];
    const completedDocumentAnalysis: any[] = [];

    // Only process the newly selected files
    for (const s of selected) {
      setBatch((b) => ({ ...b, currentFile: s.file.name }));

      if (s.kind === 'image') {
        // === Image pipeline ===
        try {
          // uploading UI
          for (let p = 0; p <= 40; p += 10) {
            await sleep(60);
            setItems((prev) => prev.map((it) => (it.id === s.id ? { ...it, progress: p } : it)));
          }

          const up: UploadResult = await uploadImage(s.file);
          if (!up.success) throw new Error(up.error || 'Upload failed');

          setItems((prev) =>
            prev.map((it) => (it.id === s.id ? { ...it, status: 'analyzing', progress: 60, ...(up.url ? { url: up.url } : {}) } : it))
          );

          const ar: AnalysisResult = await analyzeImage(up.url!);
          if (!ar.success || !ar.data) throw new Error(ar.error || 'Analyze failed');

          const fraudRisk = Math.round(parseFloat(ar.data.aiScore) * 100);
          const keyIndicators = ar.data.detectedIssues || [];
          const demoCost = Math.round(Math.random() * 9000 + 1000); // demo

          // Update item
          setItems((prev) =>
            prev.map((it) =>
              it.id === s.id
                ? {
                    ...(it as ProcessedImageItem),
                    status: 'completed',
                    progress: 100,
                    analysis: ar.data,
                    fraudRisk,
                    keyIndicators,
                    url: up.url!,
                    kind: 'image',
                    cost: demoCost,
                  }
                : it
            )
          );

          // Stats + DB rows (store only image rows in imageAnalysis table, as before)
          const row: ImageAnalysis = {
            analysis_name: analysisTitle,
            filename: up.filename || s.file.name,
            file_size: s.file.size,
            file_url: up.url || '',
            fraud_score: parseFloat(ar.data?.fraudScore || '0'),
            ai_score: parseFloat(ar.data?.aiScore || '0'),
            is_fraudulent: !!ar.data?.isFraudulent,
            risk_level: ar.data?.riskLevel || 'LOW',
            ai_analysis: ar.data?.aiAnalysis || '',
            fraud_analysis: ar.data?.fraudAnalysis || '',
            detected_issues: JSON.stringify(keyIndicators),
            cost: demoCost,
          };
          resultsForDB.push(row);

          // Update aggregations
          if (row.is_fraudulent) {
            fraudDetectedCount += 1;
            totalCost += row.cost || 0;
          }
          updateStats(row.is_fraudulent, row.is_fraudulent ? (row.cost || 0) : 0);
          
          // 통합 분석을 위한 데이터 수집
          completedImageAnalysis.push({
            analysis: ar.data,
            fraudRisk: fraudRisk
          });
        } catch (err: any) {
          setItems((prev) =>
            prev.map((it) => (it.id === s.id ? { ...it, status: 'error', error: err?.message || 'Image analysis failed' } : it))
          );
        }
      } else {
        // === PDF pipeline ===
        try {
          for (let p = 0; p <= 40; p += 10) {
            await sleep(60);
            setItems((prev) => prev.map((it) => (it.id === s.id ? { ...it, progress: p } : it)));
          }

          // Parse PDF (Gemini)
          const base64 = await toBase64(s.file);
          setItems((prev) => prev.map((it) => (it.id === s.id ? { ...it, status: 'analyzing', progress: 60 } : it)));

          const parseRes = await fetch('http://localhost:8000/api/parse-claim-pdf', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ pdf: base64, extraPrompt: 'Focus on extracting all available claim information accurately.' }),
          });
          if (!parseRes.ok) throw new Error(`PDF parsing failed: ${parseRes.status}`);
          const parsed = await parseRes.json();

          // Predict fraud
          const predictRes = await fetch('http://localhost:8000/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(parsed.canonical),
          });
          if (!predictRes.ok) throw new Error(`Fraud prediction failed: ${predictRes.status}`);
          const predicted = await predictRes.json();
          const first = Array.isArray(predicted.data) ? predicted.data[0] : predicted.data;

          setItems((prev) =>
            prev.map((it) =>
              it.id === s.id
                ? {
                    ...(it as ProcessedPdfItem),
                      status: 'completed',
                    progress: 100,
                    extractedData: parsed.canonical,
                    fraudPrediction: first,
                    kind: 'pdf',
                  }
                : it
            )
          );

          const isFraud = first?.decision === 1;
          const claimAmt = Number(parsed?.canonical?.ClaimAmount || 0);
          totalClaimAmount += claimAmt;
          if (isFraud) {
            fraudDetectedCount += 1;
            totalCost += claimAmt; // treat PDF fraud "savings" = claim amount
          }
          updateStats(isFraud, isFraud ? claimAmt : 0);
          
          // 통합 분석을 위한 데이터 수집
          completedDocumentAnalysis.push({
            extractedData: parsed.canonical,
            fraudPrediction: first
          });
        } catch (err: any) {
          setItems((prev) =>
            prev.map((it) => (it.id === s.id ? { ...it, status: 'error', error: err?.message || 'Document analysis failed' } : it))
          );
        }
      }

      completed += 1;
      setBatch((b) => ({ ...b, completedFiles: completed }));
    }

    // Persist image analysis rows
    for (const row of resultsForDB) {
      await saveImageAnalysis(row);
    }

    // Update analysis metadata totals
    const meta: AnalysisMetadata = {
      analysis_name: analysisTitle,
      total_files: batch.totalFiles, // Use the total files count from batch
      completed_files: batch.completedFiles + completed, // Add to existing completed count
      fraud_detected_count: fraudDetectedCount, // This will be updated with existing + new
      total_claim_amount: Math.round(totalClaimAmount * 100) / 100,
      total_cost: Math.round(totalCost * 100) / 100,
      file_urls: resultsForDB.map((r) => r.file_url).filter(Boolean),
    };
    await updateAnalysisMetadata(meta);

    // 통합 분석 요청
    console.log('🔍 Starting comprehensive analysis...');
    console.log('📸 Completed image analysis:', completedImageAnalysis);
    console.log('📄 Completed document analysis:', completedDocumentAnalysis);
    
    try {
      if (completedImageAnalysis.length > 0 || completedDocumentAnalysis.length > 0) {
        console.log('🚀 Sending comprehensive analysis request...');
        
        const comprehensiveResponse = await fetch('http://localhost:8000/api/analyze-comprehensive', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({
            imageAnalysis: completedImageAnalysis,
            documentAnalysis: completedDocumentAnalysis
          })
        });

        console.log('📡 Response status:', comprehensiveResponse.status);
        console.log('📡 Response ok:', comprehensiveResponse.ok);

        if (comprehensiveResponse.ok) {
          const comprehensiveResult = await comprehensiveResponse.json();
          console.log('✅ Comprehensive analysis result:', comprehensiveResult);
          setComprehensiveResult(comprehensiveResult.data);
        } else {
          const errorText = await comprehensiveResponse.text();
          console.error('❌ Comprehensive analysis failed:', comprehensiveResponse.status, errorText);
        }
      } else {
        console.log('⚠️ No completed analysis data found for comprehensive analysis');
      }
    } catch (error) {
      console.error('❌ Comprehensive analysis error:', error);
    }

    // refresh global stats
    refreshStats();
    
    // Clear selected files after processing
    setSelected([]);
    
    setBatch((b) => ({ ...b, isAnalyzing: false, currentFile: undefined }));
    toast.success('Analysis complete!');
  };

  /* ----------------- RENDER ----------------- */
  const overallProgress =
    batch.totalFiles > 0 ? Math.round((batch.completedFiles / batch.totalFiles) * 100) : 0;

  const cardTitle = (
    <div className="flex items-center mb-6">
      <div className="flex items-center">
        <FontAwesomeIcon icon={faArrowUpFromBracket} className="text-2xl text-blue-600 mr-3" />
        <h2 className="text-2xl font-semibold text-gray-900">Upload Vehicle Damage Images</h2>
      </div>
    </div>
  );

  return (
    <div className="p-6">
      <div className="max-w-full mx-auto">
        {/* Header */}
        <div className="mb-6">
          <div className="flex items-center mb-2">
            <h1 className="text-3xl font-bold text-gray-900">AI Fraud Detection</h1>
          </div>
          <p className="text-gray-600">Upload vehicle damage images and/or PDF claim documents. Analyze together with one click.</p>
        </div>

        {/* Unified card */}
          <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-8">
          {cardTitle}

          {/* Dropzone - only show when no files selected and not analyzed */}
          {selected.length === 0 && !isAnalyzed && (
            <div
              className={`border-2 border-dashed rounded-lg p-8 transition-colors duration-200 cursor-pointer ${
                dragOver ? 'border-blue-400 bg-blue-50' : 'border-gray-300 hover:border-blue-400 hover:bg-blue-50'
              }`}
              onDragOver={onDragOver}
              onDragLeave={onDragLeave}
              onDrop={onDrop}
              onClick={onPickClick}
              >
                <div className="text-center">
                  <div className="w-20 h-20 bg-blue-100 rounded-lg flex items-center justify-center mx-auto mb-6">
                    <FontAwesomeIcon icon={faArrowUpFromBracket} className="text-3xl text-blue-600" />
                  </div>
                  <h3 className="text-xl font-semibold text-gray-900 mb-2">
                    Drop Images/PDFs here, or click to browse
                  </h3>
                  <p className="text-gray-600 mb-4">
                    Select multiple images for batch processing
                  </p>
                  <button className="bg-blue-600 hover:bg-blue-700 text-white px-6 py-3 rounded-lg flex items-center space-x-2 mx-auto transition-colors">
                    <FontAwesomeIcon icon={faFile} />
                    <span>Browse Files</span>
                  </button>
                  <p className="text-sm text-gray-500 mt-3">Supports: JPG/PNG, PDF • Max size depends on backend</p>
                </div>
              <input
                ref={inputRef}
                type="file"
                multiple
                accept="image/*,.pdf,application/pdf"
                onChange={onInputChange}
                className="hidden"
              />
            </div>
          )}

          {/* Selected files cards - hide only after analysis is complete */}
          {selected.length > 0 && !items.some((i) => i.status === 'completed') && (
            <div className="mt-6">
              <div className="flex flex-wrap gap-4">
                {selected.map((s) => (
                  <div key={s.id} className="relative bg-white rounded-lg border border-gray-200 p-4 w-48">
                    <div className="aspect-square w-full rounded-lg overflow-hidden bg-gray-100 mb-3">
                      {s.kind === 'image' && s.preview ? (
                        <img src={s.preview} alt={s.file.name} className="w-full h-full object-cover" />
                      ) : (
                        <div className="w-full h-full flex items-center justify-center">
                          {s.kind === 'image' ? (
                            <FontAwesomeIcon icon={faFileImage} className="text-gray-400 text-2xl" />
                          ) : (
                            <FontAwesomeIcon icon={faFilePdf} className="text-red-400 text-2xl" />
                          )}
                        </div>
                      )}
                    </div>
                    <p className="text-sm font-medium text-gray-900 truncate mb-1" title={s.file.name}>
                      {s.file.name}
                    </p>
                    <p className="text-xs text-gray-500 mb-2">
                      {bytesToMB(s.file.size)} MB
                    </p>
                    <button
                      onClick={() => remove(s.id)}
                      className="absolute top-2 right-2 w-6 h-6 bg-red-500 text-white rounded-full flex items-center justify-center text-xs hover:bg-red-600"
                      title="Remove"
                    >
                      ×
                    </button>
                  </div>
                ))}
                
                {/* Upload More Button Card */}
                <div 
                  className="relative bg-white rounded-lg border-2 border-dashed border-gray-300 hover:border-blue-400 hover:bg-blue-50 p-4 w-48 cursor-pointer transition-colors duration-200"
                  onClick={onPickClick}
                >
                  <div className="aspect-square w-full rounded-lg overflow-hidden bg-gray-50 mb-3 flex items-center justify-center">
                    <div className="text-center">
                      <FontAwesomeIcon icon={faArrowUpFromBracket} className="text-gray-400 text-3xl mb-2" />
                      <p className="text-xs text-gray-500">Add More</p>
                    </div>
                  </div>
                  <p className="text-sm font-medium text-gray-900 mb-1">Upload More Files</p>
                  <p className="text-xs text-gray-500">Click to add more images or PDFs</p>
                  <input
                    ref={inputRef}
                    type="file"
                    multiple
                    accept="image/*,.pdf,application/pdf"
                    onChange={onInputChange}
                    className="hidden"
                  />
                </div>
              </div>
            </div>
          )}

          {/* Claim Title - hide only after analysis is complete */}
          {!items.some((i) => i.status === 'completed') && (
            <div className="mt-6 w-1/2 mx-auto">
              <label htmlFor="analysisTitle" className="block text-sm font-medium text-gray-700 mb-2">
                Claim Title
              </label>
              <input
                id="analysisTitle"
                type="text"
                value={analysisTitle}
                onChange={(e) => setAnalysisTitle(e.target.value)}
                placeholder="e.g., Claim Case Bundle - Sep 2025"
                className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
              />
            </div>
          )}
                
          {/* Single Analyze button - only show when not analyzed and not currently analyzing */}
          {!isAnalyzed && !batch.isAnalyzing && (
            <div className="mt-6 text-center">
              <button
                onClick={onAnalyze}
                disabled={selected.length === 0}
                className="bg-blue-600 hover:bg-blue-700 disabled:bg-gray-400 text-white px-8 py-4 rounded-lg inline-flex items-center space-x-3 transition-colors font-semibold"
              >
                <FontAwesomeIcon icon={faShieldAlt} />
                <span>Analyze</span>
              </button>
            </div>
          )}

          {/* Start Analysis button - show when files are added during analysis */}
          {isAnalyzed && !batch.isAnalyzing && selected.length > 0 && (
            <div className="mt-6 text-center">
              <button
                onClick={onAnalyze}
                disabled={selected.length === 0}
                className="bg-green-600 hover:bg-green-700 disabled:bg-gray-400 text-white px-8 py-4 rounded-lg inline-flex items-center space-x-3 transition-colors font-semibold"
              >
                <FontAwesomeIcon icon={faShieldAlt} />
                <span>Start Analysis</span>
              </button>
            </div>
          )}
            
          {/* Batch Analysis Progress */}
          {batch.isAnalyzing && (
            <div className="mt-8 bg-white rounded-lg border border-gray-200 p-6">
              {/* Header */}
              <div className="flex items-center mb-4">
                <FontAwesomeIcon icon={faShieldAlt} className="text-blue-600 mr-3" />
                <h3 className="text-lg font-semibold text-gray-900">Batch Analysis Progress</h3>
              </div>
              
              {/* Summary */}
              <div className="mb-4">
                <p className="text-sm text-gray-600">
                  Processing <span className="font-medium">{batch.totalFiles}</span> images • 
                  <span className="text-green-600 font-medium ml-1">{batch.completedFiles} completed</span>
                </p>
              </div>

              {/* Overall Progress */}
              <div className="mb-6">
                <div className="flex items-center justify-between mb-2">
                  <span className="text-sm font-medium text-gray-700">Overall Progress</span>
                  <span className="text-sm font-medium text-gray-900">{overallProgress}%</span>
                </div>
                <div className="w-full bg-gray-200 rounded-full h-2">
                  <div 
                    className="bg-blue-600 h-2 rounded-full transition-all duration-300"
                    style={{ width: `${overallProgress}%` }}
                  />
                </div>
              </div>

              {/* Individual File Progress Cards */}
              <div className="space-y-3">
                {items.map((it) => {
                  const isAnalyzing = it.status === 'analyzing' || it.status === 'uploading';
                  const isCompleted = it.status === 'completed';
                  const isError = it.status === 'error';
                  
                  return (
                    <div 
                      key={it.id} 
                      className={`border rounded-lg p-4 ${
                        isAnalyzing ? 'border-blue-200 bg-blue-50' : 
                        isCompleted ? 'border-green-200 bg-green-50' :
                        isError ? 'border-red-200 bg-red-50' :
                        'border-gray-200 bg-gray-50'
                      }`}
                    >
                      <div className="flex items-center">
                        {/* Thumbnail */}
                        <div className="w-12 h-12 rounded-lg overflow-hidden bg-gray-200 mr-4 flex-shrink-0">
                          {it.kind === 'image' && it.url ? (
                            <img src={it.url} alt={it.file.name} className="w-full h-full object-cover" />
                          ) : (
                            <div className="w-full h-full flex items-center justify-center">
                              {it.kind === 'image' ? (
                                <FontAwesomeIcon icon={faFileImage} className="text-gray-400" />
                              ) : (
                                <FontAwesomeIcon icon={faFilePdf} className="text-red-400" />
                              )}
                            </div>
                          )}
                        </div>
                        
                        {/* File Info */}
                        <div className="flex-1 min-w-0">
                          <div className="flex items-center justify-between mb-1">
                            <p className="text-sm font-medium text-gray-900 truncate" title={it.file.name}>
                              {it.file.name}
                            </p>
                            <div className="flex items-center space-x-2">
                              {isAnalyzing && (
                                <FontAwesomeIcon icon={faSpinner} className="text-gray-800 animate-spin" />
                              )}
                              {isCompleted && (
                                <FontAwesomeIcon icon={faCheckCircle} className="text-green-600" />
                              )}
                              {isError && (
                                <FontAwesomeIcon icon={faExclamationTriangle} className="text-red-600" />
                              )}
                            </div>
                          </div>
                          
                          {/* Status */}
                          <div className="mb-2">
                            {isAnalyzing && (
                              <span className="text-sm text-gray-800 font-medium">Analyzing</span>
                            )}
                            {isCompleted && (
                              <span className="text-sm text-green-600 font-medium">Completed</span>
                            )}
                            {isError && (
                              <span className="text-sm text-red-600 font-medium">Error</span>
                            )}
                          </div>
                          
                          {/* Progress Bar */}
                          <div className="flex items-center justify-between">
                            <div className="flex-1 mr-3">
                              <div className="w-full bg-gray-200 rounded-full h-1.5">
                                <div
                                  className={`h-1.5 rounded-full transition-all duration-300 ${
                                    isCompleted ? 'bg-green-600' : 
                                    isError ? 'bg-red-600' : 'bg-blue-600'
                                  }`}
                                  style={{ width: `${it.progress || 0}%` }}
                                />
                              </div>
                            </div>
                            <span className="text-xs text-gray-500 font-medium">
                              {it.progress || 0}%
                            </span>
                          </div>
                          
                          {/* Error Message */}
                          {isError && it.error && (
                            <p className="text-xs text-red-600 mt-1">{it.error}</p>
                          )}
                        </div>
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>
          )}

          {/* Post-Analysis Options */}
          {!batch.isAnalyzing && items.some((i) => i.status === 'completed') && (
            <div className="mt-8 bg-white rounded-lg border border-gray-200 p-6 mb-6 shadow-sm">
              <div className="text-center">
                <div className="flex flex-row justify-center mb-4 w-full">
                  <FontAwesomeIcon icon={faShieldAlt} className="text-blue-600 text-4xl bg-blue-50 rounded-xl p-2" />
                </div>
                <h3 className="text-lg font-semibold text-gray-900">Analysis Complete!</h3>
              </div>
            </div>
          )}

        {/* Post-Analysis Dashboard Layout */}
        {!batch.isAnalyzing && items.some((i) => i.status === 'completed') && (
          <div className="space-y-6">
            {/* Summary Cards */}
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
              {/* Total Analyzed */}
              <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
                <div className="flex items-center justify-between">
                  <div>
                    <p className="text-sm text-gray-600">Total Analyzed</p>
                    <p className="text-2xl font-bold text-gray-900">{items.filter(i => i.status === 'completed').length}</p>
                  </div>
                  <div className="p-3 bg-blue-100 rounded-lg">
                    <FontAwesomeIcon icon={faFileAlt} className="w-6 h-6 text-blue-600" />
                  </div>
                </div>
              </div>

              {/* Fraudulent */}
              <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
                <div className="flex items-center justify-between">
                  <div>
                    <p className="text-sm text-gray-600">Fraudulent</p>
                    <p className="text-2xl font-bold text-gray-900">
                      {items.filter(i => i.status === 'completed' && i.kind === 'image' && (i as ProcessedImageItem).analysis?.isFraudulent).length}
                    </p>
                  </div>
                  <div className="p-3 bg-red-100 rounded-lg">
                    <FontAwesomeIcon icon={faExclamationTriangle} className="w-6 h-6 text-red-600" />
                  </div>
                </div>
              </div>

              {/* AI Generated */}
              <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
                <div className="flex items-center justify-between">
                  <div>
                    <p className="text-sm text-gray-600">AI Generated</p>
                    <p className="text-2xl font-bold text-gray-900">
                      {items.filter(i => i.status === 'completed' && i.kind === 'image' && (i as ProcessedImageItem).analysis?.aiScore && parseFloat((i as ProcessedImageItem).analysis!.aiScore) > 0.7).length}
                    </p>
                  </div>
                  <div className="p-3 bg-red-100 rounded-lg">
                    <FontAwesomeIcon icon={faExclamationTriangle} className="w-6 h-6 text-red-600" />
                  </div>
                </div>
              </div>

              {/* Potential Savings */}
              <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
                <div className="flex items-center justify-between">
                  <div>
                    <p className="text-sm text-gray-600">Potential Savings</p>
                    <p className="text-2xl font-bold text-gray-900">
                      ${items.filter(i => i.status === 'completed' && i.kind === 'image' && (i as ProcessedImageItem).analysis?.isFraudulent)
                        .reduce((sum, i) => sum + ((i as ProcessedImageItem).cost || 0), 0).toLocaleString()}
                    </p>
                  </div>
                  <div className="p-3 bg-green-100 rounded-lg">
                    <FontAwesomeIcon icon={faDollarSign} className="w-6 h-6 text-green-600" />
                  </div>
                </div>
              </div>
            </div>

            {/* Success Banner */}
            <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
              <div className="flex items-center justify-between">
                <div className="flex items-center space-x-4">
                  <div className="p-3 bg-blue-100 rounded-lg">
                    <FontAwesomeIcon icon={faShieldAlt} className="w-6 h-6 text-blue-600" />
                  </div>
                  <div>
                    <h3 className="text-lg font-semibold text-gray-900">Case Created Successfully</h3>
                    <p className="text-gray-600">A new case has been created with {items.filter(i => i.status === 'completed').length} analyzed images.</p>
                  </div>
                </div>
                <div className="flex space-x-3">
                  <button
                    onClick={resetPage}
                    className="inline-flex items-center px-4 py-2 bg-gray-600 text-white rounded-lg hover:bg-gray-700 transition-colors"
                  >
                    <FontAwesomeIcon icon={faArrowUpFromBracket} className="mr-2" />
                    Analyze New Batch
                  </button>
                  <button
                    onClick={() => navigate(`/cases/${encodeURIComponent(analysisTitle)}`)}
                    className="inline-flex items-center px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors"
                  >
                    <FontAwesomeIcon icon={faFileAlt} className="mr-2" />
                    Open Case Details
                  </button>
                </div>
              </div>
            </div>

            {/* Results Display */}
            <ResultsDisplay 
              files={convertToUploadedFiles(items)}
            />
          </div>
        )}

        {/* Comprehensive Analysis Results */}
        {!batch.isAnalyzing && comprehensiveResult && (
          <div className="mt-8 border border-gray-200 rounded-lg p-6 bg-gradient-to-r from-blue-50 to-purple-50">
            <div className="flex items-center justify-between mb-4">
              <div className="flex items-center space-x-3">
                <FontAwesomeIcon icon={faShieldAlt} className="text-2xl text-blue-600" />
                <h4 className="text-xl font-bold text-gray-900">Comprehensive Fraud Risk Analysis</h4>
              </div>
              {comprehensiveResult.aggregateRisk && (
                <span className={`text-sm px-4 py-2 rounded-full font-semibold ${
                  comprehensiveResult.aggregateRisk.level === 'HIGH' 
                    ? 'text-red-700 bg-red-50' 
                    : comprehensiveResult.aggregateRisk.level === 'MEDIUM' 
                    ? 'text-amber-700 bg-amber-50' 
                    : 'text-green-700 bg-green-50'
                }`}>
                  {comprehensiveResult.aggregateRisk.level} RISK
                </span>
              )}
            </div>

            {/* Overall Risk Score Card */}
            {comprehensiveResult.aggregateRisk && (
              <div className="bg-white rounded-lg p-4 mb-4 shadow-sm">
                <div className="flex items-center justify-between">
                  <div className="flex items-center space-x-3">
                    <FontAwesomeIcon icon={faShieldAlt} className="text-blue-500" />
                    <span className="font-semibold text-gray-800">Overall Risk Score</span>
                  </div>
                  <div className="text-2xl font-bold text-gray-900">
                    {(comprehensiveResult.aggregateRisk.overall * 100).toFixed(1)}%
                  </div>
                </div>
                <div className="mt-2 w-full bg-gray-200 rounded-full h-3">
                  <div 
                    className={`h-3 rounded-full transition-all duration-500 ${
                      comprehensiveResult.aggregateRisk.level === 'HIGH' ? 'bg-red-500' : 
                      comprehensiveResult.aggregateRisk.level === 'MEDIUM' ? 'bg-yellow-500' : 'bg-green-500'
                    }`}
                    style={{ width: `${comprehensiveResult.aggregateRisk.overall * 100}%` }}
                  />
                </div>
              </div>
            )}

            {/* AI Analysis Results */}
            {comprehensiveResult.aiAnalysis ? (
              <div className="bg-white rounded-lg p-4 shadow-sm">
                <h5 className="font-semibold text-gray-800 mb-3 flex items-center">
                  <FontAwesomeIcon icon={faShieldAlt} className="text-green-600 mr-2" />
                  AI Comprehensive Analysis Report
                </h5>
                
                {/* Key Findings */}
                {comprehensiveResult.aiAnalysis.key_findings && (
                  <div className="mb-4">
                    <h6 className="font-medium text-gray-700 mb-2">🔍 Key Findings</h6>
                    <div className="bg-yellow-50 border border-yellow-200 rounded-lg p-3">
                      <p className="text-sm text-gray-800">{comprehensiveResult.aiAnalysis.key_findings}</p>
                    </div>
                  </div>
                )}

                {/* Risk Factors */}
                {comprehensiveResult.aiAnalysis.risk_factors && comprehensiveResult.aiAnalysis.risk_factors.length > 0 && (
                  <div className="mb-4">
                    <h6 className="font-medium text-gray-700 mb-2">⚠️ Risk Factors</h6>
                    <ul className="space-y-1">
                      {comprehensiveResult.aiAnalysis.risk_factors.map((factor: string, index: number) => (
                        <li key={index} className="text-sm text-gray-700 flex items-start">
                          <span className="text-red-500 mr-2">•</span>
                          {factor}
                        </li>
                      ))}
                    </ul>
                  </div>
                )}

                {/* Recommendations */}
                {comprehensiveResult.aiAnalysis.recommendations && comprehensiveResult.aiAnalysis.recommendations.length > 0 && (
                  <div className="mb-4">
                    <h6 className="font-medium text-gray-700 mb-2">💡 Recommendations</h6>
                    <ul className="space-y-1">
                      {comprehensiveResult.aiAnalysis.recommendations.map((rec: string, index: number) => (
                        <li key={index} className="text-sm text-gray-700 flex items-start">
                          <span className="text-blue-500 mr-2">•</span>
                          {rec}
                        </li>
                      ))}
                    </ul>
                  </div>
                )}

                {/* Detailed Analysis */}
                {comprehensiveResult.aiAnalysis.detailed_analysis && (
                  <div className="mb-4">
                    <h6 className="font-medium text-gray-700 mb-2">📊 Detailed Analysis</h6>
                    <div className="bg-gray-50 border border-gray-200 rounded-lg p-3">
                      <p className="text-sm text-gray-800">{comprehensiveResult.aiAnalysis.detailed_analysis}</p>
                    </div>
                  </div>
                )}
              </div>
            ) : (
              <div className="bg-amber-50 border border-amber-200 rounded-lg p-4 flex items-start">
                <FontAwesomeIcon icon={faExclamationTriangle} className="text-amber-600 mr-3 mt-0.5" />
                <div>
                  <p className="text-sm font-medium text-amber-800">Unable to generate AI analysis results.</p>
                  <p className="text-xs text-amber-700 mt-1">Google API key is not configured or network error occurred.</p>
                </div>
              </div>
            )}
          </div>
        )}
        </div>
      </div>
    </div>
  );
};

export default Upload;
