import React, { useEffect, useMemo, useRef, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome';
import { 
  faArrowLeft,
  faArrowUpFromBracket,
  faCheckCircle, 
  faExclamationTriangle,
  faFile,
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

  // Derived counts
  const counts = useMemo(() => {
    const img = selected.filter((s) => s.kind === 'image').length;
    const pdf = selected.filter((s) => s.kind === 'pdf').length;
    return { img, pdf, total: selected.length };
  }, [selected]);

  /* ------------- Select & Remove ------------- */
  const onPickClick = () => inputRef.current?.click();

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

  /* ------------- Analyze Button ------------- */
  const onAnalyze = async () => {
    if (selected.length === 0) {
      toast.info('Please add images or PDFs to analyze.');
      return;
    }
    if (!analysisTitle.trim()) {
      toast.info('Please enter an Analysis Title.');
      return;
    }
    if (names.includes(analysisTitle)) {
      toast.error('Analysis title already exists. Choose another title.');
      return;
    }
    
    // Prepare UI state
    setItems(
      selected.map<ProcessedItem>((s) =>
        s.kind === 'image'
          ? { id: s.id, file: s.file, kind: 'image', status: 'uploading', progress: 0 }
          : { id: s.id, file: s.file, kind: 'pdf', status: 'uploading', progress: 0 }
      )
    );
    setBatch({ isAnalyzing: true, totalFiles: selected.length, completedFiles: 0, currentFile: undefined });

    // Create analysis metadata shell (image table reuse – only images will be inserted below)
    const metaInit = await saveAnalysisMetadata({
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

    // Process sequentially (simple UI)
    const resultsForDB: ImageAnalysis[] = [];
    let fraudDetectedCount = 0;  // count both images+pdf
    let totalCost = 0;           // image fraud “cost” (demo) + pdf ClaimAmount for frauds
    let totalClaimAmount = 0;    // pdf sum of claim amounts (for info)
    let completed = 0;

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
            totalCost += claimAmt; // treat PDF fraud “savings” = claim amount
          }
          updateStats(isFraud, isFraud ? claimAmt : 0);
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
      total_files: selected.length,
      completed_files: completed,
      fraud_detected_count: fraudDetectedCount,
      total_claim_amount: Math.round(totalClaimAmount * 100) / 100,
      total_cost: Math.round(totalCost * 100) / 100,
      file_urls: resultsForDB.map((r) => r.file_url).filter(Boolean),
    };
    await updateAnalysisMetadata(meta);

    // refresh global stats
    refreshStats();
    
    setBatch((b) => ({ ...b, isAnalyzing: false, currentFile: undefined }));
    toast.success('Analysis complete!');
  };

  /* ----------------- RENDER ----------------- */
  const overallProgress =
    batch.totalFiles > 0 ? Math.round((batch.completedFiles / batch.totalFiles) * 100) : 0;

  const cardTitle = (
    <div className="flex items-center mb-6">
      {!batch.isAnalyzing && (
            <button 
              className="mr-4 p-2 hover:bg-gray-100 rounded-lg transition-colors"
              onClick={() => navigate('/dashboard')}
            >
              <FontAwesomeIcon icon={faArrowLeft} className="text-gray-600" />
            </button>
      )}
      <div className="flex items-center">
        <FontAwesomeIcon icon={faShieldAlt} className="text-2xl text-blue-600 mr-3" />
        <h2 className="text-2xl font-semibold text-gray-900">Upload & Analyze (Images + PDFs)</h2>
            </div>
          </div>
  );

  return (
    <div className="p-6">
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <div className="mb-6">
          <h1 className="text-3xl font-bold text-gray-900">AI Fraud Detection</h1>
          <p className="text-gray-600">Upload vehicle damage images and/or PDF claim documents. Analyze together with one click.</p>
        </div>

        {/* Unified card */}
          <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-8">
          {cardTitle}

          {/* Dropzone */}
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
                Selected: <span className="font-medium">{counts.total}</span> files
                &nbsp; (Images: {counts.img} · PDFs: {counts.pdf})
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

          {/* Selected list */}
          {selected.length > 0 && (
            <div className="mt-6">
              <h4 className="text-sm font-semibold text-gray-700 mb-3">Selected Files</h4>
              <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-3">
                {selected.map((s) => (
                  <div key={s.id} className="relative bg-gray-50 rounded-lg p-2 border border-gray-200">
                    <div className="flex items-center space-x-3">
                      <div className="w-12 h-12 rounded-lg overflow-hidden bg-gray-200 flex items-center justify-center">
                        {s.kind === 'image' ? (
                          s.preview ? <img src={s.preview} alt={s.file.name} className="w-full h-full object-cover" /> :
                          <FontAwesomeIcon icon={faFileImage} className="text-gray-500" />
                        ) : (
                          <FontAwesomeIcon icon={faFilePdf} className="text-red-500 text-lg" />
                        )}
                      </div>
                      <div className="min-w-0">
                        <p className="text-sm font-medium text-gray-900 truncate" title={s.file.name}>
                          {s.file.name}
                      </p>
                      <p className="text-xs text-gray-500">
                          {s.kind.toUpperCase()} • {bytesToMB(s.file.size)} MB
                      </p>
                      </div>
                    </div>
                      <button
                      onClick={() => remove(s.id)}
                        className="absolute top-1 right-1 w-5 h-5 bg-red-500 text-white rounded-full flex items-center justify-center text-xs hover:bg-red-600"
                      title="Remove"
                      >
                        ×
                      </button>
                    </div>
                  ))}
                  </div>
                </div>
          )}

          {/* Analysis Title */}
          <div className="mt-6">
                  <label htmlFor="analysisTitle" className="block text-sm font-medium text-gray-700 mb-2">
                    Analysis Title
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
                
          {/* Single Analyze button */}
          <div className="mt-6 text-center">
                  <button
              onClick={onAnalyze}
              disabled={batch.isAnalyzing || selected.length === 0}
              className="bg-blue-600 hover:bg-blue-700 disabled:bg-gray-400 text-white px-8 py-4 rounded-lg inline-flex items-center space-x-3 transition-colors"
            >
              {batch.isAnalyzing ? (
                <>
                  <FontAwesomeIcon icon={faSpinner} className="animate-spin" />
                  <span>Analyzing {batch.completedFiles}/{batch.totalFiles}...</span>
                </>
              ) : (
                <>
                    <FontAwesomeIcon icon={faShieldAlt} />
                  <span>Analyze {selected.length} File(s)</span>
                </>
              )}
                  </button>
            </div>
            
          {/* Batch progress */}
          {batch.isAnalyzing && (
            <div className="mt-8">
              <div className="flex items-center justify-between mb-2">
                <span className="text-sm text-gray-600">
                  Processing <span className="font-medium">{batch.totalFiles}</span> file(s)
                  {batch.currentFile ? ` • Current: ${batch.currentFile}` : ''}
                </span>
                <span className="text-sm text-blue-700 font-medium">{overallProgress}%</span>
          </div>
              <div className="w-full bg-gray-200 rounded-full h-3">
                <div 
                  className="bg-blue-600 h-3 rounded-full transition-all duration-300"
                  style={{ width: `${overallProgress}%` }}
                />
            </div>

              {/* Per-item progress */}
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3 mt-4">
                {items.map((it) => (
                  <div key={it.id} className="border border-gray-200 rounded-lg p-3 bg-gray-50">
                    <div className="flex items-center justify-between">
                      <div className="flex items-center space-x-2">
                        {it.kind === 'image' ? (
                          <FontAwesomeIcon icon={faFileImage} className="text-gray-500" />
                        ) : (
                          <FontAwesomeIcon icon={faFilePdf} className="text-red-500" />
                        )}
                        <span className="text-sm font-medium text-gray-900 truncate" title={it.file.name}>
                          {it.file.name}
                        </span>
                      </div>
                      <div className="text-right">
                        {it.status === 'completed' ? (
                          <FontAwesomeIcon icon={faCheckCircle} className="text-green-600" />
                        ) : it.status === 'error' ? (
                          <FontAwesomeIcon icon={faExclamationTriangle} className="text-red-600" />
                        ) : (
                          <FontAwesomeIcon icon={faSpinner} className="text-blue-600 animate-spin" />
                        )}
                      </div>
                    </div>
                    
                    {it.status !== 'completed' && (
                      <div className="mt-2">
                        <div className="w-full bg-gray-200 rounded-full h-2">
                          <div
                            className="bg-blue-600 h-2 rounded-full transition-all duration-300"
                            style={{ width: `${it.progress || 0}%` }}
                          />
                        </div>
                        <div className="text-right text-xs text-gray-500 mt-1">{it.progress || 0}%</div>
                      </div>
                    )}

                    {it.status === 'error' && (
                      <p className="mt-2 text-xs text-red-600">{it.error}</p>
                    )}
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Results Display */}
          {!batch.isAnalyzing && items.some((i) => i.status === 'completed') && (
          <ResultsDisplay 
              files={convertToUploadedFiles(items)}
              actionButton={
              <button
                  onClick={() => navigate('/dashboard')}
                  className="inline-flex items-center px-6 py-3 bg-gray-600 text-white rounded-lg hover:bg-gray-700 transition-colors"
              >
                  Back to Dashboard
              </button>
              }
          />
        )}
        </div>
      </div>
    </div>
  );
};

export default Upload;
