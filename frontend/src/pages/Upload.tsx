import React from 'react';
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome';
import { faUpload, faFileImage, faCloudUploadAlt } from '@fortawesome/free-solid-svg-icons';

const Upload: React.FC = () => {
  return (
    <div className="p-6">
      <div className="max-w-4xl mx-auto">
        <div className="mb-8">
          <h1 className="text-3xl font-bold text-gray-900 mb-2">Upload & Analyze</h1>
          <p className="text-gray-600">
            Upload vehicle damage images for AI-powered fraud detection analysis
          </p>
        </div>

        <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-8">
          <div className="text-center">
            <div className="w-24 h-24 bg-blue-100 rounded-full flex items-center justify-center mx-auto mb-6">
              <FontAwesomeIcon icon={faCloudUploadAlt} className="text-4xl text-blue-600" />
            </div>
            <h2 className="text-2xl font-semibold text-gray-900 mb-4">Upload Vehicle Damage Images</h2>
            <p className="text-gray-600 mb-8 max-w-2xl mx-auto">
              Drag and drop your images here, or click to browse. We support JPG, PNG, and other common image formats.
            </p>
            
            <div className="border-2 border-dashed border-gray-300 rounded-lg p-12 hover:border-blue-400 transition-colors duration-200 cursor-pointer">
              <FontAwesomeIcon icon={faFileImage} className="text-6xl text-gray-400 mb-4" />
              <p className="text-lg font-medium text-gray-700 mb-2">Drop files here or click to upload</p>
              <p className="text-sm text-gray-500">Maximum file size: 10MB per image</p>
            </div>
            
            <div className="mt-8">
              <button className="bg-blue-600 hover:bg-blue-700 text-white px-8 py-3 rounded-lg font-medium flex items-center space-x-2 mx-auto transition-colors duration-200">
                <FontAwesomeIcon icon={faUpload} />
                <span>Start Analysis</span>
              </button>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Upload;
