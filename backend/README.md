# iSpy Backend Server

Node.js Express server for AI-powered fraud detection using Hugging Face models.

## Features

- **AI Image Detection**: Uses Hugging Face models to analyze vehicle damage images
- **Fraud Detection**: Enhanced analysis specifically for fraud detection
- **Multiple Input Methods**: Support for file uploads, file paths, and image URLs
- **CORS Enabled**: Ready for frontend integration
- **File Upload Support**: Multer integration for handling image uploads

## Installation

```bash
npm install
```

## Environment Variables

Create a `.env` file in the backend directory:

```env
PORT=8000
HF_API_KEY=your_huggingface_api_key_here
```

## Running the Server

### Development Mode
```bash
npm run dev
```

### Production Mode
```bash
npm start
```

The server will start on `http://localhost:8000`

## API Endpoints

### Health Check
- **GET** `/health` - Server health status

### Image Detection
- **POST** `/detect_image` - Detect objects from file path
  ```json
  {
    "image_path": "/path/to/image.jpg"
  }
  ```

- **POST** `/detect_image_url` - Detect objects from image URL
  ```json
  {
    "image_url": "https://example.com/image.jpg"
  }
  ```

- **POST** `/upload_and_detect` - Upload image file and detect objects
  ```
  Content-Type: multipart/form-data
  image: [file]
  ```

### Fraud Detection
- **POST** `/analyze_fraud` - Analyze image URL for fraud detection
  ```json
  {
    "image_url": "https://example.com/image.jpg"
  }
  ```

- **POST** `/upload_and_analyze` - Upload image file and analyze for fraud
  ```
  Content-Type: multipart/form-data
  image: [file]
  ```

## Response Format

### Standard Detection Response
```json
{
  "result": "car damage score = 0.85, vehicle score = 0.92"
}
```

### Fraud Analysis Response
```json
{
  "success": true,
  "data": {
    "isFraudulent": false,
    "confidence": 0.85,
    "riskLevel": "LOW",
    "analysis": "No significant fraud indicators detected...",
    "detectedIssues": [],
    "rawClassification": "car damage score = 0.85",
    "fraudScore": 0.15
  }
}
```

## Testing

Run the test script to verify endpoints:

```bash
node test_endpoints.js
```

## File Upload Limits

- Maximum file size: 10MB
- Allowed file types: Images only (jpg, png, gif, etc.)
- Upload directory: `uploads/` (auto-created)

## CORS Configuration

The server is configured to allow requests from any origin. For production, update the CORS settings in `server.js`:

```javascript
res.header('Access-Control-Allow-Origin', 'https://yourdomain.com');
```

## Error Handling

All endpoints include comprehensive error handling and return appropriate HTTP status codes:

- `400` - Bad Request (missing parameters)
- `500` - Internal Server Error (processing errors)

## Dependencies

- `express` - Web framework
- `@huggingface/inference` - Hugging Face AI model integration
- `multer` - File upload handling
- `axios` - HTTP client for testing
- `nodemon` - Development server (dev dependency)
