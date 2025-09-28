# iSpy Frontend - AI Vehicle Damage Claim Fraud Detection

A React-based frontend application for AI-powered vehicle damage claim fraud detection. This application provides an intuitive interface for uploading images, analyzing them for fraud indicators, and managing case data.

## 🏗️ Architecture Overview

The frontend is built with React 19 and TypeScript, using a modern component-based architecture with the following key architectural patterns:

### Core Technologies
- **React 19** - Modern React with latest features
- **TypeScript** - Type-safe development
- **Vite** - Fast build tool and development server
- **Tailwind CSS** - Utility-first CSS framework
- **React Router DOM** - Client-side routing
- **Supabase** - Backend-as-a-Service for database and storage
- **Chart.js** - Data visualization
- **FontAwesome** - Icon library

## 📁 Project Structure

```
frontend/
├── src/
│   ├── api/                    # API layer and data services
│   │   ├── database.ts         # Supabase database operations
│   │   └── imageUpload.ts      # Image upload and analysis services
│   ├── components/             # Reusable UI components
│   │   ├── ResultsDisplay.tsx  # Analysis results display component
│   │   └── Sidenav.tsx         # Navigation sidebar component
│   ├── contexts/               # React Context providers
│   │   └── StatsContext.tsx    # Global application statistics
│   ├── lib/                    # Utility libraries
│   │   └── supabase.ts         # Supabase client configuration
│   ├── pages/                  # Main application pages
│   │   ├── Dashboard.tsx       # Analytics dashboard
│   │   ├── Upload.tsx          # File upload and analysis
│   │   ├── Cases.tsx           # Case management overview
│   │   ├── CaseDetail.tsx      # Individual case details
│   │   └── ImageReview.tsx     # Image review interface
│   ├── App.tsx                 # Main application component
│   ├── main.tsx               # Application entry point
│   └── index.css              # Global styles
├── public/                     # Static assets
├── dist/                      # Production build output
├── package.json               # Dependencies and scripts
├── vite.config.ts            # Vite configuration
└── tsconfig.json             # TypeScript configuration
```

## 🔄 Component Architecture & Data Flow

### 1. Application Structure

```
App (Root)
├── StatsProvider (Context)
│   └── Router
│       └── AppLayout
│           ├── Sidenav (Navigation)
│           └── Routes
│               ├── Dashboard
│               ├── Upload
│               ├── Cases
│               ├── CaseDetail
│               └── ImageReview
```

### 2. Key Components & Their Relationships

#### **App.tsx** - Application Root
- Wraps the entire application with `StatsProvider` for global state management
- Sets up React Router for navigation
- Configures toast notifications

#### **StatsContext.tsx** - Global State Management
- **Purpose**: Manages application-wide statistics (cases analyzed, fraud detected, money saved)
- **Data Sources**: Fetches data from Supabase database on mount
- **Key Functions**:
  - `updateStats()` - Updates stats when new analysis completes
  - `refreshStats()` - Reloads stats from database
  - `resetStats()` - Resets all statistics

#### **Sidenav.tsx** - Navigation Component
- **Dependencies**: Uses `StatsContext` to display real-time statistics
- **Features**: 
  - Responsive design (mobile/desktop)
  - Quick stats display
  - Navigation state management
- **Communication**: Receives navigation callbacks from parent

### 3. Page Components

#### **Dashboard.tsx** - Analytics Dashboard
- **Purpose**: Displays analytics charts and key metrics
- **Data Flow**: 
  - Uses `StatsContext` for real-time stats
  - Fetches analysis metadata for chart data
  - Processes data for Chart.js visualization
- **Features**:
  - Interactive charts showing fraud trends
  - Alert system for high-risk cases
  - Real-time statistics display

#### **Upload.tsx** - File Upload & Analysis
- **Purpose**: Handles file uploads and AI analysis
- **Key Features**:
  - Drag-and-drop file interface
  - Batch processing with progress tracking
  - Real-time analysis status updates
  - Integration with ML backend API
- **Data Flow**:
  - Uploads files to Supabase Storage
  - Calls ML API for fraud analysis
  - Saves results to database
  - Updates global stats via context

#### **Cases.tsx** - Case Management
- **Purpose**: Overview of all analyzed cases
- **Features**:
  - Search and filtering capabilities
  - Status-based filtering
  - Cost sorting options
- **Data Flow**: Fetches analysis metadata from database

#### **CaseDetail.tsx** - Individual Case View
- **Purpose**: Detailed view of specific case analysis
- **Features**:
  - Individual image analysis results
  - Fraud risk assessment
  - Cost breakdown
- **Navigation**: Accessed via dynamic routing (`/cases/:analysisName`)

#### **ImageReview.tsx** - Image Review Interface
- **Purpose**: Review and manage uploaded images
- **Features**:
  - Image gallery with filtering
  - Individual image analysis results
  - Batch operations

### 4. API Layer

#### **database.ts** - Database Operations
- **Supabase Integration**: Handles all database operations
- **Key Functions**:
  - `fetchAnalysisMetadata()` - Gets all case metadata
  - `fetchImageAnalyses()` - Gets individual image analyses
  - `saveAnalysisMetadata()` - Saves case metadata
  - `saveImageAnalysis()` - Saves individual analysis results
- **Data Models**:
  - `AnalysisMetadata` - Case-level data
  - `ImageAnalysis` - Individual image analysis data

#### **imageUpload.ts** - File & Analysis Services
- **File Management**: Handles Supabase Storage operations
- **ML Integration**: Communicates with backend ML API
- **Key Functions**:
  - `uploadImage()` - Uploads files to Supabase Storage
  - `analyzeImage()` - Calls ML API for fraud detection
  - `deleteImage()` - Removes files from storage

#### **supabase.ts** - Configuration
- **Purpose**: Supabase client configuration
- **Environment Variables**: Manages storage bucket names and API keys

## 🔄 Data Flow Architecture

### 1. Upload & Analysis Flow
```
User Upload → Supabase Storage → ML API → Database → Stats Update
     ↓              ↓              ↓           ↓           ↓
  File Selection → Storage URL → Analysis → Save Results → Context Update
```

### 2. Statistics Flow
```
Database → StatsContext → Components → UI Display
    ↓           ↓            ↓           ↓
Analysis Data → Global State → Real-time Updates → Dashboard/Sidenav
```

### 3. Navigation Flow
```
User Action → Router → Page Component → API Call → Database → UI Update
     ↓           ↓           ↓            ↓          ↓         ↓
Click Nav → Route Change → Component Mount → Data Fetch → State Update → Render
```

## 🚀 Getting Started

### Prerequisites
- Node.js (v18 or higher)
- npm or yarn package manager
- Supabase account and project setup

### Environment Setup
Create a `.env` file in the frontend directory with the following variables:

```env
VITE_PUBLIC_SUPABASE_URL=your_supabase_url
VITE_PUBLIC_SUPABASE_ANON_KEY=your_supabase_anon_key
VITE_STORAGE_NAME=your_storage_bucket_name
VITE_UPLOAD_FOLDER=uploads
VITE_FRAUD_FOLDER=fraud
VITE_NON_FRAUD_FOLDER=non-fraud
VITE_API_URL=your_ml_api_url
```

### Installation & Development

1. **Install Dependencies**
   ```bash
   cd frontend
   npm install
   ```

2. **Start Development Server**
   ```bash
   npm run dev
   ```
   The application will be available at `http://localhost:5173`

3. **Build for Production**
   ```bash
   npm run build
   ```

4. **Preview Production Build**
   ```bash
   npm run preview
   ```

### Available Scripts

- `npm run dev` - Start development server with hot reload
- `npm run build` - Build production bundle
- `npm run preview` - Preview production build
- `npm run lint` - Run ESLint for code quality

## 🔧 Configuration

### Vite Configuration
The project uses Vite with the following plugins:
- **React Plugin**: Enables React support with Fast Refresh
- **Tailwind CSS Plugin**: Enables Tailwind CSS processing

### TypeScript Configuration
- Strict type checking enabled
- React-specific type definitions
- Path mapping for clean imports

## 🎨 Styling & UI

### Design System
- **Tailwind CSS**: Utility-first CSS framework
- **Responsive Design**: Mobile-first approach
- **Component Library**: Custom components with consistent styling
- **Icons**: FontAwesome for consistent iconography

### Key UI Patterns
- **Sidebar Navigation**: Collapsible on mobile, persistent on desktop
- **Card-based Layout**: Consistent card components for data display
- **Progress Indicators**: Real-time progress for long-running operations
- **Toast Notifications**: User feedback for actions

## 🔌 Backend Integration

### Supabase Integration
- **Database**: PostgreSQL with real-time subscriptions
- **Storage**: File storage for uploaded images
- **Authentication**: Ready for user authentication (not currently implemented)

### ML API Integration
- **Endpoint**: `/analyze_fraud` for image analysis
- **Request Format**: JSON with image URL
- **Response Format**: Analysis results with fraud scores and explanations

## 📊 State Management

### Context Pattern
- **StatsContext**: Global application statistics
- **Local State**: Component-level state for UI interactions
- **Server State**: Data fetched from Supabase

### Data Persistence
- **Database**: Supabase PostgreSQL for persistent data
- **Storage**: Supabase Storage for file assets
- **Real-time**: Automatic updates when data changes

## 🔍 Key Features

### 1. **Intelligent File Upload**
- Drag-and-drop interface
- Multiple file format support
- Batch processing with progress tracking
- Automatic file validation

### 2. **AI-Powered Analysis**
- Real-time fraud detection
- Risk level assessment
- Detailed analysis explanations
- Cost calculation and tracking

### 3. **Comprehensive Dashboard**
- Interactive charts and graphs
- Real-time statistics
- Alert system for high-risk cases
- Trend analysis

### 4. **Case Management**
- Search and filtering
- Status tracking
- Cost analysis
- Detailed case views

### 5. **Responsive Design**
- Mobile-first approach
- Adaptive navigation
- Touch-friendly interfaces
- Cross-device compatibility

## 🚨 Error Handling

### Client-Side Error Handling
- **Toast Notifications**: User-friendly error messages
- **Loading States**: Visual feedback during operations
- **Retry Mechanisms**: Automatic retry for failed operations
- **Graceful Degradation**: Fallback UI for errors

### API Error Handling
- **Network Errors**: Connection timeout handling
- **Validation Errors**: Input validation feedback
- **Server Errors**: Backend error message display
- **Rate Limiting**: Handling of API rate limits

## 🔐 Security Considerations

### Data Protection
- **Environment Variables**: Sensitive data in environment variables
- **File Validation**: Client-side file type and size validation
- **Secure Storage**: Supabase security features
- **CORS Configuration**: Proper cross-origin resource sharing

### Best Practices
- **Input Sanitization**: All user inputs properly sanitized
- **File Upload Security**: Secure file handling
- **API Security**: Secure communication with backend
- **Data Privacy**: User data protection compliance

## 📈 Performance Optimization

### Bundle Optimization
- **Code Splitting**: Route-based code splitting
- **Tree Shaking**: Unused code elimination
- **Asset Optimization**: Image and asset optimization
- **Lazy Loading**: Component lazy loading

### Runtime Performance
- **React Optimization**: Proper use of hooks and memoization
- **Chart Performance**: Optimized chart rendering
- **Image Optimization**: Efficient image loading and display
- **State Management**: Efficient state updates

## 🧪 Testing Strategy

### Testing Approach
- **Component Testing**: Individual component testing
- **Integration Testing**: Component interaction testing
- **E2E Testing**: End-to-end user flow testing
- **API Testing**: Backend integration testing

### Quality Assurance
- **TypeScript**: Compile-time error checking
- **ESLint**: Code quality and consistency
- **Prettier**: Code formatting (recommended)
- **Manual Testing**: User experience validation

## 🚀 Deployment

### Production Build
- **Optimized Bundle**: Minified and optimized code
- **Asset Optimization**: Compressed images and assets
- **Environment Configuration**: Production environment variables
- **CDN Ready**: Static asset serving

### Deployment Options
- **Static Hosting**: Netlify, Vercel, or similar
- **Container Deployment**: Docker containerization
- **Cloud Platforms**: AWS, GCP, or Azure deployment
- **CI/CD Pipeline**: Automated deployment workflows

## 📚 Additional Resources

### Documentation
- **API Documentation**: Backend API reference
- **Database Schema**: Supabase database structure
- **Component Library**: Reusable component documentation
- **Deployment Guide**: Production deployment instructions

### Development Tools
- **React DevTools**: Browser extension for debugging
- **Supabase Dashboard**: Database and storage management
- **Vite DevTools**: Development server features
- **TypeScript Language Server**: IDE support for TypeScript

---

This frontend application provides a comprehensive interface for AI-powered fraud detection, with a focus on user experience, performance, and maintainability. The modular architecture ensures scalability and ease of development while providing powerful features for fraud detection and case management.