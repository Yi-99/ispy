import React from 'react';
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome';
import { 
  faGrip,
  faUpload, 
  faFileAlt, 
  faShieldAlt, 
  faTimes,
} from '@fortawesome/free-solid-svg-icons';
import { useStats } from '../contexts/StatsContext';

interface SidenavProps {
  isOpen: boolean;
  onToggle: () => void;
  currentPage: string;
  onNavigate: (page: string) => void;
}

const Sidenav: React.FC<SidenavProps> = ({ isOpen, onToggle, currentPage, onNavigate }) => {
  const { casesAnalyzed, fraudDetected, moneySaved } = useStats();

  const navigationItems = [
    { id: 'dashboard', label: 'Dashboard', icon: faGrip },
    { id: 'upload', label: 'Upload & Analyze', icon: faUpload },
    { id: 'cases', label: 'Case Review', icon: faShieldAlt },
    { id: 'image-review', label: 'Image Review', icon: faFileAlt },
  ];

  const quickStats = [
    { label: 'Cases Analyzed', value: casesAnalyzed.toString() },
    { label: 'Fraud Detected', value: fraudDetected.toString() },
    { label: 'Money Saved', value: `$${moneySaved.toLocaleString()}` },
  ];

  return (
    <>
      {/* Mobile overlay */}
      {isOpen && (
        <div 
          className="fixed inset-0 bg-black bg-opacity-50 z-40 lg:hidden"
          onClick={onToggle}
        />
      )}
      
      {/* Sidebar */}
      <div className={`
        fixed top-0 left-0 h-full bg-white shadow-lg z-50 transform transition-transform duration-300 ease-in-out
        ${isOpen ? 'translate-x-0' : '-translate-x-full'}
        lg:translate-x-0 lg:static lg:shadow-none
        w-80
      `}>
        <div className="flex flex-col h-full">
          {/* Header */}
          <div className="flex items-center justify-between p-6 border-b border-gray-200">
            <div className="flex items-center space-x-3">
              <div className="w-10 h-10 bg-blue-600 rounded-lg flex items-center justify-center">
                <FontAwesomeIcon icon={faShieldAlt} className="text-white text-lg" />
              </div>
              <div>
                <h1 className="text-2xl font-bold text-gray-900">iSpy</h1>
                <p className="text-xs text-gray-500 w-3/4">AI Vehicle Damage Claim Fraud Detection</p>
              </div>
            </div>
            <button
              onClick={onToggle}
              className="lg:hidden p-2 rounded-md hover:bg-gray-100"
            >
              <FontAwesomeIcon icon={faTimes} className="text-gray-500" />
            </button>
          </div>

          {/* Navigation */}
          <div className="flex-1 px-6 py-4">
            <div className="mb-8">
              <h3 className="text-xs font-semibold text-gray-500 uppercase tracking-wider mb-4">
                Navigation
              </h3>
              <nav className="space-y-2">
                {navigationItems.map((item) => (
                  <button
                    key={item.id}
                    onClick={() => onNavigate(item.id)}
                    className={`
                      w-full flex items-center space-x-3 px-4 py-3 rounded-lg text-left transition-colors duration-200
                      ${currentPage === item.id 
                        ? 'bg-blue-50 text-blue-700' 
                        : 'text-gray-700 hover:bg-gray-50 hover:text-gray-900'
                      }
                    `}
                  >
                    <FontAwesomeIcon 
                      icon={item.icon} 
                      className={`w-5 h-5 ${currentPage === item.id ? 'text-blue-700' : 'text-gray-400'}`}
                    />
                    <span className="font-medium">{item.label}</span>
                  </button>
                ))}
              </nav>
            </div>

            {/* Quick Stats */}
            <div className="mb-8">
              <h3 className="text-xs font-semibold text-gray-500 uppercase tracking-wider mb-4">
                Quick Stats
              </h3>
              <div className="space-y-3">
                {quickStats.map((stat, index) => (
                  <div key={index} className="bg-gray-50 rounded-lg p-4">
                    <div className="flex items-center justify-between">
                      <div>
                        <p className="text-xs text-gray-500">{stat.label}</p>
                        <p className={`text-sm font-medium ${stat.label === 'Money Saved' && moneySaved > 0 ? 'text-green-600' : 'text-gray-900'}`}>{stat.value}</p>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>

        </div>
      </div>
    </>
  );
};

export default Sidenav;
