import React from 'react';
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome';
import { 
  faFileAlt, 
  faSearch, 
  faFilter, 
  faExclamationTriangle, 
  faCheckCircle, 
  faClock,
  faEye
} from '@fortawesome/free-solid-svg-icons';

const Cases: React.FC = () => {
  const cases = [
    {
      id: 'CASE-001',
      status: 'fraudulent',
      statusColor: 'bg-red-100 text-red-800',
      statusIcon: faExclamationTriangle,
      statusIconColor: 'text-red-600',
      title: 'Vehicle Damage Claim - Honda Civic',
      date: '2024-01-15',
      amount: '$8,500',
      risk: 'High',
      riskColor: 'bg-red-100 text-red-800'
    },
    {
      id: 'CASE-002',
      status: 'legitimate',
      statusColor: 'bg-green-100 text-green-800',
      statusIcon: faCheckCircle,
      statusIconColor: 'text-green-600',
      title: 'Vehicle Damage Claim - Toyota Camry',
      date: '2024-01-14',
      amount: '$3,200',
      risk: 'Low',
      riskColor: 'bg-green-100 text-green-800'
    },
    {
      id: 'CASE-003',
      status: 'pending',
      statusColor: 'bg-yellow-100 text-yellow-800',
      statusIcon: faClock,
      statusIconColor: 'text-yellow-600',
      title: 'Vehicle Damage Claim - Ford F-150',
      date: '2024-01-13',
      amount: '$12,000',
      risk: 'Medium',
      riskColor: 'bg-yellow-100 text-yellow-800'
    }
  ];

  return (
    <div className="p-6">
      <div className="max-w-7xl mx-auto">
        <div className="mb-8">
          <h1 className="text-3xl font-bold text-gray-900 mb-2">Case Review</h1>
          <p className="text-gray-600">
            Review and manage vehicle damage claims for fraud detection
          </p>
        </div>

        {/* Filters and Search */}
        <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6 mb-6">
          <div className="flex flex-col sm:flex-row gap-4">
            <div className="flex-1">
              <div className="relative">
                <FontAwesomeIcon 
                  icon={faSearch} 
                  className="absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400"
                />
                <input
                  type="text"
                  placeholder="Search cases..."
                  className="w-full pl-10 pr-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                />
              </div>
            </div>
            <div className="flex gap-2">
              <button className="px-4 py-2 border border-gray-300 rounded-lg hover:bg-gray-50 flex items-center space-x-2">
                <FontAwesomeIcon icon={faFilter} />
                <span>Filter</span>
              </button>
            </div>
          </div>
        </div>

        {/* Cases Table */}
        <div className="bg-white rounded-lg shadow-sm border border-gray-200 overflow-hidden">
          <div className="overflow-x-auto">
            <table className="min-w-full divide-y divide-gray-200">
              <thead className="bg-gray-50">
                <tr>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Case ID
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Status
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Title
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Date
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Amount
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Risk Level
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Actions
                  </th>
                </tr>
              </thead>
              <tbody className="bg-white divide-y divide-gray-200">
                {cases.map((caseItem) => (
                  <tr key={caseItem.id} className="hover:bg-gray-50">
                    <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-900">
                      {caseItem.id}
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap">
                      <span className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${caseItem.statusColor}`}>
                        <FontAwesomeIcon 
                          icon={caseItem.statusIcon} 
                          className={`w-3 h-3 mr-1 ${caseItem.statusIconColor}`}
                        />
                        {caseItem.status.charAt(0).toUpperCase() + caseItem.status.slice(1)}
                      </span>
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                      {caseItem.title}
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                      {caseItem.date}
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-900">
                      {caseItem.amount}
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap">
                      <span className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${caseItem.riskColor}`}>
                        {caseItem.risk}
                      </span>
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm font-medium">
                      <button className="text-blue-600 hover:text-blue-900 flex items-center space-x-1">
                        <FontAwesomeIcon icon={faEye} className="w-4 h-4" />
                        <span>View</span>
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Cases;
