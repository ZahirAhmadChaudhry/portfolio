
import { ReactNode } from 'react';

interface StatCardProps {
  title: string;
  value: string;
  icon: ReactNode;
  color?: string;
}

const StatCard = ({ title, value, icon, color = 'bg-blue-50 text-tech-blue' }: StatCardProps) => {
  return (
    <div className="bg-white rounded-lg shadow-sm p-6 border border-gray-100 hover:shadow-md transition-shadow">
      <div className="flex items-start">
        <div className={`${color} p-3 rounded-full mr-4`}>
          {icon}
        </div>
        <div>
          <p className="text-gray-500 text-sm">{title}</p>
          <h3 className="text-2xl font-bold mt-1">{value}</h3>
        </div>
      </div>
    </div>
  );
};

export default StatCard;
