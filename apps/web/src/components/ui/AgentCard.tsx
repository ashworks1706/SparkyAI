
import { ReactNode } from "react";

interface AgentCardProps {
  name: string;
  importance: string;
  description: string;
  functions: string[];
  icon: ReactNode;
}

const AgentCard = ({ name, importance, description, functions, icon }: AgentCardProps) => {
  return (
    <div className="rounded-2xl overflow-hidden border border-gray-100 group hover:shadow-lg transition-all duration-300 h-full flex flex-col">
      <div className="bg-gradient-to-br from-sparky-maroon to-sparky-gold/40 p-4">
        <div className="bg-white/10 backdrop-blur-md rounded-xl p-3 w-10 h-10 flex items-center justify-center text-white mb-3">
          {icon}
        </div>
        <h3 className="text-lg font-semibold text-white mb-1">{name}</h3>
        <p className="text-white/80 text-xs">{importance}</p>
      </div>
      
      <div className="p-4 bg-white flex-grow flex flex-col">
        <p className="text-gray-600 mb-3 text-sm flex-grow">{description}</p>
        
        <div>
          <h4 className="text-xs font-medium text-gray-900 mb-2">Key Functions:</h4>
          <ul className="space-y-1">
            {functions.map((func, index) => (
              <li key={index} className="text-xs text-gray-600 flex items-start">
                <span className="text-sparky-maroon mr-2">•</span>
                <span>{func}</span>
              </li>
            ))}
          </ul>
        </div>
      </div>
    </div>
  );
};

export default AgentCard;
