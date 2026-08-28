
import { ReactNode } from "react";

interface FeatureCardProps {
  icon: ReactNode;
  title: string;
  description: string;
}

const FeatureCard = ({ icon, title, description }: FeatureCardProps) => {
  return (
    <div className="group relative rounded-2xl p-5 bg-white border border-gray-100 shadow-sm hover:shadow-md transition-all duration-300 h-full flex flex-col">
      <div className="mb-3 text-sparky-maroon bg-sparky-maroon/10 w-10 h-10 rounded-xl flex items-center justify-center">
        {icon}
      </div>
      <h3 className="text-lg font-semibold mb-2 text-gray-900 group-hover:text-sparky-maroon transition-colors">
        {title}
      </h3>
      <p className="text-gray-600 text-sm flex-grow">
        {description}
      </p>
    </div>
  );
};

export default FeatureCard;
