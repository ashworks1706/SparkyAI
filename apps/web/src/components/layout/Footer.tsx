
import { Heart } from "lucide-react";

const Footer = () => {
  const currentYear = new Date().getFullYear();

  return (
    <footer className="bg-gray-50 py-12 md:py-16">
      <div className="container mx-auto px-4 md:px-6">
        <div className="grid grid-cols-1 md:grid-cols-4 gap-8">
          <div className="md:col-span-2">
            <div className="flex items-center space-x-2 mb-4">
              <span className="text-xl font-bold text-sparky-maroon">Sparky<span className="text-sparky-gold">AI</span></span>
            </div>
            <p className="text-gray-600 max-w-md">
            An AI University assistant leveraging multi-agent architecture with real-time data integration. Combines Self improvement CRAG and Firebase analytics for resource retrieval across courses, shuttles, jobs, sports, events and much more.
            </p>
          </div>

          <div>
            <h3 className="font-medium text-gray-900 mb-4">Navigation</h3>
            <ul className="space-y-3">
              <li><a href="#features" className="text-gray-600 hover:text-sparky-maroon transition-colors">Features</a></li>
              <li><a href="#architecture" className="text-gray-600 hover:text-sparky-maroon transition-colors">Architecture</a></li>
              <li><a href="#agents" className="text-gray-600 hover:text-sparky-maroon transition-colors">Agents</a></li>
              <li><a href="#contact" className="text-gray-600 hover:text-sparky-maroon transition-colors">Contact</a></li>
            </ul>
          </div>

          <div>
            <h3 className="font-medium text-gray-900 mb-4">Resources</h3>
            <ul className="space-y-3">
              {/* <li><a href="#" className="text-gray-600 hover:text-sparky-maroon transition-colors">Documentation</a></li> */}
              <li><a href="https://github.com/somwrks/SparkyAI" className="text-gray-600 hover:text-sparky-maroon transition-colors">GitHub</a></li>
              {/* <li><a href="#" className="text-gray-600 hover:text-sparky-maroon transition-colors">Blog</a></li> */}
            </ul>
          </div>
        </div>

        <div className="border-t border-gray-200 mt-12 pt-8 flex flex-col md:flex-row justify-between items-center">
          <p className="text-gray-600 text-sm mb-4 md:mb-0">
            © {currentYear} SparkyAI. All rights reserved.
          </p>
          <div className="flex items-center space-x-1 text-sm text-gray-600">
            <span>Made with</span>
            <Heart className="h-4 w-4 text-sparky-maroon" />
            <span>by <a href="https://ashworks.dev" className="font-medium hover:text-sparky-maroon transition-colors">Ash</a></span>
          </div>
        </div>
      </div>
    </footer>
  );
};

export default Footer;
