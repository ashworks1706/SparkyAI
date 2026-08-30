
import { Github, Linkedin, ExternalLink } from "lucide-react";

const Contact = () => {
  return (
    <section className="py-20" id="contact">
      <div className="container mx-auto px-4 md:px-6">
        <div className="mx-auto max-w-3xl">
          <div>
            <div className="inline-block px-3 py-1 rounded-full bg-sparky-maroon/10 text-sparky-maroon text-sm font-medium mb-4">
              Get In Touch
            </div>
            <h2 className="heading-lg mb-6">Project Links</h2>
            <p className="text-gray-600 text-lg mb-8 max-w-lg">
              Have questions or want to learn more about SparkyAI? Reach out to us directly or check out our resources.
            </p>
            
            <div className="space-y-6 mb-8">
              <div className="flex items-center space-x-3">
                <div className="bg-sparky-maroon/10 p-2 rounded-full">
                  <Github className="h-5 w-5 text-sparky-maroon" />
                </div>
                <a 
                  href="https://github.com/somwrks/SparkyAI" 
                  className="text-gray-700 hover:text-sparky-maroon flex items-center space-x-1"
                  target="_blank"
                  rel="noopener noreferrer"
                >
                  <span>GitHub Repository</span>
                  <ExternalLink className="h-3 w-3" />
                </a>
              </div>
              
              <div className="flex items-center space-x-3">
                <div className="bg-sparky-maroon/10 p-2 rounded-full">
                  <Linkedin className="h-5 w-5 text-sparky-maroon" />
                </div>
                <a 
                  href="https://linkedin.com/in/ashworks" 
                  className="text-gray-700 hover:text-sparky-maroon flex items-center space-x-1"
                  target="_blank"
                  rel="noopener noreferrer"
                >
                  <span>Connect on LinkedIn</span>
                  <ExternalLink className="h-3 w-3" />
                </a>
              </div>
            </div>
            
            <div className="p-6 bg-gray-50 rounded-2xl border border-gray-100">
              <h3 className="text-lg font-medium mb-2">About the Author</h3>
              <p className="text-gray-600 mb-4">
                SparkyAI was developed by Ash, dedicated to creating intelligent solutions for campus environments.
              </p>
              <a 
                href="https://ashworks.dev" 
                className="text-sparky-maroon hover:text-sparky-maroon/80 font-medium flex items-center space-x-1"
                target="_blank"
                rel="noopener noreferrer"
              >
                <span>Visit Portfolio</span>
                <ExternalLink className="h-3 w-3" />
              </a>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
};

export default Contact;
