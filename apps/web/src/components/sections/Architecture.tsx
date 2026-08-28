
import { useState } from "react";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";

const Architecture = () => {
  const [activeTab, setActiveTab] = useState("overview");

  return (
    <section className="py-20 bg-gray-50" id="architecture">
      <div className="container mx-auto px-4 md:px-6">
        <div className="text-center max-w-3xl mx-auto mb-16">
          <div className="inline-block px-3 py-1 rounded-full bg-sparky-maroon/10 text-sparky-maroon text-sm font-medium mb-4">
            Technical Design
          </div>
          <h2 className="heading-lg mb-4">Advanced Architecture</h2>
          <p className="text-gray-600 text-lg">
            SparkyAI's sophisticated architecture combines multiple technologies to deliver an exceptional AI assistant experience.
          </p>
        </div>

        <div className="max-w-5xl mx-auto">
          <Tabs defaultValue="overview" className="w-full" onValueChange={setActiveTab}>
            <TabsList className="grid w-full grid-cols-2 md:grid-cols-4 mb-8">
              <TabsTrigger value="overview" className="text-sm md:text-base">Overview</TabsTrigger>
              <TabsTrigger value="rag" className="text-sm md:text-base">RAG System</TabsTrigger>
              <TabsTrigger value="agents" className="text-sm md:text-base">Agent Flow</TabsTrigger>
              <TabsTrigger value="technologies" className="text-sm md:text-base">Technologies</TabsTrigger>
            </TabsList>
            
            <div className="bg-white rounded-2xl shadow-sm overflow-hidden border border-gray-100">
              <TabsContent value="overview" className="p-0 m-0">
                <div className="aspect-video bg-gray-100 flex items-center justify-center">
                  <div className="text-center p-6 md:p-10">
                    <h3 className="text-xl font-semibold mb-4">System Architecture Overview</h3>
                    <p className="text-gray-600 mb-4">Multi-agent system diagram combining RAG architecture with specialized task processing.</p>
                    <div className="relative w-full max-w-3xl mx-auto rounded-lg overflow-hidden shadow-md">
                      <img 
                        src="https://github.com/user-attachments/assets/642fd6d6-5232-4347-b1dc-3e78d3d0c758" 
                        alt="SparkyAI Architecture Diagram" 
                        className="w-full h-auto"
                      />
                    </div>
                  </div>
                </div>
              </TabsContent>
              
              <TabsContent value="rag" className="p-0 m-0">
                <div className="aspect-video bg-gray-100 flex items-center justify-center">
                  <div className="text-center p-6 md:p-10">
                    <h3 className="text-xl font-semibold mb-4">RAG System Architecture</h3>
                    <p className="text-gray-600 mb-4">Retrieval-Augmented Generation system with vector database integration.</p>
                    <div className="relative w-full max-w-3xl mx-auto rounded-lg overflow-hidden shadow-md">
                      <img 
                        src="https://github.com/user-attachments/assets/18e2c5fc-b777-4b22-b63e-5bd97ffde5cd" 
                        alt="RAG System Architecture" 
                        className="w-full h-auto"
                      />
                    </div>
                  </div>
                </div>
              </TabsContent>
              
              <TabsContent value="agents" className="p-0 m-0">
                <div className="aspect-video bg-gray-100 flex items-center justify-center">
                  <div className="text-center p-6 md:p-10">
                    <h3 className="text-xl font-semibold mb-4">Agent Interaction Flow</h3>
                    <p className="text-gray-600 mb-4">How different agents interact to process user requests and retrieve information.</p>
                    <div className="relative w-full max-w-3xl mx-auto rounded-lg overflow-hidden shadow-md">
                      <img 
                        src="https://github.com/user-attachments/assets/c2228237-c7e2-4f50-84e4-30ca07c7d2f0" 
                        alt="Agent Interaction Flow" 
                        className="w-full h-auto"
                      />
                    </div>
                  </div>
                </div>
              </TabsContent>
              
              <TabsContent value="technologies" className="p-0 m-0">
                <div className="aspect-video bg-gray-100 flex items-center justify-center">
                  <div className="text-center p-6 md:p-10">
                    <h3 className="text-xl font-semibold mb-4">Technology Stack</h3>
                    <p className="text-gray-600 mb-4">The cutting-edge technologies that power SparkyAI's capabilities.</p>
                    <div className="relative w-full max-w-3xl mx-auto">
                      <div className="grid grid-cols-2 md:grid-cols-3 gap-4 mt-6">
                        <div className="bg-white p-4 rounded-lg shadow-sm border border-gray-100">
                          <h4 className="font-medium text-sm mb-2">AI/ML</h4>
                          <p className="text-xs text-gray-600">Gemini, LangChain, TensorFlow</p>
                        </div>
                        <div className="bg-white p-4 rounded-lg shadow-sm border border-gray-100">
                          <h4 className="font-medium text-sm mb-2">NLP</h4>
                          <p className="text-xs text-gray-600">Hugging Face Transformers, NLTK</p>
                        </div>
                        <div className="bg-white p-4 rounded-lg shadow-sm border border-gray-100">
                          <h4 className="font-medium text-sm mb-2">Embeddings</h4>
                          <p className="text-xs text-gray-600">BAAI/bge-large-en-v1.5 model</p>
                        </div>
                        <div className="bg-white p-4 rounded-lg shadow-sm border border-gray-100">
                          <h4 className="font-medium text-sm mb-2">Cross Encoder</h4>
                          <p className="text-xs text-gray-600">cross-encoder/ms-marco-MiniLM-L-6-v2</p>
                        </div>
                        <div className="bg-white p-4 rounded-lg shadow-sm border border-gray-100">
                          <h4 className="font-medium text-sm mb-2">Vector Search</h4>
                          <p className="text-xs text-gray-600">Qdrant</p>
                        </div>
                        <div className="bg-white p-4 rounded-lg shadow-sm border border-gray-100">
                          <h4 className="font-medium text-sm mb-2">Web Scraping</h4>
                          <p className="text-xs text-gray-600">Selenium, BeautifulSoup4</p>
                        </div>
                        <div className="bg-white p-4 rounded-lg shadow-sm border border-gray-100">
                          <h4 className="font-medium text-sm mb-2">APIs</h4>
                          <p className="text-xs text-gray-600">Discord API, Firebase API</p>
                        </div>
                        <div className="bg-white p-4 rounded-lg shadow-sm border border-gray-100">
                          <h4 className="font-medium text-sm mb-2">Databases</h4>
                          <p className="text-xs text-gray-600">Firestore, Notion</p>
                        </div>
                        <div className="bg-white p-4 rounded-lg shadow-sm border border-gray-100">
                          <h4 className="font-medium text-sm mb-2">Other</h4>
                          <p className="text-xs text-gray-600">Asyncio, Docker</p>
                        </div>
                      </div>
                    </div>
                  </div>
                </div>
              </TabsContent>
            </div>
          </Tabs>
        </div>
      </div>
    </section>
  );
};

export default Architecture;
