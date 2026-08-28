
import { Brain, Database, Cpu, Zap, BarChart, Lock, GitMerge, Search, Code, Gauge } from "lucide-react";
import FeatureCard from "../ui/FeatureCard";

const Features = () => {
  const features = [
    {
      icon: <Brain className="h-6 w-6" />,
      title: "Multi-Agent System",
      description: "Utilizes specialized AI agents including Superior, RAG Search, and domain-specific agents for targeted task execution."
    },
    {
      icon: <Database className="h-6 w-6" />,
      title: "Advanced RAG Architecture",
      description: "Combines vector search with large language models for precise information retrieval and contextually relevant responses."
    },
    {
      icon: <Cpu className="h-6 w-6" />,
      title: "RAPTOR Retrieval",
      description: "Implements hierarchical document representation for nuanced information retrieval with maximum accuracy."
    },
    {
      icon: <Search className="h-6 w-6" />,
      title: "MIPS & Vector Search",
      description: "Uses Maximum Inner Product Search and advanced vector similarity techniques for efficient large-scale dataset searches."
    },
    {
      icon: <Code className="h-6 w-6" />,
      title: "Custom Embedding Model",
      description: "Leverages BAAI/bge-large-en-v1.5 model and cross-encoder reranking for high-quality text embeddings and results."
    },
    {
      icon: <Gauge className="h-6 w-6" />,
      title: "Performance Optimizations",
      description: "Features batch processing, caching mechanisms, async operations, and result deduplication for blazing-fast responses."
    },
    {
      icon: <GitMerge className="h-6 w-6" />,
      title: "Multi-Method Search",
      description: "Combines RAPTOR, similarity search, MIPS, and ScaNN for comprehensive and efficient information retrieval."
    },
    {
      icon: <Lock className="h-6 w-6" />,
      title: "FERPA Compliance",
      description: "Ensures all student data handling is compliant with Family Educational Rights and Privacy Act regulations."
    }
  ];

  return (
    <section className="py-20 bg-gray-50" id="features">
      <div className="container mx-auto px-4 md:px-6">
        <div className="text-center max-w-3xl mx-auto mb-16">
          <div className="inline-block px-3 py-1 rounded-full bg-sparky-maroon/10 text-sparky-maroon text-sm font-medium mb-4">
            Powerful Capabilities
          </div>
          <h2 className="heading-lg mb-4">Advanced Features</h2>
          <p className="text-gray-600 text-lg">
            SparkyAI combines cutting-edge technologies to deliver an exceptional campus assistant experience.
          </p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-8">
          {features.map((feature, index) => (
            <FeatureCard
              key={index}
              icon={feature.icon}
              title={feature.title}
              description={feature.description}
            />
          ))}
        </div>
      </div>
    </section>
  );
};

export default Features;
