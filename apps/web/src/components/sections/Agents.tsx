
import { Bot, Search, Globe, Activity, MessageSquare, Library, Briefcase, Users, Trophy } from "lucide-react";
import AgentCard from "../ui/AgentCard";

const Agents = () => {
  const agents = [
    {
      name: "Superior Agent",
      importance: "Central coordinator for user interactions",
      description: "Handles main messages, decides on direct responses or function calls, and utilizes multiple agents/functions as needed.",
      functions: [
        "Coordinate all agent interactions",
        "Process user queries",
        "Determine appropriate agent routing",
        "Synthesize final responses"
      ],
      icon: <Bot className="h-6 w-6" />
    },
    {
      name: "RAG Search Agent",
      importance: "Knowledge retrieval specialist",
      description: "Performs search over RAG knowledge base with access to general Google search utility for comprehensive queries.",
      functions: [
        "Knowledge base search",
        "Semantic query processing",
        "Google search integration",
        "Context-aware responses"
      ],
      icon: <Search className="h-6 w-6" />
    },
    {
      name: "News Media Agent",
      importance: "Information dissemination",
      description: "Access to ASU social media and news sources for the latest campus updates and announcements.",
      functions: [
        "Social media monitoring",
        "News article retrieval",
        "Content summarization",
        "Media trend analysis"
      ],
      icon: <Globe className="h-6 w-6" />
    },
    {
      name: "Shuttle Status Agent",
      importance: "Real-time transportation updates",
      description: "Provides real-time access to ASU campus rider portal for shuttle status and schedule information.",
      functions: [
        "Live shuttle tracking",
        "Route information",
        "Schedule updates",
        "Delay notifications"
      ],
      icon: <Activity className="h-6 w-6" />
    },
    {
      name: "Discord Agent",
      importance: "Community engagement manager",
      description: "Manages Discord channels including announcements, feedback, customer service, polls, and posts.",
      functions: [
        "Channel management",
        "Announcement creation",
        "Poll generation",
        "Forum post coordination"
      ],
      icon: <MessageSquare className="h-6 w-6" />
    },
    {
      name: "Library Agent",
      importance: "Academic resource coordinator",
      description: "Connects to library.asu resources and provides live room status information for study spaces.",
      functions: [
        "Book availability checks",
        "Study room status",
        "Resource recommendations",
        "Library hours information"
      ],
      icon: <Library className="h-6 w-6" />
    },
    {
      name: "Sports Agent",
      importance: "Athletics information specialist",
      description: "Provides access to sundevilathletics.asu for sports schedules, news, and ticket information.",
      functions: [
        "Game schedules",
        "Team statistics",
        "Event information",
        "Ticket availability"
      ],
      icon: <Trophy className="h-6 w-6" />
    },
    {
      name: "Student Jobs Agent",
      importance: "Career opportunities manager",
      description: "Connects to workday.asu to provide information about student employment opportunities.",
      functions: [
        "Job listing retrieval",
        "Application status updates",
        "Career event information",
        "Resume assistance"
      ],
      icon: <Briefcase className="h-6 w-6" />
    },
    {
      name: "Student Clubs Events Agent",
      importance: "Campus engagement facilitator",
      description: "Accesses sundevilsync.asu and scholarships.asu to provide information on clubs, events, and funding opportunities.",
      functions: [
        "Club information retrieval",
        "Event calendar access",
        "Scholarship listings",
        "Application deadline reminders"
      ],
      icon: <Users className="h-6 w-6" />
    }
  ];

  return (
    <section className="py-20" id="agents">
      <div className="container mx-auto px-4 md:px-6">
        <div className="text-center max-w-3xl mx-auto mb-16">
          <div className="inline-block px-3 py-1 rounded-full bg-sparky-maroon/10 text-sparky-maroon text-sm font-medium mb-4">
            Intelligent Coordination
          </div>
          <h2 className="heading-lg mb-4">Multi-Agent System</h2>
          <p className="text-gray-600 text-lg">
            SparkyAI leverages a sophisticated multi-agent architecture where specialized agents work together to deliver comprehensive assistance.
          </p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8">
          {agents.map((agent, index) => (
            <AgentCard
              key={index}
              name={agent.name}
              importance={agent.importance}
              description={agent.description}
              functions={agent.functions}
              icon={agent.icon}
            />
          ))}
        </div>
      </div>
    </section>
  );
};

export default Agents;
