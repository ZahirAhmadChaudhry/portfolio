import { useEffect, useState } from 'react';
import { useParams, Link } from 'react-router-dom';
import Navbar from '@/components/Navbar';
import Footer from '@/components/Footer';
import { ArrowLeft, Github, ExternalLink, ChevronRight } from 'lucide-react';
import { projects } from '@/data/projects';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';

// Visualization component to render different types of visualizations
const Visualization = ({ data }: { data: any }) => {
  if (!data) return null;

  switch (data.type) {
    case 'image':
      return (
        <img 
          src={data.content} 
          alt={data.alt || 'Project visualization'} 
          className="max-w-full h-auto rounded-md"
        />
      );
    case 'svg':
      return (
        <div 
          className="svg-container" 
          dangerouslySetInnerHTML={{ __html: data.content }}
        />
      );
    case 'mermaid':
      // For mermaid diagrams, we'll use a placeholder that gets processed by client-side JS
      return (
        <div className="mermaid-diagram">
          <pre className="mermaid">
            {data.content}
          </pre>
        </div>
      );
    default:
      return <p>Unsupported visualization type</p>;
  }
};

const ProjectDetail = () => {
  const { id } = useParams<{ id: string }>();
  const [project, setProject] = useState<any | null>(null);
  
  useEffect(() => {
    if (id) {
      const foundProject = projects.find(p => p.id === id);
      setProject(foundProject || null);
      
      // Scroll to top when project changes
      window.scrollTo(0, 0);
      
      // Initialize mermaid if needed and available
      if (foundProject?.visualizations?.some(v => v.type === 'mermaid') && window.mermaid) {
        try {
          window.mermaid.init(undefined, document.querySelectorAll('.mermaid'));
        } catch (error) {
          console.error('Mermaid initialization failed:', error);
        }
      }
    }
  }, [id]);

  // Second useEffect specifically for mermaid rendering
  // This helps when the DOM is fully updated with mermaid content
  useEffect(() => {
    const timer = setTimeout(() => {
      if (window.mermaid && document.querySelectorAll('.mermaid').length > 0) {
        try {
          window.mermaid.contentLoaded();
        } catch (error) {
          console.error('Mermaid content loading failed:', error);
        }
      }
    }, 100);
    
    return () => clearTimeout(timer);
  }, [project]);
  
  if (!project) {
    return (
      <>
        <Navbar />
        <div className="pt-28 pb-16 px-4 min-h-screen flex items-center justify-center">
          <div className="text-center">
            <h2 className="text-2xl font-bold mb-4">Project Not Found</h2>
            <p className="text-gray-600 mb-6">The project you're looking for doesn't exist or has been moved.</p>
            <Link 
              to="/projects" 
              className="inline-flex items-center text-primary hover:underline"
            >
              <ArrowLeft className="mr-2 h-4 w-4" />
              Back to Projects
            </Link>
          </div>
        </div>
        <Footer />
      </>
    );
  }

  return (
    <>
      <Navbar />
      
      {/* Header Section with Breadcrumb */}
      <section className="pt-28 pb-12 px-4 bg-gray-50">
        <div className="max-w-5xl mx-auto">
          <div className="flex items-center text-sm text-gray-500 mb-6">
            <Link to="/" className="hover:text-primary">Home</Link>
            <ChevronRight className="h-4 w-4 mx-2" />
            <Link to="/projects" className="hover:text-primary">Projects</Link>
            <ChevronRight className="h-4 w-4 mx-2" />
            <span className="text-gray-900 font-medium">{project.title}</span>
          </div>
          
          <div className="bg-white rounded-xl overflow-hidden border border-gray-200 shadow-sm">
            {/* Project Hero Section */}
            <div className="h-72 sm:h-96 overflow-hidden relative">
              <img 
                src={project.image} 
                alt={project.title} 
                className="w-full h-full object-cover" 
              />
              <div className="absolute inset-0 bg-gradient-to-t from-black/60 to-transparent flex items-end">
                <div className="p-6 sm:p-8 text-white">
                  <div className={`inline-block px-3 py-1 rounded-full text-sm font-medium mb-3 ${
                    project.category === 'Classic ML' ? 'bg-tech-green/80 text-white' :
                    project.category === 'Computer Vision' ? 'bg-tech-blue/80 text-white' :
                    project.category === 'NLP' ? 'bg-tech-purple/80 text-white' :
                    'bg-tech-red/80 text-white'
                  }`}>
                    {project.category}
                  </div>
                  <h1 className="text-3xl sm:text-4xl font-bold">{project.title}</h1>
                </div>
              </div>
            </div>
            
            <div className="p-6 sm:p-8">
              {/* Action Buttons */}
              <div className="flex flex-wrap justify-end items-center gap-3 mb-6">
                {project.github && (
                  <a
                    href={project.github}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="inline-flex items-center px-4 py-2 border border-gray-300 rounded-lg text-sm font-medium text-gray-700 hover:bg-gray-50 transition-colors"
                  >
                    <Github className="mr-2 h-4 w-4" />
                    GitHub Repo
                  </a>
                )}
                {project.liveDemo && (
                  <a
                    href={project.liveDemo}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="inline-flex items-center px-4 py-2 bg-primary text-white rounded-lg text-sm font-medium hover:bg-primary/90 transition-colors"
                  >
                    <ExternalLink className="mr-2 h-4 w-4" />
                    Live Demo
                  </a>
                )}
              </div>
              
              {/* Tags */}
              <div className="flex flex-wrap gap-2 mb-8">
                {project.tags.map((tag: string) => (
                  <span key={tag} className="bg-gray-100 text-gray-800 px-3 py-1 rounded-full text-xs font-medium">
                    {tag}
                  </span>
                ))}
              </div>
              
              {/* Content Tabs */}
              <Tabs defaultValue="overview" className="w-full">
                <TabsList className="w-full justify-start mb-6 border-b pb-0">
                  <TabsTrigger value="overview" className="rounded-t-lg rounded-b-none">Overview</TabsTrigger>
                  {(project.challenges && project.challenges.length > 0) && (
                    <TabsTrigger value="challenges" className="rounded-t-lg rounded-b-none">Challenges</TabsTrigger>
                  )}
                  {(project.visualizations && project.visualizations.length > 0) && (
                    <TabsTrigger value="visualizations" className="rounded-t-lg rounded-b-none">Visualizations</TabsTrigger>
                  )}
                  {project.codeSnippet && (
                    <TabsTrigger value="code" className="rounded-t-lg rounded-b-none">Code</TabsTrigger>
                  )}
                </TabsList>
                
                <TabsContent value="overview" className="prose prose-blue max-w-none mt-4">
                  <h2 className="text-2xl font-bold mb-4">Project Overview</h2>
                  <p className="text-gray-700">{project.fullDescription || project.description}</p>
                  
                  {(project.solutions && project.solutions.length > 0) && (
                    <>
                      <h3 className="text-xl font-bold mt-8 mb-3">Solutions & Approach</h3>
                      <ul className="space-y-2">
                        {project.solutions.map((solution: string, index: number) => (
                          <li key={index} className="flex items-start">
                            <span className="inline-flex items-center justify-center w-6 h-6 rounded-full bg-primary/10 text-primary text-sm font-bold mr-3 flex-shrink-0 mt-0.5">
                              {index + 1}
                            </span>
                            <span>{solution}</span>
                          </li>
                        ))}
                      </ul>
                    </>
                  )}
                  
                  {(project.achievements && project.achievements.length > 0) && (
                    <>
                      <h3 className="text-xl font-bold mt-8 mb-3">Key Achievements</h3>
                      <ul className="space-y-2">
                        {project.achievements.map((achievement: string, index: number) => (
                          <li key={index} className="flex items-start">
                            <span className="inline-block w-2 h-2 rounded-full bg-green-500 mr-3 flex-shrink-0 mt-2"></span>
                            <span>{achievement}</span>
                          </li>
                        ))}
                      </ul>
                    </>
                  )}
                  
                  {(project.technologies && project.technologies.length > 0) && (
                    <>
                      <h3 className="text-xl font-bold mt-8 mb-3">Technologies Used</h3>
                      <div className="flex flex-wrap gap-2">
                        {project.technologies.map((tech: string, index: number) => (
                          <span key={index} className="px-3 py-1.5 bg-gray-100 text-gray-800 rounded-md text-sm font-medium">
                            {tech}
                          </span>
                        ))}
                      </div>
                    </>
                  )}
                </TabsContent>
                
                {(project.challenges && project.challenges.length > 0) && (
                  <TabsContent value="challenges" className="prose prose-blue max-w-none mt-4">
                    <h2 className="text-2xl font-bold mb-4">Challenges Faced</h2>
                    <div className="space-y-6">
                      {project.challenges.map((challenge: string, index: number) => (
                        <div key={index} className="bg-gray-50 p-5 rounded-lg border border-gray-200">
                          <h3 className="text-lg font-medium mb-2">Challenge {index + 1}</h3>
                          <p className="text-gray-700">{challenge}</p>
                        </div>
                      ))}
                    </div>
                  </TabsContent>
                )}
                
                {(project.visualizations && project.visualizations.length > 0) && (
                  <TabsContent value="visualizations" className="prose prose-blue max-w-none mt-4">
                    <h2 className="text-2xl font-bold mb-4">Project Visualizations</h2>
                    <div className="grid grid-cols-1 gap-8">
                      {project.visualizations.map((vis: any, index: number) => (
                        <div key={index} className="bg-gray-50 p-5 rounded-lg border border-gray-200">
                          {vis.title && <h3 className="text-lg font-medium mb-3">{vis.title}</h3>}
                          {vis.description && <p className="mb-4 text-gray-700">{vis.description}</p>}
                          <div className="bg-white p-4 rounded border border-gray-200">
                            <Visualization data={vis} />
                          </div>
                        </div>
                      ))}
                    </div>
                  </TabsContent>
                )}
                
                {project.codeSnippet && (
                  <TabsContent value="code" className="prose prose-blue max-w-none mt-4">
                    <h2 className="text-2xl font-bold mb-4">Code Snippet</h2>
                    <div className="bg-gray-900 text-gray-100 p-5 rounded-lg overflow-x-auto">
                      <pre className="whitespace-pre-wrap">
                        <code>{project.codeSnippet}</code>
                      </pre>
                    </div>
                  </TabsContent>
                )}
              </Tabs>
            </div>
          </div>
        </div>
      </section>
      
      <Footer />
    </>
  );
};

export default ProjectDetail;
