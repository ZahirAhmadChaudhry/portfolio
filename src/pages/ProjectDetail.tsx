
import { useEffect, useState } from 'react';
import { useParams, Link } from 'react-router-dom';
import Navbar from '@/components/Navbar';
import Footer from '@/components/Footer';
import { ArrowLeft, Github, ExternalLink } from 'lucide-react';
import { projects } from '@/data/projects';

const ProjectDetail = () => {
  const { id } = useParams<{ id: string }>();
  const [project, setProject] = useState<any | null>(null);
  
  useEffect(() => {
    if (id) {
      const foundProject = projects.find(p => p.id === id);
      setProject(foundProject || null);
      
      // Scroll to top when project changes
      window.scrollTo(0, 0);
    }
  }, [id]);
  
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
      
      {/* Header Section */}
      <section className="pt-28 pb-12 px-4">
        <div className="max-w-5xl mx-auto">
          <Link 
            to="/projects" 
            className="inline-flex items-center text-gray-600 hover:text-primary mb-6"
          >
            <ArrowLeft className="mr-2 h-4 w-4" />
            Back to Projects
          </Link>
          
          <div className="bg-white rounded-xl overflow-hidden border border-gray-200 shadow-sm">
            <div className="h-72 sm:h-96 overflow-hidden">
              <img 
                src={project.image} 
                alt={project.title} 
                className="w-full h-full object-cover" 
              />
            </div>
            
            <div className="p-6 sm:p-8">
              <div className="flex flex-wrap justify-between items-start mb-6">
                <div>
                  <h1 className="text-3xl font-bold mb-2">{project.title}</h1>
                  <div className={`inline-block px-3 py-1 rounded-full text-sm font-medium ${
                    project.category === 'Classic ML' ? 'bg-tech-green/10 text-tech-green' :
                    project.category === 'Computer Vision' ? 'bg-tech-blue/10 text-tech-blue' :
                    project.category === 'NLP' ? 'bg-tech-purple/10 text-tech-purple' :
                    'bg-tech-red/10 text-tech-red'
                  }`}>
                    {project.category}
                  </div>
                </div>
                
                <div className="flex space-x-3 mt-2 sm:mt-0">
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
              </div>
              
              <div className="flex flex-wrap gap-2 mb-8">
                {project.tags.map((tag: string) => (
                  <span key={tag} className="bg-gray-100 text-gray-800 px-2 py-1 rounded-full text-xs">
                    {tag}
                  </span>
                ))}
              </div>
              
              <div className="prose prose-blue max-w-none">
                <h2>Overview</h2>
                <p>{project.fullDescription || project.description}</p>
                
                {project.challenges && (
                  <>
                    <h2>Challenges</h2>
                    <ul>
                      {project.challenges.map((challenge: string, index: number) => (
                        <li key={index}>{challenge}</li>
                      ))}
                    </ul>
                  </>
                )}
                
                {project.solutions && (
                  <>
                    <h2>Solutions & Approach</h2>
                    <ul>
                      {project.solutions.map((solution: string, index: number) => (
                        <li key={index}>{solution}</li>
                      ))}
                    </ul>
                  </>
                )}
                
                {project.achievements && (
                  <>
                    <h2>Achievements</h2>
                    <ul>
                      {project.achievements.map((achievement: string, index: number) => (
                        <li key={index}>{achievement}</li>
                      ))}
                    </ul>
                  </>
                )}
                
                {project.visualization && (
                  <>
                    <h2>Results Visualization</h2>
                    <div className="bg-gray-50 p-4 rounded-lg border border-gray-200 flex justify-center">
                      <img 
                        src={project.visualization} 
                        alt="Project Results" 
                        className="max-w-full h-auto rounded" 
                      />
                    </div>
                  </>
                )}
                
                {project.codeSnippet && (
                  <>
                    <h2>Code Snippet</h2>
                    <div className="github-snippet">
                      <pre>{project.codeSnippet}</pre>
                    </div>
                  </>
                )}
                
                {project.technologies && (
                  <>
                    <h2>Technologies Used</h2>
                    <ul>
                      {project.technologies.map((tech: string, index: number) => (
                        <li key={index}>{tech}</li>
                      ))}
                    </ul>
                  </>
                )}
              </div>
            </div>
          </div>
        </div>
      </section>
      
      <Footer />
    </>
  );
};

export default ProjectDetail;
