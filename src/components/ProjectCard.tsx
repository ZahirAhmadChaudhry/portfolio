
import { Link } from 'react-router-dom';
import { ArrowUpRight, Github } from 'lucide-react';

interface ProjectCardProps {
  id: string;
  title: string;
  description: string;
  image: string;
  category: string;
  tags: string[];
  github?: string;
}

const ProjectCard = ({
  id,
  title,
  description,
  image,
  category,
  tags,
  github
}: ProjectCardProps) => {
  const getCategoryColor = (category: string) => {
    switch (category.toLowerCase()) {
      case 'classic ml':
        return 'bg-tech-green/10 text-tech-green';
      case 'computer vision':
        return 'bg-tech-blue/10 text-tech-blue';
      case 'nlp':
        return 'bg-tech-purple/10 text-tech-purple';
      case 'genai':
        return 'bg-tech-red/10 text-tech-red';
      default:
        return 'bg-gray-100 text-gray-800';
    }
  };

  return (
    <div className="group bg-white border border-gray-200 rounded-lg overflow-hidden hover:shadow-md transition-all duration-300">
      <div className="relative h-48 overflow-hidden">
        <img 
          src={image} 
          alt={title} 
          className="w-full h-full object-cover transition-transform duration-500 group-hover:scale-105"
        />
        <div className="absolute inset-0 bg-gradient-to-t from-black/50 to-transparent opacity-0 group-hover:opacity-100 transition-opacity duration-300" />
        <span className={`absolute top-3 left-3 ${getCategoryColor(category)} px-2 py-1 rounded-full text-xs font-medium`}>
          {category}
        </span>
      </div>
      
      <div className="p-5">
        <h3 className="text-xl font-bold mb-2 group-hover:text-primary transition-colors">
          {title}
        </h3>
        
        <p className="text-muted-foreground mb-4 line-clamp-2">
          {description}
        </p>
        
        <div className="flex flex-wrap gap-2 mb-4">
          {tags.slice(0, 3).map((tag) => (
            <span key={tag} className="bg-gray-100 text-gray-800 px-2 py-1 rounded-full text-xs">
              {tag}
            </span>
          ))}
          {tags.length > 3 && (
            <span className="bg-gray-100 text-gray-800 px-2 py-1 rounded-full text-xs">
              +{tags.length - 3}
            </span>
          )}
        </div>
        
        <div className="flex items-center justify-between">
          <Link
            to={`/projects/${id}`}
            className="text-primary font-medium flex items-center hover:underline"
          >
            View Details
            <ArrowUpRight className="ml-1 w-4 h-4" />
          </Link>
          
          {github && (
            <a
              href={github}
              target="_blank"
              rel="noopener noreferrer"
              className="text-gray-600 hover:text-gray-900"
              aria-label="GitHub repository"
            >
              <Github className="w-5 h-5" />
            </a>
          )}
        </div>
      </div>
    </div>
  );
};

export default ProjectCard;
