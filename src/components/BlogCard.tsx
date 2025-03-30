
import { Link } from 'react-router-dom';
import { Calendar } from 'lucide-react';

interface BlogCardProps {
  id: string;
  title: string;
  excerpt: string;
  date: string;
  tags: string[];
  image?: string;
}

const BlogCard = ({ id, title, excerpt, date, tags, image }: BlogCardProps) => {
  return (
    <div className="bg-white border border-gray-200 rounded-lg overflow-hidden hover:shadow-md transition-all duration-300">
      {image && (
        <div className="h-48 overflow-hidden">
          <img 
            src={image} 
            alt={title} 
            className="w-full h-full object-cover transition-transform duration-500 hover:scale-105"
          />
        </div>
      )}
      
      <div className="p-5">
        <div className="flex items-center text-muted-foreground text-sm mb-3">
          <Calendar className="w-4 h-4 mr-1" />
          {date}
        </div>
        
        <Link to={`/blog/${id}`}>
          <h3 className="text-xl font-bold mb-2 hover:text-primary transition-colors">
            {title}
          </h3>
        </Link>
        
        <p className="text-muted-foreground mb-4 line-clamp-2">
          {excerpt}
        </p>
        
        <div className="flex flex-wrap gap-2">
          {tags.map((tag) => (
            <Link 
              key={tag} 
              to={`/blog?tag=${tag}`}
              className="bg-gray-100 text-gray-800 px-2 py-1 rounded-full text-xs hover:bg-gray-200 transition-colors"
            >
              {tag}
            </Link>
          ))}
        </div>
      </div>
    </div>
  );
};

export default BlogCard;
