
import { useState, useEffect } from 'react';
import Navbar from '@/components/Navbar';
import Footer from '@/components/Footer';
import ProjectCard from '@/components/ProjectCard';
import SearchBar from '@/components/SearchBar';
import { projects } from '@/data/projects';

const Projects = () => {
  const [activeCategory, setActiveCategory] = useState<string>('all');
  const [searchQuery, setSearchQuery] = useState<string>('');
  const [filteredProjects, setFilteredProjects] = useState(projects);
  
  const categories = [
    { id: 'all', name: 'All Projects' },
    { id: 'classic ml', name: 'Classic ML' },
    { id: 'computer vision', name: 'Computer Vision' },
    { id: 'nlp', name: 'NLP' },
    { id: 'genai', name: 'GenAI' },
  ];

  useEffect(() => {
    let result = projects;
    
    // Filter by category
    if (activeCategory !== 'all') {
      result = result.filter(project => 
        project.category.toLowerCase() === activeCategory.toLowerCase()
      );
    }
    
    // Filter by search query
    if (searchQuery) {
      const query = searchQuery.toLowerCase();
      result = result.filter(project => 
        project.title.toLowerCase().includes(query) || 
        project.description.toLowerCase().includes(query) ||
        project.tags.some(tag => tag.toLowerCase().includes(query))
      );
    }
    
    setFilteredProjects(result);
  }, [activeCategory, searchQuery]);

  const handleCategoryChange = (category: string) => {
    setActiveCategory(category);
  };

  const handleSearch = (query: string) => {
    setSearchQuery(query);
  };

  return (
    <>
      <Navbar />
      
      {/* Header Section */}
      <section className="pt-28 pb-16 px-4 grid-pattern">
        <div className="max-w-5xl mx-auto text-center">
          <h1 className="text-4xl font-bold mb-6">My Projects</h1>
          <p className="text-lg text-gray-600 max-w-3xl mx-auto mb-8">
            A collection of machine learning projects showcasing my expertise in various domains,
            from classic ML algorithms to cutting-edge generative AI and NLP solutions.
          </p>
          
          <div className="max-w-xl mx-auto">
            <SearchBar 
              onSearch={handleSearch} 
              placeholder="Search projects by name, description, or technology..."
            />
          </div>
        </div>
      </section>
      
      {/* Projects Section */}
      <section className="py-16 bg-white">
        <div className="max-w-7xl mx-auto px-4">
          {/* Category Tabs */}
          <div className="flex flex-wrap justify-center gap-2 mb-12">
            {categories.map((category) => (
              <button
                key={category.id}
                onClick={() => handleCategoryChange(category.id)}
                className={`px-5 py-2 rounded-full text-sm font-medium transition-colors ${
                  activeCategory === category.id
                    ? 'bg-primary text-white'
                    : 'bg-gray-100 text-gray-600 hover:bg-gray-200'
                }`}
              >
                {category.name}
              </button>
            ))}
          </div>
          
          {filteredProjects.length > 0 ? (
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8">
              {filteredProjects.map((project) => (
                <ProjectCard 
                  key={project.id}
                  id={project.id}
                  title={project.title}
                  description={project.description}
                  image={project.image}
                  category={project.category}
                  tags={project.tags}
                  github={project.github}
                />
              ))}
            </div>
          ) : (
            <div className="text-center py-16">
              <h3 className="text-xl font-medium text-gray-800 mb-2">No projects found</h3>
              <p className="text-gray-500">
                Try adjusting your search or filter criteria to find what you're looking for.
              </p>
            </div>
          )}
        </div>
      </section>
      
      <Footer />
    </>
  );
};

export default Projects;
