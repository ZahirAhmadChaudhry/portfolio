
import { useState, useEffect } from 'react';
import { useLocation } from 'react-router-dom';
import Navbar from '@/components/Navbar';
import Footer from '@/components/Footer';
import BlogCard from '@/components/BlogCard';
import SearchBar from '@/components/SearchBar';
import { blogPosts } from '@/data/blog';

const Blog = () => {
  const location = useLocation();
  const [activeTag, setActiveTag] = useState<string>('all');
  const [searchQuery, setSearchQuery] = useState<string>('');
  const [filteredPosts, setFilteredPosts] = useState(blogPosts);
  
  // Get all unique tags from blog posts
  const allTags = ['all', ...Array.from(new Set(blogPosts.flatMap(post => post.tags)))];
  
  useEffect(() => {
    // Check for tag in URL query params
    const params = new URLSearchParams(location.search);
    const tagParam = params.get('tag');
    if (tagParam) {
      setActiveTag(tagParam);
    }
    
    let result = blogPosts;
    
    // Filter by tag
    if (activeTag !== 'all') {
      result = result.filter(post => 
        post.tags.includes(activeTag)
      );
    }
    
    // Filter by search query
    if (searchQuery) {
      const query = searchQuery.toLowerCase();
      result = result.filter(post => 
        post.title.toLowerCase().includes(query) || 
        post.excerpt.toLowerCase().includes(query) ||
        post.tags.some(tag => tag.toLowerCase().includes(query))
      );
    }
    
    setFilteredPosts(result);
  }, [activeTag, searchQuery, location.search]);

  const handleTagClick = (tag: string) => {
    setActiveTag(tag);
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
          <h1 className="text-4xl font-bold mb-6">Blog</h1>
          <p className="text-lg text-gray-600 max-w-3xl mx-auto mb-8">
            Thoughts, insights, and technical guides on machine learning, NLP, 
            and production-ready ML solutions.
          </p>
          
          <div className="max-w-xl mx-auto">
            <SearchBar 
              onSearch={handleSearch} 
              placeholder="Search blog posts by title, content, or tag..."
            />
          </div>
        </div>
      </section>
      
      {/* Blog Posts Section */}
      <section className="py-16 bg-white">
        <div className="max-w-7xl mx-auto px-4">
          {/* Tags */}
          <div className="flex flex-wrap justify-center gap-2 mb-12">
            {allTags.map((tag) => (
              <button
                key={tag}
                onClick={() => handleTagClick(tag)}
                className={`px-4 py-1 rounded-full text-sm font-medium transition-colors ${
                  activeTag === tag
                    ? 'bg-primary text-white'
                    : 'bg-gray-100 text-gray-600 hover:bg-gray-200'
                }`}
              >
                {tag === 'all' ? 'All Posts' : tag}
              </button>
            ))}
          </div>
          
          {filteredPosts.length > 0 ? (
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8">
              {filteredPosts.map((post) => (
                <BlogCard 
                  key={post.id}
                  id={post.id}
                  title={post.title}
                  excerpt={post.excerpt}
                  date={post.date}
                  tags={post.tags}
                  image={post.image}
                />
              ))}
            </div>
          ) : (
            <div className="text-center py-16">
              <h3 className="text-xl font-medium text-gray-800 mb-2">No blog posts found</h3>
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

export default Blog;
