
import { useEffect, useState } from 'react';
import { useParams, Link } from 'react-router-dom';
import Navbar from '@/components/Navbar';
import Footer from '@/components/Footer';
import { ArrowLeft, Calendar, User, Clock } from 'lucide-react';
import { blogPosts } from '@/data/blog';

const BlogPost = () => {
  const { id } = useParams<{ id: string }>();
  const [post, setPost] = useState<any | null>(null);
  
  useEffect(() => {
    if (id) {
      const foundPost = blogPosts.find(p => p.id === id);
      setPost(foundPost || null);
      
      // Scroll to top when post changes
      window.scrollTo(0, 0);
    }
  }, [id]);
  
  if (!post) {
    return (
      <>
        <Navbar />
        <div className="pt-28 pb-16 px-4 min-h-screen flex items-center justify-center">
          <div className="text-center">
            <h2 className="text-2xl font-bold mb-4">Blog Post Not Found</h2>
            <p className="text-gray-600 mb-6">The post you're looking for doesn't exist or has been moved.</p>
            <Link 
              to="/blog" 
              className="inline-flex items-center text-primary hover:underline"
            >
              <ArrowLeft className="mr-2 h-4 w-4" />
              Back to Blog
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
        <div className="max-w-3xl mx-auto">
          <Link 
            to="/blog" 
            className="inline-flex items-center text-gray-600 hover:text-primary mb-6"
          >
            <ArrowLeft className="mr-2 h-4 w-4" />
            Back to Blog
          </Link>
          
          <article className="bg-white rounded-xl overflow-hidden border border-gray-200 shadow-sm">
            {post.image && (
              <div className="h-72 sm:h-96 overflow-hidden">
                <img 
                  src={post.image} 
                  alt={post.title} 
                  className="w-full h-full object-cover" 
                />
              </div>
            )}
            
            <div className="p-6 sm:p-8">
              <div className="mb-6">
                <h1 className="text-3xl font-bold mb-4">{post.title}</h1>
                
                <div className="flex flex-wrap items-center text-sm text-gray-500 gap-y-2">
                  <div className="flex items-center mr-4">
                    <Calendar className="h-4 w-4 mr-1" />
                    {post.date}
                  </div>
                  
                  <div className="flex items-center mr-4">
                    <User className="h-4 w-4 mr-1" />
                    Zahir Ahmad
                  </div>
                  
                  <div className="flex items-center">
                    <Clock className="h-4 w-4 mr-1" />
                    {post.readTime || '5 min read'}
                  </div>
                </div>
              </div>
              
              <div className="flex flex-wrap gap-2 mb-8">
                {post.tags.map((tag: string) => (
                  <Link 
                    key={tag} 
                    to={`/blog?tag=${tag}`}
                    className="bg-gray-100 text-gray-800 px-2 py-1 rounded-full text-xs hover:bg-gray-200 transition-colors"
                  >
                    {tag}
                  </Link>
                ))}
              </div>
              
              <div className="prose prose-blue max-w-none">
                {/* This would be the full blog post content */}
                <p className="lead">{post.excerpt}</p>
                
                {post.content && (
                  <div dangerouslySetInnerHTML={{ __html: post.content }} />
                )}
                
                {!post.content && (
                  <>
                    <h2>Introduction</h2>
                    <p>
                      This is a placeholder for the full blog post content. In a real implementation, 
                      this would be replaced with the actual content of the blog post, which could be 
                      stored in a CMS or as markdown files.
                    </p>
                    
                    <h2>Main Content</h2>
                    <p>
                      Lorem ipsum dolor sit amet, consectetur adipiscing elit. Donec euismod, nisl eget
                      consectetur sagittis, nisl nunc consectetur nisi, euismod aliquam nisi nisl euismod.
                      Donec euismod, nisl eget consectetur sagittis, nisl nunc consectetur nisi, euismod
                      aliquam nisi nisl euismod.
                    </p>
                    
                    <p>
                      Donec euismod, nisl eget consectetur sagittis, nisl nunc consectetur nisi, euismod
                      aliquam nisi nisl euismod. Donec euismod, nisl eget consectetur sagittis, nisl nunc
                      consectetur nisi, euismod aliquam nisi nisl euismod.
                    </p>
                    
                    <h2>Code Example</h2>
                    <div className="github-snippet">
                      <pre>{`# Example Python code for ML model training
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Create a random forest classifier
clf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)

# Train the model
clf.fit(X_train, y_train)

# Make predictions
predictions = clf.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, predictions)
print(f"Model Accuracy: {accuracy:.4f}")
`}</pre>
                    </div>
                    
                    <h2>Conclusion</h2>
                    <p>
                      This placeholder content is just to demonstrate the layout and styling of a blog post.
                      In a real implementation, each blog post would have its own unique content.
                    </p>
                  </>
                )}
              </div>
              
              {/* Author Bio */}
              <div className="mt-12 pt-8 border-t border-gray-200">
                <div className="flex items-center">
                  <div className="bg-primary/10 rounded-full w-12 h-12 flex items-center justify-center mr-4">
                    <span className="text-primary font-bold text-lg">ZA</span>
                  </div>
                  <div>
                    <h4 className="font-bold">Zahir Ahmad</h4>
                    <p className="text-gray-600 text-sm">Machine Learning Engineer specializing in NLP & Production-Ready Solutions</p>
                  </div>
                </div>
              </div>
            </div>
          </article>
        </div>
      </section>
      
      <Footer />
    </>
  );
};

export default BlogPost;
