
import Navbar from '@/components/Navbar';
import Footer from '@/components/Footer';
import ContactForm from '@/components/ContactForm';
import { Github, Linkedin, Mail, MapPin } from 'lucide-react';

const Contact = () => {
  return (
    <>
      <Navbar />
      
      {/* Header Section */}
      <section className="pt-28 pb-16 px-4 grid-pattern">
        <div className="max-w-5xl mx-auto text-center">
          <h1 className="text-4xl font-bold mb-6">Get in Touch</h1>
          <p className="text-lg text-gray-600 max-w-3xl mx-auto">
            Whether you have a question about machine learning, want to discuss a project, 
            or just want to say hi, feel free to reach out!
          </p>
        </div>
      </section>
      
      {/* Contact Section */}
      <section className="py-16 bg-white">
        <div className="max-w-6xl mx-auto px-4">
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
            <div className="lg:col-span-2">
              <ContactForm />
            </div>
            
            <div>
              <div className="bg-gray-50 rounded-lg border border-gray-200 p-6 md:p-8">
                <h3 className="text-xl font-bold mb-6">Contact Information</h3>
                
                <div className="space-y-6">
                  <div className="flex">
                    <div className="bg-primary/10 p-3 rounded-full mr-4 flex-shrink-0">
                      <Mail className="h-5 w-5 text-primary" />
                    </div>
                    <div>
                      <h4 className="font-medium text-gray-900 mb-1">Email</h4>
                      <a 
                        href="mailto:zahirahmadchaudhry@gmail.com" 
                        className="text-gray-600 hover:text-primary"
                      >
                        zahirahmadchaudhry@gmail.com
                      </a>
                    </div>
                  </div>
                  
                  <div className="flex">
                    <div className="bg-primary/10 p-3 rounded-full mr-4 flex-shrink-0">
                      <MapPin className="h-5 w-5 text-primary" />
                    </div>
                    <div>
                      <h4 className="font-medium text-gray-900 mb-1">Location</h4>
                      <p className="text-gray-600">
                        Lyon, France
                      </p>
                    </div>
                  </div>
                  
                  <div className="flex">
                    <div className="bg-primary/10 p-3 rounded-full mr-4 flex-shrink-0">
                      <Linkedin className="h-5 w-5 text-primary" />
                    </div>
                    <div>
                      <h4 className="font-medium text-gray-900 mb-1">LinkedIn</h4>
                      <a 
                        href="https://www.linkedin.com/in/zahirahmadchaudhry" 
                        target="_blank" 
                        rel="noopener noreferrer"
                        className="text-gray-600 hover:text-primary"
                      >
                        linkedin.com/in/zahirahmadchaudhry
                      </a>
                    </div>
                  </div>
                  
                  <div className="flex">
                    <div className="bg-primary/10 p-3 rounded-full mr-4 flex-shrink-0">
                      <Github className="h-5 w-5 text-primary" />
                    </div>
                    <div>
                      <h4 className="font-medium text-gray-900 mb-1">GitHub</h4>
                      <a 
                        href="https://github.com/ZahirAhmadChaudhry" 
                        target="_blank" 
                        rel="noopener noreferrer"
                        className="text-gray-600 hover:text-primary"
                      >
                        github.com/ZahirAhmadChaudhry
                      </a>
                    </div>
                  </div>
                </div>
                
                <div className="mt-8 pt-6 border-t border-gray-200">
                  <h4 className="font-medium text-gray-900 mb-4">Connect with me</h4>
                  <div className="flex space-x-4">
                    <a
                      href="https://github.com/ZahirAhmadChaudhry"
                      target="_blank"
                      rel="noopener noreferrer"
                      className="bg-gray-200 hover:bg-gray-300 rounded-full p-3 transition-colors"
                      aria-label="GitHub"
                    >
                      <Github className="h-5 w-5" />
                    </a>
                    <a
                      href="https://www.linkedin.com/in/zahirahmadchaudhry"
                      target="_blank"
                      rel="noopener noreferrer"
                      className="bg-gray-200 hover:bg-gray-300 rounded-full p-3 transition-colors"
                      aria-label="LinkedIn"
                    >
                      <Linkedin className="h-5 w-5" />
                    </a>
                    <a
                      href="mailto:zahirahmadchaudhry@gmail.com"
                      className="bg-gray-200 hover:bg-gray-300 rounded-full p-3 transition-colors"
                      aria-label="Email"
                    >
                      <Mail className="h-5 w-5" />
                    </a>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </section>
      
      <Footer />
    </>
  );
};

export default Contact;
