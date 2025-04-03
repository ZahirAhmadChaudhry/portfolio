
import Navbar from '@/components/Navbar';
import Footer from '@/components/Footer';
import ContactForm from '@/components/ContactForm';
import { Github, Linkedin, Mail } from 'lucide-react';

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
                <h3 className="text-xl font-bold mb-6">Connect with me</h3>
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
      </section>
      
      <Footer />
    </>
  );
};

export default Contact;
