
import { Github, Linkedin, Mail } from 'lucide-react';
import { Link } from 'react-router-dom';

const Footer = () => {
  return (
    <footer className="bg-zinc-50 border-t">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-12">
        <div className="grid grid-cols-1 md:grid-cols-4 gap-8">
          <div className="col-span-1 md:col-span-2">
            <h3 className="text-lg font-bold mb-4">Zahir Ahmad</h3>
            <p className="text-muted-foreground mb-4 max-w-md">
              Machine Learning Engineer specializing in NLP & Production-Ready Solutions.
              Building intelligent systems to solve real-world problems.
            </p>
            <div className="flex space-x-4">
              <a
                href="https://github.com/ZahirAhmadChaudhry"
                target="_blank"
                rel="noopener noreferrer"
                className="text-gray-600 hover:text-primary transition-colors"
                aria-label="GitHub"
              >
                <Github size={20} />
              </a>
              <a
                href="https://www.linkedin.com/in/zahirahmadchaudhry"
                target="_blank"
                rel="noopener noreferrer"
                className="text-gray-600 hover:text-primary transition-colors"
                aria-label="LinkedIn"
              >
                <Linkedin size={20} />
              </a>
              <a
                href="mailto:zahirahmadchaudhry@gmail.com"
                className="text-gray-600 hover:text-primary transition-colors"
                aria-label="Email"
              >
                <Mail size={20} />
              </a>
            </div>
          </div>
          
          <div>
            <h4 className="text-sm font-semibold mb-4 text-gray-900">Site Map</h4>
            <ul className="space-y-2">
              <li>
                <Link to="/" className="text-muted-foreground hover:text-primary transition-colors">
                  Home
                </Link>
              </li>
              <li>
                <Link to="/about" className="text-muted-foreground hover:text-primary transition-colors">
                  About
                </Link>
              </li>
              <li>
                <Link to="/projects" className="text-muted-foreground hover:text-primary transition-colors">
                  Projects
                </Link>
              </li>
              <li>
                <Link to="/contact" className="text-muted-foreground hover:text-primary transition-colors">
                  Contact
                </Link>
              </li>
            </ul>
          </div>
          
          <div>
            <h4 className="text-sm font-semibold mb-4 text-gray-900">Expertise</h4>
            <ul className="space-y-2">
              <li className="text-muted-foreground">Machine Learning</li>
              <li className="text-muted-foreground">Natural Language Processing</li>
              <li className="text-muted-foreground">Computer Vision</li>
              <li className="text-muted-foreground">Generative AI</li>
              <li className="text-muted-foreground">MLOps</li>
            </ul>
          </div>
        </div>
        
        <div className="border-t border-gray-200 mt-8 pt-8 flex flex-col md:flex-row justify-between items-center">
          <p className="text-sm text-muted-foreground">
            &copy; {new Date().getFullYear()} Zahir Ahmad. All rights reserved.
          </p>
          <p className="text-sm text-muted-foreground mt-2 md:mt-0">
            Machine Learning Engineer
          </p>
        </div>
      </div>
    </footer>
  );
};

export default Footer;
