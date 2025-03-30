
import { Link } from 'react-router-dom';
import Navbar from '@/components/Navbar';
import Footer from '@/components/Footer';
import StatCard from '@/components/StatCard';
import ProjectCard from '@/components/ProjectCard';
import { ArrowDown, ArrowRight, Code, Cpu, Database, Award, BrainCircuit, LayoutGrid } from 'lucide-react';
import { projects } from '@/data/projects';

const Index = () => {
  return (
    <>
      <Navbar />
      
      {/* Hero Section */}
      <section className="pt-28 pb-16 md:pt-32 md:pb-24 px-4 grid-pattern">
        <div className="max-w-7xl mx-auto">
          <div className="grid grid-cols-1 lg:grid-cols-5 gap-8 items-center">
            <div className="lg:col-span-3 animate-fade-up">
              <div className="inline-block mb-3 px-3 py-1 bg-primary/10 text-primary rounded-full text-sm font-medium">
                Machine Learning Engineer
              </div>
              <h1 className="text-4xl md:text-5xl lg:text-6xl font-bold mb-4 leading-tight">
                Zahir Ahmad
              </h1>
              <h2 className="text-2xl md:text-3xl text-gray-700 mb-6">
                Specializing in NLP & Production-Ready Solutions
              </h2>
              <p className="text-gray-600 mb-8 max-w-2xl text-lg">
                Building intelligent systems and ML pipelines for complex data analysis tasks, 
                with expertise in NLP, deep learning, and MLOps.
              </p>
              <div className="flex flex-wrap gap-4">
                <Link 
                  to="/projects" 
                  className="bg-primary text-white px-6 py-3 rounded-lg font-medium inline-flex items-center hover:bg-primary/90 transition-colors"
                >
                  View Projects
                  <ArrowRight className="ml-2 h-5 w-5" />
                </Link>
                <Link 
                  to="/contact" 
                  className="bg-white text-gray-800 px-6 py-3 rounded-lg font-medium inline-flex items-center border border-gray-300 hover:bg-gray-50 transition-colors"
                >
                  Contact Me
                </Link>
              </div>
            </div>
            <div className="lg:col-span-2 animate-fade-in">
              <div className="relative">
                <div className="absolute -inset-0.5 rounded-lg bg-gradient-to-r from-tech-blue via-tech-purple to-tech-green opacity-75 blur"></div>
                <div className="relative bg-white p-5 rounded-lg shadow-lg">
                  <div className="github-snippet">
                    <pre className="text-gray-100">
{`# Machine Learning Pipeline
import torch
import numpy as np
from transformers import BertModel

def build_model():
    model = BertModel.from_pretrained(
        "bert-base-multilingual-cased"
    )
    return model

# Initialize model for multilingual NLP
model = build_model()
print("Model loaded successfully!")
`}
                    </pre>
                  </div>
                </div>
              </div>
            </div>
          </div>
          
          <div className="flex justify-center mt-12 animate-bounce">
            <a href="#stats" className="text-gray-500 hover:text-primary">
              <ArrowDown className="h-6 w-6" />
            </a>
          </div>
        </div>
      </section>
      
      {/* Stats Section */}
      <section id="stats" className="py-16 bg-gray-50">
        <div className="max-w-7xl mx-auto px-4">
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
            <StatCard 
              title="ML Projects" 
              value="10+" 
              icon={<Cpu className="h-6 w-6" />}
              color="bg-tech-blue/10 text-tech-blue"
            />
            <StatCard 
              title="NLP Expertise" 
              value="Advanced" 
              icon={<BrainCircuit className="h-6 w-6" />}
              color="bg-tech-purple/10 text-tech-purple"
            />
            <StatCard 
              title="Years Experience" 
              value="3+" 
              icon={<Award className="h-6 w-6" />}
              color="bg-tech-green/10 text-tech-green"
            />
            <StatCard 
              title="Tech Stack" 
              value="PyTorch, TF, MLOps" 
              icon={<LayoutGrid className="h-6 w-6" />}
              color="bg-tech-red/10 text-tech-red"
            />
          </div>
        </div>
      </section>
      
      {/* Featured Projects */}
      <section className="py-16">
        <div className="max-w-7xl mx-auto px-4">
          <div className="flex flex-col md:flex-row justify-between items-start md:items-center mb-12">
            <h2 className="section-heading">Featured Projects</h2>
            <Link 
              to="/projects" 
              className="mt-4 md:mt-0 text-primary flex items-center hover:underline"
            >
              View All Projects
              <ArrowRight className="ml-1 h-4 w-4" />
            </Link>
          </div>
          
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {projects.slice(0, 3).map((project) => (
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
        </div>
      </section>
      
      {/* About Brief */}
      <section className="py-16 bg-gray-50">
        <div className="max-w-7xl mx-auto px-4">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 items-center">
            <div>
              <h2 className="section-heading mb-8">About Me</h2>
              <p className="text-gray-600 mb-6">
                Machine Learning Engineer with expertise in developing end-to-end ML pipelines for complex 
                data analysis tasks. Strong focus on NLP, deep learning architectures (CNN, GNN, Transformers), 
                and classical ML algorithms with proven experience processing multilingual datasets and 
                implementing production-ready solutions in multidisciplinary environments.
              </p>
              <p className="text-gray-600 mb-8">
                Currently focusing on research in healthcare prediction models and opinion analysis in 
                sustainability discussions, creating specialized NLP techniques and interactive visualization tools.
              </p>
              <Link 
                to="/about" 
                className="bg-primary text-white px-5 py-2 rounded-lg font-medium inline-flex items-center hover:bg-primary/90 transition-colors"
              >
                Learn More
                <ArrowRight className="ml-2 h-4 w-4" />
              </Link>
            </div>
            <div className="grid grid-cols-2 gap-4">
              <div className="bg-white p-5 rounded-lg shadow-sm">
                <Code className="h-8 w-8 text-tech-blue mb-4" />
                <h3 className="text-lg font-semibold mb-2">Programming</h3>
                <p className="text-gray-600">Python, Linux, SQL, R</p>
              </div>
              <div className="bg-white p-5 rounded-lg shadow-sm">
                <Database className="h-8 w-8 text-tech-green mb-4" />
                <h3 className="text-lg font-semibold mb-2">Data Science</h3>
                <p className="text-gray-600">Pandas, NumPy, SciPy, Matplotlib</p>
              </div>
              <div className="bg-white p-5 rounded-lg shadow-sm">
                <BrainCircuit className="h-8 w-8 text-tech-purple mb-4" />
                <h3 className="text-lg font-semibold mb-2">Machine Learning</h3>
                <p className="text-gray-600">PyTorch, TensorFlow, SVM, GNN</p>
              </div>
              <div className="bg-white p-5 rounded-lg shadow-sm">
                <Cpu className="h-8 w-8 text-tech-red mb-4" />
                <h3 className="text-lg font-semibold mb-2">MLOps</h3>
                <p className="text-gray-600">Docker, Git, CI/CD, SLURM</p>
              </div>
            </div>
          </div>
        </div>
      </section>
      
      {/* CTA Section */}
      <section className="py-20 bg-primary text-white">
        <div className="max-w-7xl mx-auto px-4 text-center">
          <h2 className="text-3xl font-bold mb-6">Ready to collaborate?</h2>
          <p className="text-xl mb-8 max-w-2xl mx-auto">
            Let's discuss how my machine learning expertise can help solve your complex data challenges.
          </p>
          <Link 
            to="/contact" 
            className="bg-white text-primary px-6 py-3 rounded-lg font-medium inline-flex items-center hover:bg-gray-100 transition-colors"
          >
            Get in Touch
            <ArrowRight className="ml-2 h-5 w-5" />
          </Link>
        </div>
      </section>
      
      <Footer />
    </>
  );
};

export default Index;
