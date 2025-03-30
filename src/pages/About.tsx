
import Navbar from '@/components/Navbar';
import Footer from '@/components/Footer';
import SkillCategory from '@/components/SkillSection';
import { Download, Award, GraduationCap, Briefcase } from 'lucide-react';

const About = () => {
  // Define skills by category
  const skillCategories = [
    {
      title: 'Programming',
      skills: ['Python', 'Linux', 'SQL', 'R'],
    },
    {
      title: 'Machine Learning',
      skills: ['SVM', 'XGBoost', 'PyTorch', 'TensorFlow', 'scikit-learn', 'CNN', 'GNN', 'Transformers', 'LLMs', 'Optimal Transport'],
    },
    {
      title: 'Natural Language Processing',
      skills: ['Text Classification', 'Opinion Mining', 'Multilingual Processing', 'Sentiment Analysis', 'BERT', 'GPT'],
    },
    {
      title: 'Data Science',
      skills: ['Pandas', 'NumPy', 'SciPy', 'Matplotlib', 'Feature Engineering'],
    },
    {
      title: 'MLOps & Tools',
      skills: ['Docker', 'Git', 'CI/CD', 'SLURM'],
    },
    {
      title: 'Cloud & HPC',
      skills: ['Microsoft Azure', 'Distributed Systems', 'GPU Acceleration'],
    },
  ];

  // Education information
  const education = [
    {
      school: 'Université Jean Monnet Saint-Étienne',
      degree: 'Master\'s Degree in Machine Learning and Data Mining',
      location: 'Saint-Étienne, France',
      period: 'Sep. 2023 - Present',
    },
    {
      school: 'Virtual University of Pakistan',
      degree: 'Bachelor\'s Degree in Computer Science',
      location: 'Islamabad, Pakistan',
      period: 'Sep. 2020 - Dec. 2022',
    },
  ];

  // Work experience
  const experience = [
    {
      company: 'Laboratoire Coactis',
      position: 'Machine Learning Research Intern',
      location: 'Saint-Étienne, France',
      period: 'Feb. 2025 - Present',
      description: 'Designing and implementing an end-to-end NLP solution for automating opinion analysis in French sustainability discussions.',
      achievements: [
        'Developing specialized NLP techniques to identify and categorize organizational paradoxes and tensions in 300+ pages of multi-speaker transcripts',
        'Creating a complete ML pipeline for multilingual text analysis that bridges management science theory with practical computational approaches',
        'Implementing a web-based application with interactive visualization capabilities for cross-report comparison and expert annotation',
        'Collaborating with interdisciplinary team of management researchers to integrate domain expertise into algorithmic design',
        'Enhancing model interpretability through transparent classification processes and interactive result editing',
      ],
    },
    {
      company: 'Laboratoire Coactis',
      position: 'Machine Learning Research Intern',
      location: 'Saint-Étienne, France',
      period: 'Apr. 2024 - Present',
      description: 'Designed and implemented an end-to-end machine learning pipeline for predicting Healthcare-Associated Infection (HAI) risks in ICU settings.',
      achievements: [
        'Conducted comprehensive literature review on HAI prediction models and identified key success factors and limitations in existing approaches',
        'Engineered unified dataset by integrating three complex healthcare sources with 412 data points, and collaborated with medical experts to optimize feature selection',
        'Implemented and evaluated multiple ML models (Logistic Regression, Autoencoders, SVM, GBM), achieving 99.9% accuracy and 0.999 R² through cross-validation',
        'Developed an interactive clinical decision support system with risk assessment capabilities through cross-functional collaboration',
      ],
    },
    {
      company: 'Ministry of Human Rights',
      position: 'Data Processing Officer',
      location: 'Islamabad, Pakistan',
      period: 'Apr. 2020 - Aug. 2023',
      description: 'Participated in the development and implementation of the Human Rights Information Management System.',
      achievements: [
        'Architected a centralized data management system integrating information from provincial and federal departments, ensuring compliance with privacy regulations',
        'Established robust data quality frameworks and validation protocols, significantly improving cross-departmental data consistency and reporting reliability',
        'Facilitated data-driven policy recommendations through collaborative analysis with human rights experts, translating complex datasets into actionable insights',
      ],
    },
  ];

  // Certifications
  const certifications = [
    {
      name: 'Machine Learning Specialization',
      issuer: 'Stanford University (Coursera)',
    },
    {
      name: 'Deep Learning Specialization',
      issuer: 'DeepLearning.AI (Coursera)',
    },
    {
      name: 'Python for Everybody',
      issuer: 'University of Michigan (Coursera)',
    },
    {
      name: 'Diploma in Cloud Computing and Networking',
      issuer: 'NAVTTC',
    },
  ];

  return (
    <>
      <Navbar />
      
      {/* Header Section */}
      <section className="pt-28 pb-16 px-4 grid-pattern">
        <div className="max-w-4xl mx-auto text-center">
          <h1 className="text-4xl font-bold mb-6">About Me</h1>
          <p className="text-lg text-gray-600 max-w-3xl mx-auto">
            Machine Learning Engineer with expertise in developing end-to-end ML pipelines for complex data analysis tasks. 
            Strong focus on NLP, deep learning architectures, and implementing production-ready solutions in multidisciplinary environments.
          </p>
          <div className="mt-8">
            <a 
              href="/Zahir_Ahmad_Resume.pdf" 
              download
              className="bg-primary text-white px-5 py-2 rounded-lg font-medium inline-flex items-center hover:bg-primary/90 transition-colors"
            >
              Download Resume
              <Download className="ml-2 h-4 w-4" />
            </a>
          </div>
        </div>
      </section>
      
      {/* Skills Section */}
      <section className="py-16 bg-white">
        <div className="max-w-5xl mx-auto px-4">
          <h2 className="section-heading mb-10">Skills & Expertise</h2>
          
          <div className="grid grid-cols-1 md:grid-cols-2 gap-x-8 gap-y-6">
            {skillCategories.map((category) => (
              <SkillCategory 
                key={category.title}
                title={category.title}
                skills={category.skills}
              />
            ))}
          </div>
        </div>
      </section>
      
      {/* Experience Section */}
      <section className="py-16 bg-gray-50">
        <div className="max-w-5xl mx-auto px-4">
          <h2 className="section-heading mb-10">Work Experience</h2>
          
          <div className="space-y-8">
            {experience.map((job, index) => (
              <div key={index} className="bg-white p-6 rounded-lg border border-gray-200 hover:shadow-md transition-all">
                <div className="flex items-start">
                  <div className="flex-shrink-0 bg-primary/10 p-3 rounded-full mr-4">
                    <Briefcase className="h-6 w-6 text-primary" />
                  </div>
                  <div>
                    <h3 className="text-xl font-bold">{job.position}</h3>
                    <div className="text-primary font-medium mb-1">{job.company}</div>
                    <div className="text-gray-600 mb-3">{job.location} • {job.period}</div>
                    <p className="text-gray-700 mb-3">{job.description}</p>
                    
                    {job.achievements && (
                      <div>
                        <h4 className="font-semibold text-gray-800 mb-2">Key Achievements:</h4>
                        <ul className="list-disc list-inside space-y-1 text-gray-600">
                          {job.achievements.map((achievement, i) => (
                            <li key={i}>{achievement}</li>
                          ))}
                        </ul>
                      </div>
                    )}
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      </section>
      
      {/* Education Section */}
      <section className="py-16 bg-white">
        <div className="max-w-5xl mx-auto px-4">
          <h2 className="section-heading mb-10">Education</h2>
          
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            {education.map((edu, index) => (
              <div key={index} className="bg-gray-50 p-6 rounded-lg border border-gray-200 hover:shadow-sm transition-all">
                <div className="flex items-start">
                  <div className="flex-shrink-0 bg-primary/10 p-3 rounded-full mr-4">
                    <GraduationCap className="h-6 w-6 text-primary" />
                  </div>
                  <div>
                    <h3 className="text-xl font-bold">{edu.school}</h3>
                    <div className="text-primary font-medium mb-1">{edu.degree}</div>
                    <div className="text-gray-600">{edu.location} • {edu.period}</div>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      </section>
      
      {/* Certifications Section */}
      <section className="py-16 bg-gray-50">
        <div className="max-w-5xl mx-auto px-4">
          <h2 className="section-heading mb-10">Certifications</h2>
          
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-6">
            {certifications.map((cert, index) => (
              <div key={index} className="bg-white p-6 rounded-lg border border-gray-200 hover:shadow-md transition-all">
                <div className="flex items-start">
                  <div className="flex-shrink-0 bg-primary/10 p-3 rounded-full mr-3">
                    <Award className="h-5 w-5 text-primary" />
                  </div>
                  <div>
                    <h3 className="font-semibold mb-1">{cert.name}</h3>
                    <div className="text-gray-600 text-sm">{cert.issuer}</div>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      </section>
      
      <Footer />
    </>
  );
};

export default About;
