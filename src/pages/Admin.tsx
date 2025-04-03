
import { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { useAuth } from '@/contexts/AuthContext';
import { projects } from '@/data/projects';
import { ArrowLeft, Edit, Plus, Trash, Save, X, Copy } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Card, CardContent, CardHeader, CardTitle, CardDescription, CardFooter } from '@/components/ui/card';
import { Label } from '@/components/ui/label';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Dialog, DialogContent, DialogDescription, DialogFooter, DialogHeader, DialogTitle, DialogTrigger } from '@/components/ui/dialog';
import { Textarea } from '@/components/ui/textarea';
import { useToast } from '@/hooks/use-toast';
import ProtectedRoute from '@/components/ProtectedRoute';

// Define project type to match data structure
interface Project {
  id: string;
  title: string;
  description: string;
  fullDescription?: string;
  image: string;
  category: string;
  tags: string[];
  github?: string;
  challenges?: string[];
  solutions?: string[];
  achievements?: string[];
  codeSnippet?: string;
  technologies?: string[];
}

const Admin = () => {
  const { logout } = useAuth();
  const navigate = useNavigate();
  const { toast } = useToast();
  
  // Local state for projects
  const [localProjects, setLocalProjects] = useState<Project[]>([]);
  const [editingProject, setEditingProject] = useState<Project | null>(null);
  const [isDialogOpen, setIsDialogOpen] = useState(false);
  const [codeGenerated, setCodeGenerated] = useState('');
  const [activeTab, setActiveTab] = useState('projects');

  // Load projects from localStorage or use original data
  useEffect(() => {
    const savedProjects = localStorage.getItem('adminProjects');
    if (savedProjects) {
      try {
        setLocalProjects(JSON.parse(savedProjects));
      } catch (e) {
        setLocalProjects(projects);
      }
    } else {
      setLocalProjects(projects);
    }
  }, []);

  // Save to localStorage whenever projects change
  useEffect(() => {
    if (localProjects.length > 0) {
      localStorage.setItem('adminProjects', JSON.stringify(localProjects));
    }
  }, [localProjects]);

  const handleLogout = () => {
    logout();
    navigate('/');
  };

  const handleEditProject = (project: Project) => {
    setEditingProject({ ...project });
    setIsDialogOpen(true);
  };

  const handleNewProject = () => {
    const newProject: Project = {
      id: `project-${Date.now()}`,
      title: 'New Project',
      description: 'Project description',
      image: 'https://images.unsplash.com/photo-1581091226825-a6a2a5aee158?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=1170&q=80',
      category: 'Classic ML',
      tags: ['Tag 1', 'Tag 2'],
      challenges: ['Challenge 1'],
      solutions: ['Solution 1'],
      achievements: ['Achievement 1'],
      technologies: ['Technology 1']
    };
    setEditingProject(newProject);
    setIsDialogOpen(true);
  };

  const handleDeleteProject = (id: string) => {
    if (confirm('Are you sure you want to delete this project?')) {
      setLocalProjects(localProjects.filter(p => p.id !== id));
      toast({
        title: "Project Deleted",
        description: "The project has been removed."
      });
    }
  };

  const handleSaveProject = () => {
    if (!editingProject) return;
    
    // Determine if we're adding or updating
    const exists = localProjects.some(p => p.id === editingProject.id);
    
    if (exists) {
      // Update existing project
      setLocalProjects(localProjects.map(p => 
        p.id === editingProject.id ? editingProject : p
      ));
    } else {
      // Add new project
      setLocalProjects([...localProjects, editingProject]);
    }
    
    setIsDialogOpen(false);
    setEditingProject(null);
    
    toast({
      title: exists ? "Project Updated" : "Project Added",
      description: exists 
        ? "Your changes have been saved." 
        : "The new project has been added."
    });
  };

  const handleInputChange = (field: keyof Project, value: string) => {
    if (!editingProject) return;
    
    setEditingProject({
      ...editingProject,
      [field]: value
    });
  };

  const handleArrayInputChange = (field: keyof Project, index: number, value: string) => {
    if (!editingProject) return;
    
    const array = [...(editingProject[field] as string[] || [])];
    array[index] = value;
    
    setEditingProject({
      ...editingProject,
      [field]: array
    });
  };

  const handleAddArrayItem = (field: keyof Project) => {
    if (!editingProject) return;
    
    const array = [...(editingProject[field] as string[] || []), ''];
    
    setEditingProject({
      ...editingProject,
      [field]: array
    });
  };

  const handleRemoveArrayItem = (field: keyof Project, index: number) => {
    if (!editingProject) return;
    
    const array = [...(editingProject[field] as string[] || [])];
    array.splice(index, 1);
    
    setEditingProject({
      ...editingProject,
      [field]: array
    });
  };

  const generateCode = () => {
    // Format the projects array as TypeScript code
    const codeLines = [
      'export const projects = [',
    ];

    localProjects.forEach((project, index) => {
      codeLines.push('  {');
      Object.entries(project).forEach(([key, value]) => {
        if (Array.isArray(value)) {
          codeLines.push(`    ${key}: [`);
          value.forEach(item => {
            codeLines.push(`      '${item.replace(/'/g, "\\'")}',`);
          });
          codeLines.push('    ],');
        } else if (typeof value === 'string') {
          if (key === 'codeSnippet') {
            codeLines.push(`    ${key}: \`${value.replace(/`/g, '\\`')}\`,`);
          } else {
            codeLines.push(`    ${key}: '${value.replace(/'/g, "\\'")}',`);
          }
        }
      });
      codeLines.push(index === localProjects.length - 1 ? '  }' : '  },');
    });

    codeLines.push('];');
    
    setCodeGenerated(codeLines.join('\n'));
    setActiveTab('code');
    
    toast({
      title: "Code Generated",
      description: "Copy the code and update your projects.ts file."
    });
  };

  const copyToClipboard = () => {
    navigator.clipboard.writeText(codeGenerated);
    toast({
      title: "Copied to Clipboard",
      description: "The code has been copied to your clipboard."
    });
  };

  return (
    <ProtectedRoute>
      <div className="min-h-screen bg-gray-50">
        <header className="bg-white shadow">
          <div className="max-w-7xl mx-auto px-4 py-6 sm:px-6 lg:px-8 flex justify-between items-center">
            <div className="flex items-center">
              <Button 
                variant="outline" 
                size="icon" 
                onClick={() => navigate('/')} 
                className="mr-4"
              >
                <ArrowLeft className="h-4 w-4" />
              </Button>
              <h1 className="text-2xl font-bold text-gray-900">
                Project Admin Panel
              </h1>
            </div>
            <Button variant="outline" onClick={handleLogout}>
              Logout
            </Button>
          </div>
        </header>

        <main className="max-w-7xl mx-auto px-4 py-6 sm:px-6 lg:px-8">
          <Card>
            <CardHeader>
              <div className="flex justify-between items-center">
                <CardTitle>Manage Portfolio Projects</CardTitle>
                <div className="flex space-x-3">
                  <Button onClick={generateCode}>
                    Generate Code
                  </Button>
                  <Button onClick={handleNewProject}>
                    <Plus className="h-4 w-4 mr-2" /> New Project
                  </Button>
                </div>
              </div>
              <CardDescription>
                Add, edit, or remove projects from your portfolio
              </CardDescription>
            </CardHeader>
            
            <CardContent>
              <Tabs value={activeTab} onValueChange={setActiveTab}>
                <TabsList className="mb-4">
                  <TabsTrigger value="projects">Projects</TabsTrigger>
                  <TabsTrigger value="code">Generated Code</TabsTrigger>
                </TabsList>
                
                <TabsContent value="projects">
                  <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                    {localProjects.map((project) => (
                      <Card key={project.id} className="overflow-hidden">
                        <div className="h-32 overflow-hidden">
                          <img 
                            src={project.image} 
                            alt={project.title} 
                            className="w-full h-full object-cover"
                          />
                        </div>
                        <CardHeader className="p-4">
                          <CardTitle className="text-lg">{project.title}</CardTitle>
                          <div className="flex items-center space-x-2">
                            <span className="px-2 py-1 bg-gray-100 rounded-full text-xs">
                              {project.category}
                            </span>
                          </div>
                        </CardHeader>
                        <CardContent className="p-4 pt-0">
                          <p className="text-sm text-gray-600 line-clamp-2">
                            {project.description}
                          </p>
                        </CardContent>
                        <CardFooter className="p-4 flex justify-end space-x-2">
                          <Button 
                            variant="outline" 
                            size="sm" 
                            onClick={() => handleDeleteProject(project.id)}
                          >
                            <Trash className="h-4 w-4" />
                          </Button>
                          <Button 
                            variant="default" 
                            size="sm" 
                            onClick={() => handleEditProject(project)}
                          >
                            <Edit className="h-4 w-4 mr-1" /> Edit
                          </Button>
                        </CardFooter>
                      </Card>
                    ))}
                  </div>
                </TabsContent>
                
                <TabsContent value="code">
                  <div className="border rounded-md p-4 bg-gray-50">
                    <div className="flex justify-between items-center mb-2">
                      <h3 className="text-lg font-medium">Generated Code for projects.ts</h3>
                      <Button onClick={copyToClipboard} size="sm">
                        <Copy className="h-4 w-4 mr-1" /> Copy
                      </Button>
                    </div>
                    <pre className="bg-gray-900 text-gray-100 p-4 rounded-md overflow-x-auto">
                      <code>{codeGenerated}</code>
                    </pre>
                    <p className="mt-4 text-sm text-gray-600">
                      Copy this code and replace the contents of your src/data/projects.ts file with it.
                    </p>
                  </div>
                </TabsContent>
              </Tabs>
            </CardContent>
          </Card>
        </main>

        {/* Edit Project Dialog */}
        {editingProject && (
          <Dialog open={isDialogOpen} onOpenChange={setIsDialogOpen}>
            <DialogContent className="max-w-3xl max-h-[80vh] overflow-y-auto">
              <DialogHeader>
                <DialogTitle>
                  {localProjects.some(p => p.id === editingProject.id) 
                    ? 'Edit Project' 
                    : 'Add New Project'
                  }
                </DialogTitle>
                <DialogDescription>
                  Update the project information and save your changes
                </DialogDescription>
              </DialogHeader>
              
              <div className="grid gap-4 py-4">
                <div className="grid grid-cols-2 gap-4">
                  <div className="space-y-2">
                    <Label htmlFor="title">Project Title</Label>
                    <Input 
                      id="title" 
                      value={editingProject.title} 
                      onChange={(e) => handleInputChange('title', e.target.value)}
                    />
                  </div>
                  
                  <div className="space-y-2">
                    <Label htmlFor="id">Project ID (URL slug)</Label>
                    <Input 
                      id="id" 
                      value={editingProject.id} 
                      onChange={(e) => handleInputChange('id', e.target.value)}
                    />
                  </div>
                </div>
                
                <div className="grid grid-cols-2 gap-4">
                  <div className="space-y-2">
                    <Label htmlFor="category">Category</Label>
                    <Input 
                      id="category" 
                      value={editingProject.category} 
                      onChange={(e) => handleInputChange('category', e.target.value)}
                    />
                  </div>
                  
                  <div className="space-y-2">
                    <Label htmlFor="github">GitHub URL (optional)</Label>
                    <Input 
                      id="github" 
                      value={editingProject.github || ''} 
                      onChange={(e) => handleInputChange('github', e.target.value)}
                    />
                  </div>
                </div>
                
                <div className="space-y-2">
                  <Label htmlFor="image">Image URL</Label>
                  <Input 
                    id="image" 
                    value={editingProject.image} 
                    onChange={(e) => handleInputChange('image', e.target.value)}
                  />
                </div>
                
                <div className="space-y-2">
                  <Label htmlFor="description">Short Description</Label>
                  <Input 
                    id="description" 
                    value={editingProject.description} 
                    onChange={(e) => handleInputChange('description', e.target.value)}
                  />
                </div>
                
                <div className="space-y-2">
                  <Label htmlFor="fullDescription">Full Description</Label>
                  <Textarea 
                    id="fullDescription" 
                    value={editingProject.fullDescription || ''} 
                    onChange={(e) => handleInputChange('fullDescription', e.target.value)}
                    rows={3}
                  />
                </div>
                
                <div className="space-y-2">
                  <div className="flex justify-between items-center">
                    <Label>Tags</Label>
                    <Button 
                      variant="outline" 
                      size="sm" 
                      onClick={() => handleAddArrayItem('tags')}
                    >
                      <Plus className="h-3 w-3 mr-1" /> Add Tag
                    </Button>
                  </div>
                  {editingProject.tags.map((tag, index) => (
                    <div key={index} className="flex items-center space-x-2">
                      <Input 
                        value={tag} 
                        onChange={(e) => handleArrayInputChange('tags', index, e.target.value)}
                      />
                      <Button 
                        variant="ghost" 
                        size="icon" 
                        onClick={() => handleRemoveArrayItem('tags', index)}
                      >
                        <X className="h-4 w-4" />
                      </Button>
                    </div>
                  ))}
                </div>
                
                <div className="space-y-2">
                  <div className="flex justify-between items-center">
                    <Label>Challenges</Label>
                    <Button 
                      variant="outline" 
                      size="sm" 
                      onClick={() => handleAddArrayItem('challenges')}
                    >
                      <Plus className="h-3 w-3 mr-1" /> Add Challenge
                    </Button>
                  </div>
                  {(editingProject.challenges || []).map((challenge, index) => (
                    <div key={index} className="flex items-center space-x-2">
                      <Input 
                        value={challenge} 
                        onChange={(e) => handleArrayInputChange('challenges', index, e.target.value)}
                      />
                      <Button 
                        variant="ghost" 
                        size="icon" 
                        onClick={() => handleRemoveArrayItem('challenges', index)}
                      >
                        <X className="h-4 w-4" />
                      </Button>
                    </div>
                  ))}
                </div>
                
                <div className="space-y-2">
                  <div className="flex justify-between items-center">
                    <Label>Solutions</Label>
                    <Button 
                      variant="outline" 
                      size="sm" 
                      onClick={() => handleAddArrayItem('solutions')}
                    >
                      <Plus className="h-3 w-3 mr-1" /> Add Solution
                    </Button>
                  </div>
                  {(editingProject.solutions || []).map((solution, index) => (
                    <div key={index} className="flex items-center space-x-2">
                      <Input 
                        value={solution} 
                        onChange={(e) => handleArrayInputChange('solutions', index, e.target.value)}
                      />
                      <Button 
                        variant="ghost" 
                        size="icon" 
                        onClick={() => handleRemoveArrayItem('solutions', index)}
                      >
                        <X className="h-4 w-4" />
                      </Button>
                    </div>
                  ))}
                </div>
                
                <div className="space-y-2">
                  <div className="flex justify-between items-center">
                    <Label>Achievements</Label>
                    <Button 
                      variant="outline" 
                      size="sm" 
                      onClick={() => handleAddArrayItem('achievements')}
                    >
                      <Plus className="h-3 w-3 mr-1" /> Add Achievement
                    </Button>
                  </div>
                  {(editingProject.achievements || []).map((achievement, index) => (
                    <div key={index} className="flex items-center space-x-2">
                      <Input 
                        value={achievement} 
                        onChange={(e) => handleArrayInputChange('achievements', index, e.target.value)}
                      />
                      <Button 
                        variant="ghost" 
                        size="icon" 
                        onClick={() => handleRemoveArrayItem('achievements', index)}
                      >
                        <X className="h-4 w-4" />
                      </Button>
                    </div>
                  ))}
                </div>
                
                <div className="space-y-2">
                  <div className="flex justify-between items-center">
                    <Label>Technologies</Label>
                    <Button 
                      variant="outline" 
                      size="sm" 
                      onClick={() => handleAddArrayItem('technologies')}
                    >
                      <Plus className="h-3 w-3 mr-1" /> Add Technology
                    </Button>
                  </div>
                  {(editingProject.technologies || []).map((tech, index) => (
                    <div key={index} className="flex items-center space-x-2">
                      <Input 
                        value={tech} 
                        onChange={(e) => handleArrayInputChange('technologies', index, e.target.value)}
                      />
                      <Button 
                        variant="ghost" 
                        size="icon" 
                        onClick={() => handleRemoveArrayItem('technologies', index)}
                      >
                        <X className="h-4 w-4" />
                      </Button>
                    </div>
                  ))}
                </div>
                
                <div className="space-y-2">
                  <Label htmlFor="codeSnippet">Code Snippet</Label>
                  <Textarea 
                    id="codeSnippet" 
                    value={editingProject.codeSnippet || ''} 
                    onChange={(e) => handleInputChange('codeSnippet', e.target.value)}
                    rows={5}
                    className="font-mono text-sm"
                  />
                </div>
              </div>
              
              <DialogFooter>
                <Button variant="outline" onClick={() => setIsDialogOpen(false)}>
                  Cancel
                </Button>
                <Button onClick={handleSaveProject}>
                  <Save className="h-4 w-4 mr-2" /> Save Project
                </Button>
              </DialogFooter>
            </DialogContent>
          </Dialog>
        )}
      </div>
    </ProtectedRoute>
  );
};

export default Admin;
