
import { Toaster } from "@/components/ui/toaster";
import { Toaster as Sonner } from "@/components/ui/sonner";
import { TooltipProvider } from "@/components/ui/tooltip";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { BrowserRouter, Routes, Route } from "react-router-dom";
import { useEffect } from "react";
import Index from "./pages/Index";
import About from "./pages/About";
import Projects from "./pages/Projects";
import ProjectDetail from "./pages/ProjectDetail";
import Contact from "./pages/Contact";
import NotFound from "./pages/NotFound";
import AdminLogin from "./pages/AdminLogin";
import Admin from "./pages/Admin";
import ProtectedRoute from "./components/ProtectedRoute";

// Add mermaid to the window object for TypeScript
declare global {
  interface Window {
    mermaid: any;
  }
}

const queryClient = new QueryClient();

const App = () => {
  // Load mermaid library dynamically
  useEffect(() => {
    const loadMermaid = async () => {
      try {
        // Only load if not already loaded
        if (!window.mermaid) {
          const mermaid = await import('mermaid');
          window.mermaid = mermaid.default;
          window.mermaid.initialize({
            startOnLoad: true,
            theme: 'neutral',
            securityLevel: 'loose'
          });
          
          // Initialize all mermaid diagrams
          window.mermaid.contentLoaded();
        }
      } catch (error) {
        console.error('Failed to load Mermaid library:', error);
      }
    };
    
    loadMermaid();
  }, []);

  return (
    <QueryClientProvider client={queryClient}>
      <TooltipProvider>
        <Toaster />
        <Sonner />
        <BrowserRouter>
          <Routes>
            <Route path="/" element={<Index />} />
            <Route path="/about" element={<About />} />
            <Route path="/projects" element={<Projects />} />
            <Route path="/projects/:id" element={<ProjectDetail />} />
            <Route path="/contact" element={<Contact />} />
            <Route path="/admin-login" element={<AdminLogin />} />
            <Route path="/admin" element={<ProtectedRoute><Admin /></ProtectedRoute>} />
            <Route path="*" element={<NotFound />} />
          </Routes>
        </BrowserRouter>
      </TooltipProvider>
    </QueryClientProvider>
  );
};

export default App;
