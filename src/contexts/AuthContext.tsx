
import React, { createContext, useContext, useState, useEffect } from 'react';

type AuthContextType = {
  isAuthenticated: boolean;
  login: (password: string) => boolean;
  logout: () => void;
};

const AuthContext = createContext<AuthContextType | undefined>(undefined);

export const AuthProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  // Simple auth state persisted in localStorage
  const [isAuthenticated, setIsAuthenticated] = useState<boolean>(() => {
    const stored = localStorage.getItem('adminAuth');
    return stored === 'true';
  });

  // Update localStorage when auth state changes
  useEffect(() => {
    localStorage.setItem('adminAuth', isAuthenticated ? 'true' : 'false');
  }, [isAuthenticated]);

  // Simple password check (in a real app, you'd use a more secure method)
  const login = (password: string): boolean => {
    // Very basic protection - in a real app, use better authentication
    if (password === 'portfolio2024') {
      setIsAuthenticated(true);
      return true;
    }
    return false;
  };

  const logout = () => {
    setIsAuthenticated(false);
  };

  return (
    <AuthContext.Provider value={{ isAuthenticated, login, logout }}>
      {children}
    </AuthContext.Provider>
  );
};

export const useAuth = (): AuthContextType => {
  const context = useContext(AuthContext);
  if (context === undefined) {
    throw new Error('useAuth must be used within an AuthProvider');
  }
  return context;
};
