import React from 'react';
import ReactDOM from 'react-dom/client';
import { BrowserRouter, Navigate, Route, Routes } from 'react-router-dom';
import './app.css';
import App from '@/root';
import RequireSim from '@/components/RequireSim';

ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <BrowserRouter>
      <Routes>
        <Route path='/' element={<Navigate to='/simulation' replace />} />
        <Route
          path='/simulation'
          element={
            <RequireSim>
              <App />
            </RequireSim>
          }
        />
      </Routes>
    </BrowserRouter>
  </React.StrictMode>,
);
