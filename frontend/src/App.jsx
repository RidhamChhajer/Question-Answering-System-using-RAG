import { useEffect, useState } from 'react';
import { Routes, Route, useNavigate, useParams } from 'react-router-dom';
import { fetchNotebooks, touchNotebook } from './api/client';
import Dashboard from './pages/Dashboard';
import Workspace from './pages/Workspace';

function WorkspaceRoute() {
  const { id } = useParams();
  const navigate = useNavigate();
  const [notebook, setNotebook] = useState(null);

  useEffect(() => {
    let active = true;
    setNotebook(null);
    fetchNotebooks()
      .then(nbs => {
        if (!active) return;
        const nb = nbs.find(n => n.id === id);
        if (!nb) {
          navigate('/');
          return;
        }
        touchNotebook(nb.id).catch(() => {});
        setNotebook(nb);
      })
      .catch(() => {
        if (active) navigate('/');
      });
    return () => { active = false; };
  }, [id, navigate]);

  if (!notebook) return <div>Loading...</div>;
  return <Workspace notebook={notebook} initConvId={null} />;
}

export default function App() {
  return (
    <Routes>
      <Route path="/" element={<Dashboard />} />
      <Route path="/notebook/:id" element={<WorkspaceRoute />} />
    </Routes>
  );
}
