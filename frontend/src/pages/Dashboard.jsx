import { useState, useEffect, useRef, useCallback } from 'react';
import { useNavigate } from 'react-router-dom';
import { fetchNotebooks, createNotebook } from '../api/client';
import NotebookCard from '../components/NotebookCard';
import styles from './Dashboard.module.css';

export default function Dashboard() {
  const navigate = useNavigate();
  const [notebooks, setNotebooks] = useState([]);
  const [search, setSearch] = useState('');
  const [isLoading, setIsLoading] = useState(true);
  const [theme, setTheme] = useState(
    document.documentElement.getAttribute('data-theme') || 'light'
  );
  const searchRef = useRef(null);

  const loadNotebooks = useCallback(() => {
    setIsLoading(true);
    fetchNotebooks()
      .then(setNotebooks)
      .catch(console.error)
      .finally(() => setIsLoading(false));
  }, []);

  useEffect(() => { loadNotebooks(); }, [loadNotebooks]);

  const openNotebook = useCallback((nb) => {
    navigate(`/notebook/${nb.id}`);
  }, [navigate]);

  const handleCreate = useCallback(async () => {
    const nb = await createNotebook('Untitled');
    setNotebooks(prev => [nb, ...prev]);
    navigate(`/notebook/${nb.id}`);
  }, [navigate]);

  const toggleTheme = () => {
    const next = theme === 'light' ? 'dark' : 'light';
    setTheme(next);
    document.documentElement.setAttribute('data-theme', next);
    localStorage.setItem('theme', next);
  };

  const filtered = search.trim()
    ? notebooks.filter(nb => nb.title.toLowerCase().includes(search.toLowerCase()))
    : notebooks;

  const handleDelete = (id) => setNotebooks(prev => prev.filter(n => n.id !== id));
  const handleRename = (updated) => setNotebooks(prev => prev.map(n => n.id === updated.id ? { ...n, ...updated } : n));

  useEffect(() => {
    const handler = (e) => {
      const isInput = ['INPUT', 'TEXTAREA'].includes(e.target.tagName);

      if (e.ctrlKey && e.key === 'k') {
        e.preventDefault();
        searchRef.current?.focus();
      }

      if (e.ctrlKey && e.key === 'n' && !isInput) {
        e.preventDefault();
        handleCreate();
      }
    };

    document.addEventListener('keydown', handler);
    return () => document.removeEventListener('keydown', handler);
  }, [handleCreate]);

  return (
    <div className={styles.page}>
      {/* ── Header ── */}
      <header className={styles.header}>
        <div className={styles.brand}>
          <span className={styles.brandIcon}>📚</span>
          <span className={styles.brandName}>RAG Notebook</span>
        </div>
        <div className={styles.headerRight}>
          <div className={styles.searchWrap}>
            <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="var(--text-3)" strokeWidth="2">
              <circle cx="11" cy="11" r="8"/><path d="M21 21l-4.35-4.35"/>
            </svg>
            <input ref={searchRef} className={styles.search} placeholder="Search notebooks…"
              value={search} onChange={e => setSearch(e.target.value)} />
          </div>
          <button className={styles.themeBtn} onClick={toggleTheme} title="Toggle dark mode">
            {theme === 'light' ? (
              <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <path d="M21 12.79A9 9 0 1111.21 3 7 7 0 0021 12.79z"/>
              </svg>
            ) : (
              <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <circle cx="12" cy="12" r="5"/>
                <path d="M12 1v2M12 21v2M4.22 4.22l1.42 1.42M18.36 18.36l1.42 1.42M1 12h2M21 12h2M4.22 19.78l1.42-1.42M18.36 5.64l1.42-1.42"/>
              </svg>
            )}
          </button>
          <button className={styles.createBtn} onClick={handleCreate}>
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
              <path d="M12 5v14M5 12h14"/>
            </svg>
            New Notebook
          </button>
        </div>
      </header>

      {/* ── Main ── */}
      <main className={styles.main}>
        <div className={styles.sectionHead}>
          <h1 className={styles.sectionTitle}>Your Notebooks</h1>
          <span className={styles.sectionCount}>{filtered.length}</span>
        </div>

        {!isLoading && filtered.length === 0 && !search && (
          <div className={styles.empty}>
            <div className={styles.emptyIcon}>📓</div>
            <div className={styles.emptyTitle}>No notebooks yet</div>
            <div className={styles.emptyDesc}>Create your first notebook to get started.</div>
            <button className={styles.createBtn} onClick={handleCreate}>
              Create notebook
            </button>
          </div>
        )}

        {!isLoading && filtered.length === 0 && search && (
          <div className={styles.empty}>
            <div className={styles.emptyTitle}>No results for "{search}"</div>
          </div>
        )}

        <div className={styles.grid}>
          {isLoading
            ? Array.from({ length: 4 }).map((_, i) => (
                <div key={i} className={styles.skeletonCard}>
                  <div className="skeleton" style={{ width: 40, height: 40, borderRadius: 12 }} />
                  <div className="skeleton" style={{ width: '60%', height: 16, marginTop: 12 }} />
                  <div className="skeleton" style={{ width: '40%', height: 12, marginTop: 8 }} />
                </div>
              ))
            : filtered.map(nb => (
                <NotebookCard key={nb.id} notebook={nb}
                  onOpen={openNotebook}
                  onDelete={handleDelete}
                  onRename={handleRename} />
              ))}
        </div>
      </main>
    </div>
  );
}
