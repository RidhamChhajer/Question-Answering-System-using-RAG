import { useState, useRef, useEffect } from 'react';
import { deleteNotebook, renameNotebook } from '../api/client';
import styles from './NotebookCard.module.css';

export default function NotebookCard({ notebook, onOpen, onDelete, onRename }) {
  const [menuOpen, setMenuOpen]     = useState(false);
  const [renaming, setRenaming]     = useState(false);
  const [renameVal, setRenameVal]   = useState(notebook.title);
  const [confirmDel, setConfirmDel] = useState(false);
  const menuRef = useRef(null);
  const inputRef = useRef(null);

  // Close menu on outside click
  useEffect(() => {
    if (!menuOpen) return;
    const handler = (e) => { if (!menuRef.current?.contains(e.target)) setMenuOpen(false); };
    document.addEventListener('mousedown', handler);
    return () => document.removeEventListener('mousedown', handler);
  }, [menuOpen]);

  useEffect(() => { if (renaming) inputRef.current?.select(); }, [renaming]);

  const handleRename = async () => {
    const t = renameVal.trim() || 'Untitled';
    setRenaming(false);
    if (t === notebook.title) return;
    const updated = await renameNotebook(notebook.id, t).catch(() => null);
    if (updated) onRename(updated);
  };

  const handleDelete = async () => {
    await deleteNotebook(notebook.id).catch(() => {});
    onDelete(notebook.id);
  };

  return (
    <div className={styles.card} onClick={() => !renaming && onOpen(notebook)}>
      <div className={styles.emoji}>{notebook.emoji}</div>

      {/* 3-dot menu */}
      <div ref={menuRef} className={styles.menuWrap} onClick={e => e.stopPropagation()}>
        <button className={styles.menuTrigger} onClick={() => setMenuOpen(v => !v)}>
          <svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor">
            <circle cx="12" cy="5"  r="1.5"/><circle cx="12" cy="12" r="1.5"/><circle cx="12" cy="19" r="1.5"/>
          </svg>
        </button>
        {menuOpen && (
          <div className={styles.menu}>
            <button onClick={() => { setMenuOpen(false); setRenaming(true); setRenameVal(notebook.title); }}>
              ✏️ Rename
            </button>
            {!confirmDel
              ? <button className={styles.danger} onClick={() => setConfirmDel(true)}>🗑️ Delete</button>
              : <button className={styles.danger} onClick={handleDelete}>Confirm delete</button>
            }
          </div>
        )}
      </div>

      {/* Title */}
      {renaming ? (
        <input ref={inputRef} className={styles.renameInput}
          value={renameVal}
          onChange={e => setRenameVal(e.target.value)}
          onBlur={handleRename}
          onKeyDown={e => { if (e.key === 'Enter') handleRename(); if (e.key === 'Escape') setRenaming(false); }}
          onClick={e => e.stopPropagation()} autoFocus />
      ) : (
        <div className={`${styles.title} truncate`}>{notebook.title}</div>
      )}

      <div className={styles.meta}>
        <span>{notebook.source_count || 0} source{notebook.source_count !== 1 ? 's' : ''}</span>
        <span>{notebook.conversation_count || 0} chat{notebook.conversation_count !== 1 ? 's' : ''}</span>
      </div>
    </div>
  );
}
