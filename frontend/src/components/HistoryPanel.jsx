import { useState, useRef } from 'react';
import { renameConversation } from '../api/client';
import styles from './HistoryPanel.module.css';

function ConvRow({ conv, isActive, onSelect, onRenamed, onDelete }) {
  const [editing, setEditing] = useState(false);
  const [val, setVal]         = useState(conv.title);
  const inputRef = useRef(null);

  const save = async () => {
    setEditing(false);
    const t = val.trim() || conv.title;
    if (t === conv.title) return;
    const updated = await renameConversation(conv.id, t).catch(() => null);
    if (updated) onRenamed(updated);
  };

  const fmt = (iso) => {
    const d = new Date(iso + 'Z');
    return d.toLocaleDateString([], { month: 'short', day: 'numeric' })
      + ' · ' + d.toLocaleTimeString([], { hour:'2-digit', minute:'2-digit' });
  };

  return (
    <div className={`${styles.row} ${isActive ? styles.active : ''}`} onClick={() => onSelect(conv.id)}>
      <div className={styles.rowTop}>
        {editing ? (
          <input ref={inputRef} autoFocus
            className={styles.editInput}
            value={val}
            onChange={e => setVal(e.target.value)}
            onBlur={save}
            onKeyDown={e => { if (e.key==='Enter') save(); if (e.key==='Escape') setEditing(false); }}
            onClick={e => e.stopPropagation()} />
        ) : (
          <span className={`${styles.title} truncate`}>{conv.title}</span>
        )}
        <button className={styles.renameBtn}
          title="Rename"
          onClick={e => { e.stopPropagation(); setEditing(true); setVal(conv.title); }}>
          <svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <path d="M11 4H4a2 2 0 00-2 2v14a2 2 0 002 2h14a2 2 0 002-2v-7"/>
            <path d="M18.5 2.5a2.121 2.121 0 013 3L12 15l-4 1 1-4 9.5-9.5z"/>
          </svg>
        </button>
        <button
          className={styles.deleteBtn}
          onClick={(e) => {
            e.stopPropagation();
            if (!window.confirm("Delete this conversation and all its messages?")) return;
            onDelete(conv.id);
          }}
          title="Delete conversation"
        >
          <svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <polyline points="3 6 5 6 21 6"/>
            <path d="M19 6v14a2 2 0 01-2 2H7a2 2 0 01-2-2V6m3 0V4a2 2 0 012-2h4a2 2 0 012 2v2"/>
          </svg>
        </button>
      </div>
      <div className={styles.date}>{fmt(conv.created_at)}</div>
    </div>
  );
}

export default function HistoryPanel({ conversations, activeConvId, onSelect, onNew, onConvsChange, onDelete, isLoading }) {
  const handleRenamed = (updated) => {
    onConvsChange(prev => prev.map(c => c.id === updated.id ? updated : c));
  };

  return (
    <div className={styles.panel}>
      <div className={styles.header}>
        <span className={styles.label}>Conversations</span>
        <button className={styles.newBtn} onClick={onNew}>+ New</button>
      </div>

      {!isLoading && conversations.length === 0 && (
        <div className={styles.empty}>No conversations yet.</div>
      )}

      <div className={styles.list}>
        {isLoading ? (
          Array.from({ length: 3 }).map((_, i) => (
            <div key={i} className={styles.skeletonRow}>
              <div className="skeleton" style={{ width: '80%', height: 14 }} />
              <div className="skeleton" style={{ width: '40%', height: 10, marginTop: 4 }} />
            </div>
          ))
        ) : (
          conversations.map(conv => (
            <ConvRow key={conv.id} conv={conv}
              isActive={conv.id === activeConvId}
              onSelect={onSelect}
              onRenamed={handleRenamed}
              onDelete={onDelete} />
          ))
        )}
      </div>
    </div>
  );
}
