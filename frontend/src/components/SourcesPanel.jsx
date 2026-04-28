import styles from './SourcesPanel.module.css';

const formatSize = (bytes) => {
  if (!bytes) return '';
  if (bytes < 1024)    return bytes + ' B';
  if (bytes < 1048576) return (bytes / 1024).toFixed(1) + ' KB';
  return (bytes / 1048576).toFixed(1) + ' MB';
};

export default function SourcesPanel({ sources, onToggle, onUploadMore, onDelete, onRetry, isLoading }) {
  const handleDragStart = (e, filename) => {
    e.dataTransfer.setData('text/plain', filename);
    e.dataTransfer.effectAllowed = 'copy';
  };

  return (
    <div className={styles.panel}>
      <div className={styles.header}>
        <span className={styles.label}>Sources</span>
        <button className={styles.uploadBtn} onClick={onUploadMore}>+ Add</button>
      </div>

      {!isLoading && sources.length === 0 && (
        <div className={styles.empty}>No sources yet.<br/>Upload a PDF to get started.</div>
      )}

      <div className={styles.list}>
        {isLoading ? (
          Array.from({ length: 3 }).map((_, i) => (
            <div key={i} className={styles.skeletonRow}>
              <div className="skeleton" style={{ width: '70%', height: 14 }} />
              <div className="skeleton" style={{ width: 50, height: 20, marginTop: 6 }} />
            </div>
          ))
        ) : (
          sources.map(src => (
            <div key={src.id} className={styles.row}
              draggable
              onDragStart={e => handleDragStart(e, src.filename)}
              title="Drag to chat to @mention">
              {/* Drag handle */}
              <div className={styles.dragHandle}>
                <svg width="12" height="12" viewBox="0 0 24 24" fill="currentColor">
                  <circle cx="9" cy="5" r="1.5"/><circle cx="15" cy="5" r="1.5"/>
                  <circle cx="9" cy="12" r="1.5"/><circle cx="15" cy="12" r="1.5"/>
                  <circle cx="9" cy="19" r="1.5"/><circle cx="15" cy="19" r="1.5"/>
                </svg>
              </div>
              {/* Checkbox */}
              <div className={`${styles.check} ${src.selected ? styles.checked : ''}`}
                onClick={() => onToggle(src.id)} />
              {/* Filename */}
              <div className={styles.nameWrap}>
                <span className={`${styles.name} truncate ${!src.selected ? styles.dimmed : ''}`}>
                  {src.filename}
                </span>
                <span className={styles.meta}>
                  {formatSize(src.file_size)}
                  {src.page_count > 0 && ` · ${src.page_count} page${src.page_count !== 1 ? 's' : ''}`}
                </span>
              </div>
              {/* Status badge */}
              <span className={`${styles.badge} ${styles[src.status?.startsWith('error') ? 'error' : src.status]}`}>
                {src.status === 'ready' ? 'Ready'
                  : src.status?.startsWith('error') ? 'Error'
                  : <><span className={styles.spin}/>Indexing</>}
              </span>
              {src.status?.startsWith('error') && (
                <button
                  className={styles.retryBtn}
                  onClick={() => onRetry(src.id)}
                  title="Retry indexing"
                >
                  ↻ Retry
                </button>
              )}
              <button
                className={styles.deleteBtn}
                onClick={() => {
                  if (!window.confirm(`Delete "${src.filename}"? This will re-index remaining sources.`)) return;
                  onDelete(src.id);
                }}
                title="Delete source"
              >
                <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                  <path d="M18 6L6 18M6 6l12 12"/>
                </svg>
              </button>
            </div>
          ))
        )}
      </div>
    </div>
  );
}
