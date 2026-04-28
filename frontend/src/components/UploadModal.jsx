import { useRef } from 'react';
import styles from './UploadModal.module.css';

const ALLOWED = '.pdf,.doc,.docx,.txt';
const ALLOWED_DESC = 'PDF, Word (.doc, .docx), Text (.txt)';

export default function UploadModal({ isOpen, onClose, onUpload }) {
  const inputRef = useRef(null);

  const handleFiles = (files) => {
    const list = Array.from(files);
    if (list.length > 0) { onUpload(list); }
  };

  const handleDrop = (e) => {
    e.preventDefault();
    handleFiles(e.dataTransfer.files);
  };

  if (!isOpen) return null;

  return (
    <div className={styles.overlay} onClick={e => e.target === e.currentTarget && onClose()}>
      <div className={styles.modal}>
        <button className={styles.close} onClick={onClose} title="Close">
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
            <path d="M18 6L6 18M6 6l12 12"/>
          </svg>
        </button>

        <h2 className={styles.title}>Add sources</h2>
        <p className={styles.desc}>Upload documents to this notebook. The system will index them automatically.</p>

        <div className={styles.dropZone}
          onDragOver={e => e.preventDefault()}
          onDrop={handleDrop}
          onClick={() => inputRef.current?.click()}>
          <div className={styles.dropIcon}>
            <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="var(--accent)" strokeWidth="1.5">
              <path d="M21 15v4a2 2 0 01-2 2H5a2 2 0 01-2-2v-4"/>
              <polyline points="17 8 12 3 7 8"/>
              <line x1="12" y1="3" x2="12" y2="15"/>
            </svg>
          </div>
          <div className={styles.dropTitle}>Drop files here or click to browse</div>
          <div className={styles.dropAllowed}>Accepted: {ALLOWED_DESC}</div>
        </div>

        <input ref={inputRef} type="file" multiple accept={ALLOWED}
          style={{ display: 'none' }}
          onChange={e => handleFiles(e.target.files)} />

        <button className={styles.browseBtn} onClick={() => inputRef.current?.click()}>
          Browse files
        </button>
      </div>
    </div>
  );
}
