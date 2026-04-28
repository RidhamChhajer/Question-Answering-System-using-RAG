import { useState, useRef, useEffect } from 'react';
import styles from './InputBar.module.css';

const MODE_INFO = {
  standard: 'Fast · Single-pass answer from top sources',
  mapreduce: 'Thorough · Analyses each section individually (slower)',
};

export default function InputBar({ sources, onSend, mode, onModeChange, disabled,
                                   onDropMention }) {
  const [text, setText]               = useState('');
  const [mentions, setMentions]       = useState([]);
  const [mentionQuery, setMentionQuery] = useState(null); // null or string
  const [showModeMenu, setShowModeMenu] = useState(false);
  const inputRef  = useRef(null);
  const modeRef   = useRef(null);

  // Close mode menu on outside click
  useEffect(() => {
    if (!showModeMenu) return;
    const h = (e) => { if (!modeRef.current?.contains(e.target)) setShowModeMenu(false); };
    document.addEventListener('mousedown', h);
    return () => document.removeEventListener('mousedown', h);
  }, [showModeMenu]);

  // Handle drag-drop from sources panel
  const handleDrop = (e) => {
    e.preventDefault();
    const filename = e.dataTransfer.getData('text/plain');
    if (filename) addMention(filename);
  };

  const addMention = (filename) => {
    if (mentions.includes(filename)) return;
    setMentions(prev => [...prev, filename]);
    setText(prev => {
      // Remove trailing @query if present
      const atIdx = prev.lastIndexOf('@');
      if (atIdx !== -1) return prev.slice(0, atIdx);
      return prev;
    });
    setMentionQuery(null);
    inputRef.current?.focus();
  };

  const removeMention = (filename) => setMentions(prev => prev.filter(m => m !== filename));

  // Detect @-trigger in input
  const handleChange = (e) => {
    const val = e.target.value;
    setText(val);
    const atIdx = val.lastIndexOf('@');
    if (atIdx !== -1) {
      const after = val.slice(atIdx + 1);
      if (!after.includes(' ')) {
        setMentionQuery(after.toLowerCase());
        return;
      }
    }
    setMentionQuery(null);
  };

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && e.ctrlKey) { e.preventDefault(); handleSend(); return; }
    if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); handleSend(); }
    if (e.key === 'Escape') setMentionQuery(null);
  };

  const handleSend = () => {
    const q = text.trim();
    if (!q || disabled) return;
    onSend(q, mentions);
    setText('');
    setMentions([]);
    setMentionQuery(null);
  };

  // Filtered sources for dropdown
  const dropdownSources = mentionQuery !== null
    ? sources.filter(s =>
        s.filename.toLowerCase().includes(mentionQuery) && !mentions.includes(s.filename)
      )
    : [];

  const readySources = sources.filter(s => s.status === 'ready' && !mentions.includes(s.filename));

  return (
    <div className={styles.wrap}>
      {/* @mention dropdown (appears above input) */}
      {dropdownSources.length > 0 && (
        <div className={styles.mentionDropdown}>
          {dropdownSources.map(s => (
            <button key={s.id} className={styles.mentionOption}
              onMouseDown={e => { e.preventDefault(); addMention(s.filename); }}>
              📄 {s.filename}
            </button>
          ))}
        </div>
      )}

      <div className={`${styles.bar} ${disabled ? styles.disabled : ''}`}
        onDragOver={e => e.preventDefault()}
        onDrop={handleDrop}>

        {/* + button for @mention */}
        <div className={styles.plusWrap}>
          <button className={styles.plusBtn}
            title="Mention a source"
            onClick={() => setMentionQuery('')}
            disabled={disabled}>
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
              <path d="M12 5v14M5 12h14"/>
            </svg>
          </button>
          {/* + dropdown when mentionQuery === '' (clicked plus) */}
          {mentionQuery === '' && readySources.length > 0 && (
            <div className={styles.mentionDropdown} style={{ bottom: '40px', top: 'auto' }}>
              {readySources.map(s => (
                <button key={s.id} className={styles.mentionOption}
                  onMouseDown={e => { e.preventDefault(); addMention(s.filename); }}>
                  📄 {s.filename}
                </button>
              ))}
            </div>
          )}
        </div>

        {/* Mention chips + text input */}
        <div className={styles.inputArea}>
          {mentions.map(m => (
            <span key={m} className={styles.chip}>
              @{m.replace(/\.[^.]+$/, '')}
              <button onClick={() => removeMention(m)}>×</button>
            </span>
          ))}
          <input ref={inputRef}
            className={styles.input}
            placeholder={disabled ? 'Generating…' : 'Ask anything…'}
            value={text}
            onChange={handleChange}
            onKeyDown={handleKeyDown}
            disabled={disabled} />
        </div>

        {/* Mode selector */}
        <div ref={modeRef} className={styles.modeWrap}>
          <button className={styles.modeBtn}
            title="Change mode"
            onClick={() => setShowModeMenu(v => !v)}
            disabled={disabled}>
            <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <circle cx="12" cy="12" r="3"/>
              <path d="M19.07 4.93l-1.41 1.41M4.93 4.93l1.41 1.41M12 2v2M12 20v2M2 12h2M20 12h2M19.07 19.07l-1.41-1.41M4.93 19.07l1.41-1.41"/>
            </svg>
          </button>
          {showModeMenu && (
            <div className={styles.modeMenu}>
              {['standard','mapreduce'].map(m => (
                <button key={m}
                  className={`${styles.modeOption} ${mode === m ? styles.modeActive : ''}`}
                  onClick={() => { onModeChange(m); setShowModeMenu(false); }}>
                  <strong>{m === 'standard' ? 'Standard' : 'Deep'}</strong>
                  <span>{m === 'standard' ? '~5-10s' : '~2 min'}</span>
                </button>
              ))}
            </div>
          )}
        </div>

        {/* Send button */}
        <button className={styles.sendBtn} onClick={handleSend} disabled={disabled || !text.trim()}>
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
            <line x1="22" y1="2" x2="11" y2="13"/>
            <polygon points="22 2 15 22 11 13 2 9 22 2"/>
          </svg>
        </button>
      </div>

      {/* Mode description */}
      <div className={styles.modeDesc}>
        <em>{MODE_INFO[mode]}</em>
      </div>
    </div>
  );
}
