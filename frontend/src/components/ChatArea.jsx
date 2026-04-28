import { useEffect, useRef, useState } from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import styles from './ChatArea.module.css';
import PipelineStatus from './PipelineStatus';

function UserMsg({ msg }) {
  return (
    <div className={styles.userRow}>
      <div className={styles.userBubble}>{msg.content}</div>
      <div className={styles.time}>{msg.timestamp}</div>
    </div>
  );
}

function FeedbackButtons({ messageId, existingRating, onFeedback }) {
  const [selected, setSelected] = useState(existingRating || null);
  useEffect(() => {
    setSelected(existingRating || null);
  }, [existingRating]);
  const handle = (rating) => {
    setSelected(rating);
    onFeedback(messageId, rating);
  };
  return (
    <div className={styles.feedbackRow}>
      <button
        className={`${styles.feedbackBtn} ${selected === 'up' ? styles.selectedUp : ''}`}
        onClick={() => handle('up')}
        title="Good answer"
      >
        <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
          <path d="M14 9V5a3 3 0 00-3-3l-4 9v11h11.28a2 2 0 002-1.7l1.38-9a2 2 0 00-2-2.3H14z"/>
          <path d="M7 22H4a2 2 0 01-2-2v-7a2 2 0 012-2h3"/>
        </svg>
      </button>
      <button
        className={`${styles.feedbackBtn} ${selected === 'down' ? styles.selectedDown : ''}`}
        onClick={() => handle('down')}
        title="Bad answer"
      >
        <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
          <path d="M10 15v4a3 3 0 003 3l4-9V2H5.72a2 2 0 00-2 1.7l-1.38 9a2 2 0 002 2.3H10z"/>
          <path d="M17 2h2.67A2.31 2.31 0 0122 4v7a2.31 2.31 0 01-2.33 2H17"/>
        </svg>
      </button>
    </div>
  );
}

function AiMsg({ msg, existingRating, onFeedback }) {
  const [copied, setCopied] = useState(false);
  const [openSource, setOpenSource] = useState(null);

  const handleCopy = () => {
    navigator.clipboard.writeText(msg.content);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  useEffect(() => {
    const handler = (e) => { if (e.key === 'Escape') setOpenSource(null); };
    document.addEventListener('keydown', handler);
    return () => document.removeEventListener('keydown', handler);
  }, []);

  return (
    <div className={styles.aiRow}>
      <div className={styles.aiAvatar}>AI</div>
      <div className={styles.aiContent}>
        <div className={styles.aiBubble}>
          {msg.streaming && msg.content === '' ? (
            <div className={styles.typingDots}>
              <span/><span/><span/>
            </div>
          ) : (
            <ReactMarkdown remarkPlugins={[remarkGfm]}>{msg.content}</ReactMarkdown>
          )}
          {!msg.streaming && (
            <button
              className={`${styles.copyBtn} ${copied ? styles.copied : ''}`}
              onClick={handleCopy}
              title={copied ? 'Copied!' : 'Copy answer'}
            >
              {copied ? (
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                  <polyline points="20 6 9 17 4 12"/>
                </svg>
              ) : (
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                  <rect x="9" y="9" width="13" height="13" rx="2"/>
                  <path d="M5 15H4a2 2 0 01-2-2V4a2 2 0 012-2h9a2 2 0 012 2v1"/>
                </svg>
              )}
            </button>
          )}
          {msg.streaming && msg.content !== '' && (
            <span className={styles.cursor} />
          )}
        </div>
        {msg.sources?.length > 0 && !msg.streaming && (
          <div className={styles.sources}>
            {msg.sources.map((s, i) => (
              <div key={i} className={styles.sourceItem}>
                <button
                  className={styles.sourceChip}
                  onClick={() => setOpenSource(openSource === i ? null : i)}
                >
                  📄 {s.document} · p.{s.page}
                </button>
                {openSource === i && (
                  <div className={styles.chunkPreview}>
                    <div className={styles.chunkHeader}>
                      <strong>{s.document}</strong> — Page {s.page}
                      <button onClick={(e) => { e.stopPropagation(); setOpenSource(null); }}>×</button>
                    </div>
                    <div className={styles.chunkText}>{s.text}</div>
                  </div>
                )}
              </div>
            ))}
          </div>
        )}
        {!msg.streaming && (
          <FeedbackButtons
            messageId={msg.id}
            existingRating={existingRating}
            onFeedback={onFeedback}
          />
        )}
        <div className={styles.time}>{msg.timestamp}</div>
      </div>
    </div>
  );
}

export default function ChatArea({ messages, pipelineSteps, isStreaming, feedbackMap, onFeedback }) {
  const feedRef = useRef(null);

  useEffect(() => {
    if (feedRef.current) feedRef.current.scrollTop = feedRef.current.scrollHeight;
  }, [messages, pipelineSteps]);

  const isEmpty = messages.length === 0 && !isStreaming;

  return (
    <div className={styles.feed} ref={feedRef}>
      {isEmpty && (
        <div className={styles.empty}>
          <p className={styles.emptyMsg}>Ask anything about your sources</p>
        </div>
      )}

      {messages.map(msg =>
        msg.role === 'user'
          ? <UserMsg key={msg.id} msg={msg} />
          : <AiMsg  key={msg.id} msg={msg} existingRating={feedbackMap?.[msg.id]} onFeedback={onFeedback} />
      )}

      {pipelineSteps.length > 0 && (
        <PipelineStatus steps={pipelineSteps} />
      )}
    </div>
  );
}
