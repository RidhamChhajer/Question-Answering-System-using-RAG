import { useState, useEffect, useCallback, useRef } from 'react';
import { useNavigate } from 'react-router-dom';
import {
  fetchSources, uploadFiles, fetchConversations,
  createConversation, fetchMessages, streamAsk, renameNotebook,
  sendFeedback, fetchFeedback, deleteSource, deleteConversation, retrySource,
} from '../api/client';
import ChatArea    from '../components/ChatArea';
import InputBar    from '../components/InputBar';
import SourcesPanel from '../components/SourcesPanel';
import HistoryPanel from '../components/HistoryPanel';
import UploadModal from '../components/UploadModal';
import styles from './Workspace.module.css';

const MIN_RIGHT = 20; // % of viewport
const MAX_RIGHT = 40;
const DEF_RIGHT = 35;

export default function Workspace({ notebook, initConvId }) {
  const navigate = useNavigate();
  const [sources, setSources]         = useState([]);
  const [conversations, setConvs]     = useState([]);
  const [activeConvId, setActiveConvId] = useState(initConvId);
  const [messages, setMessages]       = useState([]);
  const [pipelineSteps, setPipeline]  = useState([]);
  const [isStreaming, setIsStreaming] = useState(false);
  const [mode, setMode]               = useState('standard');
  const [rightTab, setRightTab]       = useState('sources');
  const [rightPct, setRightPct]       = useState(DEF_RIGHT);
  const [showUpload, setShowUpload]   = useState(false);
  const [nbTitle, setNbTitle]         = useState(notebook?.title || 'Untitled');
  const [editingTitle, setEditingTitle] = useState(false);
  const [feedbackMap, setFeedbackMap] = useState({});
  const [theme, setTheme] = useState(
    document.documentElement.getAttribute('data-theme') || 'light'
  );
  const [isSourcesLoading, setIsSourcesLoading] = useState(true);
  const [isConvsLoading, setIsConvsLoading] = useState(true);
  const [showSidebar, setShowSidebar] = useState(true);
  const titleInputRef = useRef(null);
  const dragging      = useRef(false);
  const startX        = useRef(0);
  const startPct      = useRef(DEF_RIGHT);
  const pollRef       = useRef(null);
  // Tracks the ID of an auto-created empty conversation so we can delete it
  // if the user navigates away before sending any message.
  const emptyConvRef  = useRef(null);
  const nbId = notebook?.id;

  // ── Load sources ──────────────────────────────────────────────────────────
  const loadSources = useCallback(() => {
    fetchSources(nbId)
      .then(srcs => setSources(srcs.map(s => ({ ...s, selected: s.status === 'ready' }))))
      .catch(console.error)
      .finally(() => setIsSourcesLoading(false));
  }, [nbId]);

  useEffect(() => { loadSources(); }, [loadSources]);

  // Delete the current empty auto-created conv (if any) before switching away.
  const discardEmptyConv = useCallback(async () => {
    const emptyId = emptyConvRef.current;
    if (!emptyId) return;
    emptyConvRef.current = null;
    try {
      await deleteConversation(emptyId);
      setConvs(prev => prev.filter(c => c.id !== emptyId));
    } catch (_) { /* ignore */ }
  }, []);

  const newConv = useCallback(async () => {
    await discardEmptyConv();          // clean up previous empty chat first
    const conv = await createConversation(nbId);
    emptyConvRef.current = conv.id;    // mark as empty until a message is sent
    setConvs(prev => [conv, ...prev]);
    setActiveConvId(conv.id);
    setMessages([]);
    setFeedbackMap({});
    setPipeline([]);
  }, [nbId, discardEmptyConv]);

  useEffect(() => {
    const handler = (e) => {
      const isInput = ['INPUT', 'TEXTAREA'].includes(e.target.tagName);

      if (e.ctrlKey && e.key === 'n' && !isInput) {
        e.preventDefault();
        newConv();
      }

      if (e.key === 'Escape' && showUpload) {
        setShowUpload(false);
      }
    };

    document.addEventListener('keydown', handler);
    return () => document.removeEventListener('keydown', handler);
  }, [newConv, showUpload]);

  // Auto-poll while any source is indexing
  useEffect(() => {
    const hasProcessing = sources.some(s => s.status === 'processing');
    if (!hasProcessing) { clearInterval(pollRef.current); return; }
    pollRef.current = setInterval(() => {
      fetchSources(nbId).then(srcs => {
        setSources(prev => srcs.map(s => ({
          ...s,
          selected: prev.find(p => p.id === s.id)?.selected ?? true,
        })));
      }).catch(console.error);
    }, 3000);
    return () => clearInterval(pollRef.current);
  }, [sources, nbId]);

  // ── Load / create conversations ──────────────────────────────────────────
  const loadConvs = useCallback(() =>
    fetchConversations(nbId).then(setConvs).catch(console.error), [nbId]);

  useEffect(() => {
    fetchConversations(nbId).then(async (convs) => {
      setConvs(convs);
      if (initConvId) {
        setActiveConvId(initConvId);
        const msgs = await fetchMessages(initConvId);
        setMessages(msgs.map(m => fmtMsg(m)));
        const map = await fetchFeedback(initConvId);
        setFeedbackMap(map);
      } else {
        // Reuse the most recent conversation if it is still empty,
        // so refreshing the page doesn't keep adding blank chats.
        const newestEmpty = convs.find(c => c.message_count === 0);
        if (newestEmpty) {
          emptyConvRef.current = newestEmpty.id;
          setActiveConvId(newestEmpty.id);
          setMessages([]);
          setFeedbackMap({});
        } else {
          const conv = await createConversation(nbId);
          emptyConvRef.current = conv.id;
          setConvs([conv, ...convs]);
          setActiveConvId(conv.id);
          setMessages([]);
          setFeedbackMap({});
          if (convs.length === 0) setShowUpload(true); // first time → open upload
        }
      }
    }).catch(console.error)
      .finally(() => setIsConvsLoading(false));
  }, [nbId, initConvId]);

  const fmtMsg = (m) => ({
    id: m.id, role: m.role, content: m.content, sources: m.sources || [],
    timestamp: new Date(m.created_at + 'Z').toLocaleTimeString([], { hour:'2-digit', minute:'2-digit' }),
    streaming: false,
  });

  // ── Switch conversation ───────────────────────────────────────────────────
  const selectConv = useCallback(async (cid) => {
    await discardEmptyConv();          // delete the empty chat before switching
    setActiveConvId(cid);
    const msgs = await fetchMessages(cid);
    setMessages(msgs.map(fmtMsg));
    const map = await fetchFeedback(cid);
    setFeedbackMap(map);
    setPipeline([]);
  }, [discardEmptyConv]);

  const handleDeleteConv = useCallback(async (cid) => {
    await deleteConversation(cid);
    setConvs(prev => prev.filter(c => c.id !== cid));

    if (cid === activeConvId) {
      const conv = await createConversation(nbId);
      setConvs(prev => [conv, ...prev]);
      setActiveConvId(conv.id);
      setMessages([]);
      setFeedbackMap({});
      setPipeline([]);
    }
  }, [activeConvId, nbId]);

  const handleFeedback = useCallback(async (messageId, rating) => {
    await sendFeedback(messageId, rating);
    setFeedbackMap(prev => ({ ...prev, [messageId]: rating }));
  }, []);

  // ── Send message ──────────────────────────────────────────────────────────
  const handleSend = useCallback(async (question, mentions) => {
    if (!activeConvId || isStreaming) return;

    const checkedSources  = sources.filter(s => s.selected && s.status === 'ready').map(s => s.filename);
    const mentionedSources = mentions;

    // Validate: if no mentions and no checked → error
    if (mentionedSources.length === 0 && checkedSources.length === 0) {
      setMessages(prev => [...prev, {
        id: `err-${Date.now()}`, role: 'ai', content: '⚠️ No sources selected. Enable at least one source to get an answer.',
        sources: [], timestamp: now(), streaming: false,
      }]);
      return;
    }

    const userMsg = { id: `u-${Date.now()}`, role:'user', content:question, sources:[], timestamp:now(), streaming:false };
    const aiId    = `a-${Date.now()}`;
    const aiMsg   = { id:aiId, role:'ai', content:'', sources:[], timestamp:now(), streaming:true };

    setMessages(prev => [...prev, userMsg, aiMsg]);
    setIsStreaming(true);
    setPipeline([]);

    const steps = [];

    await streamAsk({
      question, conversationId: activeConvId, mode,
      checkedSources, mentionedSources,
      onStep: (ev) => {
        steps.push({ content: ev.content, eta: ev.eta, status: 'active' });
        const updated = steps.map((s, i) => ({
          ...s, status: i < steps.length - 1 ? 'done' : 'active',
        }));
        setPipeline([...updated]);
      },
      onToken: (token) => {
        setPipeline([]); // clear steps once tokens start
        setMessages(prev => prev.map(m => m.id === aiId ? { ...m, content: m.content + token } : m));
      },
      onSources: (srcs) => {
        setMessages(prev => prev.map(m => m.id === aiId ? { ...m, sources: srcs } : m));
      },
      onDone: () => {
        emptyConvRef.current = null;   // message sent → conversation is no longer empty
        setIsStreaming(false);
        setPipeline([]);
        setMessages(prev => prev.map(m => m.id === aiId ? { ...m, streaming: false } : m));
        // Refresh conversations (auto-title may have updated)
        fetchConversations(nbId).then(setConvs).catch(console.error);
      },
      onError: (err) => {
        setIsStreaming(false);
        setPipeline([]);
        setMessages(prev => prev.map(m => m.id === aiId
          ? { ...m, content: `⚠️ ${err}`, streaming: false } : m));
      },
    });
  }, [activeConvId, isStreaming, mode, sources, nbId]);

  // ── Upload ────────────────────────────────────────────────────────────────
  const handleUpload = useCallback(async (files) => {
    setShowUpload(false);
    const newSrcs = await uploadFiles(nbId, files);
    setSources(prev => [...prev, ...newSrcs.map(s => ({ ...s, selected: true }))]);
  }, [nbId]);

  const handleDeleteSource = useCallback(async (sid) => {
    await deleteSource(sid);
    setSources(prev => prev.filter(s => s.id !== sid));
    // Poll again after re-indexing starts
    setTimeout(loadSources, 2000);
  }, [loadSources]);

  const handleRetrySource = useCallback(async (sid) => {
    await retrySource(sid);
    setSources(prev =>
      prev.map(s => s.id === sid ? { ...s, status: 'processing' } : s)
    );
  }, []);

  // ── Source toggle ─────────────────────────────────────────────────────────
  const toggleSource = useCallback((id) => {
    setSources(prev => prev.map(s => (
      s.id === id && s.status === 'ready' ? { ...s, selected: !s.selected } : s
    )));
  }, []);

  // ── Notebook title edit ───────────────────────────────────────────────────
  const saveTitle = async () => {
    setEditingTitle(false);
    const t = nbTitle.trim() || 'Untitled';
    setNbTitle(t);
    const updated = await renameNotebook(nbId, t).catch(() => null);
    if (updated?.title) setNbTitle(updated.title);
  };

  const toggleTheme = () => {
    const next = theme === 'light' ? 'dark' : 'light';
    setTheme(next);
    document.documentElement.setAttribute('data-theme', next);
    localStorage.setItem('theme', next);
  };

  const handleExport = useCallback(() => {
    if (messages.length === 0) return;

    let md = `# ${nbTitle}\n\n`;
    md += `*Exported: ${new Date().toLocaleString()}*\n\n---\n\n`;

    for (const msg of messages) {
      if (msg.role === 'user') {
        md += `## Q: ${msg.content}\n\n`;
      } else {
        md += `${msg.content}\n\n`;
        if (msg.sources?.length > 0) {
          const srcList = msg.sources
            .map(s => `${s.document} p.${s.page}`)
            .join(', ');
          md += `**Sources:** ${srcList}\n\n`;
        }
        md += `---\n\n`;
      }
    }

    const safeTitle = nbTitle.replace(/[\\/:*?"<>|]/g, '-');
    const blob = new Blob([md], { type: 'text/markdown' });
    const url  = URL.createObjectURL(blob);
    const a    = document.createElement('a');
    a.href     = url;
    a.download = `${safeTitle} - conversation.md`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  }, [messages, nbTitle]);

  useEffect(() => { if (editingTitle) titleInputRef.current?.select(); }, [editingTitle]);

  // ── Resizable divider ─────────────────────────────────────────────────────
  const startDrag = useCallback((e) => {
    dragging.current = true;
    startX.current   = e.clientX;
    startPct.current = rightPct;
    document.body.style.cursor = 'col-resize';
    document.body.style.userSelect = 'none';

    const onMove = (ev) => {
      if (!dragging.current) return;
      const deltaPx  = startX.current - ev.clientX;
      const deltaPct = (deltaPx / window.innerWidth) * 100;
      const newPct   = Math.min(MAX_RIGHT, Math.max(MIN_RIGHT, startPct.current + deltaPct));
      setRightPct(newPct);
    };
    const onUp = () => {
      dragging.current = false;
      document.body.style.cursor = '';
      document.body.style.userSelect = '';
      document.removeEventListener('mousemove', onMove);
      document.removeEventListener('mouseup', onUp);
    };
    document.addEventListener('mousemove', onMove);
    document.addEventListener('mouseup', onUp);
  }, [rightPct]);

  return (
    <div className={styles.page}>
      {/* ── Left panel ── */}
      <div className={styles.left} style={{ width: `${100 - rightPct}%` }}>

        {/* Top bar */}
        <div className={styles.topBar}>
          <button className={styles.backBtn} onClick={() => navigate('/')} title="Back to notebooks">
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
              <path d="M19 12H5M12 19l-7-7 7-7"/>
            </svg>
          </button>
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
          <button
            className={styles.sidebarToggle}
            onClick={() => setShowSidebar(prev => !prev)}
            title={showSidebar ? 'Hide sidebar' : 'Show sidebar'}
          >
            {showSidebar ? '✕ Hide' : '☰ Sources'}
          </button>

          {editingTitle ? (
            <input ref={titleInputRef}
              className={styles.titleInput}
              value={nbTitle}
              onChange={e => setNbTitle(e.target.value)}
              onBlur={saveTitle}
              onKeyDown={e => { if (e.key==='Enter') saveTitle(); if (e.key==='Escape') { setNbTitle(notebook.title); setEditingTitle(false); } }}
              autoFocus />
          ) : (
            <h1 className={styles.title} title="Click to rename"
              onClick={() => setEditingTitle(true)}>
              {nbTitle}
              <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" className={styles.editIcon}>
                <path d="M11 4H4a2 2 0 00-2 2v14a2 2 0 002 2h14a2 2 0 002-2v-7"/>
                <path d="M18.5 2.5a2.121 2.121 0 013 3L12 15l-4 1 1-4 9.5-9.5z"/>
              </svg>
            </h1>
          )}
          <button
            className={styles.exportBtn}
            onClick={handleExport}
            title="Export conversation as Markdown"
            disabled={messages.length === 0}
          >
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <path d="M21 15v4a2 2 0 01-2 2H5a2 2 0 01-2-2v-4"/>
              <polyline points="7 10 12 15 17 10"/>
              <line x1="12" y1="15" x2="12" y2="3"/>
            </svg>
            Export
          </button>
        </div>

        {/* Chat */}
        <div className={styles.chatWrap}>
          <ChatArea messages={messages} feedbackMap={feedbackMap} onFeedback={handleFeedback}
            pipelineSteps={pipelineSteps} isStreaming={isStreaming} />
          <InputBar sources={sources} mode={mode} onModeChange={setMode}
            disabled={isStreaming} onSend={handleSend} />
        </div>
      </div>

      {/* ── Divider ── */}
      {showSidebar && (
        <>
          {/* ── Divider ── */}
          <div className={styles.divider} onMouseDown={startDrag} title="Drag to resize" />

          {/* ── Right panel ── */}
          <div className={styles.right} style={{ width: `${rightPct}%` }}>
            {/* Tab buttons */}
            <div className={styles.tabs}>
              <button className={`${styles.tab} ${rightTab==='sources' ? styles.tabActive:''}`}
                onClick={() => setRightTab('sources')}>Sources</button>
              <button className={`${styles.tab} ${rightTab==='history' ? styles.tabActive:''}`}
                onClick={() => setRightTab('history')}>History</button>
            </div>

            {rightTab === 'sources' && (
              <SourcesPanel sources={sources} onToggle={toggleSource} onDelete={handleDeleteSource}
                onRetry={handleRetrySource} isLoading={isSourcesLoading}
                onUploadMore={() => setShowUpload(true)} />
            )}
            {rightTab === 'history' && (
              <HistoryPanel conversations={conversations} activeConvId={activeConvId}
                onSelect={selectConv} onNew={newConv} onConvsChange={setConvs}
                onDelete={handleDeleteConv} isLoading={isConvsLoading} />
            )}
          </div>
        </>
      )}

      <UploadModal isOpen={showUpload} onClose={() => setShowUpload(false)} onUpload={handleUpload} />
    </div>
  );
}

const now = () => new Date().toLocaleTimeString([], { hour:'2-digit', minute:'2-digit' });
