"""
server.py — FastAPI backend for the RAG web app.
Run: uvicorn api.server:app --reload --port 8000
"""
import os, sys, json, asyncio, shutil, random, re
from pathlib import Path
from fastapi import FastAPI, HTTPException, UploadFile, File, Form, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from starlette.requests import Request

ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(ROOT))

import config

from api import database

from api.database import (
    init_db, create_notebook, get_notebook, list_notebooks, update_notebook,
    touch_notebook, delete_notebook,
    create_conversation, get_conversation, list_conversations, rename_conversation,
    auto_title_conversation, count_messages, delete_conversation,
    add_message, list_messages,
    add_source, list_sources, update_source_status, delete_source, get_source,
    update_source_metadata,
    EMOJIS,
)

limiter = Limiter(key_func=get_remote_address)

app = FastAPI(title="RAG API", version="2.0.0")
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(CORSMiddleware,
    allow_origins=["http://localhost:5173","http://127.0.0.1:5173"],
    allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

LLM_MODEL     = config.OLLAMA_MODEL

@app.on_event("startup")
def startup():
    init_db()

# ── Models ─────────────────────────────────────────────────────────────────────
class NotebookBody(BaseModel):
    title: str = "Untitled"
    emoji: str | None = None

class PatchNotebookBody(BaseModel):
    title: str | None = None
    emoji: str | None = None

class ConvBody(BaseModel):
    title: str = "New conversation"

class RenameConvBody(BaseModel):
    title: str

class AskBody(BaseModel):
    question: str
    conversation_id: str
    mode: str = "standard"
    top_k: int = 5
    model: str = LLM_MODEL
    checked_sources: list[str] = []    # filenames with checkbox ON
    mentioned_sources: list[str] = []  # @mentioned filenames (override)

class FeedbackBody(BaseModel):
    rating: str

# ── Notebooks ──────────────────────────────────────────────────────────────────
@app.get("/api/notebooks")
def get_notebooks():
    nbs = list_notebooks()
    for nb in nbs:
        sources = list_sources(nb["id"])
        nb["source_count"]       = len([s for s in sources if s["status"] == "ready"])
        nb["conversation_count"] = len(list_conversations(nb["id"]))
    return {"notebooks": nbs}

@app.post("/api/notebooks", status_code=201)
def post_notebook(body: NotebookBody):
    emoji = body.emoji or random.choice(EMOJIS)
    nb = create_notebook(title=body.title, emoji=emoji)
    return {"notebook": nb}

@app.patch("/api/notebooks/{nid}")
def patch_notebook(nid: str, body: PatchNotebookBody):
    if not get_notebook(nid): raise HTTPException(404, "Notebook not found")
    if body.title is not None:
        if not body.title.strip():
            raise HTTPException(400, "Title cannot be empty.")
        if len(body.title) > 100:
            raise HTTPException(400, "Title too long. Maximum 100 characters.")
    return {"notebook": update_notebook(nid, title=body.title, emoji=body.emoji)}

@app.delete("/api/notebooks/{nid}", status_code=204)
def del_notebook(nid: str):
    if not get_notebook(nid): raise HTTPException(404, "Notebook not found")
    delete_notebook(nid)

@app.post("/api/notebooks/{nid}/touch")
def touch_nb(nid: str):
    """Call when a notebook is opened — keeps it at top of list."""
    if not get_notebook(nid): raise HTTPException(404, "Notebook not found")
    touch_notebook(nid)
    return {"ok": True}

# ── Conversations ──────────────────────────────────────────────────────────────
@app.get("/api/notebooks/{nid}/conversations")
def get_convs(nid: str):
    if not get_notebook(nid): raise HTTPException(404)
    return {"conversations": list_conversations(nid)}

@app.post("/api/notebooks/{nid}/conversations", status_code=201)
def post_conv(nid: str, body: ConvBody):
    if not get_notebook(nid): raise HTTPException(404)
    return {"conversation": create_conversation(nid, title=body.title)}

@app.patch("/api/conversations/{cid}")
def patch_conv(cid: str, body: RenameConvBody):
    if not get_conversation(cid): raise HTTPException(404)
    if not body.title.strip():
        raise HTTPException(400, "Title cannot be empty.")
    if len(body.title) > 200:
        raise HTTPException(400, "Title too long. Maximum 200 characters.")
    return {"conversation": rename_conversation(cid, body.title)}

@app.delete("/api/conversations/{cid}", status_code=204)
def del_conv(cid: str):
    if not get_conversation(cid):
        raise HTTPException(404, "Conversation not found")
    delete_conversation(cid)

@app.get("/api/conversations/{cid}/messages")
def get_msgs(cid: str):
    if not get_conversation(cid): raise HTTPException(404)
    return {"messages": list_messages(cid)}

@app.post("/api/messages/{mid}/feedback")
def post_feedback(mid: str, body: FeedbackBody):
    if body.rating not in ("up", "down"):
        raise HTTPException(400, "Rating must be 'up' or 'down'")
    # Verify message exists
    with database._conn() as c:
        if not c.execute("SELECT 1 FROM messages WHERE id=?", (mid,)).fetchone():
            raise HTTPException(404, "Message not found")
    result = database.add_feedback(mid, body.rating)
    return {"feedback": result}

@app.get("/api/conversations/{cid}/feedback")
def get_conv_feedback(cid: str):
    return {"feedback": database.get_feedbacks_for_conversation(cid)}

# ── Ask (SSE) ──────────────────────────────────────────────────────────────────
def _sse(obj): return f"data: {json.dumps(obj)}\n\n"

async def _stream(body: AskBody):
    import ollama as _ol
    from src.retriever   import load_retriever, retrieve
    from src.reranker    import rerank_chunks
    from src.rag_engine  import assemble_context, build_prompt, FALLBACK_ANSWER, format_history

    # Resolve notebook-specific paths from conversation
    conv = get_conversation(body.conversation_id)
    if not conv:
        yield _sse({"type":"error","content":"Conversation not found."})
        return
    nb_id         = conv["notebook_id"]
    nb_index_path = config.get_index_path(nb_id)

    # Guard: index must exist
    if not os.path.isfile(nb_index_path):
        yield _sse({"type":"error","content":"No documents indexed yet. Upload a PDF first."})
        return

    # ── Determine effective sources ────────────────────────────────────────────
    if body.mentioned_sources:
        effective = body.mentioned_sources          # @mention = exclusive
    elif body.checked_sources:
        effective = body.checked_sources            # checked boxes
    else:
        yield _sse({"type":"error","content":"No sources selected. Enable at least one source to get an answer."})
        return

    try:
        # Step 1: embed
        yield _sse({"type":"step","content":"Embedding your question…","eta":"~1s"})
        load_retriever(notebook_id=nb_id)

        # Step 1b: multi-query expansion
        from src.query_expander import expand_query
        yield _sse({"type":"step","content":"Expanding query…","eta":"~2s"})
        queries = expand_query(question=body.question, model=body.model)
        yield _sse({"type":"queries","content":queries})

        # Step 2: vector search (one search per query variant, merge results)
        yield _sse({"type":"step","content":"Searching vector index…","eta":"~2s"})
        seen: dict[str, dict] = {}   # chunk_id → best chunk dict
        for q in queries:
            for chunk in retrieve(query=q, top_k=50):
                cid = chunk["chunk_id"]
                if cid not in seen or chunk["score"] > seen[cid]["score"]:
                    seen[cid] = chunk
        raw_chunks = sorted(seen.values(), key=lambda c: c["score"], reverse=True)

        # Filter to effective sources
        chunks = [c for c in raw_chunks if c["document"] in effective]
        if not chunks:
            yield _sse({"type":"error","content":f"No content found in the selected source(s): {', '.join(effective)}"})
            return

        # Step 3: re-rank
        yield _sse({"type":"step","content":"Re-ranking results…","eta":"~3s"})
        chunks = rerank_chunks(question=body.question, chunks=chunks, top_n=body.top_k)

        sources_payload = [
            {"document": c["document"], "page": c["page"],
             "score": round(float(c.get("rerank_score", c.get("score",0))), 3),
             "text": c["text"][:500]}
            for c in chunks
        ]
        yield _sse({"type":"sources","content":sources_payload})

        full_response = ""

        if body.mode == "mapreduce":
            from src.map_reduce_engine import MAP_PROMPT, REDUCE_PROMPT
            partials = []
            for i, chunk in enumerate(chunks):
                yield _sse({"type":"step",
                            "content":f"Analysing section {i+1}/{len(chunks)}…",
                            "eta":"~15s"})
                mp = MAP_PROMPT.replace("{chunk}", chunk["text"]).replace("{question}", body.question)
                resp = _ol.chat(model=body.model,
                                messages=[{"role":"user","content":mp}],
                                options=config.OLLAMA_OPTIONS,
                                stream=False)
                partials.append(resp["message"]["content"].strip())
                await asyncio.sleep(0)

            useful = [p for p in partials
                      if "not covered in this section" not in p.lower() and len(p.strip()) > 20]
            if not useful:
                yield _sse({"type":"token","content":FALLBACK_ANSWER})
                yield _sse({"type":"done","content":""})
                full_response = FALLBACK_ANSWER
            else:
                answers_block = "\n\n---\n\n".join(f"[Section {i+1}]:\n{a}" for i,a in enumerate(useful))
                reduce_prompt = REDUCE_PROMPT.replace("{answers}", answers_block).replace("{question}", body.question)
                yield _sse({"type":"step","content":"Synthesising final answer…","eta":"~15s"})
                stream = _ol.chat(model=body.model,
                                  messages=[{"role":"user","content":reduce_prompt}],
                                  options=config.OLLAMA_OPTIONS,
                                  stream=True)
                for tok in stream:
                    t = tok["message"]["content"]
                    full_response += t
                    yield _sse({"type":"token","content":t})
                    await asyncio.sleep(0)
        else:
            # Fetch recent conversation history
            raw_msgs = list_messages(body.conversation_id)
            recent = raw_msgs[-config.HISTORY_MESSAGES:]
            history = [{"role": m["role"], "content": m["content"]} for m in recent]
            history_str = format_history(history, config.HISTORY_MAX_WORDS) if history else ""

            context, _ = assemble_context(chunks)
            prompt = build_prompt(context=context, question=body.question, history=history_str)
            yield _sse({"type":"step","content":"Generating response…","eta":"~5-10s"})
            stream = _ol.chat(model=body.model,
                              messages=[{"role":"user","content":prompt}],
                              options=config.OLLAMA_OPTIONS,
                              stream=True)
            for tok in stream:
                t = tok["message"]["content"]
                full_response += t
                yield _sse({"type":"token","content":t})
                await asyncio.sleep(0)

        # Persist messages
        is_first = count_messages(body.conversation_id) == 0
        add_message(body.conversation_id, "user", body.question)
        add_message(body.conversation_id, "ai", full_response.strip(), sources=sources_payload)
        yield _sse({"type":"done","content":""})

        if is_first:
            try:
                # Generate a concise title using the LLM (non-streaming, quick call)
                answer_preview = " ".join(full_response.split()[:200])  # first 200 words
                title_prompt = (
                    "Generate a concise 5-8 word title summarizing this Q&A exchange. "
                    "Return ONLY the title text — no quotes, no numbering, no explanation.\n\n"
                    f"Question: {body.question}\nAnswer: {answer_preview}"
                )
                title_resp = _ol.chat(
                    model=body.model,
                    messages=[{"role": "user", "content": title_prompt}],
                    options=config.OLLAMA_OPTIONS,
                    stream=False
                )
                generated_title = title_resp["message"]["content"].strip()
                # Strip surrounding quotes if present
                generated_title = generated_title.strip('"\'')
                # Validate: no newlines, reasonable length
                if not generated_title or "\n" in generated_title:
                    raise ValueError("Bad output")
                if len(generated_title) > 80:
                    generated_title = generated_title[:80] + "…"
                auto_title_conversation(body.conversation_id, generated_title)
            except Exception:
                # Fallback: use the original question (old behavior)
                auto_title_conversation(body.conversation_id, body.question)

    except Exception as e:
        yield _sse({"type":"error","content":str(e)})

@app.post("/api/ask")
@limiter.limit("10/minute")
async def ask_ep(request: Request, body: AskBody):
    # Validate question
    if not body.question.strip():
        raise HTTPException(400, "Question cannot be empty.")
    if len(body.question) > 2000:
        raise HTTPException(400, "Question too long. Maximum 2000 characters.")
    return StreamingResponse(_stream(body), media_type="text/event-stream",
                             headers={"Cache-Control":"no-cache","X-Accel-Buffering":"no"})

# ── Upload ─────────────────────────────────────────────────────────────────────
ALLOWED_EXTS = {".pdf", ".doc", ".docx", ".txt"}

def _run_index(notebook_id, src_ids):
    try:
        from src.pipeline import run_pipeline, run_embedding_pipeline
        run_pipeline(notebook_id=notebook_id,
                     chunk_size=config.CHUNK_SIZE, overlap=config.OVERLAP)
        run_embedding_pipeline(notebook_id=notebook_id,
                               batch_size=config.BATCH_SIZE)
        # Count pages per document from the chunks
        import json, collections
        chunks_path = config.get_processed_dir(notebook_id) / "chunks.json"
        with open(chunks_path) as f:
            all_chunks = json.load(f)

        pages_by_doc = collections.defaultdict(set)
        for chunk in all_chunks:
            pages_by_doc[chunk["document"]].add(chunk["page"])
        indexed_docs = set(pages_by_doc)

        # Update page count for each source
        sources = list_sources(notebook_id)
        for src in sources:
            pages = pages_by_doc.get(src["filename"], set())
            update_source_metadata(src["id"], len(pages))
        source_by_id = {src["id"]: src for src in sources}
        for sid in src_ids:
            src = source_by_id.get(sid)
            if src and src["filename"] in indexed_docs:
                update_source_status(sid, "ready")
            else:
                update_source_status(sid, "error: no indexed text found for this file")
    except Exception as e:
        for sid in src_ids: update_source_status(sid, f"error: {e}")

@app.post("/api/upload", status_code=202)
async def upload(bg: BackgroundTasks,
                 notebook_id: str = Form(...),
                 files: list[UploadFile] = File(...)):
    if not get_notebook(notebook_id): raise HTTPException(404)
    nb_pdf_dir = config.get_pdf_dir(notebook_id)
    nb_pdf_dir.mkdir(parents=True, exist_ok=True)
    saved = []
    for f in files:
        if ".." in f.filename:
            continue
        safe_name = re.sub(r"[^\w\-.]", "_", f.filename)
        safe_name = safe_name.lstrip(".")
        if not safe_name.strip():
            continue
        ext = Path(safe_name).suffix.lower()
        if ext not in ALLOWED_EXTS: continue
        dest = nb_pdf_dir / safe_name
        with open(dest, "wb") as out: shutil.copyfileobj(f.file, out)
        file_size = os.path.getsize(dest)
        src = add_source(notebook_id, safe_name, status="processing", file_size=file_size)
        saved.append(src)
    if not saved: raise HTTPException(400, f"No valid files. Allowed: {', '.join(ALLOWED_EXTS)}")
    bg.add_task(_run_index, notebook_id, [s["id"] for s in saved])
    return {"sources": saved}

@app.get("/api/sources/{nid}")
def get_srcs(nid: str):
    if not get_notebook(nid): raise HTTPException(404)
    return {"sources": list_sources(nid)}

@app.delete("/api/sources/{sid}", status_code=204)
def del_source(sid: str, bg: BackgroundTasks):
    src = get_source(sid)
    if not src:
        raise HTTPException(404, "Source not found")

    notebook_id = src["notebook_id"]
    filename    = src["filename"]

    # 1. Delete physical file from disk
    file_path = config.get_pdf_dir(notebook_id) / filename
    if file_path.exists():
        file_path.unlink()

    # 2. Delete DB record
    delete_source(sid)

    # 3. Check remaining sources
    remaining = list_sources(notebook_id)
    if remaining:
        # Re-index remaining files
        for s in remaining:
            update_source_status(s["id"], "processing")
        bg.add_task(_run_index, notebook_id, [s["id"] for s in remaining])
    else:
        # Clean up stale index files
        chunks_file = config.get_processed_dir(notebook_id) / "chunks.json"
        index_dir   = config.get_vector_dir(notebook_id)
        if chunks_file.exists():
            chunks_file.unlink()
        if index_dir.exists():
            import shutil
            shutil.rmtree(index_dir)

@app.post("/api/sources/{sid}/retry")
def retry_source(sid: str, bg: BackgroundTasks):
    src = get_source(sid)
    if not src:
        raise HTTPException(404, "Source not found")
    if not src["status"].startswith("error"):
        raise HTTPException(400, "Source is not in error state")

    update_source_status(sid, "processing")
    bg.add_task(_run_index, src["notebook_id"], [sid])
    return {"ok": True}
