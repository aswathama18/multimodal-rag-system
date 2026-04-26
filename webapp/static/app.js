/* =========================================================
   app.js — Multimodal RAG Dashboard
   ========================================================= */

'use strict';

// ── Tab switching ─────────────────────────────────────────
document.querySelectorAll('.tab-btn').forEach(btn => {
  btn.addEventListener('click', () => {
    document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
    document.querySelectorAll('.tab-panel').forEach(p => p.classList.remove('active'));
    btn.classList.add('active');
    document.getElementById('panel-' + btn.dataset.tab).classList.add('active');
    if (btn.dataset.tab === 'status') loadStatus();
  });
});

// ── Toast ─────────────────────────────────────────────────
function toast(msg, type = 'info') {
  const el = document.getElementById('toast');
  el.textContent = msg;
  el.className   = `show ${type}`;
  clearTimeout(el._t);
  el._t = setTimeout(() => { el.className = ''; }, 3400);
}

// ── Log console helper ────────────────────────────────────
function appendLog(consoleEl, msg, cls = '') {
  const line = document.createElement('div');
  line.className = 'log-line ' + cls;
  line.innerHTML = `<span class="log-prompt">›</span>${escHtml(msg)}`;
  consoleEl.appendChild(line);
  consoleEl.scrollTop = consoleEl.scrollHeight;
}

function escHtml(str) {
  return String(str)
    .replace(/&/g,'&amp;').replace(/</g,'&lt;')
    .replace(/>/g,'&gt;').replace(/"/g,'&quot;');
}

function clearLog(consoleEl) {
  consoleEl.innerHTML = '';
}

// ── SSE stream ────────────────────────────────────────────
function startStream(streamUrl, logEl, onDone, onError) {
  const es = new EventSource(streamUrl);
  es.onmessage = e => {
    const data = JSON.parse(e.data);
    if (data.type === 'log')  appendLog(logEl, data.msg);
    if (data.type === 'ping') return;
    if (data.type === 'done') { appendLog(logEl, '✓ Done', 'done'); es.close(); onDone(); }
    if (data.type === 'error') { appendLog(logEl, '✗ ' + data.msg, 'error'); es.close(); onError(data.msg); }
  };
  es.onerror = () => { es.close(); onError('Connection lost.'); };
  return es;
}

// ═══════════════════════════════════════════════════════════
// DOCUMENT RAG
// ═══════════════════════════════════════════════════════════

const docFileInput   = document.getElementById('doc-file');
const docFileName    = document.getElementById('doc-file-name');
const docProcessBtn  = document.getElementById('doc-process-btn');
const docLogCard     = document.getElementById('doc-log-card');
const docLog         = document.getElementById('doc-log');
const docForce       = document.getElementById('doc-force');
const docQueryInput  = document.getElementById('doc-query-input');
const docAskBtn      = document.getElementById('doc-ask-btn');
const docAnswer      = document.getElementById('doc-answer');
const docAnswerText  = document.getElementById('doc-answer-text');

// Drag-over feedback
const docDrop = document.getElementById('doc-drop');
docDrop.addEventListener('dragover', e => { e.preventDefault(); docDrop.classList.add('dragover'); });
docDrop.addEventListener('dragleave', () => docDrop.classList.remove('dragover'));
docDrop.addEventListener('drop', e => {
  e.preventDefault(); docDrop.classList.remove('dragover');
  if (e.dataTransfer.files.length) {
    docFileInput.files = e.dataTransfer.files;
    onDocFileChosen();
  }
});

docFileInput.addEventListener('change', onDocFileChosen);

function onDocFileChosen() {
  const f = docFileInput.files[0];
  if (!f) return;
  docFileName.textContent = '📎 ' + f.name;
  docProcessBtn.disabled  = false;
}

docProcessBtn.addEventListener('click', async () => {
  const file = docFileInput.files[0];
  if (!file) return toast('Please select a PDF first.', 'error');

  docProcessBtn.disabled = true;
  docProcessBtn.innerHTML = '<span class="spinner"></span> Processing…';
  docLogCard.style.display = 'block';
  clearLog(docLog);
  appendLog(docLog, 'Uploading ' + file.name + ' …');

  const form = new FormData();
  form.append('file', file);
  form.append('force_reset', docForce.checked ? 'true' : 'false');

  try {
    const res  = await fetch('/api/doc/process', { method: 'POST', body: form });
    if (!res.ok) { const e = await res.json(); throw new Error(e.detail || 'Upload failed'); }
    const { job_id } = await res.json();
    appendLog(docLog, 'Upload OK — starting pipeline (job ' + job_id.slice(0,8) + '…)');

    startStream(
      '/api/doc/stream/' + job_id, docLog,
      () => {
        docProcessBtn.disabled = false;
        docProcessBtn.innerHTML = '▶ Process Document';
        toast('Document indexed! You can now query it.', 'success');
      },
      err => {
        docProcessBtn.disabled = false;
        docProcessBtn.innerHTML = '▶ Process Document';
        toast('Pipeline error: ' + err, 'error');
      }
    );
  } catch (err) {
    appendLog(docLog, err.message, 'error');
    docProcessBtn.disabled = false;
    docProcessBtn.innerHTML = '▶ Process Document';
    toast(err.message, 'error');
  }
});

docAskBtn.addEventListener('click', askDoc);
docQueryInput.addEventListener('keydown', e => { if (e.key === 'Enter') askDoc(); });

async function askDoc() {
  const q = docQueryInput.value.trim();
  if (!q) return toast('Enter a question first.', 'error');

  docAskBtn.disabled = true;
  docAskBtn.innerHTML = '<span class="spinner"></span>';
  docAnswer.classList.remove('visible');

  try {
    const res = await fetch('/api/doc/query', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ question: q }),
    });
    if (!res.ok) { const e = await res.json(); throw new Error(e.detail || 'Query failed'); }
    const { answer } = await res.json();
    docAnswerText.innerHTML = marked.parse(answer);
    docAnswer.classList.add('visible');
  } catch (err) {
    toast(err.message, 'error');
  } finally {
    docAskBtn.disabled = false;
    docAskBtn.innerHTML = 'Ask ▶';
  }
}

// ═══════════════════════════════════════════════════════════
// IMAGE ANALYZER
// ═══════════════════════════════════════════════════════════

const imgFileInput   = document.getElementById('img-file');
const imgFileNames   = document.getElementById('img-file-names');
const imgAnalyzeBtn  = document.getElementById('img-analyze-btn');
const imgPrompt      = document.getElementById('img-prompt');
const imgResultCard  = document.getElementById('img-result-card');
const imgResult      = document.getElementById('img-result');

const imgDrop = document.getElementById('img-drop');
imgDrop.addEventListener('dragover', e => { e.preventDefault(); imgDrop.classList.add('dragover'); });
imgDrop.addEventListener('dragleave', () => imgDrop.classList.remove('dragover'));
imgDrop.addEventListener('drop', e => {
  e.preventDefault(); imgDrop.classList.remove('dragover');
  if (e.dataTransfer.files.length) {
    imgFileInput.files = e.dataTransfer.files;
    onImgChosen();
  }
});

imgFileInput.addEventListener('change', onImgChosen);

function onImgChosen() {
  const files = Array.from(imgFileInput.files);
  if (!files.length) return;
  if (files.length > 3) { toast('Max 3 images.', 'error'); return; }
  imgFileNames.textContent = files.map(f => '🖼 ' + f.name).join('  ');
  imgAnalyzeBtn.disabled   = false;
}

imgAnalyzeBtn.addEventListener('click', async () => {
  const files = imgFileInput.files;
  if (!files.length) return toast('Upload at least one image.', 'error');

  imgAnalyzeBtn.disabled = true;
  imgAnalyzeBtn.innerHTML = '<span class="spinner"></span> Analyzing…';
  imgResultCard.style.display = 'none';

  const form = new FormData();
  for (const f of files) form.append('files', f);
  form.append('prompt', imgPrompt.value.trim() || 'Describe this image in detail. What do you see?');

  try {
    const res = await fetch('/api/image/analyze', { method: 'POST', body: form });
    if (!res.ok) { const e = await res.json(); throw new Error(e.detail || 'Analysis failed'); }
    const { analysis } = await res.json();
    imgResult.innerHTML         = marked.parse(analysis);
    imgResultCard.style.display = 'block';
    toast('Analysis complete!', 'success');
  } catch (err) {
    toast(err.message, 'error');
  } finally {
    imgAnalyzeBtn.disabled = false;
    imgAnalyzeBtn.innerHTML = '🔍 Analyze with Gemini';
  }
});

// ═══════════════════════════════════════════════════════════
// VIDEO RAG
// ═══════════════════════════════════════════════════════════

const videoUrlInput   = document.getElementById('video-url-input');
const videoProcessBtn = document.getElementById('video-process-btn');
const videoForce      = document.getElementById('video-force');
const videoLogCard    = document.getElementById('video-log-card');
const videoLog        = document.getElementById('video-log');
const videoQueryInput = document.getElementById('video-query-input');
const videoAskBtn     = document.getElementById('video-ask-btn');
const videoAnswer     = document.getElementById('video-answer');
const videoAnswerText = document.getElementById('video-answer-text');

videoProcessBtn.addEventListener('click', async () => {
  const url = videoUrlInput.value.trim();
  if (!url) return toast('Paste a YouTube URL first.', 'error');
  if (!url.startsWith('http')) return toast('Enter a valid URL starting with http.', 'error');

  videoProcessBtn.disabled = true;
  videoProcessBtn.innerHTML = '<span class="spinner"></span> Processing…';
  videoLogCard.style.display = 'block';
  clearLog(videoLog);
  appendLog(videoLog, 'Sending URL to pipeline …');

  try {
    const res = await fetch('/api/video/process', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ url, force_reset: videoForce.checked }),
    });
    if (!res.ok) { const e = await res.json(); throw new Error(e.detail || 'Failed to start pipeline'); }
    const { job_id } = await res.json();
    appendLog(videoLog, 'Pipeline started (job ' + job_id.slice(0,8) + '…)');

    startStream(
      '/api/video/stream/' + job_id, videoLog,
      () => {
        videoProcessBtn.disabled = false;
        videoProcessBtn.innerHTML = '▶ Process Video';
        toast('Video indexed! You can now query it.', 'success');
      },
      err => {
        videoProcessBtn.disabled = false;
        videoProcessBtn.innerHTML = '▶ Process Video';
        toast('Pipeline error: ' + err, 'error');
      }
    );
  } catch (err) {
    appendLog(videoLog, err.message, 'error');
    videoProcessBtn.disabled = false;
    videoProcessBtn.innerHTML = '▶ Process Video';
    toast(err.message, 'error');
  }
});

videoAskBtn.addEventListener('click', askVideo);
videoQueryInput.addEventListener('keydown', e => { if (e.key === 'Enter') askVideo(); });

async function askVideo() {
  const q = videoQueryInput.value.trim();
  if (!q) return toast('Enter a question first.', 'error');

  videoAskBtn.disabled = true;
  videoAskBtn.innerHTML = '<span class="spinner"></span>';
  videoAnswer.classList.remove('visible');

  try {
    const res = await fetch('/api/video/query', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ question: q }),
    });
    if (!res.ok) { const e = await res.json(); throw new Error(e.detail || 'Query failed'); }
    const { answer, frames_used } = await res.json();
    videoAnswerText.innerHTML = marked.parse(answer);
    videoAnswer.classList.add('visible');
    toast(`Answer generated using ${frames_used} frame(s).`, 'success');
  } catch (err) {
    toast(err.message, 'error');
  } finally {
    videoAskBtn.disabled = false;
    videoAskBtn.innerHTML = 'Ask ▶';
  }
}

// ═══════════════════════════════════════════════════════════
// TEXT RAG
// ═══════════════════════════════════════════════════════════

const textFileInput   = document.getElementById('text-file');
const textFileName    = document.getElementById('text-file-name');
const textProcessBtn  = document.getElementById('text-process-btn');
const textQueryInput  = document.getElementById('text-query-input');
const textAskBtn      = document.getElementById('text-ask-btn');
const textAnswer      = document.getElementById('text-answer');
const textAnswerText  = document.getElementById('text-answer-text');

const textDrop = document.getElementById('text-drop');
textDrop.addEventListener('dragover', e => { e.preventDefault(); textDrop.classList.add('dragover'); });
textDrop.addEventListener('dragleave', () => textDrop.classList.remove('dragover'));
textDrop.addEventListener('drop', e => {
  e.preventDefault(); textDrop.classList.remove('dragover');
  if (e.dataTransfer.files.length) {
    textFileInput.files = e.dataTransfer.files;
    onTextFileChosen();
  }
});

textFileInput.addEventListener('change', onTextFileChosen);

function onTextFileChosen() {
  const f = textFileInput.files[0];
  if (!f) return;
  textFileName.textContent = '📄 ' + f.name;
  textProcessBtn.disabled  = false;
}

textProcessBtn.addEventListener('click', async () => {
  const file = textFileInput.files[0];
  if (!file) return toast('Please select a text file first.', 'error');

  textProcessBtn.disabled = true;
  textProcessBtn.innerHTML = '<span class="spinner"></span> Processing…';

  const form = new FormData();
  form.append('file', file);

  try {
    const res = await fetch('/api/text/process', { method: 'POST', body: form });
    if (!res.ok) { const e = await res.json(); throw new Error(e.detail || 'Upload failed'); }
    const data = await res.json();
    toast(data.message, 'success');
  } catch (err) {
    toast(err.message, 'error');
  } finally {
    textProcessBtn.disabled = false;
    textProcessBtn.innerHTML = '▶ Process Text';
  }
});

textAskBtn.addEventListener('click', askText);
textQueryInput.addEventListener('keydown', e => { if (e.key === 'Enter') askText(); });

async function askText() {
  const q = textQueryInput.value.trim();
  if (!q) return toast('Enter a question first.', 'error');

  textAskBtn.disabled = true;
  textAskBtn.innerHTML = '<span class="spinner"></span>';
  textAnswer.classList.remove('visible');

  try {
    const res = await fetch('/api/text/query', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ question: q }),
    });
    if (!res.ok) { const e = await res.json(); throw new Error(e.detail || 'Query failed'); }
    const { answer } = await res.json();
    textAnswerText.innerHTML = marked.parse(answer);
    textAnswer.classList.add('visible');
  } catch (err) {
    toast(err.message, 'error');
  } finally {
    textAskBtn.disabled = false;
    textAskBtn.innerHTML = 'Ask ▶';
  }
}

// ═══════════════════════════════════════════════════════════
// IMAGE-TEXT RAG
// ═══════════════════════════════════════════════════════════

const imagetextFileInput   = document.getElementById('imagetext-file');
const imagetextFileName    = document.getElementById('imagetext-file-name');
const imagetextProcessBtn  = document.getElementById('imagetext-process-btn');

const imagetextImgFileInput = document.getElementById('imagetext-img-file');
const imagetextImgFileName  = document.getElementById('imagetext-img-name');

const imagetextQueryInput  = document.getElementById('imagetext-query-input');
const imagetextAskBtn      = document.getElementById('imagetext-ask-btn');
const imagetextAnswer      = document.getElementById('imagetext-answer');
const imagetextAnswerText  = document.getElementById('imagetext-answer-text');

const imagetextDrop = document.getElementById('imagetext-drop');
imagetextDrop.addEventListener('dragover', e => { e.preventDefault(); imagetextDrop.classList.add('dragover'); });
imagetextDrop.addEventListener('dragleave', () => imagetextDrop.classList.remove('dragover'));
imagetextDrop.addEventListener('drop', e => {
  e.preventDefault(); imagetextDrop.classList.remove('dragover');
  if (e.dataTransfer.files.length) {
    imagetextFileInput.files = e.dataTransfer.files;
    onImagetextFileChosen();
  }
});

imagetextFileInput.addEventListener('change', onImagetextFileChosen);

function onImagetextFileChosen() {
  const f = imagetextFileInput.files[0];
  if (!f) return;
  imagetextFileName.textContent = '📚 ' + f.name;
  imagetextProcessBtn.disabled  = false;
}

const imagetextImgDrop = document.getElementById('imagetext-img-drop');
imagetextImgDrop.addEventListener('dragover', e => { e.preventDefault(); imagetextImgDrop.classList.add('dragover'); });
imagetextImgDrop.addEventListener('dragleave', () => imagetextImgDrop.classList.remove('dragover'));
imagetextImgDrop.addEventListener('drop', e => {
  e.preventDefault(); imagetextImgDrop.classList.remove('dragover');
  if (e.dataTransfer.files.length) {
    imagetextImgFileInput.files = e.dataTransfer.files;
    onImagetextImgFileChosen();
  }
});

imagetextImgFileInput.addEventListener('change', onImagetextImgFileChosen);

function onImagetextImgFileChosen() {
  const f = imagetextImgFileInput.files[0];
  if (!f) return;
  imagetextImgFileName.textContent = '🖼️ ' + f.name;
}

imagetextProcessBtn.addEventListener('click', async () => {
  const file = imagetextFileInput.files[0];
  if (!file) return toast('Please select a text knowledge base first.', 'error');

  imagetextProcessBtn.disabled = true;
  imagetextProcessBtn.innerHTML = '<span class="spinner"></span> Processing…';

  const form = new FormData();
  form.append('file', file);

  try {
    const res = await fetch('/api/imagetext/process', { method: 'POST', body: form });
    if (!res.ok) { const e = await res.json(); throw new Error(e.detail || 'Upload failed'); }
    const data = await res.json();
    toast(data.message, 'success');
  } catch (err) {
    toast(err.message, 'error');
  } finally {
    imagetextProcessBtn.disabled = false;
    imagetextProcessBtn.innerHTML = '▶ Process Knowledge Base';
  }
});

imagetextAskBtn.addEventListener('click', askImagetext);
imagetextQueryInput.addEventListener('keydown', e => { if (e.key === 'Enter') askImagetext(); });

async function askImagetext() {
  const q = imagetextQueryInput.value.trim();
  if (!q) return toast('Enter a question first.', 'error');

  imagetextAskBtn.disabled = true;
  imagetextAskBtn.innerHTML = '<span class="spinner"></span>';
  imagetextAnswer.classList.remove('visible');

  const form = new FormData();
  form.append('question', q);
  const imgFile = imagetextImgFileInput.files[0];
  if (imgFile) form.append('image', imgFile);

  try {
    const res = await fetch('/api/imagetext/query', { method: 'POST', body: form });
    if (!res.ok) { const e = await res.json(); throw new Error(e.detail || 'Query failed'); }
    const { answer } = await res.json();
    imagetextAnswerText.innerHTML = marked.parse(answer);
    imagetextAnswer.classList.add('visible');
  } catch (err) {
    toast(err.message, 'error');
  } finally {
    imagetextAskBtn.disabled = false;
    imagetextAskBtn.innerHTML = 'Ask ▶';
  }
}

// ═══════════════════════════════════════════════════════════
// STATUS TAB
// ═══════════════════════════════════════════════════════════

async function loadStatus() {
  const [docRes, vidRes, txtRes, imgTxtRes] = await Promise.allSettled([
    fetch('/api/doc/status'), fetch('/api/video/status'),
    fetch('/api/text/status'), fetch('/api/imagetext/status')
  ]);

  if (docRes.status === 'fulfilled' && docRes.value.ok) {
    const d = await docRes.value.json();
    document.getElementById('s-chain').textContent  = d.chain_ready ? 'Ready ✓' : 'Not loaded';
    document.getElementById('s-chain').className    = 'stat-value ' + (d.chain_ready ? 'green' : 'muted');
    document.getElementById('s-chroma').textContent = d.chroma_vectors >= 0 ? d.chroma_vectors : 'N/A';
    document.getElementById('s-imgs').textContent   = d.extracted_images;
  }

  if (vidRes.status === 'fulfilled' && vidRes.value.ok) {
    const v = await vidRes.value.json();
    document.getElementById('s-vindex').textContent = v.lancedb_rows;
    document.getElementById('s-vindex').className   = 'stat-value ' + (v.index_ready ? 'green' : 'muted');
    document.getElementById('s-frames').textContent = v.frames;
    document.getElementById('s-vtitle').textContent = v.metadata?.Title || '—';
    document.getElementById('s-vauthor').textContent = v.metadata?.Author ? 'by ' + v.metadata.Author : '';
  }

  let textIndexed = 0;
  if (txtRes.status === 'fulfilled' && txtRes.value.ok) {
    const t = await txtRes.value.json();
    textIndexed += t.indexed_chunks >= 0 ? t.indexed_chunks : 0;
  }
  if (imgTxtRes.status === 'fulfilled' && imgTxtRes.value.ok) {
    const t = await imgTxtRes.value.json();
    textIndexed += t.indexed_chunks >= 0 ? t.indexed_chunks : 0;
  }
  document.getElementById('s-txtindex').textContent = textIndexed || '—';
  if (textIndexed > 0) document.getElementById('s-txtindex').className = 'stat-value purple';
  else document.getElementById('s-txtindex').className = 'stat-value muted';
}

document.getElementById('status-refresh-btn').addEventListener('click', () => {
  loadStatus(); toast('Status refreshed.', 'info');
});

// ── Reset buttons ─────────────────────────────────────────
document.getElementById('reset-doc-btn').addEventListener('click', async () => {
  if (!confirm('This will delete all extracted images and the ChromaDB. Continue?')) return;
  const res = await fetch('/api/doc/reset', { method: 'POST' });
  const d   = await res.json();
  toast('Document RAG reset. ' + (d.removed?.length || 0) + ' dirs removed.', 'success');
  loadStatus();
});

document.getElementById('reset-video-btn').addEventListener('click', async () => {
  if (!confirm('This will delete mixed_data/, video_data/, and LanceDB. Continue?')) return;
  const res = await fetch('/api/video/reset', { method: 'POST' });
  const d   = await res.json();
  toast('Video RAG reset. ' + (d.removed?.length || 0) + ' dirs removed.', 'success');
  loadStatus();
});

document.getElementById('reset-text-btn').addEventListener('click', async () => {
  await fetch('/api/text/reset', { method: 'POST' });
  toast('Text RAG reset.', 'success');
  loadStatus();
});

document.getElementById('reset-imagetext-btn').addEventListener('click', async () => {
  await fetch('/api/imagetext/reset', { method: 'POST' });
  toast('Image-Text RAG reset.', 'success');
  loadStatus();
});

// ── Initial status load on page open ─────────────────────
loadStatus();
