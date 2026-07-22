const fileInput = document.querySelector("#fileInput");
const fileStatus = document.querySelector("#fileStatus");
const questionInput = document.querySelector("#question");
const searchMode = document.querySelector("#searchMode");
const askButton = document.querySelector("#askButton");
const conversation = document.querySelector("#conversation");
const readyState = document.querySelector("#readyState");
const notebookList = document.querySelector("#notebookList");
const activeNotebook = document.querySelector("#activeNotebook");
const showUrlBtn = document.querySelector("#showUrlBtn");
const urlForm = document.querySelector("#urlForm");
const urlInput = document.querySelector("#urlInput");
let notebooks = [];
let activeNotebookId = null;
let feedbackHistoryIds = new Set();
const copyIcon = '<i class="bi bi-copy" aria-hidden="true"></i>';

fileInput.addEventListener("change", uploadFile);
document.querySelector("#questionForm").addEventListener("submit", (event) => { event.preventDefault(); sendCurrentQuestion(); });
questionInput.addEventListener("keydown", (event) => { if (event.key === "Enter" && !event.shiftKey) { event.preventDefault(); sendCurrentQuestion(); } });
notebookList.addEventListener("click", async (event) => { 
  const deleteBtn = event.target.closest("[data-delete-id]");
  if (deleteBtn) {
    event.stopPropagation();
    if (confirm("確定要刪除這份筆記本及其所有歷史紀錄嗎？")) {
      const id = deleteBtn.dataset.deleteId;
      const response = await fetch(`/api/notebooks/${id}`, { method: "DELETE" });
      if (response.ok) {
        if (activeNotebookId === id) {
          activeNotebookId = null;
          activeNotebook.textContent = "請選擇或建立筆記本";
          readyState.textContent = "等待建立";
          readyState.classList.remove("ready");
          conversation.innerHTML = "";
          questionInput.disabled = searchMode.disabled = askButton.disabled = true;
        }
        await loadNotebooks();
      } else {
        alert("刪除失敗");
      }
    }
    return;
  }
  const button = event.target.closest("[data-notebook-id]"); 
  if (button) selectNotebook(button.dataset.notebookId); 
});
showUrlBtn.addEventListener("click", () => {
  urlForm.style.display = urlForm.style.display === "none" ? "flex" : "none";
  if (urlForm.style.display === "flex") urlInput.focus();
});
urlForm.addEventListener("submit", (event) => { event.preventDefault(); uploadUrl(); });

function sendCurrentQuestion() { const question = questionInput.value.trim(); if (question && activeNotebookId && !askButton.disabled) ask(question); }

function renderNotebooks() {
  notebookList.innerHTML = notebooks.length ? notebooks.map((notebook) => {
    let typeStr = notebook.source_type || 'FILE';
    if (typeStr === 'file' || typeStr === 'FILE') {
      const parts = notebook.name.split('.');
      if (parts.length > 1) typeStr = parts.pop();
    }
    typeStr = typeStr.toUpperCase();
    let colorClass = 'bg-secondary text-secondary';
    if (typeStr === 'WEB') colorClass = 'bg-primary text-primary';
    else if (typeStr === 'PDF') colorClass = 'bg-danger text-danger';
    else if (['XLSX', 'XLS', 'CSV'].includes(typeStr)) colorClass = 'bg-success text-success';
    else if (['DOCX', 'TXT'].includes(typeStr)) colorClass = 'bg-info text-dark';
    
    const badge = `<span class="badge rounded-pill bg-opacity-25 ${colorClass}" style="font-size: 9px; padding: 2px 5px; font-weight: 600; vertical-align: middle;">${escapeHtml(typeStr)}</span>`;
    return `<div class="notebook-item-wrapper ${notebook.id === activeNotebookId ? "active" : ""}">
      <button type="button" class="notebook-item" data-notebook-id="${notebook.id}">
        <span style="margin-bottom: 5px;">${escapeHtml(notebook.name)}</span>
        <div>${badge}</div>
      </button>
      <button type="button" class="delete-notebook" data-delete-id="${notebook.id}" title="刪除筆記本">
        <i class="bi bi-trash"></i>
      </button>
    </div>`;
  }).join("") : "<p>尚未建立筆記本。</p>";
}

async function loadNotebooks() {
  const response = await fetch("/api/notebooks");
  const data = await response.json();
  notebooks = data.items;
  renderNotebooks();
}

async function selectNotebook(notebookId) {
  const notebook = notebooks.find((item) => item.id === notebookId);
  if (!notebook) return;
  activeNotebookId = notebookId;
  activeNotebook.textContent = notebook.name;
  readyState.textContent = "筆記本已就緒";
  readyState.classList.add("ready");
  questionInput.disabled = searchMode.disabled = askButton.disabled = false;
  renderNotebooks();
  await loadNotebookHistory();
  questionInput.focus();
}

async function loadNotebookHistory() {
  if (!activeNotebookId) return;
  const [response, feedbackResponse] = await Promise.all([fetch(`/api/notebooks/${activeNotebookId}/history`), fetch("/api/feedbacks")]);
  const [data, feedbackData] = await Promise.all([response.json(), feedbackResponse.json()]);
  if (!response.ok) return;
  feedbackHistoryIds = new Set(feedbackData.items.map((item) => item.history_id).filter(Boolean));
  conversation.innerHTML = data.items.length ? data.items.map((item) => renderHistoryItem(item)).join("") : '<div class="empty-state"><h3>開始這份文件的對話</h3><p>提出問題後，問答會保留在目前的筆記本中。</p></div>';
  conversation.scrollTop = conversation.scrollHeight;
}

function renderHistoryItem(item) {
  return `<div class="message user">${escapeHtml(item.question)}</div>${answerMessage(item.answer, item.question, item.id, item.sources)}`;
}

async function uploadFile() {
  const file = fileInput.files[0];
  if (!file) return;
  fileStatus.textContent = "正在建立 " + file.name + "…";
  const form = new FormData(); form.append("file", file);
  const response = await fetch("/api/upload", { method: "POST", body: form });
  const data = await response.json();
  if (!response.ok) { fileStatus.textContent = data.error; return; }
  fileStatus.innerHTML = '<span class="file-mark"><i class="bi bi-check-lg" aria-hidden="true"></i></span><span><b>' + escapeHtml(data.filename) + '</b><small>已建立文件筆記本</small></span>';
  await loadNotebooks();
  await selectNotebook(data.notebook_id);
}

async function uploadUrl() {
  const url = urlInput.value.trim();
  if (!url) return;
  fileStatus.textContent = "正在加入網址…";
  urlForm.style.display = "none";
  const response = await fetch("/api/upload_url", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ url }) });
  const data = await response.json();
  urlInput.value = "";
  if (!response.ok) { fileStatus.textContent = data.error; return; }
  fileStatus.innerHTML = '<span class="file-mark"><i class="bi bi-check-lg" aria-hidden="true"></i></span><span><b>' + escapeHtml(data.filename) + '</b><small>已加入網址</small></span>';
  await loadNotebooks();
  await selectNotebook(data.notebook_id);
}

async function ask(question) {
  conversation.querySelector(".empty-state")?.remove();
  conversation.innerHTML += '<div class="message user">' + escapeHtml(question) + '</div><div class="message ai loading">正在分析<span>.</span><span>.</span><span>.</span></div>';
  conversation.scrollTop = conversation.scrollHeight;
  questionInput.value = "";
  askButton.disabled = true;
  const response = await fetch("/api/ask", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ question, notebook_id: activeNotebookId, search_mode: searchMode.value }) });
  const data = await response.json();
  conversation.querySelector(".loading")?.remove();
  askButton.disabled = false;
  if (!response.ok) { conversation.innerHTML += '<div class="message ai">' + escapeHtml(data.error) + '</div>'; return; }
  conversation.innerHTML += answerMessage(data.answer, question, data.history_id, data.sources);
  conversation.scrollTop = conversation.scrollHeight;
}

function answerMessage(answer, question, historyId, sources = ["AI 文字分析"]) {
  const feedback = feedbackHistoryIds.has(historyId) ? '<div class="feedback"><b>✓ 已回饋</b></div>' : '<div class="feedback" data-history-id="' + escapeAttribute(historyId) + '" data-question="' + escapeAttribute(question) + '" data-answer="' + escapeAttribute(answer) + '"><button class="feedback-icon" data-score="good" aria-label="有幫助" title="有幫助"><i class="bi bi-hand-thumbs-up" aria-hidden="true"></i></button><button class="feedback-icon" data-score="bad" aria-label="需要改善" title="需要改善"><i class="bi bi-hand-thumbs-down" aria-hidden="true"></i></button></div>';
  return '<div class="message ai"><div class="answer markdown-body">' + renderAnswer(answer) + '</div><div class="answer-actions">' + feedback + '<button class="icon-button" data-copy aria-label="複製回答" title="複製回答">' + copyIcon + '</button></div><div class="sources">參考：' + sources.map(escapeHtml).join("、") + '</div></div>';
}

conversation.addEventListener("click", async (event) => {
  const copyButton = event.target.closest("[data-copy]");
  if (copyButton) { await navigator.clipboard?.writeText(copyButton.closest(".ai").querySelector(".answer").innerText); copyButton.classList.add("copied"); copyButton.querySelector(".bi").className = "bi bi-check2"; setTimeout(() => { copyButton.classList.remove("copied"); copyButton.querySelector(".bi").className = "bi bi-copy"; }, 1400); return; }
  const scoreButton = event.target.closest("[data-score]");
  if (!scoreButton) return;
  const feedback = scoreButton.closest(".feedback");
  if (feedbackHistoryIds.has(feedback.dataset.historyId)) return;
  if (scoreButton.dataset.score === "good") { saveFeedback(feedback, "good", ""); return; }
  feedback.innerHTML = '<input class="feedback-note" placeholder="請說明需要改善的地方" required><button data-save-score="bad">送出回饋</button>';
});
conversation.addEventListener("click", async (event) => {
  const saveButton = event.target.closest("[data-save-score]");
  if (!saveButton) return;
  const feedback = saveButton.closest(".feedback");
  const note = feedback.querySelector("input").value.trim();
  if (!note) { feedback.querySelector("input").focus(); return; }
  saveFeedback(feedback, saveButton.dataset.saveScore, note);
});

async function saveFeedback(feedback, score, note) {
  feedback.querySelectorAll("button").forEach((button) => { button.disabled = true; });
  const response = await fetch("/api/feedback", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ score, note, question: feedback.dataset.question, answer: feedback.dataset.answer, history_id: feedback.dataset.historyId }) });
  const data = await response.json();
  if (response.ok) feedbackHistoryIds.add(feedback.dataset.historyId);
  feedback.innerHTML = response.ok ? "<b>✓ 回饋已儲存</b>" : "<b>" + escapeHtml(data.error) + "</b>";
}

function renderAnswer(answer) {
  if (window.marked && window.DOMPurify) return DOMPurify.sanitize(marked.parse(answer, { breaks: true }));
  return escapeHtml(answer).replace(/\n/g, "<br>");
}

function escapeHtml(value) {
  if (typeof value === "object" && value) return formatSource(value);
  const element = document.createElement("div"); element.textContent = value; return element.innerHTML;
}
function escapeAttribute(value) { return escapeHtml(value).replace(/"/g, "&quot;"); }
function formatSource(source) {
  if (source.source_type === "pdf") {
    const page = source.page_number ? `第 ${source.page_number} 頁` : "頁碼不明";
    const score = source.score == null ? "" : ` · ${(source.score * 100).toFixed(0)}%`;
    return escapeHtml(`${source.title} · ${page}${score}`);
  }
  const label = `${source.title} · chunk ${source.chunk_index} · ${(source.score * 100).toFixed(0)}%`;
  return `<a href="${escapeAttribute(source.url)}" target="_blank" rel="noopener noreferrer">${escapeHtml(label)}</a>`;
}
loadNotebooks();
