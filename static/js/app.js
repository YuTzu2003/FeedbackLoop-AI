const fileInput = document.querySelector("#fileInput");
const fileStatus = document.querySelector("#fileStatus");
const questionInput = document.querySelector("#question");
const askButton = document.querySelector("#askButton");
const conversation = document.querySelector("#conversation");
const readyState = document.querySelector("#readyState");
const notebookList = document.querySelector("#notebookList");
const activeNotebook = document.querySelector("#activeNotebook");
let notebooks = [];
let activeNotebookId = null;
const copyIcon = '<i class="bi bi-copy" aria-hidden="true"></i>';

fileInput.addEventListener("change", uploadFile);
document.querySelector("#questionForm").addEventListener("submit", (event) => { event.preventDefault(); sendCurrentQuestion(); });
questionInput.addEventListener("keydown", (event) => { if (event.key === "Enter" && !event.shiftKey) { event.preventDefault(); sendCurrentQuestion(); } });
notebookList.addEventListener("click", (event) => { const button = event.target.closest("[data-notebook-id]"); if (button) selectNotebook(button.dataset.notebookId); });

function sendCurrentQuestion() { const question = questionInput.value.trim(); if (question && activeNotebookId && !askButton.disabled) ask(question); }

function renderNotebooks() {
  notebookList.innerHTML = notebooks.length ? notebooks.map((notebook) => `<button type="button" class="notebook-item ${notebook.id === activeNotebookId ? "active" : ""}" data-notebook-id="${notebook.id}"><span>${escapeHtml(notebook.name)}</span><small>${new Date(notebook.created_at).toLocaleDateString("zh-TW")}</small></button>`).join("") : "<p>尚未建立筆記本。</p>";
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
  questionInput.disabled = askButton.disabled = false;
  renderNotebooks();
  await loadNotebookHistory();
  questionInput.focus();
}

async function loadNotebookHistory() {
  if (!activeNotebookId) return;
  const response = await fetch(`/api/notebooks/${activeNotebookId}/history`);
  const data = await response.json();
  if (!response.ok) return;
  conversation.innerHTML = data.items.length ? data.items.map((item) => renderHistoryItem(item)).join("") : '<div class="empty-state"><div class="assistant-orb"><i class="bi bi-egg-fried" aria-hidden="true"></i></div><h3>開始這份文件的對話</h3><p>提出問題後，問答會保留在目前的筆記本中。</p></div>';
  conversation.scrollTop = conversation.scrollHeight;
}

function renderHistoryItem(item) {
  return `<div class="message user">${escapeHtml(item.question)}</div>${answerMessage(item.answer, item.question)}`;
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

async function ask(question) {
  conversation.querySelector(".empty-state")?.remove();
  conversation.innerHTML += '<div class="message user">' + escapeHtml(question) + '</div><div class="message ai loading">正在分析<span>.</span><span>.</span><span>.</span></div>';
  conversation.scrollTop = conversation.scrollHeight;
  questionInput.value = "";
  askButton.disabled = true;
  const response = await fetch("/api/ask", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ question, notebook_id: activeNotebookId }) });
  const data = await response.json();
  conversation.querySelector(".loading")?.remove();
  askButton.disabled = false;
  if (!response.ok) { conversation.innerHTML += '<div class="message ai">' + escapeHtml(data.error) + '</div>'; return; }
  conversation.innerHTML += answerMessage(data.answer, question, data.sources);
  conversation.scrollTop = conversation.scrollHeight;
}

function answerMessage(answer, question, sources = ["AI 文字分析"]) {
  return '<div class="message ai"><div class="answer">' + escapeHtml(answer).replace(/\n/g, "<br>") + '</div><div class="sources">參考：' + sources.map(escapeHtml).join("、") + '</div><div class="answer-actions"><button class="icon-button" data-copy aria-label="複製回答" title="複製回答">' + copyIcon + '</button></div><div class="feedback" data-question="' + escapeAttribute(question) + '" data-answer="' + escapeAttribute(answer) + '"><span>這個回答有幫助嗎？</span><button data-score="good">✓ 有幫助</button><button data-score="bad">× 需要改善</button></div></div>';
}

conversation.addEventListener("click", async (event) => {
  const copyButton = event.target.closest("[data-copy]");
  if (copyButton) { await navigator.clipboard?.writeText(copyButton.closest(".ai").querySelector(".answer").innerText); copyButton.classList.add("copied"); copyButton.querySelector(".bi").className = "bi bi-check2"; setTimeout(() => { copyButton.classList.remove("copied"); copyButton.querySelector(".bi").className = "bi bi-copy"; }, 1400); return; }
  const scoreButton = event.target.closest("[data-score]");
  if (!scoreButton) return;
  const feedback = scoreButton.closest(".feedback");
  feedback.innerHTML = '<input class="feedback-note" placeholder="補充說明（選填）"><button data-save-score="' + scoreButton.dataset.score + '">送出回饋</button>';
});
conversation.addEventListener("click", async (event) => {
  const saveButton = event.target.closest("[data-save-score]");
  if (!saveButton) return;
  const feedback = saveButton.closest(".feedback"); saveButton.disabled = true;
  const response = await fetch("/api/feedback", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ score: saveButton.dataset.saveScore, note: feedback.querySelector("input").value, question: feedback.dataset.question, answer: feedback.dataset.answer }) });
  const data = await response.json();
  feedback.innerHTML = response.ok ? "<b>✓ 回饋已儲存</b>" : "<b>" + escapeHtml(data.error) + "</b>";
});

function escapeHtml(value) { const element = document.createElement("div"); element.textContent = value; return element.innerHTML; }
function escapeAttribute(value) { return escapeHtml(value).replace(/"/g, "&quot;"); }
loadNotebooks();
