const fileInput = document.querySelector("#fileInput");
const fileStatus = document.querySelector("#fileStatus");
const questionInput = document.querySelector("#question");
const askButton = document.querySelector("#askButton");
const conversation = document.querySelector("#conversation");
const readyState = document.querySelector("#readyState");
const copyIcon = '<svg aria-hidden="true" viewBox="0 0 24 24"><rect x="9" y="9" width="11" height="11" rx="2"/><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/></svg>';

fileInput.addEventListener("change", uploadFile);
document.querySelector("#questionForm").addEventListener("submit", (event) => { event.preventDefault(); sendCurrentQuestion(); });
questionInput.addEventListener("keydown", (event) => { if (event.key === "Enter" && !event.shiftKey) { event.preventDefault(); sendCurrentQuestion(); } });
document.addEventListener("click", (event) => { const suggestion = event.target.closest("[data-suggestion]"); if (suggestion && !questionInput.disabled) { questionInput.value = suggestion.dataset.suggestion; questionInput.focus(); } });

function sendCurrentQuestion() { const question = questionInput.value.trim(); if (question && !askButton.disabled) ask(question); }

async function uploadFile() {
  const file = fileInput.files[0]; if (!file) return;
  fileStatus.textContent = "\u6b63\u5728\u8655\u7406 " + file.name + "\u2026";
  const form = new FormData(); form.append("file", file);
  const response = await fetch("/api/upload", { method: "POST", body: form }); const data = await response.json();
  if (!response.ok) { fileStatus.textContent = data.error; return; }
  fileStatus.innerHTML = '<span class="file-mark">\u2713</span><span><b>' + escapeHtml(data.filename) + '</b><small>\u6a94\u6848\u5df2\u6e96\u5099\u597d\u5206\u6790</small></span>';
  questionInput.disabled = askButton.disabled = false; readyState.textContent = "\u8cc7\u6599\u5df2\u5c31\u7dd2"; readyState.classList.add("ready"); questionInput.focus();
}

async function ask(question) {
  conversation.innerHTML += '<div class="message user">' + escapeHtml(question) + '</div><div class="message ai loading">\u6b63\u5728\u5206\u6790\u8cc7\u6599<span>.</span><span>.</span><span>.</span></div>';
  conversation.scrollTop = conversation.scrollHeight; questionInput.value = ""; askButton.disabled = true;
  const response = await fetch("/api/ask", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ question }) });
  const data = await response.json(); document.querySelector(".loading")?.remove(); askButton.disabled = false;
  if (!response.ok) { conversation.innerHTML += '<div class="message ai">' + escapeHtml(data.error) + '</div>'; return; }
  conversation.innerHTML += '<div class="message ai"><div class="answer">' + escapeHtml(data.answer).replace(/\n/g, "<br>") + '</div><div class="sources">\u53c3\u8003\uff1a' + data.sources.map(escapeHtml).join(" \u00b7 ") + '</div><div class="answer-actions"><button class="icon-button" data-copy aria-label="\u8907\u88fd\u56de\u7b54" title="\u8907\u88fd\u56de\u7b54">' + copyIcon + '</button></div><div class="feedback" data-question="' + escapeAttribute(question) + '" data-answer="' + escapeAttribute(data.answer) + '"><span>\u9019\u500b\u56de\u7b54\u6709\u5e6b\u52a9\u55ce\uff1f</span><button data-score="good">\u2713 \u6709\u5e6b\u52a9</button><button data-score="bad">\u2715 \u9700\u8981\u6539\u5584</button></div></div>';
  conversation.scrollTop = conversation.scrollHeight;
}

conversation.addEventListener("click", async (event) => {
  const copyButton = event.target.closest("[data-copy]");
  if (copyButton) { await navigator.clipboard?.writeText(copyButton.closest(".ai").querySelector(".answer").innerText); copyButton.classList.add("copied"); setTimeout(() => copyButton.classList.remove("copied"), 1400); return; }
  const scoreButton = event.target.closest("[data-score]"); if (!scoreButton) return;
  const feedback = scoreButton.closest(".feedback"); feedback.innerHTML = '<input class="feedback-note" placeholder="\u88dc\u5145\u8aaa\u660e\uff08\u9078\u586b\uff09"><button data-save-score="' + scoreButton.dataset.score + '">\u9001\u51fa\u56de\u994b</button>';
});
conversation.addEventListener("click", async (event) => {
  const saveButton = event.target.closest("[data-save-score]"); if (!saveButton) return;
  const feedback = saveButton.closest(".feedback"); saveButton.disabled = true;
  const response = await fetch("/api/feedback", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ score: saveButton.dataset.saveScore, note: feedback.querySelector("input").value, question: feedback.dataset.question, answer: feedback.dataset.answer }) });
  const data = await response.json(); feedback.innerHTML = response.ok ? "<b>\u2713 \u56de\u994b\u5df2\u5132\u5b58</b>" : "<b>" + escapeHtml(data.error) + "</b>";
});
function escapeHtml(value) { const element = document.createElement("div"); element.textContent = value; return element.innerHTML; }
function escapeAttribute(value) { return escapeHtml(value).replace(/"/g, "&quot;"); }
