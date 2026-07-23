const list = document.querySelector("#feedbackList");
const count = document.querySelector("#feedbackCount");
let feedbackItems = [];
let activeFilter = "all";

document.querySelector("#refresh").addEventListener("click", loadFeedback);
document.querySelector(".filter").addEventListener("click", (event) => {
  const button = event.target.closest("[data-filter]");
  if (!button) return;
  activeFilter = button.dataset.filter;
  document.querySelectorAll("[data-filter]").forEach((item) => item.classList.toggle("selected", item === button));
  renderFeedback();
});

async function loadFeedback() {
  const response = await fetch("/api/feedbacks");
  const data = await response.json();
  if (!response.ok) return;
  feedbackItems = data.items;
  document.querySelector("#total").textContent = feedbackItems.length;
  document.querySelector("#good").textContent = feedbackItems.filter((item) => item.score === "good").length;
  document.querySelector("#bad").textContent = feedbackItems.filter((item) => item.score === "bad").length;
  renderFeedback();
}

function renderFeedback() {
  const visibleItems = feedbackItems.filter((item) => activeFilter === "all" || item.score === activeFilter);
  count.textContent = `${visibleItems.length} 筆紀錄`;
  list.innerHTML = visibleItems.length ? visibleItems.map(renderRecord).join("") : '<p class="empty-log">尚無回饋紀錄。</p>';
}

function renderRecord(item) {
  const score = item.score === "good" ? "有幫助" : "需要改善";
  return `<article class="feedback-record"><div><span class="feedback-score ${item.score}">${score}</span><time>${new Date(item.created_at).toLocaleString("zh-TW")}</time></div><h2>${escapeHtml(item.question)}</h2><p>${escapeHtml(item.answer)}</p>${item.note ? `<aside><b>補充說明</b>${escapeHtml(item.note)}</aside>` : ""}</article>`;
}

function escapeHtml(value) { const element = document.createElement("div"); element.textContent = value || ""; return element.innerHTML; }
loadFeedback();
