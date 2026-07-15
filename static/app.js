const input = document.querySelector('#fileInput');
const status = document.querySelector('#fileStatus');
const question = document.querySelector('#question');
const submit = document.querySelector('#askButton');
const conversation = document.querySelector('#conversation');

input.addEventListener('change', async () => {
  if (!input.files[0]) return;
  status.textContent = '\u6b63\u5728\u8655\u7406 ' + input.files[0].name + '\u2026';
  const form = new FormData(); form.append('file', input.files[0]);
  const res = await fetch('/api/upload', {method:'POST', body:form}); const data = await res.json();
  if (!res.ok) { status.textContent = data.error; return; }
  status.innerHTML = '<b>\u2713 ' + escapeHtml(data.filename) + '</b><small>\u5df2\u6e96\u5099\u597d\u5206\u6790</small>';
  question.disabled = submit.disabled = false; question.focus();
});

document.querySelector('#questionForm').addEventListener('submit', async e => {
  e.preventDefault(); const text = question.value.trim(); if (!text) return;
  ask(text);
});

async function ask(text) {
  conversation.innerHTML += `<div class="message user">${escapeHtml(text)}</div><div class="message ai loading">\u6b63\u5728\u5206\u6790\u8cc7\u6599<span>.</span><span>.</span><span>.</span></div>`;
  conversation.scrollTop = conversation.scrollHeight; question.value=''; submit.disabled=true;
  const res = await fetch('/api/ask',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({question:text})}); const data=await res.json();
  document.querySelector('.loading')?.remove(); submit.disabled=false;
  if (!res.ok) return;
  conversation.innerHTML += `<div class="message ai"><div class="answer">${escapeHtml(data.answer).replace(/\n/g,'<br>')}</div><div class="sources">\u53c3\u8003\uff1a${data.sources.map(escapeHtml).join(' \u00b7 ')}</div><div class="answer-actions"><button data-copy>\u8907\u88fd\u56de\u7b54</button><button data-regenerate data-question="${escapeAttr(text)}">\u91cd\u65b0\u7522\u751f</button></div><div class="feedback" data-question="${escapeAttr(text)}" data-answer="${escapeAttr(data.answer)}"><span>\u9019\u500b\u56de\u7b54\u6709\u5e6b\u52a9\u55ce\uff1f</span><button data-score="good">\u2713 \u6709\u5e6b\u52a9</button><button data-score="bad">\u2715 \u9700\u8981\u6539\u5584</button></div></div>`;
  conversation.scrollTop = conversation.scrollHeight;
}

conversation.addEventListener('click', async e => {
  if (e.target.dataset.copy !== undefined) { navigator.clipboard?.writeText(e.target.closest('.ai').querySelector('.answer').innerText); e.target.textContent='\u5df2\u8907\u88fd'; return; }
  if (e.target.dataset.regenerate !== undefined) { ask(e.target.dataset.question); return; }
  if (!e.target.dataset.score) return;
  const box=e.target.parentElement; box.innerHTML=`<input class="feedback-note" placeholder="\u88dc\u5145\u8aaa\u660e\uff08\u9078\u586b\uff09"><button data-save-score="${e.target.dataset.score}">\u5132\u5b58\u56de\u994b</button>`;
});

conversation.addEventListener('click', async e => {
  if (!e.target.dataset.saveScore) return;
  const box=e.target.parentElement; const result=await fetch('/api/feedback',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({score:e.target.dataset.saveScore,note:box.querySelector('input').value,question:box.dataset.question,answer:box.dataset.answer})});
  if (result.ok) box.innerHTML='<b>\u2713 \u56de\u994b\u5df2\u5132\u5b58</b>';
});
function escapeHtml(value) { const el=document.createElement('div'); el.textContent=value; return el.innerHTML; }
function escapeAttr(value) { return escapeHtml(value).replace(/"/g, '&quot;'); }
