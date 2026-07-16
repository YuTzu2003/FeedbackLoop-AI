const refreshButton = document.querySelector("#refresh");
const message = document.querySelector("#connectionMessage");
refreshButton.addEventListener("click", loadStatus);
async function loadStatus() {
    refreshButton.disabled = true;
    const response = await fetch("/api/weaviate/status"); 
    const data = await response.json(); 
    document.querySelector("#ready").textContent = data.ready ? "已就緒" : "未就緒"; 
    document.querySelector("#live").textContent = data.live ? "正常" : "未連線"; 
    message.textContent = response.ok ? "連線成功：未執行任何 collection 或資料物件操作。" : data.error; message.classList.toggle("error", !response.ok); 
    refreshButton.disabled = false;
}
loadStatus();
