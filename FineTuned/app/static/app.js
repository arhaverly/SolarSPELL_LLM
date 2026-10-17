"use strict";

const messages = document.getElementById("messages");
const empty = document.getElementById("empty");
const form = document.getElementById("composer");
const input = document.getElementById("input");
const send = document.getElementById("send");
const status = document.getElementById("status");

function renderMarkdown(text) {
  return DOMPurify.sanitize(marked.parse(text));
}

function addMessage(role) {
  empty.hidden = true;
  const wrapper = document.createElement("section");
  wrapper.className = `message ${role}`;
  wrapper.innerHTML = `<div><span class="avatar">${role === "user" ? "You" : "AI"}</span><div class="content"></div></div>`;
  messages.appendChild(wrapper);
  return wrapper.querySelector(".content");
}

function scrollToBottom() {
  messages.scrollTop = messages.scrollHeight;
}

function showSources(content, sources) {
  if (!sources.length) return;
  const details = document.createElement("details");
  details.className = "sources";
  details.innerHTML = "<summary>Sources</summary><ol></ol>";
  const list = details.querySelector("ol");
  for (const s of sources) {
    const item = document.createElement("li");
    item.textContent = `${s.source}, p. ${s.page}`;
    list.appendChild(item);
  }
  content.after(details);
}

async function ask(question) {
  addMessage("user").textContent = question;
  const content = addMessage("assistant");
  content.classList.add("pending");
  scrollToBottom();

  let answer = "";
  let sources = [];
  try {
    const response = await fetch("api/query", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query: question }),
    });
    if (!response.ok) throw new Error(`Server error ${response.status}`);

    // Newline-delimited JSON: {"sources": [...]}, then {"token": "..."} lines.
    const reader = response.body.pipeThrough(new TextDecoderStream()).getReader();
    let buffer = "";
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += value;
      const lines = buffer.split("\n");
      buffer = lines.pop();
      for (const line of lines) {
        if (!line) continue;
        const event = JSON.parse(line);
        if (event.sources) sources = event.sources;
        if (event.token) answer += event.token;
        if (event.error) throw new Error(event.error);
      }
      content.innerHTML = renderMarkdown(answer);
      scrollToBottom();
    }
  } catch (error) {
    content.innerHTML = renderMarkdown(answer);
    const note = document.createElement("p");
    note.className = "error";
    note.textContent = error.message;
    content.appendChild(note);
  }
  content.classList.remove("pending");
  showSources(content, sources);
  scrollToBottom();
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  const question = input.value.trim();
  if (!question || send.disabled) return;
  input.value = "";
  send.disabled = input.disabled = true;
  await ask(question);
  send.disabled = input.disabled = false;
  input.focus();
});

input.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    form.requestSubmit();
  }
});

async function checkHealth() {
  try {
    const health = await (await fetch("api/health")).json();
    status.className = `status ${health.llm ? "ok" : "down"}`;
    status.title = health.llm ? "Language model ready" : "Language model not running";
  } catch {
    status.className = "status down";
  }
}

checkHealth();
setInterval(checkHealth, 30000);
