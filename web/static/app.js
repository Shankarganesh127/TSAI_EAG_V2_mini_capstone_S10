const form = document.querySelector("#query-form");
const input = document.querySelector("#query");
const submit = document.querySelector("#submit");
const conversation = document.querySelector("#conversation");
const status = document.querySelector("#status");

function appendMessage(role, text, html = null) {
  const article = document.createElement("article");
  article.className = `message ${role}`;

  const label = document.createElement("div");
  label.className = "label";
  label.textContent = role === "user" ? "You" : "Agent";

  const bubble = document.createElement("div");
  bubble.className = "bubble";
  if (role === "assistant" && html) {
    bubble.innerHTML = html;
  } else {
    bubble.textContent = text;
  }

  article.append(label, bubble);
  conversation.append(article);
  conversation.scrollTop = conversation.scrollHeight;
}

function setBusy(busy) {
  submit.disabled = busy;
  input.disabled = busy;
  status.textContent = busy
    ? "Searching memory and local knowledge…"
    : "Logs and processing details remain in the background.";
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  const query = input.value.trim();
  if (!query) return;

  appendMessage("user", query);
  input.value = "";
  setBusy(true);

  try {
    const response = await fetch("/api/query", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query }),
    });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.detail || "Request failed");
    appendMessage("assistant", payload.answer, payload.answer_html);
  } catch (error) {
    appendMessage("assistant", `Sorry, I could not complete that request. ${error.message}`);
  } finally {
    setBusy(false);
    input.focus();
  }
});

input.addEventListener("input", () => {
  input.style.height = "auto";
  input.style.height = `${Math.min(input.scrollHeight, 180)}px`;
});

input.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    form.requestSubmit();
  }
});
