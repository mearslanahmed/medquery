# API Reference

## Base URLs

- **Production:** `https://medquery-chatbot.onrender.com`
- **Local Dev:** `http://localhost:8080`

---

## 1. Streaming Chat (`POST /api/chat/stream`)

Streams the response word-by-word using Server-Sent Events (SSE). This is what the web UI uses.

### Headers
```http
Content-Type: application/json
```

### Request Payload
```json
{
  "msg": "What are the first-aid steps for chemical burns?",
  "history": [
    { "role": "user", "text": "What causes skin burns?" },
    { "role": "assistant", "text": "Burns are caused by heat, chemicals, radiation, or electricity..." }
  ]
}
```

### Stream Events Format

The server streams lines starting with `data: `:

1. **Sources (sent first, as soon as Pinecone finishes searching):**
   ```json
   data: {"type": "sources", "sources": [{"file": "Harrison_Internal_Medicine.pdf", "page": 412, "snippet": "Chemical burns require immediate and copious irrigation with water for at least 15 minutes..."}]}
   ```
2. **Tokens (streamed as generated):**
   ```json
   data: {"type": "token", "token": "### First-Aid Steps\n"}
   data: {"type": "token", "token": "- **Irrigate immediately:** Flush with cool water for 15+ minutes.\n"}
   ```
3. **Done (sent when finished):**
   ```json
   data: {"type": "done"}
   ```

---

## 2. Standard Chat (`POST /api/chat`)

If you want a normal blocking API call that returns the full answer and sources together in one JSON payload:

### Request
```json
{
  "msg": "Compare Ibuprofen and Paracetamol.",
  "history": []
}
```

### Response (`200 OK`)
```json
{
  "status": "success",
  "query": "Compare Ibuprofen and Paracetamol.",
  "answer": "### Overview\n- **Ibuprofen:** NSAID with anti-inflammatory and pain-relieving effects...\n- **Paracetamol:** Works centrally on pain and fever without reducing inflammation...",
  "sources": [
    {
      "file": "Goodman_Gilman_Pharmacology.pdf",
      "page": 680,
      "snippet": "Ibuprofen is a non-selective inhibitor of cyclooxygenase..."
    }
  ]
}
```

---

## 3. Health Check (`GET /api/health` or `GET /ping`)

Use this endpoint to verify that the app is awake and connected to Pinecone.

### Response (`200 OK`)
```json
{
  "status": "healthy",
  "system": "Multi-Agent Clinical Pipeline",
  "providers": "Groq + OpenRouter + Gemini Fallback Pool",
  "index": "medquery",
  "uptime_seconds": 1250
}
```

---

## 4. Code Examples

### cURL
```bash
curl -N -X POST https://medquery-chatbot.onrender.com/api/chat/stream \
     -H "Content-Type: application/json" \
     -d '{"msg": "What are the common symptoms of Asthma?"}'
```

### Python
```python
import json
import requests

url = "https://medquery-chatbot.onrender.com/api/chat/stream"
payload = {"msg": "What is asthma?", "history": []}

with requests.post(url, json=payload, stream=True) as r:
    for line in r.iter_lines(decode_unicode=True):
        if line and line.startswith("data: "):
            event = json.loads(line[6:])
            if event["type"] == "token":
                print(event["token"], end="", flush=True)
            elif event["type"] == "sources":
                print(f"\n[Found {len(event['sources'])} citations]")
            elif event["type"] == "done":
                print("\n[Done]")
```

### JavaScript (Node.js or Browser)
```javascript
const response = await fetch("https://medquery-chatbot.onrender.com/api/chat/stream", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ msg: "What is asthma?", history: [] })
});

const reader = response.body.getReader();
const decoder = new TextDecoder();

while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    const chunk = decoder.decode(value);
    chunk.split("\n\n").forEach(line => {
        if (line.startsWith("data: ")) {
            const data = JSON.parse(line.slice(6));
            if (data.type === "token") process.stdout.write(data.token);
        }
    });
}
```
