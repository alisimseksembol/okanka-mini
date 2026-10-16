# Okanka Mini Space — API contract

Base URL (`SPACE`): the Space's direct URL, e.g. `https://<hf-user>-okanka-mini-chat.hf.space`
(NOT `https://huggingface.co/spaces/...`). This is the value for the website's `OKANKA_MINI_URL`.

The Space is a Gradio app (Gradio SDK, ZeroGPU). It exposes one Gradio API endpoint named **`chat`**
plus a plain `GET /health`.

## 1. `chat` (Gradio HTTP API, two requests)

### Inputs (positional, always all three, in this order)

| # | name | type | meaning |
|---|---|---|---|
| 0 | `messages_json` | string | JSON **string** of an array `[{"role": "system"\|"user"\|"assistant", "content": "..."}]`; the last message must be `user`. System messages are prepended to the first user turn. Older turns are dropped if the prompt exceeds ~1.5K tokens. |
| 1 | `max_tokens` | integer | max new tokens (clamped to 1..1024 and to the context window). |
| 2 | `key` | string | shared secret. Checked only if the Space env var `OKANKA_MINI_KEY` is set; otherwise ignored (send `""`). |

### Step 1 — submit

```
POST {SPACE}/gradio_api/call/chat
Content-Type: application/json

{"data": ["[{\"role\":\"user\",\"content\":\"Merhaba, sen kimsin?\"}]", 256, "<OKANKA_MINI_KEY or empty>"]}
```

Response `200`: `{"event_id": "a1b2c3..."}`

### Step 2 — stream the result (Server-Sent Events)

```
GET {SPACE}/gradio_api/call/chat/{event_id}
Accept: text/event-stream
```

The body is an SSE stream. Each event's `data` is a JSON array with ONE string = **the full reply text so far**
(not a delta). Example:

```
event: generating
data: [""]

event: generating
data: ["Ben Okanka"]

event: generating
data: ["Ben Okanka Mini'yim, Okanka ekibinin"]

event: complete
data: ["Ben Okanka Mini'yim, Okanka ekibinin tamamen sıfırdan eğittiği deneysel yerli modeliyim."]
```

Other events:

* `event: error` — verified with Gradio 6.29.1 the payload is a JSON object, e.g.
  `data: {"error": "unauthorized: invalid key", "duration": 10, "visible": true, "title": "Error"}`
  (wrong key), `{"error": "bad request: ...", ...}` (malformed `messages_json`), or a ZeroGPU quota message.
  Older Gradio versions may send `data: null`. Treat any `error` event as failure and show `data.error` if present.
* `event: heartbeat` with `data: null` — keep-alive, ignore.

Client logic: keep the last `generating` value; the `complete` value is the final reply. To show a typing
effect, render `data[0]` of each `generating` event (replace, don't append).

### Notes for the Next.js site

* Call it **server-side** (Route Handler) so the key never reaches the browser; send `key` from `OKANKA_MINI_KEY`.
* ZeroGPU cold start / queue: the first request after idle can take 30-120 s (Space wake-up + model load).
  Use a generous timeout (e.g. 180 s) and show a "model uyanıyor" message.
* Free ZeroGPU quota is per *calling HF user*; anonymous server-side calls share the Space's limited
  anonymous quota. If you see quota errors, send `Authorization: Bearer <HF token>` on both requests
  (an HF token with read scope; PRO accounts get more ZeroGPU minutes).
* Optional: `@gradio/client` (`Client.connect(SPACE)` then `client.submit("/chat", [messages_json, max_tokens, key])`)
  implements the same protocol.

### Minimal TypeScript example

```ts
const base = process.env.OKANKA_MINI_URL!; // https://<user>-okanka-mini-chat.hf.space
const r1 = await fetch(`${base}/gradio_api/call/chat`, {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({ data: [JSON.stringify(messages), 256, process.env.OKANKA_MINI_KEY ?? ""] }),
});
const { event_id } = await r1.json();
const r2 = await fetch(`${base}/gradio_api/call/chat/${event_id}`);
const reader = r2.body!.getReader();
const dec = new TextDecoder();
let buf = "", event = "", text = "";
for (;;) {
  const { value, done } = await reader.read();
  if (done) break;
  buf += dec.decode(value, { stream: true });
  let i;
  while ((i = buf.indexOf("\n")) >= 0) {
    const line = buf.slice(0, i).trimEnd(); buf = buf.slice(i + 1);
    if (line.startsWith("event:")) event = line.slice(6).trim();
    else if (line.startsWith("data:")) {
      const payload = line.slice(5).trim();
      if (event === "error") throw new Error(payload);
      if (event === "generating" || event === "complete") {
        const arr = JSON.parse(payload);
        if (Array.isArray(arr)) text = arr[0];   // full text so far
      }
    }
  }
}
// `text` is the final reply
```

### curl example

```bash
EID=$(curl -s -X POST "$SPACE/gradio_api/call/chat" -H "Content-Type: application/json" \
  -d '{"data":["[{\"role\":\"user\",\"content\":\"Merhaba!\"}]",128,""]}' | python -c "import sys,json;print(json.load(sys.stdin)['event_id'])")
curl -N "$SPACE/gradio_api/call/chat/$EID"
```

## 2. `GET /health`

```
GET {SPACE}/health   ->  200 application/json
{"status":"ok","model_repo":"<user>/okanka-mini","device":"cuda(zerogpu)"|"cpu","n_layer":16,"n_embd":1024,
 "vocab_size":32768,"sequence_len":1024,"key_required":true}
```

No key needed. Useful for the site's availability check (a sleeping Space wakes up on this request).
