# Talk mode — a conversational layer over the memory box

Talk mode lets the agent understand free-form English ("Hey, keep in
mind the client budget is $750 for the audit") instead of only the
classic commands. It is **off by default** and completely optional:
with talk mode off — or with no API key present — the demo behaves
byte-identically to the classic version.

## How to enable

1. Put an OpenRouter API key (nothing else) in a file named
   `openrouter_key.txt` in this folder. The file is read by
   `brain.py` at runtime; the key is never hardcoded and must never
   be committed.
2. Start the server with talk mode on, from this folder:

   ```
   TALK_MODE=1 python3 server.py
   ```

3. (Optional) pick a different model with `TALK_MODEL`
   (default: `openai/gpt-4.1-nano` — chosen cheap and fast; the two
   calls per message are tiny).

## How it works

- **Interpret** (`brain.interpret`): the model turns the input into a
  JSON plan only — `store` / `correct` / `forget` / `ask` / `chat`,
  with the fact or topic extracted as a clean string.
- **Execute**: the `MemoryBox` runs the operation deterministically.
  It enforces every rule (read/change/forget locks, expiry) and
  writes every receipt, exactly as in the classic demo.
- **Phrase** (`brain.phrase`): the model turns ONLY the operation
  outcome into one natural sentence. A denied outcome contains no
  memory content, so there is nothing for the model to leak.

Explicit commands (`Remember:` / `Correct:` / `Forget:` / `Search:`)
always take the classic path, even with talk mode on.

## Fallback behaviour

Any brain failure — missing/empty key file, HTTP error, timeout
(10 s), invalid JSON, unknown action — raises `BrainUnavailable`:

- Failure while *interpreting* (before anything touched the box):
  the input falls back to the classic path, unchanged.
- Failure while *phrasing* (after the box already executed): the
  agent replies with a deterministic local phrasing of the same
  outcome. The operation is never re-executed.

A brain failure can never break a request.

## Honesty note

The box enforces every rule; the model only interprets and phrases.
The model cannot read, change, or forget anything on its own — every
operation goes through the same `MemoryBox` methods (and the same
receipts) as the classic demo.

## Tests

`python3 test_demo.py` — the classic 18 beat tests plus talk-mode
tests that inject a fake transport (`FakeTransport`): no network and
no key file are needed.
