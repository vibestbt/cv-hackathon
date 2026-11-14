# Memory You Can Govern — hackathon demo

A small Hermes agent with a **memory box** behind it, and Larry as the
**Keeper**. Built for Cerebral Valley "Agents in Context Toronto"
(Sat Nov 14, 2026), Problem Statement 2: *Memory You Can Govern* —
memory that changes what the agent does on the next run, with rules for
who can read each memory, how long it lives, and how it gets corrected
or forgotten.

Screen: chat on the left, **receipt trail** on the right. Every store,
retrieve, correct, expiry, refusal and forget writes a real receipt,
derived from the actual operation.

## Run it — zero-code operation (by design)

The operator never writes or reads code. From this repo:
`python3 server.py` → http://localhost:8765 (PORT env overrides) —
open that address and the demo is up. (On the operator's own machine
a double-click launcher runs the same command and opens the browser
by itself.)

Everything is driven from the page: plain-English chat, labelled
buttons for every beat, a persona dropdown, "New run", "Reset
everything". **Atlas connects from the page too** — paste the
connection string into the box at the top and click "Connect to
Atlas"; nothing is saved unless the connection succeeds, and
**"Use this laptop's box instead"** is the one-click fallback
(session-only; the saved Atlas setup returns next start). No installs
needed for the local version (Python 3 standard library only); the
start file quietly installs the one Atlas driver (`pymongo`) itself,
only when an Atlas setup exists.

## Backends — one config point

Default: local file store (`memories.json`), retrieval ranked by local
cosine similarity over deterministic hashing embeddings.

One config point underneath: **`atlas_uri.txt`** in this folder — but
the operator never edits it by hand; the page's "Connect to Atlas"
writes it, only after a successful connection. With a string present
at start, retrieval tries a real Atlas `$vectorSearch` (index
`vector_index`); if the index doesn't exist
yet, data still lives in Atlas and ranking is done locally — the
page's top-right corner always labels where the box lives, in plain
words: "MongoDB Atlas (cloud)", "MongoDB Atlas (cloud) — search done
on this laptop", or "this laptop".

## The six beats

1. **STORE** a fact with a receipt + read-rule ("only the Keeper").
2. **NEXT RUN** — a fresh run answers from the box; behaviour changed.
3. **CORRECT** $750 → $900; old version superseded with a receipt.
4. **EXPIRY** — a door code with a 20-second lifetime, honoured.
5. **READ-RULE** — a second persona (visitor) is refused, on the record
   — for reading AND for changing: the memory's card governs both.
6. **FORGET + PROVE** — content hard-deleted (whole version chain);
   chat, direct search and vector search all return nothing, and the
   receipt trail itself is scrubbed of the forgotten content.

## Tests

```
python3 test_demo.py
```

Result, Sat Nov 14, 2026: **14 tests, all passing** — one per beat
plus persistence-across-restart, a regression guard that an unrelated
memory is never served as the answer to a question it doesn't match,
and the change-lock tests: a visitor can neither correct nor forget a
keeper-only memory, while the Keeper corrects freely and a visitor can
still correct a memory whose rule names the visitor.
The Atlas path was verified live against the sandbox cluster
(Nov 13-14): connect from the page, store, vector search, restart
persistence, and a 15-point live battery through the page — all
passing. A bad connection string is refused with a plain-English
message and nothing is saved; an unreachable cluster is refused the
same way; "Use this laptop's box instead" switches the backend and
reports it.
Two real bugs were found by testing the beats end-to-end and fixed
before this was called done: forgetting originally left the superseded
version's value quoted in the trail, and an expired memory's question
could be answered by an unrelated memory (similarity threshold too low).

## Files

- `memory_box.py` — the box + agent (all logic, both backends)
- `server.py` — stdlib web server + JSON API (chat, connect-to-Atlas, fallback switch)
- `index.html` — the demo page (chat left, receipts right, labelled beat buttons, Atlas paste box)
- `test_demo.py` — headless beat tests
- `atlas_uri.txt` — local only, never committed: written by the app on a successful Atlas connect; the one config point
