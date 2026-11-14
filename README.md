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

The operator never writes or reads code. **Double-click
`START-DEMO.command`** in this folder (or `bash start.sh`) — it brings
the demo up and opens it in the browser by itself. Under the hood it
runs `python3 server.py` → http://localhost:8765 (PORT env overrides).

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

Practice script (exact words, exact clicks, Saturday plan):
**[PRACTICE.md](PRACTICE.md)**.

## Backends — one config point

Default: local file store (`memories.json`), retrieval ranked by local
cosine similarity over deterministic hashing embeddings.

One config point underneath: **`atlas_uri.txt`** in this folder — but
the operator never edits it by hand; the page's "Connect to Atlas"
writes it, only after a successful connection. With a string present
at start, retrieval tries a real Atlas `$vectorSearch` (index
`vector_index`, see PRACTICE.md step 2); if the index doesn't exist
yet, data still lives in Atlas and ranking is done locally — the
page's top-right corner always labels where the box lives, in plain
words: "MongoDB Atlas (cloud)", "MongoDB Atlas (cloud) — search done
on this laptop", or "this laptop".

## The six beats

1. **STORE** a fact with a receipt + read-rule ("only the Keeper").
2. **NEXT RUN** — a fresh run answers from the box; behaviour changed.
3. **CORRECT** $750 → $900; old version superseded with a receipt.
4. **EXPIRY** — a door code with a 20-second lifetime, honoured.
5. **READ-RULE** — a second persona (visitor) is refused, on the record.
6. **FORGET + PROVE** — content hard-deleted (whole version chain);
   chat, direct search and vector search all return nothing, and the
   receipt trail itself is scrubbed of the forgotten content.

## Tests

```
python3 test_demo.py
```

Result, Wed Nov 11, 2026: **8 tests, all passing** (2.4s) — one per beat
plus persistence-across-restart and a regression guard that an unrelated
memory is never served as the answer to a question it doesn't match.
The page's two new endpoints were verified live the same evening: a
bad connection string is refused with a plain-English message and
nothing is saved; an unreachable cluster is refused the same way;
"Use this laptop's box instead" switches the backend and reports it.
(The Atlas-connect success path shares the constructor code exercised
at startup; it gets its first live run against Larry's real cluster in
tonight's practice, Step 2.)
Two real bugs were found by testing the beats end-to-end and fixed
before this was called done: forgetting originally left the superseded
version's value quoted in the trail, and an expired memory's question
could be answered by an unrelated memory (similarity threshold too low).

## Files

- `memory_box.py` — the box + agent (all logic, both backends)
- `server.py` — stdlib web server + JSON API (chat, connect-to-Atlas, fallback switch)
- `index.html` — the demo page (chat left, receipts right, labelled beat buttons, Atlas paste box)
- `START-DEMO.command` / `start.sh` — the one double-click / one command that starts everything
- `test_demo.py` — headless beat tests
- `PRACTICE.md` — Larry's practice + Saturday runbook
- `atlas_uri.txt` — (written by the app on a successful connect) the one config point; never commit it
