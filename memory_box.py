"""
memory_box.py — the "memory box" behind the little Hermes agent.

Hackathon demo for Problem Statement 2: "Memory You Can Govern".
Implements, for real (no theatre):
  - STORE      a memory with a receipt and a read-rule (who may read it)
  - RETRIEVE   semantic search (vector similarity) with read-rule enforcement
  - CORRECT    a new version supersedes the old one, both get receipts;
               the memory's rule gates CHANGING as well as reading
  - EXPIRY     memories can carry a lifetime; expired memories stop answering
  - FORGET     hard delete of the content + a tombstone; every recall path
               (chat retrieve, direct search, vector search) returns nothing,
               and the receipt trail redacts the forgotten content

Backends:
  - Local file store (default): memories live in memories.json next to this
    file, ranked by a local cosine similarity over hashing embeddings.
  - MongoDB Atlas (when configured): if a connection string is present in
    the file atlas_uri.txt in this folder AND the `pymongo` package is
    installed, memories are stored in Atlas and retrieval first tries a
    real Atlas $vectorSearch aggregation. If the vector index does not
    exist yet, retrieval falls back to ranking the Atlas documents locally
    and says so (backend label "atlas-storage-local-rank"). The on-stage
    target is Atlas with the vector index created (see PRACTICE.md).

Embeddings are a deterministic local hashing embedding (bag of words ->
128-dim vector). No API key, works offline, and the SAME vectors are what
get stored in Atlas, so $vectorSearch ranks exactly what the local path
ranks. This is deliberate: the demo is about governing memory, and the
embedding model is interchangeable.
"""

import hashlib
import json
import math
import os
import re
import time

HERE = os.path.dirname(os.path.abspath(__file__))
LOCAL_STORE = os.path.join(HERE, "memories.json")
ATLAS_URI_FILE = os.path.join(HERE, "atlas_uri.txt")
ATLAS_DB = "hackathon"
ATLAS_COLLECTION = "memories"
VECTOR_INDEX_NAME = "vector_index"
EMB_DIM = 128
SIM_THRESHOLD = 0.40  # cosine similarity; below this, search says "nothing found"
# (Atlas $vectorSearch returns (1+cos)/2 for cosine — _rank converts back,
#  so this threshold means the same thing on both backends.)


# ---------------------------------------------------------------- embeddings
def norm_token(tok):
    """Canonicalise a token so the same VALUE hashes the same way however
    it was typed: "$900", "900", "$900.00" and "900.00" are one token.
    (Found live, Nov 13 rehearsal: a bare "900" query scored cosine 0.0
    against a stored "$900" memory — different tokens, different buckets —
    and a bare "$900" query scored 0.32, under the 0.40 threshold.)"""
    tok = tok.strip("$',")
    tok = tok.replace(",", "")
    if re.fullmatch(r"\d+\.0+", tok):
        tok = tok.split(".")[0]
    return tok


def norm_tokens(text):
    return [t for t in (norm_token(tok) for tok in
                        re.findall(r"[a-z0-9$']+", text.lower())) if t]


def embed(text):
    """Deterministic hashing embedding: token -> bucket, count, L2-normalise."""
    vec = [0.0] * EMB_DIM
    for tok in norm_tokens(text):
        h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
        vec[h % EMB_DIM] += 1.0
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / norm for v in vec]


def cosine(a, b):
    return sum(x * y for x, y in zip(a, b))


# ------------------------------------------------------------------- topics
def topic_key(text):
    """A stable key so CORRECT and FORGET can find 'the same' memory."""
    t = text.lower()
    if "budget" in t:
        return "client-budget"
    if "door code" in t or "doorcode" in t:
        return "door-code"
    words = re.findall(r"[a-z0-9']+", t)
    words = [w for w in words if w not in
             ("the", "a", "an", "is", "remember", "that", "for", "of")]
    return "-".join(words[:3]) or "misc"


def fmt_money(text):
    m = re.search(r"\$\s?\d[\d,]*", text)
    return m.group(0).replace(" ", "") if m else None


# -------------------------------------------------------------------- store
class MemoryBox:
    """One memory box. `backend` is 'local', 'atlas', or
    'atlas-storage-local-rank' (set honestly at runtime)."""

    def __init__(self, store_path=LOCAL_STORE, atlas_uri=None):
        self.store_path = store_path
        self.memories = []   # list of dicts (includes tombstones)
        self.receipts = []   # list of dicts {n, ts, op, actor, detail}
        self.backend = "local"
        self._atlas = None
        uri = atlas_uri
        if uri is None and os.path.exists(ATLAS_URI_FILE):
            with open(ATLAS_URI_FILE) as f:
                uri = f.read().strip() or None
        if uri:
            self._try_atlas(uri)
        if self._atlas is None:
            self._load_local()
        else:
            self._load_atlas()

    # ------------------------------------------------------------ backends
    def _try_atlas(self, uri):
        try:
            from pymongo import MongoClient
        except ImportError:
            return  # pymongo not installed -> stay local, honestly
        try:
            client = MongoClient(uri, serverSelectionTimeoutMS=4000)
            client.admin.command("ping")
            self._atlas = client[ATLAS_DB][ATLAS_COLLECTION]
            self.backend = "atlas"
        except Exception:
            self._atlas = None

    def _load_local(self):
        if os.path.exists(self.store_path):
            with open(self.store_path) as f:
                data = json.load(f)
            self.memories = data.get("memories", [])
            self.receipts = data.get("receipts", [])

    def _save_local(self):
        with open(self.store_path, "w") as f:
            json.dump({"memories": self.memories,
                       "receipts": self.receipts}, f, indent=1)

    def _load_atlas(self):
        docs = list(self._atlas.find({"kind": "memory"}))
        for d in docs:
            d.pop("_id", None)
            d.pop("kind", None)
        self.memories = docs
        rdocs = list(self._atlas.find({"kind": "receipt"}).sort("n", 1))
        for d in rdocs:
            d.pop("_id", None)
            d.pop("kind", None)
        self.receipts = rdocs

    def _persist_memory(self, mem):
        if self._atlas is not None:
            doc = dict(mem)
            doc["kind"] = "memory"
            self._atlas.replace_one({"kind": "memory", "id": mem["id"]},
                                    doc, upsert=True)
        else:
            self._save_local()

    def _persist_all(self):
        if self._atlas is not None:
            for m in self.memories:
                self._persist_memory(m)
            self._atlas.delete_many({"kind": "receipt"})
            for r in self.receipts:
                doc = dict(r)
                doc["kind"] = "receipt"
                self._atlas.insert_one(doc)
        else:
            self._save_local()

    def wipe(self):
        self.memories = []
        self.receipts = []
        if self._atlas is not None:
            self._atlas.delete_many({})
        else:
            self._save_local()

    # ------------------------------------------------------------ receipts
    def _receipt(self, op, actor, detail):
        r = {"n": len(self.receipts) + 1,
             "ts": time.strftime("%H:%M:%S"),
             "op": op, "actor": actor, "detail": detail}
        self.receipts.append(r)
        if self._atlas is not None:
            doc = dict(r)
            doc["kind"] = "receipt"
            self._atlas.insert_one(doc)
        return r

    # ----------------------------------------------------------- lifecycle
    def _expire_pass(self, actor="keeper"):
        """Mark newly-expired memories, with one receipt each."""
        now = time.time()
        for m in self.memories:
            if (m.get("status") == "active" and m.get("expires_at")
                    and now >= m["expires_at"]):
                m["status"] = "expired"
                self._persist_memory(m)
                self._receipt("EXPIRED", actor,
                              "Memory #%d (%s) reached the end of its "
                              "lifetime and stopped answering."
                              % (m["id"], m["key"]))

    def _active(self):
        return [m for m in self.memories if m.get("status") == "active"]

    # ------------------------------------------------------------ operations
    def store(self, text, actor="keeper", readers=None, ttl_seconds=None):
        self._expire_pass(actor)
        readers = readers or ["keeper"]
        key = topic_key(text)
        mem = {"id": (max([m["id"] for m in self.memories], default=0) + 1),
               "key": key, "text": text, "embedding": embed(text),
               "readers": readers, "status": "active",
               "stored_at": time.time(),
               "expires_at": (time.time() + ttl_seconds) if ttl_seconds else None}
        self.memories.append(mem)
        self._persist_memory(mem)
        rule = "readable by: " + ", ".join(readers)
        life = ("lifetime: %ds" % ttl_seconds) if ttl_seconds else "no expiry"
        self._receipt("STORE", actor,
                      "Memory #%d stored: \"%s\" (%s; %s)"
                      % (mem["id"], text, rule, life))
        return mem

    def correct(self, text, actor="keeper"):
        """A correction supersedes the active memory with the same key.
        The memory's card governs CHANGING as well as reading: an actor
        the rule doesn't name cannot supersede it. Returns the new
        memory, or None when the change is refused."""
        self._expire_pass(actor)
        key = topic_key(text)
        old = [m for m in self._active() if m["key"] == key]
        if old and actor not in old[0]["readers"]:
            self._receipt("REFUSED", actor,
                          "%s tried to change memory #%d (%s) — rule says "
                          "changeable by: %s. Refused."
                          % (actor, old[0]["id"], old[0]["key"],
                             ", ".join(old[0]["readers"])))
            return None
        new = self.store(text, actor=actor,
                         readers=old[0]["readers"] if old else ["keeper"])
        for m in old:
            m["status"] = "superseded"
            self._persist_memory(m)
            self._receipt("CORRECT", actor,
                          "Memory #%d superseded by #%d: \"%s\" -> \"%s\""
                          % (m["id"], new["id"], m["text"], text))
        return new

    def forget(self, query, actor="keeper"):
        """Hard-delete content for the memory matching `query`.
        Returns the forgotten memory's id, None if nothing matched, or
        "refused" if the memory's rule doesn't name this actor — the
        card governs forgetting too, same lock as correct()."""
        self._expire_pass(actor)
        key = topic_key(query)
        # Forgetting a fact forgets its whole version chain — the active
        # version AND any superseded/expired versions of the same topic.
        targets = [m for m in self.memories
                   if m["key"] == key and m["status"] in
                   ("active", "expired", "superseded")]
        if not targets:
            self._receipt("FORGET-MISS", actor,
                          "Forget asked for '%s' — nothing matched." % key)
            return None
        gate = next((m for m in targets if m["status"] == "active"),
                    targets[0])
        if actor not in gate["readers"]:
            self._receipt("REFUSED", actor,
                          "%s tried to forget memory #%d (%s) — rule says "
                          "changeable by: %s. Refused."
                          % (actor, gate["id"], gate["key"],
                             ", ".join(gate["readers"])))
            return "refused"
        gone_texts = [m["text"] for m in targets if m.get("text")]
        for m in targets:
            m["text"] = None            # content is GONE, not hidden
            m["embedding"] = []
            m["status"] = "forgotten"
            self._persist_memory(m)
            self._receipt("FORGET", actor,
                          "Memory #%d (%s) FORGOTTEN: content deleted. "
                          "Only this tombstone remains." % (m["id"], m["key"]))
        # Scrub the trail: no earlier receipt may still quote the content.
        for r in self.receipts:
            for gt in gone_texts:
                if gt in r["detail"]:
                    r["detail"] = r["detail"].replace(
                        gt, "«content forgotten»")
        self._persist_all()
        return targets[0]["id"]

    # ------------------------------------------------------------- retrieval
    def _rank(self, query):
        """Return [(score, memory)] over active memories, best first.
        Uses Atlas $vectorSearch when available; otherwise local cosine."""
        act = self._active()
        if not act:
            return []
        qv = embed(query)
        if self._atlas is not None:
            try:
                pipeline = [
                    {"$vectorSearch": {
                        "index": VECTOR_INDEX_NAME,
                        "path": "embedding",
                        "queryVector": qv,
                        "numCandidates": 50, "limit": 10,
                        "filter": {"status": "active", "kind": "memory"}}},
                    {"$project": {"id": 1, "score": {"$meta": "vectorSearchScore"}}}]
                # Atlas cosine score is (1+cos)/2 — convert back to cosine
                # so the threshold and receipts mean the same as local.
                hits = {d["id"]: 2 * d["score"] - 1
                        for d in self._atlas.aggregate(pipeline)}
                if hits:
                    # Index-lag guard (seen live Nov 13): Atlas Search
                    # indexes new docs asynchronously, so a memory stored
                    # seconds ago can be MISSING from $vectorSearch hits
                    # while local cosine over the same stored embeddings
                    # scores it well above threshold. If the local best
                    # is missing from the hits, serve the local ranking
                    # rather than answering "I don't know" on stage.
                    local = sorted(
                        [(cosine(qv, m["embedding"]), m) for m in act],
                        key=lambda x: x[0], reverse=True)
                    if (local and local[0][0] >= SIM_THRESHOLD
                            and local[0][1]["id"] not in hits):
                        return local
                    self.backend = "atlas"
                    by_id = {m["id"]: m for m in act}
                    return sorted([(hits[m["id"]], by_id[m["id"]])
                                    for m in hits if m["id"] in by_id],
                                   reverse=True)
            except Exception:
                # No vector index yet -> honest local ranking over Atlas docs.
                self.backend = "atlas-storage-local-rank"
        return sorted([(cosine(qv, m["embedding"]), m) for m in act],
                      key=lambda x: x[0], reverse=True)

    _STOP = {"the", "a", "an", "is", "for", "of", "what", "whats", "s"}

    def _exact_match(self, query):
        """Value/short-query fallback: if every meaningful token in a
        SHORT query (<=3 tokens, or any token is a number) appears
        verbatim in an active memory's text, that memory IS the answer —
        vector similarity alone can never serve a bare value like "900"
        (single-token cosine against a full sentence tops out ~0.32).
        Only active memories are scanned, so expired / superseded /
        forgotten content can never resurface through this path, and the
        caller's read-rule check still happens in search() afterwards.
        Newest matching memory wins."""
        q = [t for t in norm_tokens(query) if t not in self._STOP]
        if not q or (len(q) > 3 and not any(t.isdigit() for t in q)):
            return None
        best = None
        for m in self._active():
            mtoks = set(norm_tokens(m["text"] or ""))
            if all(t in mtoks for t in q):
                if best is None or m["id"] > best["id"]:
                    best = m
        return best

    def search(self, query, actor="keeper"):
        """Direct search path (the 'look in the box' path).
        Returns dict {found, memory, denied}."""
        self._expire_pass(actor)
        exact = self._exact_match(query)
        if exact is not None:
            ranked = [(1.0, exact)]
        else:
            ranked = self._rank(query)
        if not ranked or ranked[0][0] < SIM_THRESHOLD:
            self._receipt("SEARCH", actor,
                          "Search \"%s\" -> nothing found." % query)
            return {"found": False, "denied": False, "memory": None}
        score, mem = ranked[0]
        if actor not in mem["readers"]:
            self._receipt("REFUSED", actor,
                          "%s asked for memory #%d (%s) — read-rule says "
                          "readable by: %s. Refused."
                          % (actor, mem["id"], mem["key"],
                             ", ".join(mem["readers"])))
            return {"found": True, "denied": True, "memory": None}
        self._receipt("RETRIEVE", actor,
                      "Memory #%d (%s) retrieved for %s (match %.2f)."
                      % (mem["id"], mem["key"], actor, score))
        return {"found": True, "denied": False, "memory": mem}


# --------------------------------------------------------------------- agent
class Agent:
    """The little Hermes. Parses plain-English demo commands and answers
    questions from the box. Session state is ONLY the run counter — memory
    lives in the box, which is the whole point of 'next run'."""

    def __init__(self, box):
        self.box = box
        self.run = 1

    def new_run(self, actor="keeper"):
        self.run += 1
        self.box._receipt("NEW RUN", actor,
                          "Run #%d started. The agent remembers only what "
                          "is in the box." % self.run)
        return ("New run started (run #%d). I only know what's in my "
                "memory box." % self.run)

    def handle(self, text, actor="keeper"):
        t = text.strip()
        low = t.lower()

        m = re.match(r"remember\s+for\s+(\d+)\s+seconds?\s*:\s*(.+)", low)
        if m:
            ttl, fact = int(m.group(1)), m.group(2)
            fact = t[t.index(":") + 1:].strip()
            self.box.store(fact, actor=actor, readers=["keeper", "visitor"],
                           ttl_seconds=ttl)
            return ("Stored for %d seconds: \"%s\". After that, it's gone "
                    "and I stop answering it." % (ttl, fact))

        if low.startswith("remember"):
            fact = t.split(":", 1)[1].strip() if ":" in t else \
                re.sub(r"^remember\s+", "", t, flags=re.I)
            self.box.store(fact, actor=actor, readers=["keeper"])
            return ("Stored, with a receipt. Rule on it: only the Keeper "
                    "(you) can read it back.")

        if low.startswith("correct"):
            fact = t.split(":", 1)[1].strip() if ":" in t else \
                re.sub(r"^correct\s+", "", t, flags=re.I)
            new = self.box.correct(fact, actor=actor)
            if new is None:
                return ("I have a memory about that, but only the Keeper "
                        "can change it. I can't do that.")
            return ("Corrected. The old version is superseded — from now "
                    "on I answer with the new one: \"%s\"" % fact)

        if low.startswith("forget"):
            what = re.sub(r"^forget\s+", "", t, flags=re.I)
            fid = self.box.forget(what, actor=actor)
            if fid == "refused":
                return ("I do have a memory about that, but only the "
                        "Keeper can forget it. I can't do that.")
            if fid is None:
                return "I couldn't find a memory matching that to forget."
            return ("Forgotten. Memory #%d is deleted — content gone, "
                    "only a tombstone receipt remains. Ask me again and "
                    "watch: nothing comes back." % fid)

        if low.startswith("search"):
            q = t.split(":", 1)[1].strip() if ":" in t else \
                re.sub(r"^search\s+", "", t, flags=re.I)
            res = self.box.search(q, actor=actor)
            if res["denied"]:
                return "That memory exists, but its read-rule refuses you."
            if not res["found"]:
                return "Direct search of the box: nothing found."
            return "Direct search found: \"%s\"" % res["memory"]["text"]

        # Anything else is treated as a question to answer from memory.
        res = self.box.search(t, actor=actor)
        if res["denied"]:
            return ("I do have a memory about that, but its rule says only "
                    "the Keeper can read it. I can't tell you.")
        if not res["found"]:
            return ("I don't know — there's nothing about that in my "
                    "memory box. (Run #%d)" % self.run)
        return "From my memory box: \"%s\"" % res["memory"]["text"]
