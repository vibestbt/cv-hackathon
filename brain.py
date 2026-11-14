"""
brain.py — the OPTIONAL talk-mode brain for the demo agent.

Talk mode lets a small LLM do two narrow jobs and NOTHING else:
  1. INTERPRET — turn free-form input into a structured intent
     (store / correct / forget / ask / chat) with the fact or topic
     extracted as a clean string.
  2. PHRASE — turn the DETERMINISTIC outcome of a MemoryBox operation
     into one natural sentence, using only what the outcome contains.

The MemoryBox stays the only authority: it executes the operation,
enforces every read/change rule, and writes every receipt. The model
never sees a memory it isn't handed in an outcome, so a denied outcome
contains no content for it to leak.

Config:
  - API key: read from the file `openrouter_key.txt` in THIS folder
    (never hardcoded, never committed). Missing/empty file -> talk
    mode is unavailable and the agent behaves exactly like the
    classic demo.
  - Model: env TALK_MODEL, default "openai/gpt-4.1-nano" (cheap+fast).
  - Stdlib only (urllib). Timeout 10s.

Failure contract: EVERY failure — no key, HTTP error, timeout,
invalid JSON, unknown action — raises BrainUnavailable. The caller
(the Agent) catches it and falls back to the classic behaviour, so a
brain failure can never break a request.

Tests inject a fake transport: Brain(key="test", transport=fake),
where fake(url, headers, payload, timeout) -> parsed-JSON dict.
No network, no key file needed.
"""

import json
import os
import urllib.error
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
KEY_FILE = os.path.join(HERE, "openrouter_key.txt")
API_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_MODEL = "openai/gpt-4.1-nano"
TIMEOUT_SECONDS = 10

ACTIONS = ("store", "correct", "forget", "ask", "chat")


class BrainUnavailable(Exception):
    """Sentinel for any brain failure. The caller falls back to the
    classic behaviour — talk mode must never break a request."""


INTERPRET_SYSTEM = (
    "You route messages for a memory-box agent. Reply with ONLY a JSON "
    "object, no other text, in exactly this shape:\n"
    '{"action": "store"|"correct"|"forget"|"ask"|"chat", '
    '"fact": <string|null>, "query": <string|null>, '
    '"ttl_seconds": <int|null>}\n'
    "- store: the user wants something remembered. fact = the fact as "
    "one clean standalone statement (e.g. \"the client's budget is "
    "$750 for the audit\"). If the user gives the memory a lifetime, "
    "ttl_seconds = that lifetime in seconds, else null.\n"
    "- correct: the user updates an existing fact. fact = the NEW fact "
    "as one clean standalone statement.\n"
    "- forget: the user wants a memory deleted. query = its topic.\n"
    "- ask: the user asks about a stored fact — their facts, their "
    "client, their project. query = its topic. Questions about the "
    "outside world (weather, news, general knowledge) are NOT ask; "
    "they are chat. If known memory topics are listed and the "
    "question is about one of them, query = that topic's words. "
    "Keep the user's own nouns — never swap in synonyms.\n"
    "- chat: anything else — including questions about the demo "
    "itself (including who built it), questions about the agent's "
    "identity or what model it "
    "is, and general questions. fact and query are null."
)

PHRASE_SYSTEM = (
    "You phrase a memory agent's reply. Reply with ONE natural "
    "sentence, nothing else.\n"
    "You are Hermes, a memory agent. Never name or hint at any AI "
    "model, provider, or company behind you.\n"
    "The outcome's detail field states what happened — phrase "
    "it, never contradict it. You are given ONLY the outcome, as "
    "JSON. Never add facts that are not in the outcome. Never "
    "invent stored "
    "content.\n"
    "- outcome \"denied\": politely say the memory exists but its "
    "rule limits it to the Keeper.\n"
    "- outcome \"change_refused\" or \"forget_refused\": politely say "
    "the memory's rule reserves that change to the Keeper.\n"
    "- outcome \"not_found\" or \"forget_not_found\": say you don't "
    "know / have no memory of that.\n"
    "- any other outcome: state it plainly, using only the details "
    "given (a \"fact\" field may be quoted)."
)


DEMO_KNOWLEDGE = (
    "THE DEMO: 'Memory You Can Govern', built by Larry Moi "
    "(Labuilds / Nexa Labs) for the Agents in Context Toronto "
    "MongoDB hackathon, Problem Statement 2: Memory You Can Govern. "
    "The agent (Hermes) has NO memory of its own: every answer is "
    "searched from the memory box at that moment. A 'new run' starts "
    "blank and knows only what the box holds.\n"
    "THE BOX: every memory carries a governing card — its content, "
    "a readers list, an optional expiry, a status (active, "
    "superseded, expired, forgotten), and a topic key linking its "
    "versions. The readers list governs READING, CHANGING, and "
    "FORGETTING alike. Every operation writes a receipt: STORE, "
    "RETRIEVE, CORRECT, EXPIRED, REFUSED, FORGET, SEARCH, NEW RUN.\n"
    "CORRECTION: the old version is superseded on the record and "
    "never answered again; the new version answers from then on.\n"
    "EXPIRY: a memory can carry a lifetime; when it lapses the "
    "memory stops answering and an EXPIRED receipt is written.\n"
    "FORGETTING: hard deletion, not hiding — the content is deleted "
    "across every version, and the receipt trail itself is scrubbed "
    "of the content; only a tombstone remains. It is provable three "
    "ways: ask in chat, search the box directly, or run a vector "
    "search — all return nothing.\n"
    "REFUSALS: a visitor asking for a Keeper-only memory is refused "
    "for reading and for changing, and each refusal writes its own "
    "REFUSED receipt. The on-screen box panel also hides any memory "
    "the current viewer may not read.\n"
    "STORAGE: MongoDB Atlas. Each memory is a document holding the "
    "content, a 128-dimension embedding, its rule, expiry, and "
    "status. Retrieval runs Atlas Vector Search (cosine similarity) "
    "with status as a filter, so superseded, expired, and forgotten "
    "memories never surface. If the search index lags on a "
    "just-written memory, the app ranks the same Atlas data locally "
    "until the index catches up, and the STORE receipt reports how "
    "long indexing took.\n"
    "HANDOFF: because the memory lives in the box rather than "
    "inside any one agent, a fresh agent — or a different engine, "
    "when one runs out of credits — can pick the work up from the "
    "box, seeing only what the rules allow. Engines are rented; "
    "the memory is owned.\n"
    "PROOF OF BEHAVIOUR: ask the same question in three states — "
    "before a memory is stored (I don't know), after it is stored "
    "(it answers), and after it is forgotten (I don't know again). "
    "The memory is the only thing that changed.\n"
    "PLATFORM: the memory engine underneath is Larry's existing "
    "platform, the Memory Layer; this demo is the layer built for "
    "today."
)

DEMO_ANSWER_SYSTEM = (
    "You are Hermes, the memory agent in the 'Memory You Can "
    "Govern' demo. Answer the user's question using ONLY the "
    "knowledge below. One or two short sentences, plain and "
    "confident.\n"
    "If the question is not covered by the knowledge — anything "
    "outside this demo, its memory box, and what you can do, "
    "including what AI model or provider you are — say that is "
    "outside your scope, and name one thing you CAN do with "
    "memory instead.\n"
    "Never mention any AI model, provider, or company name. "
    "Never invent demo facts beyond the knowledge.\n"
    "\nKNOWLEDGE:\n" + DEMO_KNOWLEDGE
)


def _urllib_transport(url, headers, payload, timeout):
    """Default transport: one POST via urllib, parsed JSON back.
    Any transport-level problem becomes BrainUnavailable."""
    req = urllib.request.Request(url,
                                 data=json.dumps(payload).encode("utf-8"),
                                 headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except BrainUnavailable:
        raise
    except Exception as e:  # HTTPError, URLError, timeout, bad JSON...
        raise BrainUnavailable("OpenRouter call failed: %s" % e)


class Brain:
    def __init__(self, key=None, transport=None, model=None,
                 key_file=KEY_FILE):
        if key is None:
            key = self._read_key(key_file)
        self.key = key or None
        self.transport = transport or _urllib_transport
        self.model = model or os.environ.get("TALK_MODEL") or DEFAULT_MODEL

    @staticmethod
    def _read_key(path):
        try:
            with open(path) as f:
                return f.read().strip() or None
        except OSError:
            return None

    def available(self):
        return bool(self.key)

    # ------------------------------------------------------------- plumbing
    def _chat(self, system, user, max_tokens):
        if not self.key:
            raise BrainUnavailable("no API key (openrouter_key.txt "
                                   "missing or empty)")
        payload = {"model": self.model, "temperature": 0,
                   "max_tokens": max_tokens,
                   "messages": [{"role": "system", "content": system},
                                {"role": "user", "content": user}]}
        headers = {"Authorization": "Bearer " + self.key,
                   "Content-Type": "application/json"}
        try:
            resp = self.transport(API_URL, headers, payload,
                                  TIMEOUT_SECONDS)
            content = resp["choices"][0]["message"]["content"]
        except BrainUnavailable:
            raise
        except Exception as e:
            raise BrainUnavailable("bad OpenRouter response: %s" % e)
        if not isinstance(content, str) or not content.strip():
            raise BrainUnavailable("empty model reply")
        return content.strip()

    # ------------------------------------------------------------ step 1
    def interpret(self, text, topics=None):
        """Free-form input -> a validated plan dict:
        {action, fact, query, ttl_seconds}. Raises BrainUnavailable
        on ANY problem (the caller then falls back to classic).
        `topics` = the box's active memory keys, given to the model
        so an ask can be matched to a real topic instead of the
        model inventing synonyms for what the user said."""
        if topics:
            text = ("Known memory topics: %s\nMessage: %s"
                    % (", ".join(topics), text))
        raw = self._chat(INTERPRET_SYSTEM, text, 160)
        s = raw.strip()
        if s.startswith("```"):  # tolerate a fenced reply
            s = s.strip("`")
            if s.lower().startswith("json"):
                s = s[4:]
            s = s.strip()
        try:
            plan = json.loads(s)
        except ValueError:
            raise BrainUnavailable("interpret did not return JSON: "
                                   "%r" % raw[:80])
        if not isinstance(plan, dict):
            raise BrainUnavailable("interpret did not return an object")
        action = plan.get("action")
        if action not in ACTIONS:
            raise BrainUnavailable("unknown action: %r" % (action,))

        def _clean(v):
            return v.strip() if isinstance(v, str) and v.strip() else None

        fact = _clean(plan.get("fact"))
        query = _clean(plan.get("query"))
        ttl = plan.get("ttl_seconds")
        if not (isinstance(ttl, int) and not isinstance(ttl, bool)
                and ttl > 0):
            ttl = None
        if action in ("store", "correct") and not fact:
            raise BrainUnavailable("interpret gave no fact for %s"
                                   % action)
        if action == "ask" and not query:
            query = text
        if action == "forget" and not query:
            query = fact or text
        return {"action": action, "fact": fact, "query": query,
                "ttl_seconds": ttl}

    # ------------------------------------------------------------ step 2
    def phrase(self, result):
        """Operation outcome (a small dict) -> one natural sentence.
        The outcome is ALL the model is given: a denied outcome
        carries no memory content, so there is nothing to leak.
        Raises BrainUnavailable on any failure; the caller then uses
        a deterministic local phrasing of the same outcome."""
        content = self._chat(PHRASE_SYSTEM, json.dumps(result), 80)
        if len(content) >= 2 and content[0] == '"' and content[-1] == '"':
            content = content[1:-1].strip()
        if not content:
            raise BrainUnavailable("empty phrasing")
        return content

    # ------------------------------------------------------------ step 3
    def answer_demo(self, question):
        """Answer a question about the demo from DEMO_KNOWLEDGE
        only. Off-demo questions — including what model or provider
        the agent is — get the scope deflection, per the system
        prompt. Raises BrainUnavailable on any failure; the caller
        falls back to the canned chat reply."""
        content = self._chat(DEMO_ANSWER_SYSTEM, question, 160)
        if len(content) >= 2 and content[0] == '"' and content[-1] == '"':
            content = content[1:-1].strip()
        if not content:
            raise BrainUnavailable("empty demo answer")
        return content
