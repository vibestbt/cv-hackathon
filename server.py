"""
server.py — tiny local web server for the hackathon demo (stdlib only).

Run:  python3 server.py          -> open http://localhost:8765
      PORT=9000 python3 server.py  (if 8765 is taken)

ATLAS CONFIG — the ONE config point: to run against MongoDB Atlas,
put the Atlas connection string (nothing else) in a file named
atlas_uri.txt in this folder and `pip install pymongo`, then restart.
No string -> local file store. The top of the page always shows which
backend is live, honestly.
"""
import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from memory_box import MemoryBox, Agent

HERE = os.path.dirname(os.path.abspath(__file__))
BOX = MemoryBox()
AGENT = Agent(BOX)


def state():
    mems = []
    for m in BOX.memories:
        mems.append({
            "id": m["id"], "key": m["key"], "status": m["status"],
            "text": m["text"] if m["status"] != "forgotten" else None,
            "readers": m["readers"],
            "expires_at": m.get("expires_at")})
    return {"backend": BOX.backend, "run": AGENT.run,
            "memories": mems, "receipts": BOX.receipts}


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _json(self, obj, code=200):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _body(self):
        n = int(self.headers.get("Content-Length", 0))
        return json.loads(self.rfile.read(n) or b"{}")

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            with open(os.path.join(HERE, "index.html"), "rb") as f:
                body = f.read()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif self.path == "/api/state":
            self._json(state())
        else:
            self._json({"error": "not found"}, 404)

    def do_POST(self):
        global BOX, AGENT
        data = self._body()
        actor = data.get("actor", "keeper")
        if self.path == "/api/message":
            before = len(BOX.receipts)
            reply = AGENT.handle(data.get("text", ""), actor=actor)
            self._json({"reply": reply,
                        "new_receipts": BOX.receipts[before:],
                        "state": state()})
        elif self.path == "/api/newrun":
            before = len(BOX.receipts)
            reply = AGENT.new_run(actor=actor)
            self._json({"reply": reply,
                        "new_receipts": BOX.receipts[before:],
                        "state": state()})
        elif self.path == "/api/search":
            before = len(BOX.receipts)
            res = BOX.search(data.get("query", ""), actor=actor)
            out = {"found": res["found"], "denied": res["denied"],
                   "text": res["memory"]["text"] if res["memory"] else None,
                   "new_receipts": BOX.receipts[before:],
                   "state": state()}
            self._json(out)
        elif self.path == "/api/reset":
            BOX.wipe()
            AGENT.run = 1
            self._json({"state": state()})
        elif self.path == "/api/connect-atlas":
            # The operator pastes the Atlas connection string into the
            # page — no files edited by hand. Only saved if it connects.
            uri = (data.get("uri") or "").strip()
            if not uri.startswith("mongodb"):
                self._json({"ok": False,
                            "message": "That doesn't look like a MongoDB "
                                       "connection string — copy the whole "
                                       "thing from Atlas and paste it again."})
                return
            from memory_box import MemoryBox as _MB, Agent as _AG
            candidate = _MB(atlas_uri=uri)
            if candidate.backend.startswith("atlas"):
                with open(os.path.join(HERE, "atlas_uri.txt"), "w") as f:
                    f.write(uri)
                BOX, AGENT = candidate, _AG(candidate)
                self._json({"ok": True,
                            "message": "Connected. The memory box now "
                                       "lives in MongoDB Atlas.",
                            "state": state()})
            else:
                self._json({"ok": False,
                            "message": "Couldn't connect with that string. "
                                       "Check the password inside it and "
                                       "that Atlas network access allows "
                                       "this connection — then paste again. "
                                       "Nothing was saved."})
        elif self.path == "/api/use-local":
            # Fallback switch, session-only: the saved Atlas string is
            # left untouched, so tomorrow's start goes back to Atlas.
            from memory_box import MemoryBox as _MB, Agent as _AG
            BOX = _MB(atlas_uri="")
            AGENT = _AG(BOX)
            self._json({"ok": True,
                        "message": "Switched: the memory box is on this "
                                   "laptop for now. (Your Atlas setup is "
                                   "saved and comes back next start.)",
                        "state": state()})
        else:
            self._json({"error": "not found"}, 404)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8765"))
    print("Memory You Can Govern — demo server")
    print("Backend:", BOX.backend)
    print("Open: http://localhost:%d" % port)
    ThreadingHTTPServer(("127.0.0.1", port), Handler).serve_forever()
