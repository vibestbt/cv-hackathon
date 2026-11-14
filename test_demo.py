"""
test_demo.py — headless tests for the 6 demo beats.
Run:  python3 test_demo.py        (stdlib unittest, no installs needed)
Each test gets a fresh box in a temp file — the real memories.json
and any Atlas config are never touched.
"""
import json
import os
import sys
import tempfile
import time
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from memory_box import MemoryBox, Agent
from brain import Brain

BUDGET_Q = "What's the client's budget?"
STORE_750 = "Remember: the client's budget is $750 for the audit"
FIX_900 = "Correct: the client's budget is $900 for the audit"


class Beats(unittest.TestCase):
    def setUp(self):
        fd, self.path = tempfile.mkstemp(suffix=".json")
        os.close(fd)
        os.unlink(self.path)
        # atlas_uri="" forces the local backend even if atlas_uri.txt exists.
        self.box = MemoryBox(store_path=self.path, atlas_uri="")
        self.ag = Agent(self.box)

    def tearDown(self):
        if os.path.exists(self.path):
            os.unlink(self.path)

    def ops(self):
        return [r["op"] for r in self.box.receipts]

    def trail_text(self):
        return "\n".join(r["detail"] for r in self.box.receipts)

    # Beat 0/2 — an empty box knows nothing; after a store + a NEW RUN,
    # the same question is answered from the box. Memory changes behaviour.
    def test_next_run_change(self):
        self.assertIn("I don't know", self.ag.handle(BUDGET_Q))
        self.ag.handle(STORE_750)
        self.ag.new_run()
        answer = self.ag.handle(BUDGET_Q)
        self.assertIn("$750", answer)
        self.assertIn("NEW RUN", self.ops())

    # Beat 1 — STORE writes a real receipt carrying the read-rule.
    def test_store_receipt_and_rule(self):
        mem = self.box.store("the client's budget is $750 for the audit")
        self.assertEqual(mem["readers"], ["keeper"])
        self.assertIn("STORE", self.ops())
        self.assertIn("readable by: keeper", self.trail_text())

    # Beat 3 — CORRECT supersedes; new answers use the new value only.
    def test_correct_supersedes(self):
        self.ag.handle(STORE_750)
        self.ag.handle(FIX_900)
        answer = self.ag.handle(BUDGET_Q)
        self.assertIn("$900", answer)
        self.assertNotIn("$750", answer)
        self.assertIn("CORRECT", self.ops())
        statuses = {m["id"]: m["status"] for m in self.box.memories}
        self.assertIn("superseded", statuses.values())
        self.assertIn("active", statuses.values())

    # Beat 4 — EXPIRY: alive before its lifetime ends, silent after,
    # with an EXPIRED receipt. (Uses a 1-second lifetime to stay fast;
    # the stage demo uses the 20-second button.)
    def test_expiry_honoured(self):
        self.box.store("the door code is 4417",
                       readers=["keeper", "visitor"], ttl_seconds=1)
        alive = self.box.search("What's the door code?")
        self.assertTrue(alive["found"])
        time.sleep(1.2)
        gone = self.box.search("What's the door code?")
        self.assertFalse(gone["found"])
        self.assertIn("EXPIRED", self.ops())

    # Beat 4b — regression guard: an unrelated memory must NOT be served
    # as the answer to the door-code question once the code has expired.
    def test_no_wrong_memory_served(self):
        self.ag.handle(STORE_750)
        self.box.store("the door code is 4417",
                       readers=["keeper", "visitor"], ttl_seconds=1)
        time.sleep(1.2)
        answer = self.ag.handle("What's the door code?")
        self.assertIn("I don't know", answer)
        self.assertNotIn("4417", answer)
        self.assertNotIn("$750", answer)

    # Beat 5 — READ-RULE: the visitor is refused, the receipt shows it,
    # and the keeper is still served.
    def test_read_rule_refusal(self):
        self.ag.handle(STORE_750)
        refused = self.ag.handle(BUDGET_Q, actor="visitor")
        self.assertIn("only the Keeper can read it", refused)
        self.assertNotIn("$750", refused)
        self.assertIn("REFUSED", self.ops())
        allowed = self.ag.handle(BUDGET_Q, actor="keeper")
        self.assertIn("$750", allowed)

    # Beat 5b — the card governs CHANGING too: a visitor cannot correct
    # a keeper-only memory. The stored value is unchanged afterwards
    # and a REFUSED receipt lands in the trail.
    def test_visitor_cannot_correct_keeper_memory(self):
        self.ag.handle(STORE_750)
        reply = self.ag.handle(FIX_900, actor="visitor")
        self.assertIn("only the Keeper can change it", reply)
        self.assertNotIn("$900", reply)
        self.assertIn("REFUSED", self.ops())
        self.assertIn("tried to change", self.trail_text())
        answer = self.ag.handle(BUDGET_Q, actor="keeper")
        self.assertIn("$750", answer)
        self.assertNotIn("$900", answer)
        statuses = [m["status"] for m in self.box.memories]
        self.assertNotIn("superseded", statuses)

    # Beat 5c — the Keeper correcting still works (the lock is a rule
    # on the card, not a wall around the box).
    def test_keeper_correct_still_works(self):
        self.ag.handle(STORE_750)
        self.ag.handle(FIX_900, actor="keeper")
        answer = self.ag.handle(BUDGET_Q, actor="keeper")
        self.assertIn("$900", answer)
        self.assertIn("CORRECT", self.ops())

    # Beat 5d — a visitor CAN correct a memory whose rule names the
    # visitor (the door code: readable by keeper + visitor).
    def test_visitor_can_correct_shared_memory(self):
        self.box.store("the door code is 4417",
                       readers=["keeper", "visitor"])
        self.ag.handle("Correct: the door code is 9999", actor="visitor")
        res = self.box.search("What's the door code?", actor="visitor")
        self.assertTrue(res["found"])
        self.assertIn("9999", res["memory"]["text"])
        self.assertIn("CORRECT", self.ops())

    # Beat 6 — FORGET + PROVE on every recall path:
    # chat answer, direct search, box search, and the receipt trail.
    def test_forget_and_prove(self):
        self.ag.handle(STORE_750)
        self.ag.handle(FIX_900)
        self.ag.handle("Forget the client's budget")
        # chat path
        self.assertIn("I don't know", self.ag.handle(BUDGET_Q))
        # direct-search path
        direct = self.ag.handle("Search: client's budget")
        self.assertIn("nothing found", direct)
        # box search path (this is the vector-search path locally)
        self.assertFalse(self.box.search("client budget")["found"])
        # tombstone: content really gone, not hidden
        for m in self.box.memories:
            if m["key"] == "client-budget":
                self.assertEqual(m["status"], "forgotten")
                self.assertIsNone(m["text"])
        # trail: FORGET receipt exists and NO receipt still quotes a value
        self.assertIn("FORGET", self.ops())
        self.assertNotIn("$750", self.trail_text())
        self.assertNotIn("$900", self.trail_text())

    # Beat 6b — FORGET is gated by the same card: a visitor cannot
    # forget a keeper-only memory; the content survives untouched.
    def test_visitor_cannot_forget_keeper_memory(self):
        self.ag.handle(STORE_750)
        reply = self.ag.handle("Forget the client's budget",
                               actor="visitor")
        self.assertIn("only the Keeper can forget it", reply)
        self.assertIn("REFUSED", self.ops())
        self.assertNotIn("FORGET", self.ops())
        answer = self.ag.handle(BUDGET_Q, actor="keeper")
        self.assertIn("$750", answer)

    # Regression (Nov 13 rehearsal, Larry's photos): asking with a bare
    # value — "900" or "$900" — found NOTHING even though the active
    # memory was "the client's budget is $900 for the audit". Cause:
    # "$900" and "900" hashed to different tokens (cosine 0.0), and a
    # bare "$900" scored 0.32, under the 0.40 threshold. A bare value
    # must find the memory that carries that exact value.
    def test_bare_value_search_finds_money_memory(self):
        self.ag.handle(STORE_750)
        self.ag.handle(FIX_900)
        for q in ("900", "$900"):
            res = self.box.search(q)
            self.assertTrue(res["found"], q)
            self.assertIn("$900", res["memory"]["text"])
            self.assertIn("$900", self.ag.handle(q))
        # the superseded value must NOT come back on a bare ask
        self.assertFalse(self.box.search("750")["found"])
        self.assertIn("I don't know", self.ag.handle("750"))

    # The bare-value path obeys the same rules as every other path:
    # keeper-only stays keeper-only, expiry and forget still win.
    def test_bare_value_respects_rules_expiry_forget(self):
        self.ag.handle(STORE_750)
        denied = self.box.search("750", actor="visitor")
        self.assertTrue(denied["denied"])
        self.assertIsNone(denied["memory"])
        self.box.store("the door code is 4417",
                       readers=["keeper", "visitor"], ttl_seconds=1)
        self.assertTrue(self.box.search("4417")["found"])
        time.sleep(1.2)
        self.assertFalse(self.box.search("4417")["found"])
        self.ag.handle("Forget the client's budget")
        self.assertFalse(self.box.search("750")["found"])

    # Beat 5e (audit, Nov 14) — the card governs STORING OVER too: a
    # visitor cannot plant a rival version of a keeper-only topic with
    # a fresh "Remember". The keeper's value survives untouched and a
    # REFUSED receipt lands in the trail.
    def test_visitor_cannot_store_over_keeper_memory(self):
        self.ag.handle(STORE_750)
        reply = self.ag.handle(
            "Remember: the client's budget is $1 for the audit",
            actor="visitor")
        self.assertIn("only the Keeper can change it", reply)
        self.assertIn("REFUSED", self.ops())
        self.assertIn("tried to store over", self.trail_text())
        answer = self.ag.handle(BUDGET_Q, actor="keeper")
        self.assertIn("$750", answer)
        actives = [m for m in self.box.memories
                   if m["key"] == "client-budget" and m["status"] == "active"]
        self.assertEqual(len(actives), 1)

    # The store-lock is scoped to topics that already exist: a visitor
    # CAN still store a brand-new topic (it lands keeper-only), and the
    # keeper CAN re-store his own topic — the lock is the card's rule,
    # not a wall around the box.
    def test_store_lock_scoped(self):
        reply = self.ag.handle("Remember: the wifi password is guest123",
                               actor="visitor")
        self.assertIn("Stored", reply)
        self.ag.handle(STORE_750)
        again = self.box.store("the client's budget is $750 for the audit",
                               actor="keeper")
        self.assertIsNotNone(again)

    # Lag bookkeeping on the local backend: _rank records the local
    # path in last_search, and no Atlas note is written to receipts
    # (the poll only runs when Atlas is the backend).
    def test_last_search_local(self):
        self.ag.handle(STORE_750)
        self.ag.handle(BUDGET_Q)
        self.assertEqual(self.box.last_search["path"], "local")
        self.assertNotIn("Searchable in Atlas", self.trail_text())

    # The screen obeys the rule too (audit, Nov 14): server state for a
    # visitor hides keeper-only content in the pane AND redacts it from
    # receipt details; the keeper's own view is untouched.
    def test_state_redacts_for_visitor(self):
        import server
        old_box, old_agent = server.BOX, server.AGENT
        try:
            server.BOX = self.box
            server.AGENT = self.ag
            self.ag.handle(STORE_750)
            self.ag.handle(FIX_900)
            vis = server.state(actor="visitor")
            blob = str(vis)
            self.assertNotIn("$750", blob)
            self.assertNotIn("$900", blob)
            self.assertTrue(any(m["hidden"] for m in vis["memories"]))
            keep = server.state(actor="keeper")
            self.assertIn("$900", str(keep))
            self.assertFalse(any(m["hidden"] for m in keep["memories"]))
        finally:
            server.BOX, server.AGENT = old_box, old_agent

    # Persistence — the box survives a restart (that's what makes
    # "next run" real on stage, not a trick of one process).
    def test_persistence_across_reopen(self):
        self.ag.handle(STORE_750)
        box2 = MemoryBox(store_path=self.path, atlas_uri="")
        ag2 = Agent(box2)
        self.assertIn("$750", ag2.handle(BUDGET_Q))


# ---------------------------------------------------------------- talk mode
class FakeTransport:
    """Injectable brain transport: returns scripted reply contents,
    one per call, and records every payload. No network, no key file.
    With always_raise=True every call fails (a dead brain)."""

    def __init__(self, replies=(), always_raise=False):
        self.replies = list(replies)
        self.always_raise = always_raise
        self.calls = []

    def __call__(self, url, headers, payload, timeout):
        self.calls.append(payload)
        if self.always_raise or not self.replies:
            raise OSError("fake transport: no scripted reply")
        content = self.replies.pop(0)
        return {"choices": [{"message": {"content": content}}]}


def plan_json(action, fact=None, query=None, ttl_seconds=None):
    return json.dumps({"action": action, "fact": fact, "query": query,
                       "ttl_seconds": ttl_seconds})


class TalkMode(unittest.TestCase):
    def setUp(self):
        fd, self.path = tempfile.mkstemp(suffix=".json")
        os.close(fd)
        os.unlink(self.path)
        self.box = MemoryBox(store_path=self.path, atlas_uri="")
        self.transport = FakeTransport()
        self.brain = Brain(key="test-key", transport=self.transport,
                           model="test/model")

    def tearDown(self):
        if os.path.exists(self.path):
            os.unlink(self.path)

    def ops(self):
        return [r["op"] for r in self.box.receipts]

    def talk_agent(self):
        return Agent(self.box, talk=True, brain=self.brain)

    # (a) Free-form store: no command word, the brain extracts the
    # fact; the box stores it keeper-only, with a STORE receipt.
    def test_freeform_store(self):
        self.transport.replies = [
            plan_json("store",
                      fact="the client's budget is $750 for the audit"),
            "Got it — I'll remember that the client's budget is $750 "
            "for the audit."]
        ag = self.talk_agent()
        reply = ag.handle(
            "Hey, keep in mind the client budget is $750 for the audit")
        self.assertIn("$750", reply)
        actives = [m for m in self.box.memories if m["status"] == "active"]
        self.assertEqual(len(actives), 1)
        self.assertEqual(actives[0]["text"],
                         "the client's budget is $750 for the audit")
        self.assertEqual(actives[0]["readers"], ["keeper"])
        self.assertIn("STORE", self.ops())
        # ...and the stored fact answers through the classic path too.
        self.assertIn("$750", ag.handle(BUDGET_Q))

    # (b) Free-form ask: the answer is the stored fact, phrased.
    def test_freeform_ask_returns_fact_phrased(self):
        self.talk_agent().handle(STORE_750)  # classic store, no brain
        self.transport.replies = [
            plan_json("ask", query="client's budget"),
            "The client's budget is $750 for the audit."]
        reply = self.talk_agent().handle(
            "So, what did we land on for the client's budget?")
        self.assertEqual(reply,
                         "The client's budget is $750 for the audit.")
        self.assertIn("RETRIEVE", self.ops())

    # (c) Visitor free-form ask on a keeper-only memory: polite
    # denial, the fact is NOT in the reply — and since Nov 14 the
    # denial is phrased deterministically, so the model is never
    # even called with the outcome: only the interpret call runs,
    # and it never contained the fact either.
    def test_visitor_ask_denied_without_leak(self):
        self.talk_agent().handle(STORE_750)
        self.transport.replies = [
            plan_json("ask", query="client's budget"),
            "I'm sorry — that memory exists, but its rule limits it "
            "to the Keeper."]
        reply = self.talk_agent().handle("What's the client's budget?",
                                         actor="visitor")
        self.assertNotIn("$750", reply)
        self.assertNotIn("750", reply)
        self.assertIn("Keeper", reply)
        self.assertIn("REFUSED", self.ops())
        self.assertEqual(len(self.transport.calls), 1)  # interpret
        # ...whose user message holds only the topic key + the
        # visitor's own question — never the stored fact.
        user_msg = self.transport.calls[0]["messages"][1]["content"]
        self.assertNotIn("750", user_msg)

    # (d) Unknown topic: an honest don't-know.
    def test_unknown_topic_dont_know(self):
        self.transport.replies = [
            plan_json("ask", query="the wifi password"),
            "I don't know — I have no memory of that."]
        reply = self.talk_agent().handle("What's the wifi password?")
        self.assertIn("don't know", reply.lower())
        self.assertIn("SEARCH", self.ops())

    # (e) A dead brain must never break a request: interpret raises,
    # the classic path answers exactly as the golden demo would.
    def test_brain_failure_falls_back_to_classic(self):
        dead = Brain(key="test-key",
                     transport=FakeTransport(always_raise=True))
        ag = Agent(self.box, talk=True, brain=dead)
        ag.handle(STORE_750)
        self.assertIn("$750", ag.handle(BUDGET_Q))
        self.assertIn("I don't know",
                      ag.handle("What's the wifi password here?"))

    # (f) Talk mode OFF (the default): free-form text is NOT routed
    # to any brain — it takes the classic question path and stores
    # nothing. (The 18 classic Beats tests above pin the rest.)
    def test_talk_off_by_default(self):
        with mock.patch.dict(os.environ):
            os.environ.pop("TALK_MODE", None)
            ag = Agent(self.box)
        self.assertFalse(ag.talk)
        reply = ag.handle(
            "Hey, keep in mind the client budget is $750 for the audit")
        self.assertIn("I don't know", reply)
        self.assertEqual(self.box.memories, [])
        self.assertNotIn("STORE", self.ops())

    # (g) Free-form correct from a visitor on a keeper-only memory:
    # the box refuses, the keeper's value survives untouched.
    def test_visitor_freeform_correct_refused(self):
        self.talk_agent().handle(STORE_750)
        self.transport.replies = [
            plan_json("correct",
                      fact="the client's budget is $1 for the audit"),
            "I can't change that one — its rule reserves changes to "
            "the Keeper."]
        reply = self.talk_agent().handle(
            "Actually make the client budget $1 for the audit",
            actor="visitor")
        self.assertIn("Keeper", reply)
        self.assertIn("REFUSED", self.ops())
        statuses = [m["status"] for m in self.box.memories]
        self.assertNotIn("superseded", statuses)
        answer = self.talk_agent().handle(BUDGET_Q, actor="keeper")
        self.assertIn("$750", answer)

    # (h) Explicit commands bypass the brain entirely in talk mode:
    # the transport is never called for them.
    def test_classic_commands_bypass_brain(self):
        ag = self.talk_agent()
        ag.handle(STORE_750)
        ag.handle(FIX_900)
        self.assertEqual(self.transport.calls, [])
        self.assertIn("$900", ag.handle(BUDGET_Q))

    # Free-form store WITH a lifetime mirrors the classic door code:
    # readable by keeper + visitor, with an expiry set.
    def test_freeform_store_with_lifetime(self):
        self.transport.replies = [
            plan_json("store", fact="the door code is 4417",
                      ttl_seconds=20),
            "Stored for 20 seconds: the door code is 4417."]
        reply = self.talk_agent().handle(
            "The door code is 4417 — keep it for just 20 seconds")
        self.assertIn("4417", reply)
        mem = self.box.memories[0]
        self.assertEqual(mem["readers"], ["keeper", "visitor"])
        self.assertIsNotNone(mem["expires_at"])

    # Free-form forget: the box forgets, and the fact stops answering.
    def test_freeform_forget(self):
        self.talk_agent().handle(STORE_750)
        self.transport.replies = [
            plan_json("forget", query="client's budget"),
            "Forgotten — it's gone."]
        reply = self.talk_agent().handle(
            "Please forget the client's budget")
        self.assertIn("Forgotten", reply)
        self.assertIn("FORGET", self.ops())
        self.assertIn("I don't know", self.talk_agent().handle(BUDGET_Q))

    # (h) Identity rule: model-identity questions are deflected
    # deterministically — the brain is never even consulted, and
    # the reply names no model or provider.
    def test_identity_questions_deflected_without_brain(self):
        ag = self.talk_agent()
        for q in ("What model are you?",
                  "Are you Claude or GPT?",
                  "Who made you?"):
            reply = ag.handle(q)
            self.assertIn("outside my scope", reply)
            for banned in ("claude", "gpt", "openai", "anthropic",
                           "openrouter"):
                self.assertNotIn(banned, reply.lower())
        self.assertEqual(len(self.transport.calls), 0)

    # (i) Demo questions are answered from the demo knowledge: the
    # answer call's system prompt carries the knowledge and the
    # scope rule.
    def test_demo_question_uses_knowledge(self):
        self.transport.replies = [
            plan_json("chat"),
            "Forgetting is hard deletion: the content is deleted "
            "across every version and the receipt trail is scrubbed; "
            "only a tombstone remains."]
        reply = self.talk_agent().handle(
            "How does forgetting work in this demo?")
        self.assertIn("tombstone", reply)
        systems = [c["messages"][0]["content"]
                   for c in self.transport.calls]
        self.assertTrue(any("Memory You Can Govern" in s
                            and "MongoDB Atlas" in s for s in systems))
        self.assertTrue(any("outside your scope" in s for s in systems))

    # (k) Ask robustness: the interpreter is told the box's active
    # topics, and if its extracted query still misses (paraphrase
    # drift), the agent retries once with the user's own words
    # before answering don't-know.
    def test_ask_retries_with_raw_words_and_gets_topics(self):
        self.talk_agent().handle(STORE_750)  # classic store, no brain
        self.transport.replies = [
            plan_json("ask", query="client's assets or holdings"),
            "The client's budget is $750 for the audit."]
        reply = self.talk_agent().handle(
            "so what did we say about the client's budget, I forgot")
        self.assertIn("$750", reply)
        first_user = self.transport.calls[0]["messages"][1]["content"]
        self.assertIn("Known memory topics:", first_user)
        self.assertIn("client-budget", first_user)

    # (l) A successful forget is phrased DETERMINISTICALLY — the
    # model is never given the chance to reword a deletion into
    # "I don't have any memory of that" (the live Nov 14 bug).
    def test_forget_reply_is_deterministic(self):
        self.talk_agent().handle(STORE_750)  # classic store, no brain
        self.transport.replies = [
            plan_json("forget", query="client's budget"),
            "I do not have any memory of that."]  # must NOT be used
        reply = self.talk_agent().handle(
            "please forget the client's budget")
        self.assertTrue(reply.startswith("Forgotten."))
        self.assertEqual(len(self.transport.calls), 1)  # interpret
        self.assertIn("FORGET", self.ops())

    # (m) Topic probe: when both the extracted query and the raw
    # words miss, a word from an active topic key in the user's
    # message triggers a direct search of that topic.
    def test_ask_topic_probe_fallback(self):
        self.talk_agent().handle(STORE_750)  # classic store, no brain
        self.transport.replies = [
            plan_json("ask", query="zzq unrelated words"),
            "The client's budget is $750 for the audit."]
        reply = self.talk_agent().handle(
            "um so earlier at the office everyone was talking and "
            "somebody mentioned the budget thing from before")
        self.assertIn("$750", reply)

    # (n) Memory before small talk: even if the interpreter routes
    # a memory question to "chat", a stored fact that matches the
    # words is answered from the box — never improvised over.
    def test_chat_route_still_answers_from_box(self):
        self.talk_agent().handle(STORE_750)  # classic store, no brain
        self.transport.replies = [
            plan_json("chat"),
            "The client's budget is $750 for the audit."]
        reply = self.talk_agent().handle(
            "so what was the budget again, i forgot")
        self.assertIn("$750", reply)
        self.assertIn("RETRIEVE", self.ops())

    # (j) The phrase prompt itself carries the identity rule, so an
    # operation reply can never name a provider either.
    def test_phrase_prompt_bans_provider_names(self):
        self.transport.replies = [
            plan_json("store", fact="the client's budget is $750 "
                                    "for the audit"),
            "Stored it."]
        self.talk_agent().handle(
            "Please keep the client budget of $750 in mind")
        systems = [c["messages"][0]["content"]
                   for c in self.transport.calls]
        self.assertTrue(any("Never name or hint at any AI model"
                            in s for s in systems))


if __name__ == "__main__":
    unittest.main(verbosity=2)
