"""
Travel Planner Agent - full multi-event pipeline.

===========================================================================
ARCHITECTURE OVERVIEW
===========================================================================
This pipeline is split into a PARENT graph and a BRANCH SUBGRAPH, because a
single itinerary document can contain several distinct activities (e.g. a
flight, a hotel stay, a museum ticket), and each one needs to be checked
against the calendar and resolved independently.

PARENT graph (AgentState) - runs once per document:
    extract -> [per-event dispatch] -> one Send() per extracted event,
    routed to either "needs_clarification" (low-confidence event) or
    "branch" (the subgraph below). All outcomes merge into a single
    `results` list.

BRANCH subgraph (BranchState) - runs once per event, in parallel:
    check_calendar -> [no conflict]  -> create_event  -> finalize_created
                    -> [conflict]    -> generate_candidates
                                     -> evaluate_candidates
                                     -> draft_alert -> finalize_drafted

WHY TWO SEPARATE STATE SCHEMAS (this was a real bug found during design):
LangGraph does not know that a field like `conflicting_events` is meant to
be "private" to one branch. If N branches run concurrently and all write
to the same state key, LangGraph raises InvalidUpdateError, because it only
sees N concurrent writes to one shared channel. The fix is to give each
branch its own state schema (BranchState) so those fields never exist in
the parent's channels at all. The ONLY field that crosses back into the
parent is `results`, which is why only `results` carries a reducer
(Annotated[list, operator.add]) -- it is the single true collection point.

LLM CALLS run against a LOCAL Ollama model (gemma3:4b) -- no API key, no
cost, nothing leaves this machine. If Ollama is not running or the model is
not pulled, a deterministic stub is used instead (clearly logged), so the
pipeline still runs end-to-end for testing.

FUTURE UI HOOK:
`process_itinerary_file` and `process_itinerary_folder` at the bottom of
this file are the intended entry points for a future UI (upload a document
to a local folder, press a button to run the agent). They are plain
functions with no notebook/CLI-specific code, so a UI layer can call them
directly.
"""
from dotenv import load_dotenv
load_dotenv()

import asyncio
import json
import operator
import os
import re
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import TypedDict, Optional, Annotated
from zoneinfo import ZoneInfo

from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.graph import StateGraph, END
from langgraph.types import Send

MIN_BUFFER_MINUTES = 30
USER_EMAIL = os.environ.get("USER_EMAIL", "you@example.com")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "gemma3:4b")
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_TIMEOUT_S = 90  # local inference on CPU can be slower than a cloud API

# Real Google Calendar data returns a mix of formats: date-only all-day
# events ("2026-09-10"), offset-aware timed events
# ("2026-09-10T18:00:00-04:00"), and our own naive local extraction times
# ("2026-09-10T10:30"). DEFAULT_TIMEZONE is the assumed zone for anything
# naive -- adjust via the DEFAULT_TIMEZONE env var if this isn't your home
# timezone.
DEFAULT_TIMEZONE = ZoneInfo(os.environ.get("DEFAULT_TIMEZONE", "America/New_York"))

# Real Google Workspace MCP server -- the default, since this is now the
# actual intended integration, verified end-to-end against a real account.
REAL_WORKSPACE_CFG = {
    "command": "uvx",
    "args": ["workspace-mcp", "--tools", "gmail", "calendar"],
    "transport": "stdio",
    "env": {
        "GOOGLE_OAUTH_CLIENT_ID": os.environ.get("GOOGLE_OAUTH_CLIENT_ID", ""),
        "GOOGLE_OAUTH_CLIENT_SECRET": os.environ.get("GOOGLE_OAUTH_CLIENT_SECRET", ""),
    },
}

# Mock servers -- kept as an explicit opt-in (--mock flag, see __main__)
# for offline structure testing without real credentials.
#MOCK_CALENDAR_CFG = {"command": sys.executable, "args": ["mock_calendar_server.py"], "transport": "stdio"}
#MOCK_GMAIL_CFG = {"command": sys.executable, "args": ["mock_gmail_server.py"], "transport": "stdio"}

import ollama
from pydantic import BaseModel, Field


_ollama_client = ollama.Client(host=OLLAMA_HOST, timeout=OLLAMA_TIMEOUT_S)


def _ollama_available() -> bool:
    """Checked once at import time. If Ollama isn't running or gemma3:4b
    isn't pulled, every LLM-backed node below falls back to a deterministic
    stub instead of failing outright."""
    try:
        _ollama_client.list()
        return True
    except Exception:
        return False


_HAS_OLLAMA = _ollama_available()

if _HAS_OLLAMA:
    class ExtractedEvent(BaseModel):
        title: str = Field(description="Short human-readable name for this travel event")
        start: Optional[str] = Field(default=None, description="Start datetime, ISO 8601, e.g. 2026-09-10T10:30")
        end: Optional[str] = Field(default=None, description="End datetime, ISO 8601, e.g. 2026-09-10T12:00")
        location: Optional[str] = Field(default=None, description="City, airport code, or venue name")
        confirmation_number: Optional[str] = Field(default=None, description="Booking code, if present in the source text")

    class ExtractedEventList(BaseModel):
        events: list[ExtractedEvent] = Field(
            description="Every distinct travel activity found in the document "
                        "(flights, hotel stays, tours, tickets each count separately)"
        )

    class Candidate(BaseModel):
        strategy: str = Field(description="One of: reschedule_existing, alternate_time, shrink_buffer, manual_review")
        description: str
        target: Optional[str] = None
        proposed_start: Optional[str] = None
        proposed_end: Optional[str] = None

    class CandidateList(BaseModel):
        candidates: list[Candidate]
else:
    print(f"[WARN] Ollama not reachable at {OLLAMA_HOST} (or model '{OLLAMA_MODEL}' not "
          f"pulled) - extraction and candidate generation will use deterministic stubs "
          f"instead of real LLM calls. Run `ollama serve` and `ollama pull {OLLAMA_MODEL}` "
          f"to use the real model. See README.")


# ===========================================================================
# PARENT graph state -- deliberately tiny. Only fields that must survive
# across the whole document's processing live here.
# ===========================================================================
class AgentState(TypedDict):
    raw_itinerary_text: str
    candidate_events: list[dict]              # extract's output: one dict per activity
    results: Annotated[list, operator.add]    # the ONE field branches write back to


# ===========================================================================
# BRANCH subgraph state -- private to one event's own journey through the
# pipeline. None of these field names exist in AgentState, so there is no
# shared channel for concurrent branches to collide on.
# ===========================================================================
class BranchState(TypedDict):
    candidate_event: dict
    observation: list             # full calendar read -- needed by evaluate_candidates
    conflicting_events: list
    conflict: bool
    candidates: list
    selected_resolutions: list
    created_event: dict
    drafted_email: dict
    results: list                 # plain list -- no reducer needed inside one branch's own run


def parse(t: str) -> datetime:
    """Parses any of the three real-world shapes we've seen: date-only
    all-day events ("2026-09-10"), offset-aware timed events
    ("2026-09-10T18:00:00-04:00"), and naive local times from our own
    extraction ("2026-09-10T10:30"). Naive values are assumed to be in
    DEFAULT_TIMEZONE, since our itinerary extraction doesn't yet capture
    timezone information."""
    dt = datetime.fromisoformat(t)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=DEFAULT_TIMEZONE)
    return dt


def overlaps(a_start: str, a_end: str, b_start: str, b_end: str) -> bool:
    return parse(a_start) < parse(b_end) and parse(b_start) < parse(a_end)


def fmt(dt: datetime) -> str:
    return dt.isoformat(timespec="minutes")


# ===========================================================================
# PARENT node: deduplicate extracted events
# ===========================================================================

def _deduplicate_events(events: list[dict]) -> list[dict]:
    """Small local models occasionally return the same event twice in one
    extraction call (sampling variance). Drop exact duplicates (same
    title, start, end) before they can be dispatched as separate branches
    and written to the calendar twice."""
    seen = set()
    deduped = []
    for event in events:
        key = (event.get("title"), event.get("start"), event.get("end"))
        if key not in seen:
            seen.add(key)
            deduped.append(event)
    return deduped

# ===========================================================================
# PARENT node: extract
# ===========================================================================

def extract_node(state: AgentState) -> AgentState:
    """Parses the raw document into a LIST of structured events -- one call,
    one node, regardless of how many activities the document describes."""
    raw_text = state["raw_itinerary_text"]

    if not _HAS_OLLAMA:
        raise RuntimeError(
            f"Ollama is not reachable at {OLLAMA_HOST} (or model '{OLLAMA_MODEL}' "
            f"is not pulled). Extraction requires a real LLM call and cannot "
            f"proceed. Run `ollama serve` and `ollama pull {OLLAMA_MODEL}`, then "
            f"try again."
        )

    prompt = f"""Extract every distinct travel activity from this itinerary
text (flights, hotel stays, museum tickets, tours each count as separate
activities).

Policy:
- Only extract information explicitly present in the text. Never guess or
  infer a plausible-looking date, time, or confirmation number.
- If a field is not clearly stated for an activity, omit it entirely
  rather than estimating.
- start and end MUST be in exactly this format: YYYY-MM-DDTHH:MM:SS
  (e.g., "2026-09-10T14:30:00"). Do NOT return a natural-language date
  like "Wednesday, September 10, 2026" or "10:30 AM" -- convert it to
  the numeric format above before returning it.  

Return a JSON list of events, each with title, start, end, location, and
confirmation_number.

Itinerary text:
---
{raw_text}
---"""
    response = _ollama_client.chat(
        model=OLLAMA_MODEL,
        messages=[{"role": "user", "content": prompt}],
        format=ExtractedEventList.model_json_schema(),
    )
    events = ExtractedEventList.model_validate_json(response.message.content).model_dump()["events"]

    #Print here.
    print(f"\n[DEBUG] Extracted {len(events)} event(s): {events}")  # <-- add this line temporarily

    events = _deduplicate_events(events)
    
    return {**state, "candidate_events": events}


def _is_confident(event: dict) -> bool:
    """Extraction-confidence checkpoint, applied PER EVENT."""
    if not (event.get("title") and event.get("start") and event.get("end")):
        return False
    try:
        # Confirms start/end are actually parseable ISO datetimes, not
        # just non-empty strings -- a small local model can return
        # natural-language text in these fields despite the schema.
        if parse(event["start"]) >= parse(event["end"]):
            return False
    except (ValueError, TypeError):
        return False
    return True


def dispatch_events(state: AgentState) -> list[Send]:
    """Conditional-entry-point routing function. For each extracted event,
    independently decides whether it is confident enough to proceed to the
    branch subgraph, or should be routed to escalation instead. Returns one
    Send per event -- this IS the fan-out."""
    sends = []
    for event in state["candidate_events"]:
        if _is_confident(event):
            sends.append(Send("branch", {
                "candidate_event": event, "observation": [], "conflicting_events": [],
                "conflict": False, "candidates": [], "selected_resolutions": [],
                "created_event": {}, "drafted_email": {}, "results": [],
            }))
        else:
            sends.append(Send("needs_clarification", {"candidate_event": event}))
    return sends


def needs_clarification_node(state: dict) -> AgentState:
    """Lives in the PARENT graph, not the subgraph -- the confidence check
    happens before a branch is ever dispatched. Writes directly to the
    parent's reducer-backed `results` field."""
    print(f"\n[Action: escalate to human - ambiguous event] {state['candidate_event']}")
    return {"results": [{"status": "needs_clarification", "event": state["candidate_event"]}]}


# ===========================================================================
# BRANCH subgraph nodes -- unchanged logic from the single-event design,
# now operating on BranchState instead of the old flat AgentState.
# ===========================================================================
async def check_calendar_node(state: BranchState, tools: dict) -> BranchState:
    """Real-tool-based conflict check.

    Two Google Workspace tools are used for two different purposes:
    - query_freebusy is the AUTHORITATIVE busy/free signal. It already
      excludes events marked "free" (e.g. all-day reminders like "Coursera
      learning time"), so checking the candidate event against its busy
      periods directly is what actually determines `conflict` -- this
      avoids the bug found during testing, where checking an existing
      event's own span (which for an all-day event is the whole day)
      against busy periods made every all-day event look "busy" no matter
      what it actually was.
    - get_events is only used to fetch human-readable titles, and ONLY
      when a real conflict was already found -- purely for naming what
      the conflict is, for the ToT step and the eventual email draft.
    """
    event = state["candidate_event"]
    time_min = f"{event['start'][:10]}T00:00:00Z"
    time_max = f"{event['end'][:10]}T23:59:59Z"

    freebusy_tool = tools["query_freebusy"]
    freebusy_result = await freebusy_tool.ainvoke({
        "user_google_email": USER_EMAIL,
        "time_min": time_min,
        "time_max": time_max,
    })
    freebusy_text = freebusy_result[0]["text"]
    busy_periods = [
        (m.group("start"), m.group("end"))
        for m in re.finditer(r"- (?P<start>\S+) to (?P<end>\S+)", freebusy_text)
    ]

    conflict = any(
        overlaps(event["start"], event["end"], b_start, b_end)
        for b_start, b_end in busy_periods
    )

    observation = []
    conflicting_events = []

    if conflict:
        get_events_tool = tools["get_events"]
        events_result = await get_events_tool.ainvoke({
            "user_google_email": USER_EMAIL,
            "time_min": time_min,
            "time_max": time_max,
            "detailed": False,
        })
        events_text = events_result[0]["text"]
        event_pattern = re.compile(
            r'- "(?P<title>[^"]+)" \(Starts: (?P<start>[^\[]+) \[.*?\], '
            r'Ends: (?P<end>[^\[]+) \[.*?\]\) ID: (?P<id>\S+)'
        )
        observation = [m.groupdict() for m in event_pattern.finditer(events_text)]

        for e in observation:
            is_all_day = "T" not in e["start"]  # date-only strings have no time component
            if is_all_day:
                continue  # never blame an all-day entry for a real conflict
            if overlaps(event["start"], event["end"], e["start"], e["end"]):
                conflicting_events.append({
                    "title": e["title"],
                    "start": e["start"],
                    "end": e["end"],
                    # Real Calendar events carry no such field. Defaulting
                    # to False is the safe choice: never assume a real
                    # event can be moved without being told so explicitly.
                    "flexible": False,
                })

    return {
        **state,
        "observation": observation,
        "conflicting_events": conflicting_events,
        "conflict": conflict,
    }


def route_on_conflict(state: BranchState) -> str:
    return "conflict_found" if state["conflict"] else "no_conflict"


def _with_seconds(iso_datetime: str) -> str:
    """Ensures a datetime string has seconds precision. Google's real
    Calendar API rejected "2026-09-20T09:00" (400 Bad Request) but
    accepted "2026-09-20T09:00:00" -- found via testing against the real
    manage_event tool."""
    return iso_datetime if iso_datetime.count(":") == 2 else iso_datetime + ":00"

def _tool_call_failed(response_text: str) -> bool:
    """MCP tool errors come back as text content, not raised exceptions --
    this must be checked explicitly or a failure gets silently reported
    as a success."""
    return response_text.strip().lower().startswith(("error", "error calling tool"))


async def create_event_node(state: BranchState, tools: dict) -> BranchState:
    """Uses the real manage_event tool (action="create"). Two things found
    only by testing against the real API: it requires full seconds
    precision on start_time/end_time, and it rejects an explicitly-passed
    location=None -- omitting the key entirely when there's no location
    works correctly."""
    event = state["candidate_event"]
    manage_event_tool = tools["manage_event"]

    args = {
        "user_google_email": USER_EMAIL,
        "action": "create",
        "summary": event["title"],
        "start_time": _with_seconds(event["start"]),
        "end_time": _with_seconds(event["end"]),
        "timezone": DEFAULT_TIMEZONE.key,
    }
    if event.get("location"):
        args["location"] = event["location"]
    
    
    
    #put debug print statement
    #print(f"\n[DEBUG] Sending to manage_event: {args}")  # <-- add this line temporarily
    

    result = await manage_event_tool.ainvoke(args)
    confirmation_text = result[0]["text"]

    if _tool_call_failed(confirmation_text):
        print(f"\n[Action: FAILED to create calendar event] {confirmation_text}")
        return {**state, "created_event": {"confirmation_text": confirmation_text, "status": "error"}}

    print(f"\n[Action: calendar event created] {confirmation_text}")
    return {**state, "created_event": {"confirmation_text": confirmation_text}}


def finalize_created_node(state: BranchState) -> BranchState:
    """Only node whose output crosses the subgraph boundary for this path.
    Reflects whether create_event_node actually succeeded -- previously
    this always reported "created" regardless of what the tool call
    actually returned."""
    tool_status = state["created_event"].get("status", "ok")
    result_status = "created" if tool_status == "ok" else "error"
    return {**state, "results": [{
        "status": result_status,
        "event": state["candidate_event"],
        "created_event": state["created_event"],
    }]}

def _generate_candidates_stub(event: dict, conflicts: list) -> list:
    """Deterministic fallback used only when Ollama isn't reachable."""
    candidates = []
    for existing in conflicts:
        if existing.get("flexible"):
            new_start = parse(event["end"]) + timedelta(minutes=MIN_BUFFER_MINUTES)
            duration = parse(existing["end"]) - parse(existing["start"])
            candidates.append({
                "strategy": "reschedule_existing",
                "description": f"Move '{existing['title']}' to start at {fmt(new_start)}",
                "target": existing["title"], "proposed_start": fmt(new_start),
                "proposed_end": fmt(new_start + duration),
            })
    latest_end = max(parse(e["end"]) for e in conflicts)
    alt_start = latest_end + timedelta(minutes=MIN_BUFFER_MINUTES)
    duration = parse(event["end"]) - parse(event["start"])
    candidates.append({
        "strategy": "alternate_time",
        "description": f"Shift '{event['title']}' to start at {fmt(alt_start)}",
        "proposed_start": fmt(alt_start), "proposed_end": fmt(alt_start + duration),
    })
    candidates.append({"strategy": "manual_review",
                        "description": "No confident automated resolution - flag for manual review"})
    return candidates


def generate_candidates_node(state: BranchState) -> BranchState:
    """ToT candidate generation. This is a stand-in for a full Tree-of-Thought
    search -- it generates 2-4 sibling candidates (the tree's only branch
    level; depth is capped at 1, see design doc Section 3) via a single LLM
    call, or the deterministic stub if Ollama is unavailable."""
    event = state["candidate_event"]
    conflicts = state["conflicting_events"]

    if _HAS_OLLAMA:
        prompt = f"""A new calendar event conflicts with existing events. Propose
2-4 distinct resolution strategies as structured candidates.

Policy:
- Never propose moving or modifying an event marked "flexible": false.
- Prefer resolutions that keep the new event at its original time when a
  reasonable option exists.
- If no resolution seems clearly safe, prefer "manual_review" over guessing.

New event: {event}
Conflicting existing events: {conflicts}

Each candidate must have a "strategy" field (one of: reschedule_existing,
alternate_time, shrink_buffer, manual_review), a "description", and if it
proposes a schedule change, "target"/"proposed_start"/"proposed_end".
Return as JSON."""
        response = _ollama_client.chat(
            model=OLLAMA_MODEL,
            messages=[{"role": "user", "content": prompt}],
            format=CandidateList.model_json_schema(),
        )
        candidates = CandidateList.model_validate_json(response.message.content).model_dump()["candidates"]
        if not any(c["strategy"] == "manual_review" for c in candidates):
            candidates.append({"strategy": "manual_review",
                                "description": "No confident automated resolution - flag for manual review"})
    else:
        candidates = _generate_candidates_stub(event, conflicts)

    return {**state, "candidates": candidates}


def evaluate_candidates_node(state: BranchState) -> BranchState:
    """Deterministic, tool-grounded constraint check -- always runs the same
    way regardless of whether candidates came from the LLM or the stub. Kept
    deterministic (not another LLM call) so this step stays verifiable; see
    design doc Section on search/evaluation strategy for the full rationale."""
    existing_events = state["observation"]
    scored = []

    for c in state["candidates"]:
        if c["strategy"] == "reschedule_existing" and c.get("proposed_start"):
            collides = any(
                e["title"] != c.get("target") and
                overlaps(c["proposed_start"], c["proposed_end"], e["start"], e["end"])
                for e in existing_events
            )
            if not collides:
                scored.append((3, c))
        elif c["strategy"] == "alternate_time" and c.get("proposed_start"):
            collides = any(
                overlaps(c["proposed_start"], c["proposed_end"], e["start"], e["end"])
                for e in existing_events
            )
            if not collides:
                scored.append((2, c))
        elif c["strategy"] == "shrink_buffer":
            scored.append((1, c))
        elif c["strategy"] == "manual_review":
            scored.append((0, c))

    if not scored:
        scored = [(0, {"strategy": "manual_review",
                        "description": "No confident automated resolution - flag for manual review"})]

    scored.sort(key=lambda pair: pair[0], reverse=True)
    top_score = scored[0][0]
    selected = [c for score, c in scored if score == top_score] or [scored[0][1]]

    return {**state, "selected_resolutions": selected[:2]}


async def draft_alert_node(state: BranchState, tools: dict) -> BranchState:
    event = state["candidate_event"]
    conflicts = state["conflicting_events"]
    resolutions = state["selected_resolutions"]

    lines = [
        "A new travel event conflicts with your calendar:",
        f"  New event: {event['title']} ({event['start']} - {event['end']})",
        "  Conflicts with: " + "; ".join(f"{e['title']} ({e['start']}-{e['end']})" for e in conflicts),
        "", "Suggested resolution(s):",
    ]
    for r in resolutions:
        lines.append(f"  - [{r['strategy']}] {r['description']}")
    body = "\n".join(lines)

    create_draft = tools["draft_gmail_message"]
    result = await create_draft.ainvoke({
        "user_google_email": USER_EMAIL,
        "to": USER_EMAIL,
        "subject": f"Travel conflict detected: {event['title']}",
        "body": body,
        "body_format": "plain",
    })
    confirmation_text = result[0]["text"]
    
    if _tool_call_failed(confirmation_text):
        print(f"\n[Action: FAILED to create draft email] {confirmation_text}")
        return {**state, "drafted_email": {"confirmation_text": confirmation_text, "status": "error"}}
    
    print(f"\n[Action: draft created in Gmail] {confirmation_text}")
    return {**state, "drafted_email": {"confirmation_text": confirmation_text}}


def finalize_drafted_node(state: BranchState) -> BranchState:
    """Only node whose output crosses the subgraph boundary for this path."""
    tool_status = state["drafted_email"].get("status", "ok")
    result_status = "conflict" if tool_status == "ok" else "error"
    return {**state, "results": [{
        "status": result_status,
        "event": state["candidate_event"],
        "selected_resolutions": state["selected_resolutions"],
        "drafted_email": state["drafted_email"],
    }]}


# ===========================================================================
# Graph assembly
# ===========================================================================
async def build_branch_subgraph(tools: dict):
    """One compiled subgraph, reused for every branch. Because it has its
    own BranchState schema, none of its intermediate fields ever touch the
    parent graph's channels -- see the module docstring for why that matters."""
    async def check_calendar_step(state: BranchState) -> BranchState:
        return await check_calendar_node(state, tools)

    async def create_event_step(state: BranchState) -> BranchState:
        return await create_event_node(state, tools)

    async def draft_alert_step(state: BranchState) -> BranchState:
        return await draft_alert_node(state, tools)

    g = StateGraph(BranchState)
    g.add_node("check_calendar", check_calendar_step)
    g.add_node("create_event", create_event_step)
    g.add_node("finalize_created", finalize_created_node)
    g.add_node("generate_candidates", generate_candidates_node)
    g.add_node("evaluate_candidates", evaluate_candidates_node)
    g.add_node("draft_alert", draft_alert_step)
    g.add_node("finalize_drafted", finalize_drafted_node)

    g.set_entry_point("check_calendar")
    g.add_conditional_edges("check_calendar", route_on_conflict, {
        "no_conflict": "create_event", "conflict_found": "generate_candidates",
    })
    g.add_edge("create_event", "finalize_created")
    g.add_edge("finalize_created", END)
    g.add_edge("generate_candidates", "evaluate_candidates")
    g.add_edge("evaluate_candidates", "draft_alert")
    g.add_edge("draft_alert", "finalize_drafted")
    g.add_edge("finalize_drafted", END)

    return g.compile()


async def build_app(calendar_server_cmd=None, gmail_server_cmd=None):
    """
    Defaults to the real Google Workspace MCP server. Pass
    calendar_server_cmd=MOCK_CALENDAR_CFG, gmail_server_cmd=MOCK_GMAIL_CFG
    explicitly for offline testing against the mock servers instead.
    """
    calendar_cfg = calendar_server_cmd or REAL_WORKSPACE_CFG
    gmail_cfg = gmail_server_cmd or REAL_WORKSPACE_CFG

    client = MultiServerMCPClient({"calendar": calendar_cfg, "gmail": gmail_cfg})
    tool_list = await client.get_tools()
    tools = {t.name: t for t in tool_list}

    branch_subgraph = await build_branch_subgraph(tools)

    g = StateGraph(AgentState)
    g.add_node("extract", extract_node)
    g.add_node("branch", branch_subgraph)          # compiled subgraph used directly as a node
    g.add_node("needs_clarification", needs_clarification_node)

    g.set_entry_point("extract")
    g.add_conditional_edges("extract", dispatch_events, {
        "branch": "branch", "needs_clarification": "needs_clarification",
    })
    g.add_edge("branch", END)
    g.add_edge("needs_clarification", END)

    return g.compile()


async def run_pipeline(raw_itinerary_text: str) -> dict:
    """Runs the full pipeline once for one document's worth of text, which
    may describe several activities. Returns the merged results list --
    one entry per extracted event, regardless of which path it took."""
    
    #if use_mock:
    #    app = await build_app(calendar_server_cmd=MOCK_CALENDAR_CFG, gmail_server_cmd=MOCK_GMAIL_CFG)
    #else:
    
    app = await build_app()
    final_state = await app.ainvoke({"raw_itinerary_text": raw_itinerary_text, "results": []})

    print(f"\n[Summary] {len(final_state['results'])} event(s) processed:")
    for r in final_state["results"]:
        print(f"  - {r['status']}: {r['event'].get('title')}")
    return final_state


# ===========================================================================
# UI ENTRY POINTS
# A UI that lets someone drop itinerary files into a local folder and press
# "start" calls process_itinerary_folder(). These functions have no
# notebook/CLI-specific code, so they can be called directly from a UI layer
# (e.g. a button's on-click handler) without modification.
#
# NOTE: these currently assume plain text files. Real PDF/email parsing is
# a known next step (see README "out of scope" section) -- when that's
# added, only load_itinerary_text() needs to change; run_pipeline and
# everything upstream of it already only deal in raw text.
# ===========================================================================
#def load_itinerary_text(path: str) -> str:
#    """Single point of change for when real PDF/email parsing is added."""
#    return Path(path).read_text()

def load_itinerary_text(path: str) -> str:
    """Reads itinerary content from a plain text file or a PDF."""
    p = Path(path)
    if p.suffix.lower() == ".pdf":
        from pypdf import PdfReader
        reader = PdfReader(str(p))
        text = "\n".join(page.extract_text() or "" for page in reader.pages)
        if not text.strip():
            raise RuntimeError(
                f"No extractable text found in '{path}'. This usually means "
                f"the PDF is scanned/image-based rather than a text-based "
                f"document. OCR support is not implemented -- try a "
                f"text-based PDF or a .txt file instead."
            )
        return text
    return p.read_text()


async def process_itinerary_file(path: str) -> dict:
    """Entry point for processing one uploaded document."""
    text = load_itinerary_text(path)
    print(f"\n=== Processing {path} ===")
    return await run_pipeline(text)

async def process_itinerary_folder(folder_path: str) -> list[dict]:
    """Entry point for a 'watch this folder, press start' UI. Processes
    every file in the folder and returns one final_state per file."""
    folder = Path(folder_path)
    results = []
    for file_path in sorted(folder.iterdir()):
        if file_path.is_file() and not file_path.name.startswith("."):
            results.append(await process_itinerary_file(str(file_path)))
    return results

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Process itinerary file(s) through the Travel Planner Agent pipeline."
    )
    parser.add_argument(
        "path", nargs="?", default="uploads",
        help="Path to an itinerary text file, or a folder of them (default: ./uploads)",
    )
    #parser.add_argument(
    #    "--mock", action="store_true",
    #    help="Use the local mock Calendar/Gmail servers instead of the real "
    #         "Google Workspace MCP server (for offline testing, no credentials needed).",
    #)
    
    args = parser.parse_args()

    target = Path(args.path)
    if not target.exists():
        print(f"Error: '{args.path}' does not exist.")
        sys.exit(1)

    if target.is_dir():
        asyncio.run(process_itinerary_folder(str(target)))
    else:
        asyncio.run(process_itinerary_file(str(target)))
