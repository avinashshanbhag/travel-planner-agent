# Travel Planner Agent

An agentic pipeline that reads travel itinerary documents, checks them
against your real Google Calendar, and either adds a conflict-free event
automatically or drafts a conflict-alert email with suggested resolutions
-- never sending anything without your review.

Built with **LangGraph** (multi-agent orchestration), a **local LLM**
(gemma3:4b via Ollama -- no API key, no cost, nothing leaves your machine),
and **real Google Calendar and Gmail integration** via the
[Google Workspace MCP server](https://github.com/taylorwilsdon/google_workspace_mcp).

## What it does

1. **Extraction Agent** -- parses an itinerary document (PDF or plain
   text) into one or more structured events (title, start, end, location,
   confirmation number). Ambiguous events are escalated for human review
   rather than guessed at.
2. **Conflict Resolution Agent** -- checks each event against your real
   calendar's free/busy status. If the time is genuinely free, the event
   is created directly. If it conflicts with something real, a bounded
   Tree-of-Thought step generates and evaluates a few candidate
   resolutions (reschedule, shift time, or flag for manual review).
3. **Communication Agent** -- drafts a conflict-alert email summarizing
   the conflict and the suggested resolution. **It only ever creates a
   draft -- it never has the ability to send.**

A single document can describe several activities (a flight, a hotel
stay, a museum ticket); each is extracted and processed independently.

## Setup

**1. Install Ollama and pull the model:**
```bash
ollama pull gemma3:4b
ollama serve   # if not already running
```

**2. Create a Python virtual environment and install dependencies:**
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

**3. Set up Google OAuth credentials:**
- Create a Google Cloud project, enable the Gmail API and Calendar API
- Under Google Auth Platform -> Data Access, add scopes for Gmail drafts
  (`gmail.drafts.create` or `gmail.compose`) and Calendar events
  (`calendar.events.owned`)
- Create an OAuth Client ID (Desktop app type) under the Clients tab
- Add your own email as a test user under Audience, since the app stays
  in "Testing" publishing status for personal use

**4. Create a `.env` file** in the project root:
```
USER_EMAIL=you@example.com
GOOGLE_OAUTH_CLIENT_ID=your-client-id.apps.googleusercontent.com
GOOGLE_OAUTH_CLIENT_SECRET=your-client-secret
```

**5. Install the Google Workspace MCP server** (a separate tool, not a
Python dependency -- installed via `uv`, not `pip`):
```bash
pip install uv
uv tool install workspace-mcp
```

## Running it

Drop one or more itinerary files (`.txt` or `.pdf`) into an `uploads/`
folder, then run:

```bash
mkdir -p uploads
# copy your itinerary file(s) into uploads/
python travel_agent_graph.py
```

This processes every file in `uploads/` by default. To process a single
file instead:
```bash
python travel_agent_graph.py path/to/one_itinerary.pdf
```

The first Calendar or Gmail call opens a browser for one-time OAuth
consent; after that, the cached token is reused automatically.

## Architecture

Three agents implemented as node groups within one LangGraph
`StateGraph`, with a parent graph that fans out one independent branch
per extracted event (so one document with several activities doesn't
block them against each other):

```
Extract (LLM) -> per-event dispatch -> [ambiguous] -> escalate to human
                                     -> [confident] -> Conflict Resolution
                                                          |
                                          [no conflict] --+-- [conflict]
                                                |                |
                                         Calendar write    Tree of Thought
                                        (manage_event)    (generate + evaluate)
                                                                 |
                                                          Communication Agent
                                                        (draft_gmail_message,
                                                          never sends)
```

See the accompanying design document for the full architecture rationale,
the multi-agent design decisions, the Tree-of-Thought structure, the
safety guardrails, and the real-integration debugging history.

## Files

- `travel_agent_graph.py` -- the full pipeline
- `requirements.txt` -- Python dependencies
- `.env` -- your credentials (never commit this -- see `.gitignore`)

## Guardrails

- The Communication Agent is never given a tool capable of sending email
  -- only drafting. This is enforced structurally, not by instruction.
- Ambiguous or malformed extraction escalates to a human rather than
  guessing -- including a checkpoint that validates extracted dates are
  actually parseable, not just present.
- Conflict detection is grounded in a live calendar free/busy query, not
  an LLM's assumption about your schedule.
- Real calendar events carry no "flexible" signal, so the system never
  assumes an existing commitment can be moved without being told so.
- Duplicate events from a single extraction call (an occasional small
  local model quirk) are detected and removed before anything is
  dispatched to the calendar.

## Known limitations

- PDF support requires a text layer (born-digital PDFs). Scanned/image
  PDFs are not supported and will raise a clear error rather than fail
  silently.
- No long-term memory yet -- re-running the pipeline on the same
  document will re-process it rather than recognizing it as already
  handled.
- No runtime monitoring for slow-developing patterns across many runs
  (e.g., the same document type repeatedly failing extraction).
