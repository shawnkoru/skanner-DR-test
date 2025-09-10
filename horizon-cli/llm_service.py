import openai
import config
import json
import time
import os
import re
from typing import Any, Dict, List

# Initialize the OpenAI client
client = openai.Client(api_key=config.OPENAI_API_KEY)

DEBUG_DEEP_RESEARCH = os.getenv("DEBUG_DEEP_RESEARCH", "0") == "1"

POLL_INTERVAL_SECONDS = 8
MAX_POLL_CYCLES = 120  # ~16 minutes max wait


def _dump_debug(status_response: Any):
    if not DEBUG_DEEP_RESEARCH:
        return
    try:
        with open("deep_research_debug.json", "w") as dbg:
            json.dump(json.loads(status_response.model_dump_json()), dbg, indent=2)
    except Exception as e:
        print(f"Debug dump failed: {e}")


def _extract_text_from_output(status_response: Any) -> str:
    """Extract concatenated textual content from a completed responses API object.

    Strategy:
    1. Prefer assistant message blocks (type == 'message') collecting parts with type in {'output_text','text'}.
    2. If none found, fall back to reasoning summaries (type == 'reasoning', summary[].text).
    3. Return error string if still empty.
    """
    texts = []
    # Pass 1: assistant / message blocks
    for item in getattr(status_response, 'output', []) or []:
        item_type = getattr(item, 'type', None)
        if item_type == 'message':
            content_list = getattr(item, 'content', None)
            if isinstance(content_list, list):
                for part in content_list:
                    p_type = getattr(part, 'type', None)
                    if p_type in ("output_text", "text"):
                        t = getattr(part, 'text', None)
                        if isinstance(t, str):
                            texts.append(t)
        elif item_type in ("text", "output_text"):
            # Some SDK variants might surface direct text blocks
            content_list = getattr(item, 'content', None)
            if isinstance(content_list, list):
                for part in content_list:
                    t = getattr(part, 'text', None)
                    if isinstance(t, str):
                        texts.append(t)
    if texts:
        return "\n\n".join(texts).strip()

    # Pass 2: reasoning summaries
    reasoning_texts = []
    for item in getattr(status_response, 'output', []) or []:
        if getattr(item, 'type', None) == 'reasoning':
            summary_list = getattr(item, 'summary', None)
            if isinstance(summary_list, list):
                for s in summary_list:
                    t = getattr(s, 'text', None)
                    if isinstance(t, str):
                        reasoning_texts.append(t)
    if reasoning_texts:
        return "\n\n".join(reasoning_texts).strip()

    return "Error: No textual content extracted from deep research output."  # final fallback


def configure_timings(poll_interval: int = None, max_cycles: int = None, debug: bool = None):
    """Optionally override runtime timing/debug controls (used by CLI flags)."""
    global POLL_INTERVAL_SECONDS, MAX_POLL_CYCLES, DEBUG_DEEP_RESEARCH
    if poll_interval is not None:
        POLL_INTERVAL_SECONDS = max(1, int(poll_interval))
    if max_cycles is not None:
        MAX_POLL_CYCLES = max(1, int(max_cycles))
    if debug is not None:
        DEBUG_DEEP_RESEARCH = bool(debug)


def _call_responses_api(system_message: str, user_query: str) -> str:
    """Call Deep Research via Responses API and return extracted text."""
    try:
        response = client.responses.create(
            model=config.OPENAI_MODEL,
            input=[
                {"role": "developer", "content": [{"type": "input_text", "text": system_message}]},
                {"role": "user", "content": [{"type": "input_text", "text": user_query}]}
            ],
            tools=[{"type": "web_search_preview"}],
            reasoning={"summary": "auto"},
            background=True
        )
        response_id = response.id

        cycles = 0
        while cycles < MAX_POLL_CYCLES:
            status_response = client.responses.retrieve(response_id)
            if status_response.status == 'completed':
                _dump_debug(status_response)
                return _extract_text_from_output(status_response)
            if status_response.status in ('failed', 'cancelled'):
                _dump_debug(status_response)
                return f"Error: Task {status_response.status}. Details: {getattr(status_response, 'error', None)}"
            cycles += 1
            time.sleep(POLL_INTERVAL_SECONDS)
        return "Error: Deep research timed out waiting for completion."
    except openai.APIError as e:
        print(f"An OpenAI API error occurred: {e}")
        raise


def generate_deep_research(topic: str) -> str:
    system_message = (
        "You are a world-class deep research analyst. Produce a comprehensive, structured, multi-section foresight report. "
        "Use clear headings, numbered sections, trend analysis, uncertainties, emerging signals, scenario framings, and a concise executive summary."
    )
    user_query = (
        f"Deep research on: {topic}. Include: 1) Executive Summary 2) Key Drivers 3) Emerging Signals 4) Opportunity/Risk Matrix 5) Scenarios (2–3) 6) Strategic Implications. "
        "Cite sources inline (short domain form)."
    )
    return _call_responses_api(system_message, user_query)


def parse_research(dr_text: str) -> dict:
    system_message = (
        "You extract structured foresight metadata ONLY as JSON. No prose, no markdown. If input looks like an error, still return valid JSON with empty arrays."
    )
    user_query = (
        "Parse the following research text and extract JSON with keys: 'topics' (list of high-level thematic clusters), 'entities' (organizations, projects, actors), "
        "'concepts' (methods, technological paradigms, domain concepts). Return ONLY JSON.\n\n" + dr_text
    )
    json_string = _call_responses_api(system_message, user_query)
    data = None
    # Attempt direct parse
    try:
        data = json.loads(json_string)
    except json.JSONDecodeError:
        match = re.search(r"```json\s*(\{[\s\S]*?\})\s*```", json_string)
        if match:
            try:
                data = json.loads(match.group(1))
            except Exception:
                data = None
    if not isinstance(data, dict):
        data = {"topics": [], "entities": [], "concepts": []}

    # Heuristic enrichment if topics empty but we have a large deep research text
    if not data.get("topics") and dr_text and not dr_text.startswith("Error:"):
        data["topics"] = _heuristic_topics(dr_text)
    return data


def _normalize_domain_map(raw: Dict[str, Any], category: str) -> Dict[str, Any]:
    """Normalize variable LLM shapes into the structure agents expect.

    Agents expect: {
        "topics": {
            "Peripheral": {"<Category>": [...]},
            "Adjacent": {"<Category>": [...]},
            "Core": {"<Category>": [...]}
        }
    }
    Accepts inputs like {"Core": [...], "Adjacent": [...], "Peripheral": [...]} or already-normalized forms.
    """
    if not isinstance(raw, dict):
        return {"topics": {"Core": {category: []}, "Adjacent": {category: []}, "Peripheral": {category: []}}}

    # If already nested with 'topics'
    if 'topics' in raw and isinstance(raw['topics'], dict):
        return raw

    core = raw.get('Core') or []
    adjacent = raw.get('Adjacent') or []
    peripheral = raw.get('Peripheral') or []

    # Some models might return dicts with category keys already
    if isinstance(core, dict):
        core_map = core
    else:
        core_map = {category: core if isinstance(core, list) else []}
    if isinstance(adjacent, dict):
        adjacent_map = adjacent
    else:
        adjacent_map = {category: adjacent if isinstance(adjacent, list) else []}
    if isinstance(peripheral, dict):
        peripheral_map = peripheral
    else:
        peripheral_map = {category: peripheral if isinstance(peripheral, list) else []}

    return {"topics": {"Core": core_map, "Adjacent": adjacent_map, "Peripheral": peripheral_map}}


def generate_domain_map(topics: list, category: str) -> dict:
    system_message = (
        "You categorize topics by foresight horizon. Return ONLY JSON with keys Core, Adjacent, Peripheral. No explanations."
    )
    user_query = (
        f"Category: {category}. Input topics: {topics}. Output JSON with arrays for Core, Adjacent, Peripheral (each array 3–8 concise topic strings)."
    )
    json_string = _call_responses_api(system_message, user_query)
    try:
        raw = json.loads(json_string)
    except json.JSONDecodeError:
        import re
        match = re.search(r"```json\s*(\{[\s\S]*?\})\s*```", json_string)
        if match:
            try:
                raw = json.loads(match.group(1))
            except Exception:
                raw = {}
        else:
            raw = {}
    normalized = _normalize_domain_map(raw, category)
    # Fallback: ensure at least 1 topic lands in each band to drive downstream scanning
    cat_key = category
    bands = ["Core", "Adjacent", "Peripheral"]
    source_topics = topics or []
    if source_topics:
        for i, band in enumerate(bands):
            band_list = normalized["topics"].get(band, {}).get(cat_key, [])
            if not band_list:
                # pick a topic deterministically based on index
                choice = source_topics[i % len(source_topics)]
                normalized["topics"].setdefault(band, {}).setdefault(cat_key, []).append(choice)
    return normalized


# ---------------- Heuristic Utilities ---------------- #
HEADING_PATTERN = re.compile(r"^(#{1,6}\s+|\d+\.|[A-Z][A-Za-z0-9 &/-]{3,}\:)")

def _heuristic_topics(dr_text: str, max_topics: int = 12) -> List[str]:
    """Extract candidate topics from headings / bullet points / capitalized phrases.
    Very lightweight heuristic to bootstrap when parsing failed.
    """
    lines = dr_text.splitlines()
    topics: List[str] = []
    seen = set()
    for ln in lines:
        line = ln.strip()
        if not line:
            continue
        if HEADING_PATTERN.match(line):
            cleaned = re.sub(r"^(#{1,6}\s+|\d+\.\s*)", "", line)
            cleaned = cleaned.rstrip(':').strip()
            key = cleaned.lower()
            if (3 <= len(cleaned) <= 120 and
                key not in {"introduction", "conclusion", "executive summary"} and
                key not in seen):
                seen.add(key)
                topics.append(cleaned)
                if len(topics) >= max_topics:
                    break
    return topics
