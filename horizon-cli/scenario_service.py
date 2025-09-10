"""Scenario extraction and quality scoring utilities."""

import re
import json
from typing import List, Dict, Any
import llm_service

SCENARIO_HEADING_PATTERN = re.compile(r"^#{2,4}\s+Scenario(?:\s+\d+)?[:\-]?\s*(.+)$", re.IGNORECASE)


def extract_scenarios(dr_text: str) -> List[Dict[str, str]]:
    """Extract scenarios from deep research markdown by heading heuristic.

    Returns list of dicts: {title, body}
    """
    lines = dr_text.splitlines()
    scenarios = []
    current = None
    for line in lines:
        m = SCENARIO_HEADING_PATTERN.match(line.strip())
        if m:
            # start new scenario
            if current and current.get('body').strip():
                scenarios.append(current)
            title = m.group(1).strip() or "Untitled Scenario"
            current = {"title": title, "body": ""}
        else:
            if current is not None:
                # stop when encountering another heading of same or higher level not starting with 'Scenario'
                if re.match(r"^#{2,3}\s+", line) and not SCENARIO_HEADING_PATTERN.match(line):
                    # finalize current scenario
                    if current.get('body').strip():
                        scenarios.append(current)
                    current = None
                else:
                    current['body'] += line + "\n"
    if current and current.get('body').strip():
        scenarios.append(current)
    return scenarios


def score_scenarios(scenarios: List[Dict[str, str]]) -> List[Dict[str, Any]]:
    """Score scenarios using LLM. Returns list with metrics per scenario.

    Metrics: novelty, plausibility, impact, clarity, uncertainty_coverage (1-5),
    plus explanation and overall_score (weighted).
    """
    if not scenarios:
        return []
    system_message = (
        "You are a foresight evaluation assistant. Score each scenario on novelty, plausibility, impact, clarity, and uncertainty coverage (1-5 integers). "
        "Return ONLY JSON: an array; each item: {title, novelty, plausibility, impact, clarity, uncertainty_coverage, explanation, overall_score}. "
        "overall_score is weighted: impact*0.3 + plausibility*0.25 + novelty*0.2 + clarity*0.15 + uncertainty_coverage*0.1 (rounded to 2 decimals)."
    )
    # Prepare scenario payload truncated if huge
    compact = [
        {"title": s["title"][:140], "body": s["body"][:2000]} for s in scenarios
    ]
    user_query = "Score these scenarios:\n" + json.dumps(compact, ensure_ascii=False)
    raw = llm_service._call_responses_api(system_message, user_query)
    data = []
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        # try fenced
        m = re.search(r"```json\s*(\[.*?\])\s*```", raw, re.DOTALL)
        if m:
            try:
                data = json.loads(m.group(1))
            except Exception:
                data = []
    # Heuristic fallback if still empty
    if not isinstance(data, list) or not data:
        data = []
        for s in scenarios:
            body = s.get('body', '')
            length_score = min(5, max(1, len(body.split()) // 150 + 1))
            impact = 3
            novelty = 3
            plausibility = 3
            clarity = 3
            uncertainty = 2
            data.append({
                "title": s.get('title'),
                "novelty": novelty,
                "plausibility": plausibility,
                "impact": impact,
                "clarity": clarity,
                "uncertainty_coverage": uncertainty,
                "explanation": "Heuristic fallback scoring due to LLM parse failure.",
                "overall_score": round(impact*0.3 + plausibility*0.25 + novelty*0.2 + clarity*0.15 + uncertainty*0.1, 2)
            })
    # Ensure overall_score computed if missing
    for item in data:
        if 'overall_score' not in item:
            try:
                item['overall_score'] = round(
                    item.get('impact',0)*0.3 + item.get('plausibility',0)*0.25 + item.get('novelty',0)*0.2 + item.get('clarity',0)*0.15 + item.get('uncertainty_coverage',0)*0.1, 2)
            except Exception:
                item['overall_score'] = 0
    return data
