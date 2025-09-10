import types
import json
import pytest

import llm_service
import logger_service
import cache_service
import scenario_service
import web_search_service
from agents.base_agent import STEEPV_Agent


class DummyStatus:
    def __init__(self, status, output=None, error=None):
        self.status = status
        self.output = output or []
        self.error = error
    def model_dump_json(self):  # for debug path
        return json.dumps({"status": self.status})


class DummyCreate:
    def __init__(self, _id="abc123"):
        self.id = _id


class FakeClient:
    def __init__(self, statuses):
        self._statuses = list(statuses)
        self._retrieves = 0
        self.responses = types.SimpleNamespace(create=self.create, retrieve=self.retrieve)
    def create(self, **kwargs):
        return DummyCreate()
    def retrieve(self, _id):  # noqa: D401
        resp = self._statuses[min(self._retrieves, len(self._statuses)-1)]
        self._retrieves += 1
        return resp


def make_message_block(texts):
    return types.SimpleNamespace(type='message', content=[types.SimpleNamespace(type='output_text', text=t) for t in texts])


def make_reasoning_block(texts):
    inner = [types.SimpleNamespace(text=t) for t in texts]
    return types.SimpleNamespace(type='reasoning', summary=inner)


def test__extract_text_paths(monkeypatch, tmp_path):
    # Cover message extraction
    mb = make_message_block(["Hello", "World"])
    status_done = DummyStatus('completed', output=[mb])
    llm_service.client = FakeClient([status_done])
    out = llm_service._call_responses_api("sys", "user")
    assert "Hello" in out and "World" in out

    # Cover reasoning fallback path
    rb = make_reasoning_block(["Reason A", "Reason B"])
    status_done2 = DummyStatus('completed', output=[rb])
    llm_service.client = FakeClient([status_done2])
    out2 = llm_service._call_responses_api("sys", "user")
    assert "Reason A" in out2 and "Reason B" in out2

    # Cover timeout path (never reaches completed)
    status_in_progress = DummyStatus('in_progress')
    llm_service.client = FakeClient([status_in_progress])
    llm_service.configure_timings(poll_interval=0, max_cycles=2, debug=True)
    out3 = llm_service._call_responses_api("sys", "user")
    assert "timed out" in out3


def test_parse_research_heuristic(monkeypatch):
    # Return invalid JSON to trigger heuristic topics
    def fake_call(sys, user):
        return "not json"
    monkeypatch.setattr(llm_service, "_call_responses_api", fake_call)
    dr_text = """# Executive Summary\nSomething\n## Quantum Acceleration\nDetails\n## Bio Convergence\nDetails"""
    parsed = llm_service.parse_research(dr_text)
    assert parsed["topics"]  # heuristic filled


def test_generate_domain_map_fallback(monkeypatch):
    def fake_call(sys, user):
        return "{invalid json"
    monkeypatch.setattr(llm_service, "_call_responses_api", fake_call)
    topics = ["Edge AI", "Synthetic Biology", "Green Hydrogen"]
    dm = llm_service.generate_domain_map(topics, "Tech")
    # Ensure each band has at least one topic via fallback injection
    for band in ["Core", "Adjacent", "Peripheral"]:
        assert dm["topics"][band]["Tech"], f"Band {band} empty"


def test_normalize_domain_map_variants():
    minimal = llm_service._normalize_domain_map({"Core": ["A"]}, "Tech")
    assert minimal["topics"]["Core"]["Tech"] == ["A"]
    already = {"topics": {"Core": {"Tech": ["B"]}, "Adjacent": {"Tech": []}, "Peripheral": {"Tech": []}}}
    assert llm_service._normalize_domain_map(already, "Tech") is already
    bad = llm_service._normalize_domain_map(None, "Tech")
    assert bad["topics"]["Core"]["Tech"] == []


def test_heuristic_topics():
    dr_text = """# Introduction\n# Mega Trend Alpha\n## Beta Driver\nRandom text\n### Gamma Factor:\nMore text"""
    topics = llm_service._heuristic_topics(dr_text)
    assert any("Mega Trend" in t for t in topics)


def test_logger_json_and_log_event(capfd):
    logger_service.init_logger(log_json=True, log_level="INFO")
    logger_service.log_event("sample_event", message="Custom message", extra_field=123)
    captured = capfd.readouterr().err.strip().splitlines()[-1]
    data = json.loads(captured)
    assert data["event"] == "sample_event" and data["extra_field"] == 123


def test_cache_roundtrip(tmp_path):
    topic = "Test Topic"
    dr = "Deep research text"
    parsed = {"topics": ["A"], "entities": [], "concepts": []}
    cache_service.save(tmp_path, topic, dr, parsed)
    dr_loaded, parsed_loaded = cache_service.load(tmp_path, topic)
    assert dr_loaded == dr and parsed_loaded == parsed


def test_cache_load_error(monkeypatch, tmp_path):
    topic = "Err Topic"
    dr_path, parsed_path = cache_service.cache_paths(tmp_path, topic)
    tmp_path.mkdir(parents=True, exist_ok=True)
    dr_path.write_text("content", encoding="utf-8")
    parsed_path.write_text("{not json", encoding="utf-8")  # corrupt
    dr_loaded, parsed_loaded = cache_service.load(tmp_path, topic)
    assert dr_loaded == "content" and parsed_loaded is None


class DummyAgent(STEEPV_Agent):
    def __init__(self):
        super().__init__(topics=["Alpha", "Beta"])


def test_agent_paths(monkeypatch):
    a = DummyAgent()
    # generate_domain_map uses llm_service.generate_domain_map -> patch to controlled map
    def fake_dm(topics, category):
        return {"topics": {"Peripheral": {"Dummy": ["Alpha"]}, "Adjacent": {"Dummy": ["Beta"]}, "Core": {"Dummy": []}}}
    monkeypatch.setattr(llm_service, "generate_domain_map", fake_dm)
    a.generate_domain_map()
    # Patch search to raise for one topic to cover exception path
    calls = {"Alpha": 0, "Beta": 0}
    def fake_search(q):
        calls[q] += 1
        if q == "Alpha":
            raise RuntimeError("boom")
        return [{"title": "T", "snippet": "S", "link": "L"}]
    monkeypatch.setattr(web_search_service, "search", fake_search)
    signals = a.scan_for_signals()
    assert len(signals) == 1 and calls["Alpha"] == 1 and calls["Beta"] == 1


def test_scenario_extraction_and_scoring(monkeypatch):
    md = """## Scenario 1: Growth\nBody text here.\nMore.\n## Scenario 2: Decline\nWords\n### Not a Scenario Heading\nMore text\n## Another Heading\nEnd"""
    scenarios = scenario_service.extract_scenarios(md)
    assert len(scenarios) == 2
    # Force scoring fallback by returning invalid JSON
    monkeypatch.setattr(llm_service, "_call_responses_api", lambda s,u: "invalid")
    scores = scenario_service.score_scenarios(scenarios)
    assert scores and all('overall_score' in s for s in scores)


def test_scenario_direct_valid_json(monkeypatch):
    scenarios = [{"title": "T", "body": "Some body"}]
    # Provide direct JSON response
    resp = json.dumps([{"title": "T", "novelty": 5, "plausibility": 4, "impact": 3, "clarity": 2, "uncertainty_coverage": 1}])
    monkeypatch.setattr(llm_service, "_call_responses_api", lambda s,u: resp)
    scores = scenario_service.score_scenarios(scenarios)
    assert scores[0]['overall_score'] >= 0


class DummyResp:
    def __init__(self, status_code, json_data=None, raise_http=False):
        self.status_code = status_code
        self._json = json_data or {}
        self.text = json.dumps(self._json)
        self._raise_http = raise_http
    def raise_for_status(self):
        if self._raise_http:
            import requests
            http_err = requests.exceptions.HTTPError(response=self)
            raise http_err
    def json(self):
        return self._json


def test_web_search_404_fallback(monkeypatch):
    # Provide first 404 then success
    calls = {"count": 0}
    def fake_post(url, headers, json, timeout):  # noqa: A002 - shadow
        calls["count"] += 1
        if calls["count"] == 1:
            return DummyResp(404, {}, raise_http=True)
        return DummyResp(200, {"results": [{"title": "A", "url": "http://a", "excerpts": ["E1", "E2"]}]})
    monkeypatch.setattr(web_search_service.requests, "post", fake_post)
    monkeypatch.setattr(web_search_service.config, "PARALLEL_AI_API_KEY", "k")
    results = web_search_service.search("test query")
    assert results and results[0]["snippet"].startswith("E1")


def test_web_search_non_retryable_http(monkeypatch):
    def fake_post(url, headers, json, timeout):
        return DummyResp(400, {"error": "bad"}, raise_http=True)
    monkeypatch.setattr(web_search_service.requests, "post", fake_post)
    monkeypatch.setattr(web_search_service.config, "PARALLEL_AI_API_KEY", "k")
    with pytest.raises(Exception):
        web_search_service.search("boom", max_retries=1)


def test_web_search_skip_no_key(monkeypatch):
    monkeypatch.setattr(web_search_service.config, "PARALLEL_AI_API_KEY", "")
    results = web_search_service.search("query")
    assert results == []


def test_web_search_retry_5xx(monkeypatch):
    calls = {"n": 0}
    class DummyResp2(DummyResp):
        pass
    def fake_post(url, headers, json, timeout):
        calls["n"] += 1
        if calls["n"] == 1:
            return DummyResp2(500, {}, raise_http=True)
        return DummyResp2(200, {"results": [{"title": "B", "url": "http://b", "excerpts": ["X"]}]})
    monkeypatch.setattr(web_search_service.requests, "post", fake_post)
    monkeypatch.setattr(web_search_service.config, "PARALLEL_AI_API_KEY", "k")
    res = web_search_service.search("retry test", max_retries=2)
    assert res and res[0]["title"] == "B"


def test_llm_call_failed_status(monkeypatch):
    from llm_service import _call_responses_api
    class FailStatus:
        def __init__(self):
            self.status = 'failed'
            self.error = {"message": "bad"}
            self.output = []
        def model_dump_json(self):
            return json.dumps({"status": self.status})
    class FakeClient2:
        def __init__(self):
            self.responses = types.SimpleNamespace(create=lambda **k: DummyCreate("z"), retrieve=lambda _id: FailStatus())
    llm_service.client = FakeClient2()
    out = _call_responses_api("sys","user")
    assert out.startswith("Error: Task failed")


def test_llm_call_api_error(monkeypatch):
    import openai
    class DummyAPIError(Exception):
        pass
    monkeypatch.setattr(openai, "APIError", DummyAPIError)
    class FakeClient3:
        def __init__(self):
            def boom(**k):
                raise openai.APIError("oops")
            self.responses = types.SimpleNamespace(create=boom)
    llm_service.client = FakeClient3()
    with pytest.raises(DummyAPIError):
        llm_service._call_responses_api("a","b")


def test_llm_extract_top_level_text(monkeypatch):
    # Cover item_type == 'text'
    class TopLevelText:
        def __init__(self):
            self.status = 'completed'
            part = types.SimpleNamespace(type='text', text='Direct Text')
            self.output = [types.SimpleNamespace(type='text', content=[part])]
        def model_dump_json(self):
            return json.dumps({"status":"completed"})
    llm_service.client = FakeClient([TopLevelText()])
    out = llm_service._call_responses_api("s","u")
    assert "Direct Text" in out


def test_parse_research_fenced(monkeypatch):
    fenced = """Some preamble```json\n{\n  \"topics\": [\"T1\"], \"entities\": [], \"concepts\": []\n}\n``` trailing"""
    monkeypatch.setattr(llm_service, "_call_responses_api", lambda s,u: fenced)
    parsed = llm_service.parse_research("irrelevant dr text")
    assert parsed['topics'] == ["T1"]


def test_generate_domain_map_fenced(monkeypatch):
    fenced = """```json\n{\n \"Core\": [\"A\"], \"Adjacent\": [\"B\"], \"Peripheral\": [\"C\"]\n}\n```"""
    monkeypatch.setattr(llm_service, "_call_responses_api", lambda s,u: fenced)
    dm = llm_service.generate_domain_map(["A"], "Tech")
    assert dm['topics']['Core']['Tech'] == ["A"]


def test_parse_research_error_input(monkeypatch):
    # If dr_text starts with Error: heuristic shouldn't inject topics
    monkeypatch.setattr(llm_service, "_call_responses_api", lambda s,u: "{}")
    parsed = llm_service.parse_research("Error: something bad")
    # parse_research ensures dict with topics key even if empty
    assert parsed.get('topics', []) == []


def test_scenario_scoring_fenced(monkeypatch):
    fenced = """```json\n[ { \"title\": \"T\", \"novelty\":1, \"plausibility\":1, \"impact\":1, \"clarity\":1, \"uncertainty_coverage\":1 } ]\n```"""
    monkeypatch.setattr(llm_service, "_call_responses_api", lambda s,u: fenced)
    scores = scenario_service.score_scenarios([{"title":"T","body":"B"}])
    assert scores[0]['title'] == 'T'


def test_scenario_empty_list():
    assert scenario_service.score_scenarios([]) == []


def test_main_skip_and_scoring(monkeypatch, tmp_path):
    # Patch LLM functions
    monkeypatch.setattr(llm_service, "generate_deep_research", lambda topic: "# Report\n## Scenario 1: Future\nBody")
    monkeypatch.setattr(llm_service, "parse_research", lambda txt: {"topics": ["Alpha"], "entities": [], "concepts": []})
    monkeypatch.setattr(scenario_service, "extract_scenarios", lambda txt: [{"title":"Scenario A","body":"Body"}])
    monkeypatch.setattr(scenario_service, "score_scenarios", lambda sc: [{"title":"Scenario A","novelty":1,"plausibility":1,"impact":1,"clarity":1,"uncertainty_coverage":1,"overall_score":1.0}])
    # Skip web search zeros signals
    from main import horizon_scan
    horizon_scan.callback = None  # defensive for typer
    # Provide explicit primitive parameters to avoid Typer OptionInfo interference
    horizon_scan(
        topic="Test",
        output_dir=tmp_path,
        poll_interval=1,
        max_cycles=2,
        debug_dr=False,
        cache_dir=None,
        no_cache=False,
        refresh_cache=False,
        log_level="INFO",
        log_json=False,
        no_scenario_scoring=False,
        dr_file=None,
        skip_web_search=True,
    )
    # Ensure report exists
    results_dir = tmp_path / 'results'
    reports = list(results_dir.glob('horizon_scan_results_*.json'))
    assert reports


def test_main_with_dr_file_and_no_scenarios(monkeypatch, tmp_path):
    dr_path = tmp_path / 'existing.md'
    dr_path.write_text("# Existing\nNo scenarios here", encoding='utf-8')
    monkeypatch.setattr(llm_service, "parse_research", lambda txt: {"topics": [], "entities": [], "concepts": []})
    from main import horizon_scan
    horizon_scan(topic="Topic", output_dir=tmp_path, dr_file=dr_path, skip_web_search=True, no_scenario_scoring=True)
    results_dir = tmp_path / 'results'
    assert any(p.name.startswith('horizon_scan_results_') for p in results_dir.iterdir())


def test_main_cache_hit(monkeypatch, tmp_path):
    cache_dir = tmp_path / 'cache'
    topic = "Cache Topic"
    # Pre-save cache
    cache_service.save(cache_dir, topic, "# DR", {"topics": ["A"], "entities": [], "concepts": []})
    monkeypatch.setattr(llm_service, "generate_deep_research", lambda topic: "# SHOULD NOT RUN")
    monkeypatch.setattr(llm_service, "parse_research", lambda txt: {"topics": ["A"], "entities": [], "concepts": []})
    from main import horizon_scan
    horizon_scan(
        topic=topic,
        output_dir=tmp_path,
        poll_interval=1,
        max_cycles=2,
        debug_dr=False,
        cache_dir=cache_dir,
        no_cache=False,
        refresh_cache=False,
        log_level="INFO",
        log_json=False,
        no_scenario_scoring=True,
        dr_file=None,
        skip_web_search=True,
    )
    # Should have horizon report
    assert (tmp_path / 'results').is_dir()


def test_cache_save_exceptions(monkeypatch, tmp_path):
    # Force write_text to raise to execute except blocks
    class BoomPath(type(tmp_path)):
        pass
    from pathlib import Path as _P
    def boom_write(self, *a, **k): raise OSError("boom")
    monkeypatch.setattr(_P, 'write_text', boom_write)
    cache_service.save(tmp_path / 'c', 'Topic', 'DR', {'topics':[], 'entities':[], 'concepts':[]})
    # No exception should propagate

