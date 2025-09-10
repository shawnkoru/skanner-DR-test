import pytest
from unittest.mock import patch

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import scenario_service

def test_extract_scenarios_basic():
    text = """## Scenario 1: Solar Supply Crunch\nDetails line 1\nMore detail\n## Scenario 2: Hydrogen Breakout\nStuff\n## Not a Scenario Heading\nIgnore body\n"""
    scenarios = scenario_service.extract_scenarios(text)
    assert len(scenarios) == 2
    assert scenarios[0]['title'].startswith('Solar Supply Crunch')
    assert 'Details line' in scenarios[0]['body']

def test_score_scenarios_parses_json():
    fake_json = '[{"title":"A","novelty":5,"plausibility":3,"impact":4,"clarity":4,"uncertainty_coverage":3,"explanation":"x","overall_score":4.0}]'
    with patch('llm_service._call_responses_api', return_value=fake_json):
        scores = scenario_service.score_scenarios([{"title":"A","body":"text"}])
    assert scores[0]['overall_score'] == 4.0

def test_score_scenarios_fallback():
    with patch('llm_service._call_responses_api', return_value='not json'):    
        scores = scenario_service.score_scenarios([{"title":"B","body":"short body words"}])
    assert scores and scores[0]['overall_score'] > 0