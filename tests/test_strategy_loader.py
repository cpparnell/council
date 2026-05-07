"""Tests for strategy YAML loading and config validation."""

import textwrap
from pathlib import Path

import pytest
import yaml

from src.strategies.config import StrategyConfig
from src.strategies.loader import load_strategy


# ---------------------------------------------------------------- helpers


def _write_yaml(tmp_path: Path, content: str) -> Path:
    p = tmp_path / "strategy.yaml"
    p.write_text(textwrap.dedent(content))
    return p


def _write_prompts(tmp_path: Path, *names: str) -> dict[str, str]:
    """Create stub prompt files and return a mapping of name → absolute path str."""
    paths = {}
    for name in names:
        p = tmp_path / name
        p.write_text("stub prompt")
        paths[name] = str(p)
    return paths


def _yaml_with_abs_prompts(tmp_path: Path, template: str, *names: str) -> Path:
    """Write a YAML that references absolute prompt paths created in tmp_path."""
    paths = _write_prompts(tmp_path, *names)
    content = template
    for name in names:
        content = content.replace(name, paths[name])
    return _write_yaml(tmp_path, content)


# ---------------------------------------------------------------- valid configs


def test_load_valid_single_agent(tmp_path):
    path = _yaml_with_abs_prompts(tmp_path, """
        name: "Test"
        agents:
          - name: alpha
            model: claude-haiku-4-5
            prompt: agent.txt
            weight: 1.0
        risk:
          stop_loss:
            type: atr_multiple
            value: 2.0
          take_profit:
            type: fixed_pct
            value: 0.06
    """, "agent.txt")
    config = load_strategy(path)
    assert config.name == "Test"
    assert len(config.agents) == 1
    assert config.agents[0].name == "alpha"


def test_load_valid_multi_agent_with_veto(tmp_path):
    path = _yaml_with_abs_prompts(tmp_path, """
        name: "Multi"
        agents:
          - name: trend
            model: claude-haiku-4-5
            prompt: a.txt
            weight: 0.6
          - name: macro
            model: claude-haiku-4-5
            prompt: b.txt
            weight: 0.4
          - name: guard
            model: claude-sonnet-4-6
            prompt: veto.txt
            is_veto: true
        risk:
          stop_loss:
            type: none
            value: 0.0
          take_profit:
            type: none
            value: 0.0
    """, "a.txt", "b.txt", "veto.txt")
    config = load_strategy(path)
    assert len(config.directional_agents) == 2
    assert config.veto_agent is not None
    assert config.veto_agent.name == "guard"


def test_no_veto_agent_is_valid(tmp_path):
    path = _yaml_with_abs_prompts(tmp_path, """
        name: "NoVeto"
        agents:
          - name: x
            model: claude-haiku-4-5
            prompt: a.txt
            weight: 0.5
          - name: y
            model: claude-haiku-4-5
            prompt: b.txt
            weight: 0.5
        risk:
          stop_loss:
            type: none
            value: 0.0
          take_profit:
            type: none
            value: 0.0
    """, "a.txt", "b.txt")
    config = load_strategy(path)
    assert config.veto_agent is None


def test_defaults_applied(tmp_path):
    path = _yaml_with_abs_prompts(tmp_path, """
        name: "Defaults"
        agents:
          - name: solo
            model: claude-haiku-4-5
            prompt: a.txt
            weight: 1.0
        risk:
          stop_loss:
            type: none
            value: 0.0
          take_profit:
            type: none
            value: 0.0
    """, "a.txt")
    config = load_strategy(path)
    assert config.scoring.confidence_floor == 30
    assert config.scoring.trade_threshold == 0.25
    assert config.validation.max_drawdown_pct == 15.0
    assert config.validation.atr_spike_multiplier == 2.0
    assert config.deliberation is None


def test_validation_config_overrides(tmp_path):
    path = _yaml_with_abs_prompts(tmp_path, """
        name: "CustomValidation"
        agents:
          - name: solo
            model: claude-haiku-4-5
            prompt: a.txt
            weight: 1.0
        risk:
          stop_loss:
            type: none
            value: 0.0
          take_profit:
            type: none
            value: 0.0
        validation:
          max_drawdown_pct: 10.0
          atr_spike_multiplier: 3.0
          min_news_count: 2
    """, "a.txt")
    config = load_strategy(path)
    assert config.validation.max_drawdown_pct == 10.0
    assert config.validation.atr_spike_multiplier == 3.0
    assert config.validation.min_news_count == 2


# ---------------------------------------------------------------- validation failures


def test_weights_not_summing_to_one_raises(tmp_path):
    path = _yaml_with_abs_prompts(tmp_path, """
        name: "BadWeights"
        agents:
          - name: x
            model: claude-haiku-4-5
            prompt: a.txt
            weight: 0.3
          - name: y
            model: claude-haiku-4-5
            prompt: b.txt
            weight: 0.3
        risk:
          stop_loss:
            type: none
            value: 0.0
          take_profit:
            type: none
            value: 0.0
    """, "a.txt", "b.txt")
    with pytest.raises(Exception, match="[Ww]eight"):
        load_strategy(path)


def test_multiple_veto_agents_raises(tmp_path):
    path = _yaml_with_abs_prompts(tmp_path, """
        name: "TwoVetos"
        agents:
          - name: v1
            model: claude-haiku-4-5
            prompt: a.txt
            is_veto: true
          - name: v2
            model: claude-haiku-4-5
            prompt: b.txt
            is_veto: true
        risk:
          stop_loss:
            type: none
            value: 0.0
          take_profit:
            type: none
            value: 0.0
    """, "a.txt", "b.txt")
    with pytest.raises(Exception, match="[Vv]eto"):
        load_strategy(path)


def test_missing_required_field_raises(tmp_path):
    # No risk section — schema validation should fail
    path = _write_yaml(tmp_path, """
        name: "MissingRisk"
        agents:
          - name: x
            model: claude-haiku-4-5
            prompt: a.txt
            weight: 1.0
    """)
    with pytest.raises(Exception):
        load_strategy(path)


def test_prompt_file_not_found_raises(tmp_path):
    path = _write_yaml(tmp_path, """
        name: "MissingPrompt"
        agents:
          - name: x
            model: claude-haiku-4-5
            prompt: /nonexistent/path/prompt.txt
            weight: 1.0
        risk:
          stop_loss:
            type: none
            value: 0.0
          take_profit:
            type: none
            value: 0.0
    """)
    with pytest.raises(ValueError, match="prompt not found"):
        load_strategy(path)


def test_deliberation_prompt_not_found_raises(tmp_path):
    paths = _write_prompts(tmp_path, "a.txt")
    path = _write_yaml(tmp_path, f"""
        name: "MissingDelib"
        agents:
          - name: x
            model: claude-haiku-4-5
            prompt: {paths["a.txt"]}
            weight: 1.0
        deliberation:
          enabled: true
          model: claude-sonnet-4-6
          prompt: /nonexistent/delib.txt
        risk:
          stop_loss:
            type: none
            value: 0.0
          take_profit:
            type: none
            value: 0.0
    """)
    with pytest.raises(ValueError, match="[Dd]eliberation prompt"):
        load_strategy(path)


def test_default_yaml_loads(monkeypatch):
    """strategies/default.yaml must load without errors."""
    from src.strategies.config import PROJECT_ROOT
    monkeypatch.chdir(PROJECT_ROOT)
    config = load_strategy("strategies/default.yaml")
    assert config.name == "BTC Council v2"
    assert len(config.directional_agents) == 3
    assert config.veto_agent is not None


def test_example_yaml_loads(monkeypatch):
    """strategies/example.yaml must load without errors."""
    from src.strategies.config import PROJECT_ROOT
    monkeypatch.chdir(PROJECT_ROOT)
    config = load_strategy("strategies/example.yaml")
    assert config.name is not None
    assert len(config.agents) >= 1
