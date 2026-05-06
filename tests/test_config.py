"""Testes das configurações compartilhadas."""

from pathlib import Path

from tech_challenge_churn import config


def test_project_root_can_be_overridden_by_environment(
    monkeypatch,
    tmp_path: Path,
) -> None:
    """Garante resolução correta da raiz do projeto em containers."""
    monkeypatch.setenv("TECH_CHALLENGE_PROJECT_ROOT", str(tmp_path))

    assert config._resolve_project_root() == tmp_path.resolve()
