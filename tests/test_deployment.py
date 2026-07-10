"""Fast CI checks for Railway deployment readiness."""

import ast
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

SPECIALIST_FILES = (
    "brain_genetic_custom_lite_.pth",
    "infectious_custom_specialist.pth",
    "developmental_malformations_lite_92plus.pth",
    "metabolic_custom_specialist.pth",
    "neoplastic_custom_specialist.pth",
    "liver_custom_malignant_classifier.pth",
    "liver_custom_Ductual_micro_final.pth",
)

DEPLOYMENT_FILES = (
    "Dockerfile",
    "railway.toml",
    "requirements-railway.txt",
    ".env.example",
    "app.py",
)


def test_app_py_parses():
    source = (ROOT / "app.py").read_text(encoding="utf-8")
    ast.parse(source)


def test_deployment_files_exist():
    for name in DEPLOYMENT_FILES:
        assert (ROOT / name).is_file(), f"Missing deployment file: {name}"


def test_generalist_weights_exist():
    path = ROOT / "rl_agent_weights (1).pth"
    assert path.is_file()
    assert path.stat().st_size > 0


def test_specialist_models_exist():
    specialist_dir = ROOT / "specialist_models"
    assert specialist_dir.is_dir()
    for name in SPECIALIST_FILES:
        path = specialist_dir / name
        assert path.is_file(), f"Missing specialist model: {name}"
        assert path.stat().st_size > 0


def test_qtable_json_valid():
    path = ROOT / "rl_training_metadata (1).json"
    assert path.is_file()
    data = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(data.get("Q"), dict)
    assert len(data["Q"]) > 0


def test_dockerfile_includes_model_assets():
    dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")
    assert "specialist_models/" in dockerfile
    assert "rl_agent_weights (1).pth" in dockerfile
    assert "rl_training_metadata (1).json" in dockerfile


def test_app_supports_railway_port():
    source = (ROOT / "app.py").read_text(encoding="utf-8")
    assert 'os.getenv("PORT")' in source or "os.environ[\"PORT\"]" in source


def test_app_finds_main_branch_generalist_name():
    source = (ROOT / "app.py").read_text(encoding="utf-8")
    assert "rl_agent_weights (1).pth" in source


def test_env_example_documents_mongo_uri():
    env_example = (ROOT / ".env.example").read_text(encoding="utf-8")
    assert "MONGO_URI" in env_example
