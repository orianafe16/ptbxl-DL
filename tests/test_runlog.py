"""Tests of the git_dirty flag of the run logs (D14; AI Error Bounty B5).

Each test builds a small temporary git repository with the same layout as
ours (src/ and results/). Skipped when git is not installed.

Run from the repository root:  python -m pytest -q tests
"""
import shutil
import subprocess

import pytest

from src import runlog

pytestmark = pytest.mark.skipif(shutil.which("git") is None,
                                reason="git not installed")


def git(repo, *args):
    subprocess.run(["git", "-c", "user.name=test", "-c",
                    "user.email=test@example.com", *args],
                   cwd=repo, check=True, capture_output=True)


@pytest.fixture
def repo(tmp_path):
    """A committed repository with one source file and an empty results/."""
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "model.py").write_text("x = 1\n")
    (tmp_path / "results" / "predictions").mkdir(parents=True)
    git(tmp_path, "init", "-q")
    git(tmp_path, "add", "src")
    git(tmp_path, "commit", "-q", "-m", "first")
    return tmp_path


def test_clean_repository_is_not_dirty(repo):
    assert runlog.git_dirty(repo) is False


def test_new_outputs_of_a_run_do_not_count(repo):
    """What happened in every full run before the fix: the run's own
    predictions file, untracked, made the log say dirty."""
    (repo / "results" / "predictions" / "run.npz").write_bytes(b"0")
    (repo / "results" / "run.json").write_text("{}")
    assert runlog.git_dirty(repo) is False


def test_changed_source_file_counts(repo):
    (repo / "src" / "model.py").write_text("x = 2\n")
    assert runlog.git_dirty(repo) is True


def test_new_source_file_not_added_counts(repo):
    """A new model file used by a run before `git add` is code that no
    commit contains."""
    (repo / "src" / "cnn.py").write_text("y = 1\n")
    assert runlog.git_dirty(repo) is True


def test_outside_a_repository_gives_none(tmp_path):
    assert runlog.git_dirty(tmp_path) is None
