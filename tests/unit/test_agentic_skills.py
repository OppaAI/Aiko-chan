from __future__ import annotations

from pathlib import Path

from agentic.skills import SkillDoc, search_skillsets


class NeverUsedEmbedder:
    def embed_query(self, *args, **kwargs):
        return [0.0]


def test_search_skillsets_falls_back_to_keywords_when_semantic_has_no_hits(monkeypatch):
    docs = [
        SkillDoc(
            skill_id="repo_patch",
            name="Repository Patch",
            path=Path("repo.md"),
            summary="Inspect and change code safely.",
            triggers=("patch code",),
            tools=("repo_read_file",),
        )
    ]
    monkeypatch.setattr("agentic.skills.discover_skill_docs", lambda: docs)
    monkeypatch.setattr("agentic.skills._semantic_rank_skills", lambda *args, **kwargs: None)

    matches = search_skillsets("repo_patch", embedder=NeverUsedEmbedder())

    assert [doc.skill_id for doc in matches] == ["repo_patch"]


def test_resolve_skill_doc_matches_directory_case_insensitively(monkeypatch, tmp_path):
    """Skill dirs keep their on-disk casing (e.g. JOB_HUNT/); resolution must
    still find them from a case-folded name on case-sensitive filesystems."""
    from agentic import skills

    skillsets = tmp_path / "skillsets"
    (skillsets / "JOB_HUNT").mkdir(parents=True)
    (skillsets / "JOB_HUNT" / "SKILL.md").write_text(
        "---\nid: job_hunt\nname: Job Hunt\n---\n\n# Job Hunt\n\nGuidance.\n"
    )
    monkeypatch.setattr(skills, "SKILL_ROOT", tmp_path)
    monkeypatch.setattr(skills, "_user_skillsets_path", lambda: tmp_path / "nope")
    monkeypatch.setattr(skills, "discover_skill_docs", lambda: [])

    doc = skills._resolve_skill_doc("job_hunt")
    assert doc is not None
    assert doc.path.parent.name == "JOB_HUNT"
