"""Piece 5: Jobhunt Lane D skill cleanup.

- JOB_HUNT is discovered via frontmatter name/description (not full-body scan).
- Full instructions load lazily via load_skill, only when selected.
- The skill body carries judgment/constraints/exceptions, not the DAG's
  step-by-step procedure (gen_job_post encodes ingest → store → synth →
  verify → output).
"""
from agentic.skills import discover_skill_docs, load_skill, search_skillsets


def _job_hunt_doc():
    docs = {d.skill_id: d for d in discover_skill_docs()}
    assert "JOB_HUNT" in docs, "JOB_HUNT skill not discovered"
    return docs["JOB_HUNT"]


def test_job_hunt_discovered_via_frontmatter():
    doc = _job_hunt_doc()
    assert doc.name == "Job Hunt"
    assert "Lane D" in doc.summary
    assert "never auto-posts" in doc.summary.lower() or "never auto-post" in doc.summary.lower()
    assert any("job" in t for t in doc.triggers)


def test_job_hunt_loads_full_instructions_lazily():
    text = load_skill("JOB_HUNT")
    assert text.startswith('<skill id="JOB_HUNT"')
    assert "gen_job_post" in text
    assert "auto_post=false" in text


def test_job_hunt_resolves_by_name():
    assert 'id="JOB_HUNT"' in load_skill("job hunt")


def test_job_hunt_searchable_by_trigger():
    hits = search_skillsets("find me jobs hiring")
    assert any(d.skill_id == "JOB_HUNT" for d in hits)


def test_job_hunt_body_has_judgment_not_dag_procedure():
    body = load_skill("JOB_HUNT")
    # judgment / constraints / exceptions present
    assert "Never auto-post" in body
    assert "Vancouver" in body
    assert "human" in body.lower()
    # no step-by-step re-encoding of the DAG's node sequence
    assert "1. ingest" not in body.lower()
    assert "step 1" not in body.lower()


def test_job_hunt_unknown_skill_reports_known():
    out = load_skill("no_such_skill_xyz")
    assert out.startswith("[skill not found:")
    assert "Job Hunt" in out
