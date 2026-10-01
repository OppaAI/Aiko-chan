---
id: JOB_HUNT
name: Job Hunt
summary: Find job listings for Oppa via the validated Lane D graph — fetch from configured RSS/email sources, draft posts, and save drafts for human review. Never auto-posts.
triggers: find jobs, job hunt, job listings, job search, hiring, career, today's jobs, daily jobs
tools: fetch_rss_and_email_into_state, get_next_job, draft_single_job, save_single_job_draft, check_jobs_remaining, report_job_run, search_jobs
---
# Job Hunt

Use this skill when Oppa asks Aiko to find jobs, run the job pipeline, or draft job posts.

## Route to the trusted DAG

Do not hand-chain the domain tools in ReAct. The validated path is the
`gen_job_post` graph (Lane D): ingest → store → synthesis → verify →
output. It is the matching trusted DAG for this skill — run it and let it
drive the tools in order.

`gen_job_post_legacy` (the old fetch → get_next → draft → save → report
loop) exists for rollback/comparison only. Do not use it for new runs.

## Judgment

- **Location:** Vancouver-area defaults from the workflow config unless Oppa
  gives another location. Never switch cities on a guess.
- **Volume:** `max_items` defaults to 30; tune it only when Oppa asks for
  more/fewer results.
- **Drafts are for review:** the verify step produces drafts for a human.
  Present the drafts and wait — do not post, publish, or send anything.
- **Tone of drafts:** plain and factual — role, company, location, source
  link. No hype, no invented perks.

## Constraints

- **Never auto-post to Threads or any social/email destination.**
  `save_single_job_draft` stays on `auto_post=false`. Oppa approves later,
  outside this skill.
- **Email source is disabled by default** in the workflow config. Do not
  enable it without Oppa asking.
- **No invented listings.** If a source returns nothing, say which source
  came back empty.

## Exceptions

- **Source fetch fails:** report which source failed, continue with the
  sources that worked, and note the gap in the final report.
- **Zero jobs found:** say so plainly in one line. Do not pad with stale or
  off-topic listings.
- **Oppa asks for a one-off search** (not the pipeline): use `search_jobs`
  directly with explicit location/keywords instead of running the graph.
