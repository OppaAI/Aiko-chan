# Calendar Studio

Calendar Studio is Aiko's local-first chief-of-staff view at
`/studio/calendar/`. It combines appointments, a lightweight project board,
message/email drafts, and full scheduler task management in one authenticated
workspace. It uses the shared purple studio theme
(`studio/shared/css/`).

Planning items are stored per user in `~/.aiko/<user>/tasks/calendar_studio.json`.
When **Schedule with Aiko** is selected, the studio also creates a real record
in the existing `tasks/schedule.json` scheduler store. Deleting that plan
disables its linked scheduler record.

## Scheduler task management

The **Aiko scheduler** section lists every `tasks/schedule.json` record for the
signed-in user — including paused (disabled) ones, which render dimmed with a
badge. Clicking a row opens the schedule editor; **+ Scheduled task** creates a
new record directly.

Editable fields: title, instructions, time of day, repeat (once / every N
minutes / hourly / daily / weekdays / weekly / custom weekdays / biweekly /
monthly), interval minutes, weekdays, timezone, action (announce / agentic /
tool), enabled, and the idle gate ("only when I'm idle" + idle minutes).
Timing changes recalculate `next_due` from now. Records can be permanently
deleted.

API (`/studio/calendar/api/`):

- `GET /api/schedules` — all records, including disabled
- `POST /api/schedules` — create a record
- `PUT /api/schedules/{id}` — edit a record in place
- `DELETE /api/schedules/{id}` — permanently delete a record

The four hardcoded internal jobs (daily reflection/dream, monthly
consolidation, conscience-ledger prune, fly replay) are not schedule records
and are intentionally not shown here — their timing is owned by the scheduler
itself.

The API deliberately takes its user identity only from the signed web session;
it does not accept a `user_id` query parameter.
