# Calendar Studio

Calendar Studio is Aiko's local-first chief-of-staff view at
`/studio/calendar/`. It combines appointments, a lightweight project board,
message/email drafts, and scheduler-backed reminders in one authenticated
workspace.

Planning items are stored per user in `~/.aiko/<user>/tasks/calendar_studio.json`.
When **Schedule with Aiko** is selected, the studio also creates a real record
in the existing `tasks/schedule.json` scheduler store. Deleting that plan
disables its linked scheduler record.

The API deliberately takes its user identity only from the signed web session;
it does not accept a `user_id` query parameter.
