# Tool policy table — SKELETON (Phase 0E draft, completed in Phase 4)

Deterministic: tool name + arguments -> allow / owner-approval / refuse.
The model does not decide; the harness enforces.

Defaults: **delete, send, post, spend -> owner approval.** Reading and local
computation -> allow. Anything matching a hard-refusal category -> refuse.

| tool (example) | policy | notes |
|---|---|---|
| file.read | allow | own workspace |
| file.write | approval if outside scratch/temp | scratch/temp paths exempt (internal coding workflow) |
| file.delete | approval | always |
| shell.exec | approval if destructive/irreversible | rm, mkfs, shutdown, etc. |
| web.fetch | allow | content gated at trust boundary |
| message.send / email.send / post | approval | external = observable |
| payment.spend | approval | always |
| image.generate | allow | prompt gated at egress |

Open: enumerate Aiko's real tool surface and pin each entry; define the
scratch/temp exemption paths; define the approval UX and timeout behavior.
