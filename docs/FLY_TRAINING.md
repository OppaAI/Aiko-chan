# Train and evaluate Aiko's fly action layer

Aiko already includes the MaleCNS v1.0 mushroom-body subcircuit, FlyWorld, and
Phase 10A dopamine/eligibility credit. `scripts/train_flyworld.py` is a
bounded command-line entry point for the existing simulation-training path.

By default it forces FlyWorld, dopamine, and MB writes into `shadow`. This
evaluates a policy, runs synthetic practice episodes, and compares held-out
scores before and after without saving plasticity changes:

```bash
python scripts/train_flyworld.py --episodes 20 --steps 8 --eval-episodes 20
```

To persist simulated learning for one specific Aiko identity, opt in and
provide both live gates explicitly. Substitute the identity Aiko uses for the
intended user:

```bash
MEMORY_FLYMB_MODE=live AIKO_FLY_DOPAMINE_MODE=live \
python scripts/train_flyworld.py --apply --user-id <aiko-user-id> \
  --episodes 20 --steps 8 --eval-episodes 20 --report /tmp/fly-training.json
```

The script refuses `--apply` without both `--user-id` and the two live gates.
Each applied episode flushes the learned KC→MBON plastic overlay to that
identity's `FLY_PLASTICITY_DB` store. The base MaleCNS connectome remains
read-only. Held-out seeds are separate from training seeds; the JSON report
includes the before/after reward delta and applied pulse count. Start with the
default shadow run, then review that report before opting into writes.

This is simulator practice, not human demonstration training. FlyWorld has a
small categorical action space and a hand-authored reward model, so it can
exercise action choice, clarification, refusal, and escalation biases. It does
not generate or score natural-language responses, fine-tune the LLM, or make
Aiko's writing style more human. Simulated dopamine is scoped away from the
real-turn eligibility traces, episodic memory, and persona trait plasticity.
For Aiko to learn a real user's preferences, use the explicit feedback/teaching
paths (`teach_preference` and the existing praise/correction hooks); synthetic
rewards should not be treated as human feedback.
