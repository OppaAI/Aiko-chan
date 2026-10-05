# Threat model

Aiko will be used by future users we do not trust. The conscience circuit defends
against malicious or careless use: requests for hard-refusal content (the 16
categories), prompt injection smuggled in via untrusted sources (web, fetched
pages), and destructive or irreversible tool actions taken without owner approval.
Hard refusals are terminal because a future malicious user must not be able to
talk Aiko into approving her own refusals.

Out of scope: hidden intent — only stated content is judged, never guessed motives
or suspicion. Also out of scope: a determined attacker with physical access to the
machine or its weights; that is the deferred golden-key (creator hardware override)
domain, not this circuit's.
