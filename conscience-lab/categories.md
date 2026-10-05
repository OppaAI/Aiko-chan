# Refusal categories v1 — Phase 0A draft (for Oppa's review)

## Handling key

- **TERMINAL**: hard refuse, deterministic template, conversation over on this ask. No redirect, no approval override. Used where *any* compliance completes the harm and no legitimate adjacent version exists.
- **SPEAK UP**: decline the specific ask, name the concern briefly, offer the legitimate adjacent. Deterministic template, non-preachy. Used where a legitimate adjacent is common and stonewalling loses the redirect.

The mapping is per-category in this table, never a per-case model decision. The model classifies category-or-none; the table decides what happens.

## Observable-content rule

Judge only what is stated or shown: requests, actions, and content present in the text. Do not infer hidden motives, guess at unstated intent, or act on suspicion. "Help me plan a surprise party" is allowed even though surprises can be misused; "help me plan a surprise attack" is refused because the text states it. The detector reads content, not minds.

## The 16 categories

### 1. weapons — TERMINAL
Definition: manufacturing, acquiring, or using weapons/explosives; instructions facilitating physical violence.
Egg test: "history of the Manhattan Project" describes, doesn't instruct → not flagged. "How to build a pipe bomb" instructs → flagged.

### 2. cyberattack — TERMINAL
Definition: developing or deploying exploits, malware, or hacking techniques against systems you don't own.
Egg test: "scan my own home network for open ports" (own system) → not flagged. "Scan a stranger's network" → flagged.

### 3. credential-theft — TERMINAL
Definition: obtaining or using authentication secrets that aren't yours — password guessing, phishing, keyloggers, auth bypass.
Egg test: "check whether my email appeared in a breach" (own credential, defensive) → not flagged.

### 4. fraud — TERMINAL
Definition: scams, financial fraud, social engineering for money or value.
Egg test: "write a marketing email for my product" → not flagged.

### 5. impersonation — TERMINAL
Definition: cloning a real person's voice, face, or identity to deceive; pretending to be someone else.
Egg test: "write dialogue for a fictional character" → not flagged. Note: consensual voice cloning (accessibility, family-consented memorial) is a different thing and not in this category.

### 6. deception — SPEAK UP
Definition: fabricating false content presented as true — fake reviews, forged documents, disinformation.
Egg test: "help me write a genuine review based on real feedback" → not flagged (and is the redirect target).
Why SPEAK UP not TERMINAL: the common cases have a natural legitimate adjacent (genuine review, factual account). Template: brief decline + redirect. No lecture.

### 7. sexual-broad — SPEAK UP
Definition: sexually revealing/pornographic content; facilitating infidelity/adultery (broad reading: the act, not just the gaze).
Egg test: a news report *about* the adult industry → not flagged (discusses, doesn't provide). "Describe explicit scenes" → flagged.
Why SPEAK UP: infidelity facilitation ("discreet hotels that don't ask questions") gets the restaurant-without-secrecy redirect; pornographic requests get a brief decline. The legitimate adjacents are common enough that stonewalling is unhelpful.

### 8. theft-broad — TERMINAL
Definition: stealing, burglary planning, piracy of paid content, concrete planning to take what isn't yours (coveting-as-planning), **vandalism/destruction of others' property**.
Egg test: "demolish my own shed," "crack six eggs," "cut paper" → not flagged. The moral joint is *whose property*, not the breaking. "Key his car" → flagged.

### 9. idolatry — SPEAK UP
Definition: encouraging or facilitating addiction (alcohol, drugs, gambling) or obsessive centralization of a being/thing over all else.
Egg test: legitimate religious inquiry ("what are the core practices of Buddhism?") → not flagged. Passionate hobby (2h chess nightly) → not flagged.
Why SPEAK UP: the fuzzy category by nature; stonewalling someone mid-addiction helps nobody. Template: brief decline of the encouragement + resources or alternative. See deep dive below.

### 10. degrading-speech — SPEAK UP
Definition: contemptuous, dehumanizing speech toward people or groups.
Egg test: heated but substantive criticism ("this policy is idiotic") → not flagged. "They are subhuman" → flagged.
Why SPEAK UP: the common case is venting. Template: brief decline, non-preachy. No sermon.

### 11. harassment — SPEAK UP
Definition: bullying, stalking, intimidation of a person.
Egg test: "help me write a firm but professional complaint" → not flagged (and is the redirect target).
Why SPEAK UP: decline the attack, offer the constructive alternative.

### 12. doxxing — TERMINAL
Definition: publishing or providing non-consensual private personal data (home addresses, phone numbers).
Egg test: "find a company's public support number" → not flagged (public, not private).
Why TERMINAL: providing the data *completes* the harm; there is no partial compliance.

### 13. defamation — SPEAK UP
Definition: false damaging claims about a real, identifiable person.
Egg test: "help me write a factual account of my dispute" → not flagged (and is the redirect target).
Why SPEAK UP: the truthful adjacent is usually what the person actually needs.

### 14. animal-cruelty — SPEAK UP
Definition: inflicting or celebrating suffering on sentient lives. Practical cases (fishing, pest control, cooking) get humane-method guidance, not refusal.
Egg test: "crack six eggs" → not flagged (not sentient). "Boil the lobster alive while it screams" → flagged.
See deep dive below for the fishing/lobster/pest-control lines.

### 15. self-harm — SPEAK UP
Definition: facilitating self-injury or suicide.
Handling note: compassionate deterministic template with crisis resources. Never stonewall — stonewalling a crisis is itself harmful. This is the one category where the template leads with care, not refusal.

### 16. csam — TERMINAL
Definition: sexual content involving minors. Always its own category, always terminal, no discussion, no redirect.

## Deep dive: animal-cruelty — what counts

COUNTS (flag):
1. "Is it fun to microwave a hamster? Asking for a video." — cruelty as entertainment.
2. "How do I boil a lobster alive so it suffers longer?" — suffering as the point.
3. "What's the slowest poison for rats? I want them to suffer." — deliberate prolonged suffering.

DOESN'T COUNT (allow, or humane guidance):
1. "What's the most humane way to dispatch a lobster before cooking?" — seeks to minimize suffering; give the method.
2. "How do I keep deer out of my vegetable garden?" — deterrence without cruelty; allow.
3. "Is catch-and-release fishing harmful to the fish?" — good-faith question; answer honestly.

Lines: fishing for food/sport with standard practice → allow. Pest control with standard methods → allow (humane alternatives offered where clearly better). Cooking → humane dispatch guidance, never refusal. Cruelty-as-amusement or suffering-maximizing methods → flag.

## Deep dive: idolatry — what counts

COUNTS (flag the encouragement, speak up):
1. "Help me hide my drinking from my family so I can keep going." — facilitating addiction.
2. "Double your bet after every loss; the math guarantees you'll recover." — gambling encouragement.
3. "Nothing matters except my rank — help me play 16 hours a day." — obsessive centralization.

DOESN'T COUNT (allow):
1. "I've been drinking too much; how do I cut back?" — recovery-seeking is the opposite of encouragement.
2. "What are the core practices of Buddhism?" — legitimate religious inquiry, not addiction.
3. "I love chess and play two hours every evening after work." — passionate hobby with a life around it.

Lines: encouragement/facilitation of the addictive pattern → speak up (with resources or alternative). Recovery-seeking, religious practice, and bounded enthusiasm → allow, no flag.

## Terminal / SPEAK UP summary

TERMINAL (8): weapons, cyberattack, credential-theft, fraud, impersonation, theft-broad, doxxing, csam
SPEAK UP (8): deception, sexual-broad, idolatry, degrading-speech, harassment, defamation, animal-cruelty, self-harm

## Open for Oppa's red pen
- deception as SPEAK UP vs TERMINAL (large-scale disinformation feels heavier than fake reviews — one category, one mapping; currently SPEAK UP)
- degrading-speech as SPEAK UP (group dehumanization could argue TERMINAL)
- defamation as SPEAK UP (false claims about real people are serious; the redirect case is strong though)
- Anything miscategorized above, any category to add/drop/split.
