"""Shared anatomical aliases for functional seeding and sensory readout."""
from __future__ import annotations

SEED_ALIASES: dict[str, tuple[str, ...]] = {
    "sensory": ("ORN", "OSN", "DA1", "VA7", "DP1", "DM", "VS", "HS",
                "Mi", "Tm", "L1", "L2", "L3", "L4", "L5", "R7", "R8",
                "LoV", "MeV", "AVLP", "PVLP", "PLP", "OCG", "GNG",
                "VES", "SMP", "AMMC", "DCH", "VCH", "CT1"),
    "visual": ("VS", "HS", "HSE", "HSN", "HSS", "H2", "Dm", "Mi", "Tm",
               "L1", "L2", "L3", "L4", "L5", "R7", "R8", "LoV", "MeV",
               "LPi", "DCH", "VCH", "CT1", "Am", "Li"),
    "auditory": ("AMMC", "JO", "aSP", "AVLP"),
    "input": ("ORN", "OSN", "DA1", "VA7", "DP1", "DM", "VS", "HS",
              "Mi", "Tm", "L1", "L2", "R7", "R8", "LoV", "MeV",
              "AVLP", "PVLP", "GNG", "AMMC"),
    "olfactory": ("ORN", "OSN", "DA1", "VA7", "DP1", "lPN", "adPN",
                  "OA-AL", "OA-V"),
    "memory": ("MBON", "KC", "PAM", "PPL", "DAN", "APL"),
    "decide": ("ER", "ExR", "EPG", "PFN", "PEN", "CX", "CB", "PB"),
    "action": ("DN", "MN"),
    "arousal": ("OA", "5-HT", "DA", "DH44", "AstA", "sNPF", "VUM", "VPM"),
}


def is_sensory_type(cell_type: str) -> bool:
    value = str(cell_type or '').strip().lower()
    return value in {'sensory', 'al', 't4', 't5'} or any(
        value.startswith(prefix.lower()) for prefix in SEED_ALIASES['sensory'])
