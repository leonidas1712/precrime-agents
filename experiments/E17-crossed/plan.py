"""Episode lists for the E17 pilot and main run (draft; sizes from WORKSHOP_PLAN.md §3.1).

    pilot : 2 scenarios x (10 PF + 10 PA), secret in file, name set n1          = 40 (20 PF + 20 PA; Leo, 3 Nov)
    pilot2: 6 scenarios x (15 PF + 5 PA file, 3 PF sysprompt), name sets alternate = 138 (Leo, 4 Nov)
    check4: 6 scenarios x (2 PF + 2 PA), draft-4 live check                     = 24 (Leo, 4 Nov)
    main  : 4 balanced scenarios x (125 PF + 30 PA), secret in file, name sets alternate = 620 (frozen 5 Nov;
            see README "Main run: rules fixed before the data"). Run in two stages:
            stage 1 = PF 001-032 and PA 001-008 in each scenario (160 episodes), stage 2 = the rest.
"""
from __future__ import annotations

from crossed import MAIN_SCENARIOS, PILOT_SCENARIOS, SCENARIOS

PLANS = {
    'pilot': {'scenarios': PILOT_SCENARIOS, 'cells': [('PF', 'FILE', 10), ('PA', 'FILE', 10)]},
    # Pilot 2 (Leo, 4 Nov): all six scenarios, PA from message 1, PF keeps going after an allowed write.
    'pilot2': {'scenarios': SCENARIOS, 'cells': [('PF', 'FILE', 15), ('PA', 'FILE', 5), ('PF', 'SYSPROMPT', 3)]},
    # draft-4 check: the peer can now read its one file. 2 + 2 per scenario, ~$0.08.
    'check4': {'scenarios': SCENARIOS, 'cells': [('PF', 'FILE', 2), ('PA', 'FILE', 2)]},
    'main': {'scenarios': MAIN_SCENARIOS, 'cells': [('PF', 'FILE', 125), ('PA', 'FILE', 30)]},
}
STAGE1 = {'PF': 32, 'PA': 8}   # main run, stage 1: episode numbers up to these, in every scenario


def episodes(plan: str, stage: int | None = None) -> list[dict]:
    """stage (main only): 1 = the first STAGE1 episode numbers of each cell, 2 = the rest, None = all."""
    spec = PLANS[plan]
    out = []
    for scenario in spec['scenarios']:
        for arm, location, n in spec['cells']:
            for i in range(1, n + 1):
                name_set = 'n1' if plan == 'pilot' else ('n1' if i % 2 else 'n2')
                if stage is not None and (i <= STAGE1[arm]) != (stage == 1):
                    continue
                out.append({'episode_id': f'{plan}-{scenario}-{arm}-{location}-{i:03d}', 'scenario': scenario,
                            'arm': arm, 'location': location, 'name_set': name_set})
    return out
