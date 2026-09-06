"""Task sentences for collection / Qwen probe. English, as OWL query later."""

from __future__ import annotations

import random

TASKS = (
    "Follow the {color} boat and keep {d} meters.",
    "Stay {d} m behind the {color} boat.",
    "Maintain a {d}-meter standoff from the {color} vessel.",
    "Keep {d} meters from the {color} boat.",
    "Track the {color} boat at {d} m.",
    "Hold a {d} meter gap to the {color} boat.",
    "Follow the {color} vessel, standoff {d} meters.",
    "Remain {d} m from the {color} boat.",
    "Approach the {color} boat and stop {d} meters away.",
    "Keep station {d} m from the {color} craft.",
    "The task is to follow the {color} boat at {d} meters.",
    "Stay {d} meters off the {color} boat.",
    "Shadow the {color} boat with a {d}-meter standoff.",
    "Do not close within {d} m of the {color} boat; follow it.",
    "Match the {color} boat and keep {d} meters.",
    "Follow that {color} boat. Distance {d} m.",
    "Maintain {d} m spacing behind the {color} vessel.",
    "Keep the {color} boat ahead at {d} meters.",
    "Trail the {color} boat by {d} meters.",
    "Hold {d} meters from the {color} boat and follow.",
)

assert len(TASKS) == 20


def render_task(color: str, standoff: float, index: int = 0) -> str:
    tmpl = TASKS[int(index) % len(TASKS)]
    d = int(standoff) if float(standoff) == int(standoff) else standoff
    return tmpl.format(color=str(color), d=d)


def sample_task(color: str, standoff: float, rng: random.Random) -> str:
    return render_task(color, standoff, rng.randrange(len(TASKS)))
