"""Time-based (Boulware) concession curves."""

from __future__ import annotations


def boulware(t: int, T: int, start: float, floor: float, beta: float) -> float:
    """Value of the concession curve at round t of a T-round schedule.

    Starts at `start` (t=1), ends at `floor` (t=T); beta > 1 holds near the
    start and concedes late, beta < 1 concedes early.
    """
    if T <= 1 or t >= T:
        return floor
    t = max(t, 1)
    progress = ((t - 1) / (T - 1)) ** beta
    return floor + (start - floor) * (1.0 - progress)
