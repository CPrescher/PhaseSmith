"""Independent integer-work scheduling reference; no native calls or atom priors."""


def phase_cycle_quotas(cycles, trials, *, short=False):
    if not isinstance(cycles, int) or isinstance(cycles, bool) or not 1 <= cycles <= 20_000:
        raise ValueError("cycles must be an integer in [1,20000]")
    if not isinstance(trials, int) or isinstance(trials, bool) or not 1 <= trials <= 128:
        raise ValueError("trials must be an integer in [1,128]")
    if not isinstance(short, bool):
        raise ValueError("short must be boolean")
    early = trials // 2
    divisor = 4 if short else 2
    half = min(cycles, max(32, (cycles + divisor - 1) // divisor))
    return [half] * early + [cycles] * (trials - early)


def phase_plateau_ready(scores, *, omit_every_third=True):
    """Pure Python early-map plateau heuristic; no native calculation."""
    import math

    values = list(scores)
    n = len(values)
    if n < 128 or n % 32 or any(not math.isfinite(x) for x in values):
        return False
    allowed = [(i, x) for i, x in enumerate(values) if not omit_every_third or i % 3 != 2]
    best = max(x for _, x in allowed)
    earlier = max(x for i, x in allowed if i < n - 32)
    return best >= 0.8 and best - earlier <= 0.002


def late_phase_plateau_ready(scores, *, omit_every_third=True):
    """Independent longer-window predicate for intermediate later starts."""
    import math

    v = list(scores)
    n = len(v)
    if n < 256 or n % 32 or any(not math.isfinite(x) for x in v):
        return False
    eligible = [(i, x) for i, x in enumerate(v) if not omit_every_third or i % 3 != 2]
    best = max(x for _, x in eligible)
    past = max(x for i, x in eligible if i < n - 128)
    return best >= 0.8 and best - past <= 0.002
