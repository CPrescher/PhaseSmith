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
