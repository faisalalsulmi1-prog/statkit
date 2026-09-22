"""Number / statistic formatting for sentences and the report (PLAN §6).

Pure functions, no state. These are the single place rounding and APA style
live, so the sentence layer and the report layer can never drift apart.

Rules (PLAN §6, §9.1):
  p(x)   -> "p < .001" or "p = .032"  (no leading zero, never "p = .000")
  num(x) -> int-valued floats print as ints; otherwise d decimals
  ci     -> "95% CI [1.20, 3.40]"     (fixed 2 decimals, leading zero kept)
  pct    -> proportion in [0, 1] -> "52.3%"
  stat   -> "t = 2.31"

A non-finite input (inf, -inf, nan) can never raise and never leaks a bare
"inf"/"nan": every numeric slot renders as the em dash below. The real app
reaches these when a statistic is not estimable -- an empty 2x2 cell gives
Fisher's exact an infinite odds ratio, a perfect-fit regression gives Cohen's
f2 = inf. The contextual wording ("not estimable because ...") is the sentence
layer's job; here the formatters only stay crash-proof and clean.
"""
import math

NON_FINITE = "—"  # em dash: the deterministic render for inf/-inf/nan


def p(x: float) -> str:
    """APA p-value. Anything below .001 collapses to 'p < .001' (so a raw 0.0
    or 0.0004 never prints the meaningless 'p = .000'); otherwise three
    decimals with the leading zero dropped. A non-finite p (a guard -- a real
    p is always finite) renders as the em dash, never 'p = nan' and never the
    misleading 'p < .001' that -inf would otherwise hit."""
    x = float(x)
    if not math.isfinite(x):
        return NON_FINITE
    if x < 0.001:
        return "p < .001"
    s = f"{x:.3f}"
    if s.startswith("0."):
        s = s[1:]  # ".032", not "0.032"
    return f"p = {s}"


def num(x: float, d: int = 2) -> str:
    """A number for prose/tables: an int-valued float prints as an int ('2'),
    anything else with exactly d decimals ('2.35', '1.20'). Negative zero is
    normalised to '0'. A non-finite value renders as the em dash instead of
    raising (int(inf)/int(nan) would blow up)."""
    x = float(x)
    if not math.isfinite(x):
        return NON_FINITE
    r = round(x, d) + 0.0  # + 0.0 turns -0.0 into 0.0
    if r == int(r):
        return str(int(r))
    return f"{r:.{d}f}"


def ci(lo: float, hi: float, level: int = 95) -> str:
    """A confidence-interval string, fixed two decimals on each bound. Leading
    zeros are kept (a CI can hold a mean difference that exceeds 1). A bound
    that is non-finite (e.g. Fisher's infinite odds-ratio upper bound) renders
    as the em dash; the finite bound still prints normally."""
    lo_s = f"{lo:.2f}" if math.isfinite(lo) else NON_FINITE
    hi_s = f"{hi:.2f}" if math.isfinite(hi) else NON_FINITE
    return f"{level}% CI [{lo_s}, {hi_s}]"


def pct(x: float, d: int = 1) -> str:
    """A proportion in [0, 1] rendered as a percentage: pct(0.523) -> '52.3%'.
    A non-finite proportion renders as the em dash, never a bare 'nan%'."""
    if not math.isfinite(x):
        return NON_FINITE
    return f"{x * 100:.{d}f}%"


def stat(name: str, val: float, d: int = 2) -> str:
    """A named test statistic: stat('t', 2.31) -> 't = 2.31'."""
    return f"{name} = {num(val, d)}"
