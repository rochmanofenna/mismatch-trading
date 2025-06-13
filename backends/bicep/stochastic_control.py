# ── stochastic_control.py ─────────────────────────────────────────────
import numpy as np
import logging
from functools import lru_cache

# ---------------------------------------------------------------------
#  logging
# ---------------------------------------------------------------------
logging.basicConfig(
    filename="results/logs/stochastic_control.log",
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)

# ---------------------------------------------------------------------
#  hyper-parameters
# ---------------------------------------------------------------------
HIGH_THRESHOLD = 10
LOW_THRESHOLD = 2
DECAY_RATE = 0.1
FEEDBACK_WARNING_THRESHOLD = 0.9
ADAPTIVE_ADJUSTMENT = 0.05
MAX_HIGH_THRESHOLD = 100
DIAGNOSTIC_LOGGING = True


def _log(msg):
    if DIAGNOSTIC_LOGGING:
        logging.info(msg)


# ---------------------------------------------------------------------
#  stochastic controls
# ---------------------------------------------------------------------
def adjust_variance(base_var, factor=1.0):
    new = base_var * factor
    _log(f"variance {base_var}→{new} (×{factor})")
    return new


@lru_cache(maxsize=128)
def adaptive_randomness_control(x, feedback, rng=(0.2, 1.0)):
    if feedback >= FEEDBACK_WARNING_THRESHOLD:
        logging.warning("feedback %.3f ≥ %.2f", feedback, FEEDBACK_WARNING_THRESHOLD)
    scale = np.clip(0.5 + 0.5 * feedback, *rng)
    return x * scale


def control_randomness_by_state(cnt, total):
    if total <= 0:
        raise ValueError("total_steps must be > 0")
    norm = cnt / total

    global HIGH_THRESHOLD
    if norm > 0.8 * HIGH_THRESHOLD and HIGH_THRESHOLD < MAX_HIGH_THRESHOLD:
        HIGH_THRESHOLD = min(MAX_HIGH_THRESHOLD, HIGH_THRESHOLD + ADAPTIVE_ADJUSTMENT)

    if norm < LOW_THRESHOLD:
        return 1.5
    if norm > HIGH_THRESHOLD:
        return 0.5
    return 1.0


def combined_variance_control(cnt, total, t, base_var=1.0):
    return (
        base_var
        * control_randomness_by_state(cnt, total)
        * np.exp(-DECAY_RATE * t)
    )


def apply_stochastic_controls(x):
    """Light-weight control used in unit-tests."""
    if np.abs(x) > 4:
        x *= 0.5
    elif np.abs(x) < 0.1:
        x *= 2.0
    return x
