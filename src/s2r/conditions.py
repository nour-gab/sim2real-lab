"""Target conditions. 'real' is the hidden proxy for hardware."""
from .envs import Physics

TARGETS = {
    "nominal":  Physics(),
    "heavy":    Physics(mass=1.4),
    "slippery": Physics(friction=0.6),
    "weak":     Physics(gain=0.75),
    "delayed":  Physics(delay=2),
    "noisy":    Physics(obs_noise=0.02),
    "real":     Physics(mass=1.25, friction=0.7, damping=1.3, gain=0.8,
                        delay=1, obs_noise=0.01),
}
# Same as 'real' but without delay/noise: what system identification can recover.
REAL_IDENTIFIABLE = Physics(mass=1.25, friction=0.7, damping=1.3, gain=0.8)
