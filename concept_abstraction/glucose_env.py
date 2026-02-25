import numpy as np
from simglucose.envs import T1DSimGymnaisumEnv
import gymnasium.spaces as spaces


def compute_reward(bg, dose, delta_bg):
    """Intermediate reward for glucose control.

    - +1 if BG in [80, 140], 0 in mild range, -1 in dangerous range
    - Small penalty for insulin dose
    - Small penalty for rapid BG change

    Args:
        bg: Current blood glucose (mg/dL)
        dose: Insulin dose administered (U)
        delta_bg: Change in BG since last step

    Returns:
        Total reward (float)
    """
    if 80 <= bg <= 140:
        r_glucose = 1.0
    elif 60 <= bg < 80 or 140 < bg <= 180:
        r_glucose = 0.0
    else:
        r_glucose = -1.0

    r_insulin = -0.01 * dose
    r_trend   = -0.001 * abs(delta_bg)

    return r_glucose + r_insulin + r_trend


def compute_delta(bg_history):
    if len(bg_history) < 2:
        return 0.0
    return bg_history[-1] - bg_history[-2]


def compute_iob(past_doses, tau=50):
    """Exponential decay insulin-on-board model.

    Args:
        past_doses: List of (dose, minutes_ago) tuples
        tau: Insulin activity half-life in minutes (~50-60 min)

    Returns:
        Estimated IOB (float)
    """
    iob = 0.0
    for dose, minutes_ago in past_doses:
        iob += dose * np.exp(-minutes_ago / tau)
    return iob


def time_features(dt):
    """Encode time of day as sin/cos pair.

    Args:
        dt: datetime object

    Returns:
        (sin, cos) tuple
    """
    minutes = dt.hour * 60 + dt.minute
    angle   = 2 * np.pi * minutes / (24 * 60)
    return np.sin(angle), np.cos(angle)


def build_observation(info, bg_history, past_doses):
    """Build the 6-dimensional normalised observation vector.

    Features: [bg_norm, delta_bg, meal_norm, iob, sin_time, cos_time]

    Args:
        info: Simulator info dict with keys 'bg', 'meal', 'time'
        bg_history: List of recent BG readings
        past_doses: List of (dose, minutes_ago) tuples

    Returns:
        np.ndarray of shape (6,), dtype float32
    """
    bg   = info["bg"]
    meal = info["meal"]
    dt   = info["time"]

    bg_norm   = bg / 200.0
    delta_bg  = compute_delta(bg_history) / 50.0
    meal_norm = meal / 100.0
    iob       = compute_iob(past_doses)
    sin_t, cos_t = time_features(dt)

    return np.array([bg_norm, delta_bg, meal_norm, iob, sin_t, cos_t], dtype=np.float32)


NUM_ACTIONS = 6

_ACTION_TO_DOSE = {0: 0, 1: 2, 2: 4, 3: 6, 4: 8, 5: 10}


def action_to_dose(action):
    return _ACTION_TO_DOSE[action]


class GlucoseEnvironment(T1DSimGymnaisumEnv):
    """Discrete-action glucose control environment.

    Wraps the simglucose T1D simulator with:
    - 6 discrete insulin dose levels (0–10 U)
    - Normalised 6-feature observation vector
    - Shaped intermediate reward function

    Args:
        patient_name: Simglucose patient ID, e.g. 'adolescent#002'
    """

    def __init__(self, patient_name="adolescent#002", **kwargs):
        super().__init__(patient_name=patient_name, **kwargs)

        self.bg_history  = []
        self.past_doses  = []   # list of (dose, minutes_ago)

        # Observation space: 6 normalised features
        self.observation_space = spaces.Box(
            low =np.array([0, -5,  0,  0, -1, -1], dtype=np.float32),
            high=np.array([3,  5,  2, 20,  1,  1], dtype=np.float32),
        )
        self.action_space = spaces.Discrete(NUM_ACTIONS)
        self.step_minutes = 5   # 5 minutes per simulator step

    def reset(self, *, seed=None, options=None):
        obs_raw, info = super().reset(seed=seed, options=options)
        self.bg_history = [info["bg"]]
        self.past_doses = []
        return build_observation(info, self.bg_history, self.past_doses), info

    def step(self, action):
        dose = action_to_dose(action)

        obs_raw, _, terminated, truncated, info = super().step(dose)

        # Age all previous doses by one step
        self.past_doses = [(d, t + self.step_minutes) for d, t in self.past_doses]
        if dose > 0:
            self.past_doses.append((dose, 0))

        bg = info["bg"]
        self.bg_history.append(bg)
        if len(self.bg_history) > 10:
            self.bg_history.pop(0)

        delta_bg = compute_delta(self.bg_history)
        reward   = compute_reward(bg, dose, delta_bg)
        obs      = build_observation(info, self.bg_history, self.past_doses)

        return obs, reward, terminated, truncated, info