import numpy as np
from simglucose.envs import T1DSimGymnaisumEnv
import gymnasium.spaces as spaces

def glucose_risk(bg):
    """
    Compute risk index from mg/dL glucose.
    Normalized to be more RL-friendly.
    """
    # Target range: 70-180 mg/dL
    if 70 <= bg <= 180:
        return 0.0  # Perfect range
    elif bg < 70:
        # Penalize hypoglycemia more severely
        return ((70 - bg) / 70) ** 2 * 10
    else:
        # Penalize hyperglycemia
        return ((bg - 180) / 180) ** 2 * 5
def compute_reward(bg, dose, delta_bg, action):
    """
    Intermediate reward function for glucose control:
    - Encourages BG in range [80, 140]
    - Mildly penalizes insulin usage
    - Mildly penalizes fast BG changes
    - Gives action-dependent small bonus
    """
    # --- 1. Glucose range reward ---
    if 80 <= bg <= 140:
        r_glucose = 1.0  # in range
    elif 60 <= bg < 80 or 140 < bg <= 180:
        r_glucose = 0.0  # slightly out of range
    else:
        r_glucose = -1.0  # dangerous range

    # --- 2. Insulin penalty ---
    r_insulin = -0.01 * dose  # discourage over-dosing

    # --- 3. BG trend penalty ---
    r_trend = -0.001 * abs(delta_bg)  # mild penalty for rapid changes

    # --- 4. Small bonus for “safe” action choices ---
    # Example: action 0 is “no insulin”, action 1 is “some insulin”, etc.
    r_action = 0.1 if action == 0 else 0.0

    # --- Total reward ---
    total_reward = r_glucose + r_insulin + r_trend + 0*r_action
    return total_reward

def compute_delta(bg_history):
    if len(bg_history) < 2:
        return 0.0
    return bg_history[-1] - bg_history[-2]


def compute_iob(past_doses, tau=50):
    """
    Simple exponential decay IOB model.
    tau ~ duration of insulin activity (50-60 min)
    """
    if len(past_doses) == 0:
        return 0.0

    iob = 0
    for dose, minutes_ago in past_doses:
        decay = np.exp(-minutes_ago / tau)
        iob += dose * decay

    return iob


def time_features(dt):
    """
    Encode time of day with sin/cos.
    """
    minutes = dt.hour * 60 + dt.minute
    angle = 2 * np.pi * minutes / (24*60)
    return np.sin(angle), np.cos(angle)

def build_observation(info, bg_history, past_doses):
    bg = info["bg"]
    meal = info["meal"]
    dt = info["time"]
    
    # Normalize features for better learning
    bg_norm = bg / 200.0  # Normalize around typical range
    delta_bg = compute_delta(bg_history) / 50.0  # Normalize delta
    meal_norm = meal / 100.0  # Normalize meal size
    iob = compute_iob(past_doses)
    sin_t, cos_t = time_features(dt)
    
    return np.array([bg_norm, delta_bg, meal_norm, iob, sin_t, cos_t], dtype=np.float32)

NUM_ACTIONS = 6

def action_to_dose(action):
    action_to_dose_dict = {
        0: 0,
        1: 2,
        2: 4,
        3: 6,
        4: 8,
        5: 10,
    }

    return action_to_dose_dict[action]
class GlucoseEnvironment(T1DSimGymnaisumEnv):
    def __init__(self, patient_name="adolescent#002", **kwargs):
        super().__init__(patient_name=patient_name, **kwargs)

        # Track BG + dose history
        self.bg_history = []
        self.past_doses = []   # list of (dose, minutes_ago)
        self.episode_rewards = []  # Track rewards
        # Observation space: 6 normalized features
        low = np.array([0, -5, 0, 0, -1, -1], dtype=np.float32)
        high = np.array([3, 5, 2, 20, 1, 1], dtype=np.float32)
        self.observation_space = spaces.Box(low=low, high=high)
        
        # Discrete actions
        self.action_space = spaces.Discrete(NUM_ACTIONS)
        
        # 3 minutes per step (from simulator)
        self.step_minutes = 5


    # ----------------------
    #        Reset
    # ----------------------
    def reset(self, *, seed=None, options=None):
        obs_raw, info = super().reset(seed=seed, options=options)
        
        # Reset histories
        self.bg_history = [info["bg"]]
        self.past_doses = []
        
        # Build observation vector
        obs = build_observation(info, self.bg_history, self.past_doses)
        return obs, info
    


    # ----------------------
    #         Step
    # ----------------------
    def step(self, action):

        # Convert discrete → dose (U)
        dose = action_to_dose(action)
        
        # Step simulator
        obs_raw, _, terminated, truncated, info = super().step(dose)
        
        # Update time counters for doses
        for i in range(len(self.past_doses)):
            d, t = self.past_doses[i]
            self.past_doses[i] = (d, t + self.step_minutes)
        
        # Add current dose
        if dose > 0:
            self.past_doses.append((dose, 0))
        
        # Update BG history
        bg = info["bg"]
        self.bg_history.append(bg)
        if len(self.bg_history) > 10:  # Keep more history for smoothing
            self.bg_history.pop(0)
        
        # Compute features
        delta_bg = compute_delta(self.bg_history)
        
        # FIXED: Correct argument order!
        reward = compute_reward(bg, dose, delta_bg,action)
        self.episode_rewards.append(reward)
        # Build final observation
        obs = build_observation(info, self.bg_history, self.past_doses)
        
        return obs, reward, terminated, truncated, info

