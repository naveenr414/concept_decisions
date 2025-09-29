import numpy as np
import pandas as pd
import tqdm
import torch
import torch.nn.functional as F
from concept_abstraction.mimic_constants import *
from sklearn.model_selection._split import train_test_split

def fit_action_bins(input_amounts, vaso_doses, n_action_bins=5):
    """
    Groups the given fluid inputs and vasopressor doses into discrete bins based
    on percentile ranks. The resulting set of actions has n_action_bins *
    n_action_bins possible actions.
    
    Returns: an assignment of each patient step (same length as input_amounts 
        and vaso_doses) to an action number; a tuple of the median values of
        input and vaso for each bin; and a tuple of the bin boundaries for each
        bin that can be used in transform_actions.
    """
    bin_percentiles = np.linspace(0, 100, n_action_bins - 1, endpoint=False)

    # io = input actions from 0 to n_actions - 1, where 0 is no fluid and 1 is
    # the next smallest amount of fluid
    input_cutoffs = [0.0] + np.percentile(input_amounts[input_amounts > 0], bin_percentiles).tolist()
    io = np.digitize(input_amounts, input_cutoffs)
    median_inputs = [
        np.median(input_amounts[io == bin_num])
        for bin_num in range(1, n_action_bins + 1)
    ]
    
    # vc = vasopressor actions, same rule as input
    vaso_cutoffs = [0.0] + np.percentile(vaso_doses[vaso_doses > 0], bin_percentiles).tolist()
    vc = np.digitize(vaso_doses, vaso_cutoffs)
    median_vaso = [
        np.median(vaso_doses[vc == bin_num])
        for bin_num in range(1, n_action_bins + 1)
    ]
    
    med = np.array([io, vc])
    actions = (med[0] - 1) * n_action_bins + (med[1] - 1)
        
    return actions, (median_inputs, median_vaso), (input_cutoffs, vaso_cutoffs)
    
def transform_actions(input_amounts, vaso_doses, cutoffs):
    """
    Transforms a set of continuous fluid and vasopressor actions into discrete
    bins using the given set of cutoffs. The cutoffs are a tuple of bin
    boundaries for fluids and vasopressors, such as those produced by the last
    return value of fit_action_bins.
    """
    input_cutoffs, vaso_cutoffs = cutoffs
    return (
        len(input_cutoffs) * (np.digitize(input_amounts, input_cutoffs) - 1) +
        (np.digitize(vaso_doses, vaso_cutoffs) - 1)
    )

def shift_actions(metadata, actions):
    """
    Shifts the actions backward so that each row provides the *next* action for
    the given state. Actions for the last observed state in each trajectory
    are set to -1.
    """
    new_actions = np.concatenate([actions[1:], np.array([-1])])
    new_actions[np.argwhere(metadata[C_BLOC].values == 1).flatten() - 1] = -1
    return new_actions
    
def build_complete_record_sequences(metadata, states, actions, absorbing_states, reward_values):
    """
    Builds a dataframe of timestepped records, adding a bloc at the end of each
    record sequence for the absorbing state (discharge or death) that occurs for
    the patient.
    """
    stay_ids = metadata[C_ICUSTAYID].values
    blocs = metadata[C_BLOC].values
    outcomes = metadata[C_OUTCOME].values
    
    qldata3 = []
    for i in range(len(blocs)):
        if i == len(blocs) - 1 or blocs[i + 1] == 1: # end of trace for this patient (next bloc is 1 for the next patient)
            qldata3.append({
                C_BLOC: blocs[i],
                C_ICUSTAYID: stay_ids[i],
                C_STATE: absorbing_states[int(outcomes[i])],
                C_ACTION: -1,
                C_OUTCOME: int(outcomes[i]),
                C_REWARD: reward_values[int(outcomes[i])]
            })
        else:
            qldata3.append({
                C_BLOC: blocs[i],
                C_ICUSTAYID: stay_ids[i],
                C_STATE: states[i],
                C_ACTION: actions[i + 1],
                C_OUTCOME: outcomes[i],
                C_REWARD: 0
            })
    return pd.DataFrame(qldata3)

def compute_transition_counts(qldata3, n_states, n_actions, transition_threshold=None):
    """
    qldata3: Dataframe of blocs, states, actions, and outcomes
    n_states: Number of states (including both clustered states and absorbing states)
    n_actions: Number of actions
    transition_threshold: If not None, the number of occurrences of an (S', S, A)
        transition required to include in the transition matrix
        
    Returns: Transition counts matrix T(S', S, A).
    """
    transitionr = np.zeros([n_states, n_states, n_actions])  #this is T(S',S,A)
    
    for _, group in qldata3.groupby(C_ICUSTAYID):
        group_states = group[C_STATE].values
        group_actions = group[C_ACTION].values
        transitionr[group_states[1:], group_states[:-1], group_actions[:-1]] += 1

    # Zero out rare transitions
    if transition_threshold is not None:
        transition_sums = np.sum(transitionr, axis=0)
        print("Zeroing out {}/{} transitions".format(
            (transition_sums <= transition_threshold).sum(),
            transitionr.shape[1] * transitionr.shape[2]
        ))
        transitionr = np.where(transition_sums <= transition_threshold,
                               0, transitionr)

    return transitionr
    
def compute_physician_policy(qldata3, n_states, n_actions, absorbing_states, reward_val=100, transition_threshold=5):
    """
    Computes the physician policy based on the given set of trajectories.
    
    Returns: the physician policy as an (S, A) probability matrix; a transition
        matrix T(S', S, A); and the reward matrix R(S, A).
    """
    # Average transition counts over S, A
    transitionr = compute_transition_counts(qldata3,
                                             n_states,
                                             n_actions,
                                             transition_threshold=transition_threshold)
    action_counts = np.sum(transitionr, axis=0)
    transitionr = np.nan_to_num(np.where(transitionr > 0, transitionr / action_counts, 0))
    physpol = np.nan_to_num(action_counts / action_counts.sum(axis=1, keepdims=True))

    # CF sutton& barto bottom 1998 page 106 - compute R[S,A] from R[S'SA] and T[S'SA]
    transition_rewards = np.zeros((n_states, n_states, n_actions))
        
    transition_rewards[absorbing_states[0], :, :] = reward_val
    transition_rewards[absorbing_states[1], :, :] = -reward_val
    

    R = (transitionr * transition_rewards).sum(axis=0)
    
    return physpol, transitionr, R
    
def build_record_sequences_with_policies(qldata3, predicted_actions, physpol, n_cluster_states, soften_factor=0.01):
    """
    Creates a copy of the given qldata3 containing softened physician and model
    actions, as well as the optimal predicted action for the state.
    """
    qldata3 = qldata3.copy()
    
    n_states, n_actions = physpol.shape
    soft_physpol = physpol.copy() # behavior policy = clinicians'

    for i in range(n_cluster_states):
        ii = soft_physpol[i,:] == 0
        z = soften_factor / ii.sum()
        nz = soften_factor / (~ii).sum()
        soft_physpol[i, ii] = z
        soft_physpol[i, ~ii] = soft_physpol[i,~ii] - nz
    
    # "optimal" policy = target policy = evaluation policy
    soft_modelpol = np.ones([n_states, n_actions]) * soften_factor / (n_actions - 1)
    soft_modelpol[np.arange(n_states), predicted_actions] = 1 - soften_factor

    qldata3[C_SOFTENED_PHYSICIAN_PROBABILITY] = soft_physpol[
        qldata3[C_STATE],
        qldata3[C_ACTION]
    ]
    qldata3[C_SOFTENED_MODEL_PROBABILITY] = soft_modelpol[
        qldata3[C_STATE],
        qldata3[C_ACTION]
    ]
    qldata3[C_OPTIMAL_ACTION] = predicted_actions[qldata3[C_STATE]]
    return qldata3


### Evaluation
def _q_learn_trace(Q, trace, gamma, alpha):
    tracelength = len(trace)
    return_t = trace[tracelength - 1, 0] # get last reward as return for penultimate state and action.

    for t in range(tracelength - 2, -1, -1):       #Step through time-steps in reverse order
        s = trace[t, 1] # get state index from trace at time t
        a = trace[t, 2] # get action index
        if a >= 0:
            Q[s, a] = (1 - alpha) * Q[s, a] + alpha * return_t # update Q.
        return_t = return_t * gamma + trace[t, 0] # return for time t-1 in terms of return and reward at t
    return Q


def off_policy_q_learning(traces, n_states, n_actions, gamma, alpha, num_traces):
    # OFF POLICY Q LEARNING

    sumQ = []
    Q = np.zeros((n_states, n_actions))
    maxavgQ = 1
    check_step = 50000
    
    stay_ids = list(traces.keys())
    
    for j in range(num_traces):
        stay_id = stay_ids[np.random.choice(len(stay_ids))]
        trace = traces[stay_id]
        Q = _q_learn_trace(Q, trace, gamma, alpha)
        sumQ.append(Q.sum())
        
        if j > 0 and j % check_step == 0:  #check if can stop iterating (when no more improvement is seen)
            s = np.mean(sumQ[-check_step:])
            d = (s - maxavgQ) / maxavgQ
            if abs(d) < 0.001:
                break
            maxavgQ = s

    return Q, np.array(sumQ)


def evaluate_physician_policy_td(qldata3, physpol, gamma, num_iter, n_cluster_states, alpha=0.1, num_traces=300000):
    """
    Performs off-policy evaluation on the physician policy.
    """
    # V value averaged over state population
    # hence the difference with mean(V) stored in recqvi(:,3)

    n_states, n_actions = physpol.shape
    bootql = np.zeros(num_iter)
    a = qldata3.loc[qldata3[C_BLOC] == 1, C_STATE].values
    initial_state_dist = np.array([(a == i).sum() for i in range(n_cluster_states)])

    # Changed the implementation to sample from all patients
    traces = {
        stay_id: trace[[C_REWARD, C_STATE, C_ACTION]].values
        for stay_id, trace in qldata3.groupby(C_ICUSTAYID)
    }
    
    for i in tqdm.tqdm(range(num_iter), desc='TD evaluation'):
        Qoff, _ = off_policy_q_learning(
            traces,
            n_states, n_actions,
            gamma, alpha, num_traces)

        # Value is sum of Q values over each action
        V = (physpol * Qoff).sum(axis=1)
        bootql[i] = np.nansum(V[:n_cluster_states] * initial_state_dist) / initial_state_dist.sum()

    return bootql

def compute_wis_estimator(sequences, gamma):
    """
    Performs off-policy evaluation on the predicted policy using weighted
    importance sampling (WIS). For each trajectory, if the reward is provided at
    a terminal state, this state *must* be included as one of the values in each
    of the four array inputs.
    """
    # WIS estimator of AI policy
    # Thanks to Omer Gottesman (Harvard) for his assistance

    # Compute the cumulative importance ratio rho for each trajectory
    def compute_rho(trajectory):
        physician_probs = trajectory[:, 0]
        model_probs = trajectory[:, 1]
        return np.prod(model_probs / physician_probs)
    
    rho_array = np.array([compute_rho(trace) for trace in sequences])
    num_nonzero_rhos = (rho_array > 0).sum()
    ess = (rho_array.sum() ** 2) / np.sum(rho_array ** 2)
    rho = np.array(rho_array, dtype=float)
    rho = np.nan_to_num(rho, nan=0.0, posinf=0.0, neginf=0.0)

    # ESS (unnormalized form)
    ess = (rho.sum()**2) / (np.sum(rho**2) + 1e-30)
    normalization = np.nansum(rho_array)

    # Compute the individual trial estimators V_WIS, which are the discounted
    # rewards over each trajectory weighted by rho
    def compute_trial_estimator(trajectory):
        rewards = trajectory[1:, 2]
        discounts = gamma ** np.arange(-1, len(rewards) - 1)
        return np.sum(discounts * rewards)
        
    individual_trial_estimators = np.array([compute_trial_estimator(trace) for trace in sequences])

    # Normalize by rhos
    bootwis = np.nansum(individual_trial_estimators * rho_array) / normalization

    return bootwis, num_nonzero_rhos, individual_trial_estimators

def evaluate_policy_wis(metadata, physician_probabilities, model_probabilities, reward_vals, gamma, num_iter):
    """
    Computes a bootstrapped WIS estimator 
    """
    bootwis = np.zeros(num_iter)
    stay_ids = metadata[C_ICUSTAYID].values
    assert len(metadata) == len(physician_probabilities) == len(model_probabilities), "Mismatched lengths"
    
    p = np.unique(stay_ids)
    num_patients = min(25000, int(len(p) * 0.75))  # 25000 patients or 75% of samples, whichever is less

    metadata = metadata.copy()
    metadata[C_SOFTENED_PHYSICIAN_PROBABILITY] = physician_probabilities
    metadata[C_SOFTENED_MODEL_PROBABILITY] = model_probabilities
    def _make_wis_record(group):
        # Structure:
        # phys_prob, model_prob, reward (= 0)
        # plus terminal state with reward from reward_vals
        return np.vstack([
            np.hstack([
                group[[C_SOFTENED_PHYSICIAN_PROBABILITY,
                       C_SOFTENED_MODEL_PROBABILITY]],
                np.zeros((len(group), 1))]),
            np.array([1, 1, reward_vals[group[C_OUTCOME].iloc[0]]])
        ])
    
    traces = {
        stay_id: _make_wis_record(trace)
        for stay_id, trace in metadata.groupby(C_ICUSTAYID)
    }
    
    for jj in tqdm.tqdm(range(num_iter), desc='WIS estimation'):
        # Sample the population
        sample_ids = np.random.choice(p, size=num_patients, replace=False)
        
        # Compute WIS estimator
        bw, num_nonzero_rhos, individual_trial_estimators = compute_wis_estimator(
            [traces[pid] for pid in sample_ids],
            gamma
        )
        bootwis[jj] = bw

    return bootwis, num_nonzero_rhos, individual_trial_estimators

def compute_physician_probabilities(physician_policy,n_cluster_states,X=None, states=None, actions=None, soften=True,soften_factor=0.01):
    """
    Returns the probabilities for each state and action according to the
    physician policy, which is learned using the same state clustering model
    as the AI Clinician.
    
    If actions is provided, any values where the action is -1 will result in
    a zero probability.
    """
    assert X is not None or states is not None, "At least one of states or X must not be None"
    if soften:
        soft_physpol = physician_policy.copy() # behavior policy = clinicians'

        for i in range(n_cluster_states):
            ii = soft_physpol[i,:] == 0
            z = soften_factor / ii.sum()
            coef = soft_physpol[i,~ii].sum()
            soft_physpol[i, ii] = z
            soft_physpol[i, ~ii] = soft_physpol[i,~ii] * (1 - soften_factor / coef)
    else:
        soft_physpol = physician_policy

    assert states is not None 
    probs = soft_physpol[states]
    if actions is not None:
        # Add a column of zeros at the end for the '-1' action (no action recorded)
        return np.hstack([probs, np.zeros(probs.shape[0]).reshape(-1, 1)])[np.arange(len(actions)), actions]
    return probs

def compute_Q(self, X=None, states=None, actions=None):
    assert X is not None or states is not None, "At least one of states or X must not be None"
    if states is None:
        states = self.compute_states(X)

    if actions is not None:
        assert len(actions) == len(states)
        # Add a column of zeros at the end for the '-1' action (no action recorded)
        return np.hstack([self.Q, np.zeros(self.Q.shape[0]).reshape(-1, 1)])[states, actions]
    return self.Q[states]


def compute_model_probabilities(model, concept_list, states=None, X=None, actions=None, soften_factor=0.01):
    """
    Returns the probability of each action under the model's policy for given states.
    If actions is provided, returns the probability of the action actually taken.
    """
    assert X is not None or states is not None, "At least one of states or X must not be None"

    assert states is not None 
    # Get the action probabilities from the model
    # stable-baselines3 stores logits in the policy
    obs_tensor = torch.Tensor([[concept(s) for concept in concept_list] for s in states]).to(model.device)
    with torch.no_grad():
        # get distribution for actions
        if not hasattr(model,"policy"):
            action_probs = np.array([[1/25 for i in range(25)] for _ in states])
        else:
            dist = model.policy.get_distribution(obs_tensor.to(model.device)) 
            action_probs = F.softmax(dist.distribution.logits).cpu().numpy()
    # Optional: soften the distribution (like AI Clinician)
    if soften_factor > 0:
        action_probs = action_probs * (1 - soften_factor)
        action_probs += soften_factor / action_probs.shape[1]

    if actions is not None:
        # pick the probability of the action actually taken
        return action_probs[np.arange(len(actions)), actions]

    return action_probs

def get_mimic_eval_info(additional_info,seed):
    c = additional_info['concept_list']
    clusterer = additional_info['clusterer']
    physpol = additional_info['physpol']
    MIMICraw = pd.read_csv("../../data/mimic_github/ai_clinician/data/mimic_model/train/MIMICraw.csv")
    metadata = pd.read_csv("../../data/mimic_github/ai_clinician/data/mimic_model/train/metadata.csv")
    MIMICzs = pd.read_csv("../../data/mimic_github/ai_clinician/data/mimic_model/train/MIMICzs.csv")

    C_ICUSTAYID = "icustayid"
    unique_icu_stays = metadata[C_ICUSTAYID].unique()

    n_action_bins = 5
    gamma = 0.99

    # Actions actually taken 
    all_actions, _, _ = fit_action_bins(
        MIMICraw[C_INPUT_STEP],
        MIMICraw[C_MAX_DOSE_VASO],
        n_action_bins=n_action_bins
    )

    train_ids, val_ids = train_test_split(unique_icu_stays, test_size=0.1,random_state=seed)
    train_indexes = metadata[metadata[C_ICUSTAYID].isin(train_ids)].index
    val_indexes = metadata[metadata[C_ICUSTAYID].isin(val_ids)].index

    X_train = MIMICzs.iloc[train_indexes]
    X_val = MIMICzs.iloc[val_indexes]

    metadata_train = metadata.iloc[train_indexes]
    metadata_val = metadata.iloc[val_indexes]

    actions_train = all_actions[train_indexes]
    actions_val  = all_actions[val_indexes]
    states_train = clusterer.predict(X_train.values)
    states_val   = clusterer.predict(X_val.values)
    trajectories = metadata_val[C_ICUSTAYID].values

    SOFA_COL = "SOFA"        # <- replace with actual SOFA column name in MIMICraw
    LACTATE_COL = "Arterial_lactate"  # <- replace with actual lactate column name in MIMICraw

    C0 = -0.025
    C1 = -0.125
    C2 = -2.0

    # scaling factor in case shaped rewards are too small/large relative to terminal rewards
    SHAPE_SCALE = 1.0  # tune this (e.g., 0.5, 2.0) after you inspect distributions

    # -------------------------
    # 2. prepare aligned arrays / dataframe
    # -------------------------
    # train_indexes already defined earlier in your snippet
    # metadata_train, states_train, actions_train are aligned with train_indexes

    # choose where to pull raw SOFA/lactate values from - I use MIMICraw here.
    sofa_vals = MIMICraw[SOFA_COL].iloc[val_indexes].astype(float).values
    lactate_vals = MIMICraw[LACTATE_COL].iloc[val_indexes].astype(float).values

    # Build a helper dataframe that preserves the exact order used to create states_train/actions_train
    df_steps = pd.DataFrame({
        "icustayid": metadata_val[C_ICUSTAYID].values,
        "state": states_val,
        "action": actions_val,
        "sofa": sofa_vals,
        "lactate": lactate_vals,
    })

    n_cluster_states = np.max(states_train)+1
    absorbing_states =  [n_cluster_states + 1, n_cluster_states]
    n_rows = len(df_steps)
    step_rewards = np.zeros(n_rows, dtype=float)

    qldata3_shaped = []
    idx = 0
    for icu, grp in df_steps.groupby("icustayid"):
        pos = grp.index.values
        for i_idx, p in enumerate(pos):
            s = int(df_steps.loc[p, "state"])
            a = int(df_steps.loc[p, "action"])
            if i_idx < len(pos) - 1:
                next_p = pos[i_idx + 1]
                s_next = int(df_steps.loc[next_p, "state"])
                sofa_seq = sofa_vals[idx]
                sofa_next = sofa_vals[idx+1]
                lactate_seq = lactate_vals[idx]
                lactate_next = lactate_vals[idx+1]

                indicator = ((sofa_seq == sofa_next) & (sofa_seq > 0)).astype(float)

                # diffs: sofa_t - sofa_{t+1}, lactate_t - lactate_{t+1}
                diff_sofa = sofa_seq - sofa_next
                diff_lactate = lactate_seq - lactate_next

                # replace NaNs (last pos) with 0 for computing shaped contribution — last step will be terminal reward
                diff_sofa = np.nan_to_num(diff_sofa, nan=0.0)
                diff_lactate = np.nan_to_num(diff_lactate, nan=0.0)

                r = C0 * indicator + C1 * diff_sofa + C2 * np.tanh(diff_lactate)
                r = r * SHAPE_SCALE
                done = False
            else:
                # last step in stay -> treat as terminal transition to absorbing state
                # choose which absorbing-state id you want (your script used two absorbing states: [n_cluster_states + 1, n_cluster_states])
                # Here I set next state to absorbing_states[1] (adjust if your convention differs)
                s_next = absorbing_states[1]
                # terminal reward: keep your previous terminal rewards array (rewards variable)
                # but you must map actual mortality flag to decide +15 or -15. Replace MORT_COL below with your outcome column.
                MORT_COL = "outcome" 
                try:
                    death_flag = int(metadata_val.iloc[p][MORT_COL])
                    # death_flag==1 => negative terminal reward; death_flag==0 => positive terminal reward
                    if death_flag == 1:
                        r = float(-15)  # -15 (example)
                    else:
                        r = float(15)  # +15 (example)
                except Exception:
                    # fallback: use the positive terminal reward (change as you see fit)
                    r = float(15)
                done = True

            # Record format: (icustayid, pos, state, action, reward, next_state, done)
            qldata3_shaped.append((int(df_steps.loc[p, "icustayid"]), int(p), s, a, r, int(s_next), done))
            idx += 1
        rewards = np.array([i[4] for i in qldata3_shaped])
    phys_probs = compute_physician_probabilities(additional_info['physpol'],np.max(states_train)+1,states=states_val, actions=actions_val)

    return states_val,actions_val,trajectories,rewards,phys_probs