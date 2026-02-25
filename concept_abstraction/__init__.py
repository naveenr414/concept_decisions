"""concept_abstraction: Decision-Relevant Concept Selection for RL.

Quick start
-----------
    import concept_abstraction as ca

    # Ground-truth (perfect) concepts
    idx = ca.DRS(policy, concepts, env, k=5)

    # Learned (imperfect) concept predictor
    predictor, acc_list = ca.train_concept_predictor(env, policy, concepts)
    idx = ca.DRS_log(policy, concepts, env, k=5, acc_list=acc_list)

    # Fast baselines (no Gurobi required)
    idx = ca.variance(policy, concepts, env, k=5)
    idx = ca.random(concepts, k=5)

Each function returns a sorted list of integer indices into `concepts`.
"""

from concept_abstraction._api import DRS, DRS_log, variance, random, train_concept_predictor

__all__ = ["DRS", "DRS_log", "variance", "random", "greedy", "train_concept_predictor"]