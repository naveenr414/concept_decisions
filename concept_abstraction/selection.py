import numpy as np

def greedy_selection(env,num_concepts):
    total_concepts = []
    all_concepts = env.concepts
    reward_by_state = env.rewards
    groups = [0 for i in range(len(reward_by_state))]

    for _ in range(num_concepts):
        scores_by_group = []
        for k in range(len(all_concepts)):
            if k in total_concepts:
                scores_by_group.append(np.inf)
                continue   
            total_groups = max(groups)
            total_score = 0
            for g in range(total_groups+1):
                states_in_group = [i for i in range(len(groups)) if groups[i] == g]
                partition_0 = [i for i in states_in_group if env.concepts[k][i] == 0]
                rewards_0 = [reward_by_state[i] for i in partition_0]
                partition_1 = [i for i in states_in_group if env.concepts[k][i] == 1]
                rewards_1 = [reward_by_state[i] for i in partition_1]

                if len(rewards_0) > 0:
                    total_score = max(max(rewards_0)-min(rewards_0),total_score)
                if len(rewards_1) > 0:
                    total_score = max(max(rewards_1)-min(rewards_1),total_score)
            scores_by_group.append(total_score)
        selected_idx = np.argmin(scores_by_group)

        total_groups = max(groups)
        new_groups = max(groups)
        for g in range(total_groups+1):
            states_in_group = [i for i in range(len(groups))  if groups[i] == g]
            partition_0 = [i for i in states_in_group if env.concepts[selected_idx][i] == 0]
            partition_1 = [i for i in states_in_group if env.concepts[selected_idx][i] == 1]
            if len(partition_0) > 0 and len(partition_1) > 0:
                for i in partition_1:
                    groups[i] = new_groups + 1
                new_groups += 1
        total_concepts.append(selected_idx)
    return total_concepts 