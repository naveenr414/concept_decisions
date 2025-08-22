def cyclic_concept_mod(i):
    def get_concept(state):
        return int((state+1)%i == 0)
    return get_concept 

def get_all_cyclic_concepts(env_nodes):
    return [cyclic_concept_mod(i) for i in range(2,env_nodes+1)]