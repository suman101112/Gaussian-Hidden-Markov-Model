def initialize_transition_matrix(transition_matrix):
    
    transition_matrix[0][0] = 0.7
    transition_matrix[0][1] = 0.3
    transition_matrix[1][0] = 0.3
    transition_matrix[1][1] = 0.7
    
    return transition_matrix
    
def initialize_emission_matrix(emission_matrix):
    
    emission_matrix[0][0] = 0.9
    emission_matrix[0][1] = 0.1
    emission_matrix[1][0] = 0.2
    emission_matrix[1][1] = 0.8
    
    return emission_matrix

def initialize_initial_probabilities(initial_probabilities):
    
    initial_probabilities[0] = 0.5
    initial_probabilities[1] = 0.5
    
    return initial_probabilities
