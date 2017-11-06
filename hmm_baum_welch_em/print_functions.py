def print_trans_emission_matrices(transition_matrix,emission_matrix):
    print "transition matrix"
    print transition_matrix
    print "emission matrix"
    print emission_matrix

def print_forward_backward_probabilities(forward_prob_list,backward_prob_list):
    print "forward"
    for item in forward_prob_list:
        print item
    print "backward"
    for item in backward_prob_list:
        print item
        
def print_gamma_delta_probabilities(gamma_prob_list,delta_prob_list):
    print "gamma"
    for item in gamma_prob_list:
        print item
    print "delta"
    for item in delta_prob_list:
        print item
