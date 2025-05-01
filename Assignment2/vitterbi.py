import numpy as np 
                #   Exon, Intron
transition_matrix = [[0.9, 0.1], # Exon
                     [0.2,0.8]] # Intron
                #   A,    U,    G,    C
emission_matrix = [[0.25, 0.25, 0.25, 0.25], # Exon
                   [0.4, 0.4, 0.05, 0.15]] # Intron

Initial_probabilities = [0.5, 0.5, 0, 0] # Exon, Intron

# NOTE: STATES AND OBSERVATIONS NEED TO BE ENUMERATED
# A: 0, U: 1, G: 2, C: 3
def Viterbi(states, init, transition, emission, obs):
    prob = np.zeros((len(obs), len(states)))
    prev = np.empty((len(obs), len(states)))
    for s in states:
        prob[0][s] = init[s] * emission[s][obs[0]]
    
    for t in range(1, len(obs)):
        for s in states:
            for r in states:
                new_prob = prob[t - 1][r] * transition[r][s] * emission[s][obs(t)]
                if new_prob > prob[t][s]:
                    prob[t][s] = new_prob
                    prev[t][s] = r
                    
    path = np.empty((len(obs)))
    path[(len(obs)-1)] = np.argmax(prob[len(obs)-1])
    for t in range(len(obs)-2, 0):
        path[t] = prev[t + 1][path[t + 1]]
        
    return path

states = [0,1,2,3]
obs = [0,1,1,0,1]

print(Viterbi(states,Initial_probabilities,
              transition_matrix,emission_matrix, obs))