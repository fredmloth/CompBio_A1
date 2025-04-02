import numpy as np 
from scipy import curvefit 
# region type 1
def type1kinetics(S1, S2, Vmax, Km1, Km2):
    return (Vmax * S1 * S2) / (Km1 * Km2 + Km2 * S1 + Km1 * S2 + S1 * S2)

# region type 2
def type1kinetics(S1, S2, Vmax, Km1, Km2):
    return (Vmax * S1 * S2) / (Km1 * S2 + Km2 * S1 + S1 * S2)


