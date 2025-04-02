import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

# region 1.1 
file_path = '/home/matijs/CLS/CompBio/Assignment1/Kinetics.csv'
data = pd.read_csv(file_path)

S1 = data['S1'].values
S2 = data['S2'].values
Rate = data['Rate'].values

unique_S2 = np.unique(S2)
split_data = {s2: {'S1': S1[S2 == s2], 'Rate': Rate[S2 == s2]} for s2 in unique_S2}

plt.scatter(y = Rate, x = S1, label='Experimental Data', color='red')
plt.xlabel('Substrate 1 Concentration (1/mM)')
plt.ylabel('Reaction Rate (mM/s)')
plt.legend()
plt.title('Two-Substrate Kinetics')
plt.show()

for s2, values in split_data.items():
    S1_inv = 1 / values['S1']
    Rate_inv = 1 / values['Rate']
           
    coeffs = np.polyfit(S1_inv, Rate_inv, 1)
    fit_line = np.poly1d(coeffs)
    
    plt.plot(S1_inv, fit_line(S1_inv), label=f'Fit for S2={s2}')

plt.scatter(y = 1/Rate, x = 1/S1, label='Experimental Data', color='red')
plt.xlabel('Substrate 1 Concentration (1/mM)')
plt.ylabel('Reaction Rate (mM/s)')
plt.legend()
plt.title('Two-Substrate Kinetics')
plt.show()


# Shows behaviour consistent with type 2 enzyme reactions
# TODO using CHi-sqr / R-sqr measures?

# region 1.2

# Print the different S2 values
print("Unique S2 values in split_data:", list(split_data.keys()))