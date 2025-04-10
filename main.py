import numpy as np 
from scipy.optimize import curve_fit 
import pandas as pd
import matplotlib.pyplot as plt 

plt.rcParams.update({'font.size': 10})

# region mechanisms
# type 1
def type1akinetics(S1S2, Vmax, Kis1, Km1):
    S1, S2 = S1S2
    return (Vmax * S1 * S2) / (Kis1*Km1 + Km1*S1 + S1*S2)

def type1bkinetics(S1S2, Vmax, Kis1, Km1, Km2):
    S1, S2 = S1S2
    return (Vmax * S1 * S2) / (Kis1*Km1 + Km1*S1 + Km2*S2 + S1*S2)

# type 2
def type2kinetics(S1S2, Km1, Km2):
    S1, S2 = S1S2
    return (S1 * S2) / (Km1 * S2 + Km2 * S1 + S1 * S2)

# region fitmeasure
def chi_squared(y, y_fit):
    return np.sum(((y - y_fit) ** 2) / y_fit)

# region 1.1
file_path = './Kinetics.csv' # File path is relative, just add the kinetics.csv to the same folder as this file
data = pd.read_csv(file_path)

s1 = data['S1'].values
s2 = data['S2'].values
Rate = data['Rate'].values

# error on type 1a fitting is normal, just including it for the sake of completeness 
try:
    popt, _ = curve_fit(type1akinetics, (s1, s2), Rate)
    chisq = chi_squared(Rate, type1akinetics((s1, s2), *popt))
    print("Type 1a Chi-squared:", chisq)
    print("   Vmax:", popt[0], "Kis1:", popt[1], "Km1:", popt[2])
    
except Exception as error:
    print('Type 1a fitting failed:')
    print("  ",error)
    
popt, _ = curve_fit(type1bkinetics, (s1, s2), Rate)
chisq = chi_squared(Rate, type1bkinetics((s1, s2), *popt))

print("Type 1b Chi-squared:", chisq)
print("   Vmax:", popt[0], "Kis1:", popt[1], "Km1:", popt[2], "Km2:", popt[3])

popt, _ = curve_fit(type2kinetics, (s1, s2), Rate)
chisq = chi_squared(Rate, type2kinetics((s1, s2), *popt))
    
plt.figure(dpi=300, figsize=(5, 3))
plt.scatter(s1, Rate, label='Data', color='blue')
plt.scatter(s1, type2kinetics((s1, s2), *popt), label='Fit', marker="x", color='red')
plt.xlabel("S1")
plt.ylabel("Rate")
plt.legend()
plt.tight_layout()
plt.show()

print("Type 2 Chi-squared:", chisq)
print("   Km1:", popt[0], "Km2:", popt[1])

# region 1.2 & 1.3
s2_values = [1.5, 2.5, 5]
lineweaver_burke_data = []
eadie_hofstee_data = []

for s2 in s2_values:
    
    # Lineweaver-Burke data
    s1 = np.linspace(0.5, 1, 10)
    v = type2kinetics((s1, s2), *popt)
    lineweaver_burke_data.append((1/s1, 1/v, f"$S_2$={s2}"))
    
    # Eadie-Hofstee data and parameter extraction
    s1 = np.linspace(0.001, 100, 100)
    v = type2kinetics((s1, s2), *popt)
    v_over_s1 = v / s1
    slope, intercept = np.polyfit(v_over_s1, v, 1) # fitting a line to read slope and intercept
    y_intercept = intercept
    eadie_hofstee_data.append((v_over_s1, v, f"$S_2$={s2}, Slope: {slope:.3g}"))

    print(f"Eadie-Hofstee for s2={s2}:")
    km2 = 1 / (y_intercept / -slope)
    print(f"  Km1: {-slope*(1+km2/s2)}")
    print(f"  Km2: {km2}")

# Plot Lineweaver-Burke
plt.figure(dpi=300, figsize=(5, 3))
for x, y, label in lineweaver_burke_data:
    plt.plot(x, y, label=label)
plt.xlabel("1/[$S_1$]")
plt.ylabel("1/v")
plt.legend()
plt.tight_layout()
plt.show()

# Plot Eadie-Hofstee
plt.figure(dpi=300, figsize=(5, 3))
for x, y, label in eadie_hofstee_data:
    plt.plot(x, y, label=label)
plt.xlabel("v/[$S_1$]")
plt.ylabel("v")
plt.legend()
plt.tight_layout()
plt.show()

#region 2.d
# Code used to generate solution space plot
D = 2
xl = np.linspace(0, 10, 201)
yl = xl
x2 = np.linspace(0, 6, 201)
y2 = x2 + D
y3 = np.repeat(8, 201)
y4 = xl + D

fig, ax = plt.subplots(sharex=True, figsize=(6, 6))
# ax.set_title("Title")
ax.plot(xl, yl, color='b', label="$v_6=v_1$", linestyle="--")
ax.plot(x2, y2, color='r')
ax.plot(x2, y2, color='r')
ax.plot(xl, y4, color='r', label="$v_6=v_1+D$")

ax.axhline(y=0, color="black", alpha=0.7)
ax.axvline(x=0, color="black", alpha=0.7)
ax.axhline(y=8, xmin=0.095, xmax=0.63, color="green", alpha=0.5)

ax.fill_between(x2, y2, y3, alpha=0.7)

ax.text(-1.5, 8, r'$v_{6, max}$', fontsize=12)
ax.text(-0.5, 2, r'D', fontsize=12)
ax.text(10, 0, r'$v_1$', fontsize=12)
ax.text(-0.5, 10, r'$v_6$', fontsize=12)
ax.arrow(0, 0, 0, 10, length_includes_head=True, head_width=0.2, head_length=0.4)
ax.arrow(0, 0, 10, 0, length_includes_head=True, head_width=0.2, head_length=0.4)
ax.set(xlim=(-1, 10), 
       ylim=(-1, 10), 
       xlabel="$v_1$",
       ylabel="$v_6$",
       xticks=[],
       yticks=[])
ax.axis('off')
ax.legend()
plt.show()