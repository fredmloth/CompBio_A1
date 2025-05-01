import numpy as np
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp


plt.rcParams.update({'font.size': 10})

# region a - ODE for type 1
# Repressilator dynamics
def gene_regulation_ode(t, y0, params):
    """
    Constructs a model two-gene regulation system using ODEs with Hill functions. 
    To be used with scipy.integrate.solve_ivp

    Parameters:
    - t: dummy parameter used for scipy.integrate.solve_ivp
    - y0: Initial values for concentrations
    - params: Dictionary containing parameters:
        - mA, mB: Max transcription rates for genes A and B.
        - gammaA, gammaB: mRNA degradation rates for genes A and B.
        - kPA, kPB: Translation rates for proteins A and B.
        - thetaA, thetaB: Expression thresholds for proteins A and B.
        - nA, nB: Hill coefficients for proteins A and B.
        - deltaPA, deltaPB: Degradation rates for proteins A and B.

    Returns:
    - dydt: Array of derivatives [dmRNA_A/dt, dmRNA_B/dt, dprotein_A/dt, dprotein_B/dt].
    """
    mRNA_A, mRNA_B, protein_A, protein_B = y0
    mA = params['mA']
    mB = params['mB']
    gammaA = params['gammaA']
    gammaB = params['gammaB']
    kPA = params['kPA']
    kPB = params['kPB']
    thetaA = params['thetaA']
    thetaB = params['thetaB']
    nA = params['nA']
    nB = params['nB']
    deltaPA = params['deltaPA']
    deltaPB = params['deltaPB']

    # Hill functions for regulation
    activ_hill_by_B = protein_B**nB / (protein_B**nB + thetaB**nB) # Protein B promotes transcription of gene A
    inhib_hill_by_A = thetaA**nA / (protein_A**nA + thetaA**nA) # Protein A inhibit transcription of gene B
    
    # ODEs
    dmRNA_A_dt = mA * activ_hill_by_B - gammaA * mRNA_A
    dmRNA_B_dt = mB * inhib_hill_by_A - gammaB * mRNA_B
    dprotein_A_dt = kPA * mRNA_A - deltaPA * protein_A
    dprotein_B_dt = kPB * mRNA_B - deltaPB * protein_B

    return np.array([dmRNA_A_dt, dmRNA_B_dt, dprotein_A_dt, dprotein_B_dt])

# region a - plotting 
params = {
    'mA': 2.35, 'mB': 2.35,  # Max transcription rates
    'gammaA': 1.0, 'gammaB': 1.0,  # mRNA degradation rates
    'kPA': 1.0, 'kPB': 1.0,  # Translation rates
    'thetaA': 0.21, 'thetaB': 0.21,  # Expression thresholds
    'nA': 3.0, 'nB': 3.0,  # Hill coefficients
    'deltaPA': 1.0, 'deltaPB': 1.0  # Protein degradation rates
}

# Initial conditions
y0 = [0.8, 0.8, 0.8, 0.8]

t_span = (0, 100)
t_eval = np.linspace(t_span[0], t_span[1], 5000) 

solution = solve_ivp(
    gene_regulation_ode, t_span, y0, args=(params,), t_eval=t_eval, method='RK45'
)
t = solution.t
mRNA_A, mRNA_B, protein_A, protein_B = solution.y

plt.figure(figsize=(8, 6), dpi = 100)
plt.plot(t, mRNA_A, label='mRNA A', color='red')
plt.plot(t, mRNA_B, label='mRNA B', color='black')
plt.xlabel('Time')
plt.ylabel('Concentration')
plt.legend()
plt.grid()
plt.show()

plt.figure(figsize=(8, 6), dpi = 100)
plt.plot(protein_A, protein_B, color='purple')
plt.xlabel('Protein A Concentration')
plt.ylabel('Protein B Concentration')
plt.grid()
plt.show()


# region b - SDEVeloProtein
def gene_regulation_sde(t, y0, params):
    """
    Simulates the stochastic gene regulation system using SDEs with Wiener processes via 
    Euler-Maruyama method.

    Parameters:
    - t: Time array
    - y0: Initial values for concentrations
    - params: Dictionary containing parameters (as in table of assignemnt document)

    Returns:
    - results: Array of concentrations over time
    """
    # Unpack parameters
    aA, aB = params['aA'], params['aB']
    bA, bB = params['bA'], params['bB']
    cA, cB = params['cA'], params['cB']
    betaA, betaB = params['betaA'], params['betaB']
    gammaA, gammaB = params['gammaA'], params['gammaB']
    nA, nB = params['nA'], params['nB']
    thetaA, thetaB = params['thetaA'], params['thetaB']
    kPA, kPB = params['kPA'], params['kPB']
    mA, mB = params['mA'], params['mB']
    deltaPA, deltaPB = params['deltaPA'], params['deltaPB']
    sigma1A, sigma2A = params['sigma1A'], params['sigma2A']
    sigma1B, sigma2B = params['sigma1B'], params['sigma2B']

    # Initialize variables
    num_steps = len(t)
    dt = np.diff(t)[0]
    results = np.zeros((num_steps, len(y0)))
    results[0] = y0

    for i in range(1, num_steps):
        mRNA_A, mRNA_B, protein_A, protein_B, pre_mRNA_A, pre_mRNA_B = results[i - 1]

        # Hill functions
        activ_hill_by_B = protein_B**nB / (protein_B**nB + thetaB**nB)
        inhib_hill_by_A = thetaA**nA / (protein_A**nA + thetaA**nA)
        
        modified_betaA = activ_hill_by_B*betaA # promoting pre-mRNA_A splicing
        modified_betaB = inhib_hill_by_A*betaB # promoting pre-mRNA_B splicing

        # 0-1 = mRNA_A, mRNA_B
        results[i, 0] = mRNA_A + (modified_betaA*pre_mRNA_A - gammaA*mRNA_A)*dt + sigma2A*np.sqrt(dt)*np.random.normal(0, 1)
        results[i, 1] = mRNA_B + (modified_betaB*pre_mRNA_B - gammaB*mRNA_B)*dt + sigma2B*np.sqrt(dt)*np.random.normal(0, 1)
        
        # 4-5 = pre_mRNA_A, pre_mRNA_B 
        results[i, 4] = pre_mRNA_A + (mA*(cA / ( 1 + np.exp(bA * (t[i] - aA)))) - betaA*pre_mRNA_A)*dt + sigma1A*np.sqrt(dt)*np.random.normal(0, 1)
        results[i, 5] = pre_mRNA_B + (mB*(cB / ( 1 + np.exp(bB * (t[i] - aB)))) - betaB*pre_mRNA_B)*dt + sigma1B*np.sqrt(dt)*np.random.normal(0, 1)
        
        # Deterministic calculation for protein translation
        results[i, 2] = protein_A + (kPA * mRNA_A - deltaPA * protein_A)*dt
        results[i, 3] = protein_B + (kPB * mRNA_B - deltaPB * protein_B)*dt
        
    return results

# region b - plotting 
sde_params = {
    'aA': 1.0, 'aB': 0.25,
    'bA': 0.0005, 'bB': 0.0005,
    'cA': 2.0, 'cB': 0.5,
    'betaA': 2.35, 'betaB': 2.35,
    'gammaA': 1.0, 'gammaB': 1.0,
    'nA': 3.0, 'nB': 3.0,
    'thetaA': 0.21, 'thetaB': 0.21,
    'kPA': 1.0, 'kPB': 1.0,
    'mA': 2.35, 'mB': 2.35,
    'deltaPA': 1.0, 'deltaPB': 1.0,
    'sigma1A': 0.05, 'sigma2A': 0.05,
    'sigma1B': 0.05, 'sigma2B': 0.05
}
y0_sde = [0.8, 0.8, 0.8, 0.8, 0.8, 0.8]

t_sde = np.linspace(0, 100, 5000)

sde_results = gene_regulation_sde(t_sde, y0_sde, sde_params)

mRNA_A_sde, mRNA_B_sde, protein_A_sde, protein_B_sde, pre_mRNA_A_sde, pre_mRNA_B_sde = sde_results.T

plt.figure(figsize=(10, 6), dpi = 100)
plt.plot(t_sde, mRNA_A_sde, label='mRNA A', color='black')
plt.plot(t_sde, mRNA_B_sde, label='mRNA B', color='red')
plt.xlabel('Time')
plt.ylabel('Concentration')
plt.legend()
plt.grid()
plt.show()

plt.figure(figsize=(8, 6), dpi = 100)
plt.plot(protein_A_sde, protein_B_sde, color='purple')
plt.xlabel('Protein A Concentration')
plt.ylabel('Protein B Concentration')
plt.grid()
plt.show()

# region Q2 Streamplot

def dx_dt(x, y, alpha, beta):
    return alpha*x - beta*x*y

def dy_dt(x, y, gamma, delta):
    return -gamma*y + delta*x*y

def XYUV(alpha, beta, gamma, delta):
    # computes derivatives for a grid of points
    x = np.linspace(-1, 10, 20)
    y = np.linspace(-1, 10, 20)
    X, Y = np.meshgrid(x, y)

    U = dx_dt(X, Y, alpha, beta)
    V = dy_dt(X, Y, gamma, delta)
    return X, Y, U, V

alpha = 2
beta = 1.1
gamma = 1
delta = 0.9

X, Y, U, V = XYUV(2, 1.1, 1, 0.9)

# stream plot
plt.figure(figsize=(8, 6), dpi=100)
plt.streamplot(X, Y, U, V, color=np.sqrt(U**2 + V**2), cmap='viridis')

# nullclines
x_nullcline = np.linspace(-1, 10, 200)
y_nullcline_x = alpha / beta  # dx/dt = 0 -> alpha*x - beta*x*y = 0 -> y = alpha/beta
y_nullcline_y = np.linspace(-1, 10, 200)
x_nullcline_y = gamma / delta  # dy/dt = 0 -> -gamma*y + delta*x*y = 0 -> x = gamma/delta

plt.plot(x_nullcline, [y_nullcline_x] * len(x_nullcline), 'r--', label='x-nullcline')
plt.plot([x_nullcline_y] * len(y_nullcline_y), y_nullcline_y, 'b--', label='y-nullcline')
plt.plot(x_nullcline_y, y_nullcline_x, 'ko', label=f'Equilibrium Point @ ({y_nullcline_x:.3g},{x_nullcline_y:.3g})')
plt.xlabel('x [Metabolite]')
plt.ylabel('y [Enzyme]')
plt.legend()
plt.grid()
plt.show()