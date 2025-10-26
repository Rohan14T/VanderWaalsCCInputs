# save as app.py
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

st.set_page_config(page_title="Van der Waals Carnot Cycle", layout="wide")
st.title("Van der Waals Carnot Cycle Simulator")

# --- User inputs via Streamlit ---
st.sidebar.header("Molecule & Critical Properties")
molecule = st.sidebar.text_input("Molecule name", "Argon")
Tc = st.sidebar.number_input("Critical temperature (K)", min_value=0.1, value=150.0)
Pc_kPa = st.sidebar.number_input("Critical pressure (kPa)", min_value=0.1, value=5000.0)
Vc_L = st.sidebar.number_input("Critical volume (L/mol)", min_value=0.0001, value=0.05)

st.sidebar.header("Cycle Temperatures")
Th = st.sidebar.number_input("Hot body temperature (K)", min_value=0.1, max_value=Tc-0.1, value=120.0)
Tcold = st.sidebar.number_input("Cold body temperature (K)", min_value=0.1, max_value=Th-0.1, value=80.0)

st.sidebar.header("Cycle Volumes")
V1_L = st.sidebar.number_input("Initial volume (L/mol)", min_value=Vc_L*1.01, value=Vc_L*1.05)
V2_L = st.sidebar.number_input("End hot isothermal volume (L/mol)", min_value=V1_L*1.01, value=V1_L*1.1)

st.sidebar.header("Molecule Structure")
mol_type = st.sidebar.selectbox("Molecule type", ["monatomic", "linear", "nonlinear"])
num_atoms = st.sidebar.number_input("Number of atoms", min_value=1, value=1, step=1)

# --- Convert units ---
Vc = Vc_L * 1e-3
Pc = Pc_kPa * 1e3
V1 = V1_L * 1e-3
V2 = V2_L * 1e-3

# --- Constants ---
R = 8.314
if mol_type == "monatomic":
    f = 3
elif mol_type == "linear":
    f = 5
else:
    f = 6

if num_atoms > 2:
    vibrational_modes = 3*num_atoms - (5 if mol_type=="linear" else 6)
    f += 0.2 * vibrational_modes

Cv = (f/2) * R
Cp = Cv + R
gamma = Cp / Cv

# --- van der Waals constants ---
a = (27*R**2 * Tc**2)/(64*Pc)
b = R*Tc/(8*Pc)

# --- van der Waals pressure ---
def pressure(T, V):
    return (R*T)/(V-b) - a/V**2  # Pa

# --- ODE for adiabatic ---
def adiabatic_ode(V, T):
    P = pressure(T[0], V)
    return [- (P + a/V**2)/Cv]

points = 300

# --- Hot isothermal expansion ---
V_hot = np.linspace(V1, V2, points)
P_hot = pressure(Th, V_hot)
P2_end = P_hot[-1]

# --- Adiabatic expansion to Tcold ---
def expand_to_Tcold(V_start, T_start, T_target):
    def event_Tcold(V, T):
        return T[0] - T_target
    event_Tcold.terminal = True
    event_Tcold.direction = -1
    sol = solve_ivp(adiabatic_ode, [V_start, V_start*5], [T_start],
                    events=event_Tcold, max_step=1e-5)
    V_adiab = sol.t
    T_adiab = sol.y[0]
    P_adiab = pressure(T_adiab, V_adiab)
    return V_adiab, P_adiab, T_adiab

V_adiab1, P_adiab1, T_adiab1 = expand_to_Tcold(V2, Th, Tcold)
V3 = V_adiab1[-1]
P3 = P_adiab1[-1]

# --- Adiabatic from initial to Tcold for cold isotherm end ---
V_adiab_check, P_adiab_check, T_adiab_check = expand_to_Tcold(V1, Th, Tcold)
V4 = V_adiab_check[-1]
P4 = P_adiab_check[-1]

# --- Cold isothermal compression ---
V_cold = np.linspace(V3, V4, points)
P_cold = pressure(Tcold, V_cold)
P_cold[0] = P3
P_cold[-1] = P4

# --- Adiabatic compression to initial ---
def compress_to_Th(V_start, T_start, V_end, T_end):
    def event_Th(V, T):
        return T[0] - T_end
    event_Th.terminal = True
    event_Th.direction = 1
    sol = solve_ivp(adiabatic_ode, [V_start, V_end], [T_start],
                    events=event_Th, max_step=1e-5)
    V_adiab = sol.t
    T_adiab = sol.y[0]
    P_adiab = pressure(T_adiab, V_adiab)
    P_adiab[-1] = pressure(T_end, V_end)
    return V_adiab, P_adiab, T_adiab

V_adiab2, P_adiab2, T_adiab2 = compress_to_Th(V4, Tcold, V1, Th)

# --- Work calculations ---
W_hot = np.trapz(P_hot, V_hot)
W_adiab1 = np.trapz(P_adiab1, V_adiab1)
W_cold = np.trapz(P_cold, V_cold)
W_adiab2 = np.trapz(P_adiab2, V_adiab2)
W_net = W_hot + W_adiab1 + W_cold + W_adiab2
efficiency = W_net / W_hot

# --- PV diagram with arrows ---
fig, ax = plt.subplots(figsize=(8,6))
ax.plot(V_hot*1e3, P_hot/1000, 'r', label="Isothermal Expansion (Th)")
ax.plot(V_adiab1*1e3, P_adiab1/1000, 'orange', label="Adiabatic Expansion")
ax.plot(V_cold*1e3, P_cold/1000, 'b', label="Isothermal Compression (Tcold)")
ax.plot(V_adiab2*1e3, P_adiab2/1000, 'g', label="Adiabatic Compression")
ax.set_xlabel("Volume (L/mol)")
ax.set_ylabel("Pressure (kPa)")
ax.set_title(f"Van der Waals Carnot Cycle: {molecule}")
ax.grid(True)
ax.legend()

# Add arrows to indicate cycle direction
def add_arrow(x, y, color='k'):
    step = max(len(x)//5, 1)
    for i in range(0, len(x)-1, step):
        ax.annotate('', xy=(x[i+1], y[i+1]), xytext=(x[i], y[i]),
                    arrowprops=dict(arrowstyle='->', color=color, lw=1.5))

add_arrow(V_hot*1e3, P_hot/1000, 'r')
add_arrow(V_adiab1*1e3, P_adiab1/1000, 'orange')
add_arrow(V_cold*1e3, P_cold/1000, 'b')
add_arrow(V_adiab2*1e3, P_adiab2/1000, 'g')

# Display in Streamlit
st.pyplot(fig)
plt.close(fig)

# --- Summary ---
st.subheader("Cycle Summary")
st.write(f"Molecule: {molecule}")
st.write(f"a = {a:.3e} Pa·m^6/mol^2, b = {b:.3e} m^3/mol")
st.write(f"Cv = {Cv:.2f} J/mol·K, Cp = {Cp:.2f} J/mol·K, γ = {gamma:.3f}")
st.write(f"T_hot = {Th:.2f} K, T_cold = {Tcold:.2f} K")
st.write(f"V1 = {V1*1e3:.3f} L/mol, V2 = {V2*1e3:.3f} L/mol, V3 = {V3*1e3:.3f} L/mol, V4 = {V4*1e3:.3f} L/mol")
st.write(f"Net Work = {W_net:.2f} J/mol, Efficiency ≈ {efficiency*100:.2f}%")
