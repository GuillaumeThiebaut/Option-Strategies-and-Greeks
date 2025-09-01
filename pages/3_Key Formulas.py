import streamlit as st

st.title('🧠 Key Formulas')

# Formulas for Vanilla Option Payoffs
st.subheader('Option Payoffs')
st.latex(r" Long \; Call \; Payoff = \max(S - K, \; 0)")
st.latex(r" Short \; Call \; Payoff = -\max(S - K, \; 0) = min(0, \; K-S)")
st.latex(r" Long \; Put \; Payoff = \max(K-S, \; 0)")
st.latex(r" Short \; Put \; Payoff = -\max(K - S, \; 0) = min(0, \; S-K)")
st.text('''S = Spot Price
K = Strike Price''')

# BSM Pricing Formulas
st.subheader('Black Scholes Model Formulas Adjusted (with Continuous Dividend Included)')
st.latex(r'Call = Se^{-qt}N(d1) - Ke^{-rt}N(d2)')
st.latex(r'Put = Ke^{-rt}N(-d2) - Se^{-qt}N(-d1)')
st.latex(r"d1 = \frac{\ln(\frac{S}{K})+(r-q+ \frac{\sigma^2}{2})t}{\sigma \sqrt{t}}")
st.latex(r"d2 = d1 - \sigma \sqrt{t}")

delta_activated = st.checkbox('Delta Formulas')

if delta_activated:
    st.latex('\Delta_{Call} = e^{-qt}N(d1)')
    st.latex('\Delta_{Put} = -e^{-qt}N(-d1)')

gamma_activated = st.checkbox('Gamma Formula')
    
if gamma_activated:
    st.latex(r"\Gamma_{Call} = \Gamma_{Put} = \frac{N(d_1) \, e^{-qt}}{S \, \sigma \sqrt{t}}")
    
vega_activated = st.checkbox('Vega Formula')
    
if vega_activated:
    st.latex(r"\nu_{Call} = \nu_{Put} = Se^{-qt}N(d1) \sqrt{t}")
    
theta_activated = st.checkbox('Theta Formulas')
    
if theta_activated:
    st.latex(r"\Theta_{Call} = - \frac{SN'(d_1) \sigma}{2 \sqrt(t)} - rKe^{-rt}N(d2)")
    st.latex(r"\Theta_{Put} = - \frac{SN'(d_1) \sigma}{2 \sqrt(t)} + rKe^{-rt}N(-d2)")

rho_activated = st.checkbox('Rho Formulas')
    
if rho_activated:
    st.latex(r"\rho_{Call} = \frac{1}{100}Kte^{-rt}N(d2)")
    st.latex(r"\rho_{Put} = - \frac{1}{100}Kte^{-rt}N(-d2) ")

st.text('''S = Spot Price
K = Strike Price
r = Risk Free Rate
q = Dividend Yield
t = Time to Maturity
σ = Volatility
        ''')