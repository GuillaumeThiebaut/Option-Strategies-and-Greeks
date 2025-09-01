import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# Defining the formulas needed
def option_payoff(S, K, premium, option_type, position):
    if option_type == "Call":
        payoff = np.maximum(S - K, 0) - premium
    elif option_type == "Put":
        payoff = np.maximum(K - S, 0) - premium
    else:
        raise ValueError("Option_type must be 'call' or 'put'")
    return payoff if position == "Long" else -payoff

# Defining a function that caculates d1 and d2 
def calculation_d1_d2(S, K, t, r, q, sigma):
    d1 = (np.log(S/K) + (r - q + 0.5 * sigma**2) * t) / (sigma * np.sqrt(t))
    d2 = d1 - sigma * np.sqrt(t)
    return d1, d2

# Defining the Normal Probability density function
def probability_density_function(x): 
    return np.exp(-x**2/2) / np.sqrt(2*np.pi)

# Defining the Black-Scholes pricing formula
def bsm_option(S, K, t, r, q, sigma, option_type): 
    d1, d2 = calculation_d1_d2(S, K, t, r, q, sigma)
    if option_type == 'Call': # Formula for a call pricing
        option_price = S * np.exp(-q * t) * norm.cdf(d1) - K * np.exp(-r * t) * norm.cdf(d2)
    elif option_type == 'Put': # Formula for a put pricing
        option_price = K * np.exp(-r * t) * norm.cdf(-d2) - S * np.exp(-q * t) * norm.cdf(-d1)
    return option_price

S = np.linspace(50, 150, 200)

# Initialising buttons and input fields
st.title('📈 Building your Option Strategies')

# Selecting a strategy and position hold
st.write('### Type of Option Strategy')

col1, col2 = st.columns(2)

strategies = ['Call', 'Put', 'Call Spread', 'Put Spread', 'Call Backspread', 'Put Backspread', 'Call Butterfly', 
              'Put Butterfly', 'Straddle', 'Strangle', 'Call Condor', 'Put Condor', 'Call Ladder',
              'Put Ladder', 'Strap', 'Strip']
option_strategy = col1.selectbox('Choose your Strategy:', strategies)
position_investor = col2.selectbox('Choose your Position:', ['Long', 'Short'])
opposition_position = 'Short' if position_investor == 'Long' else "Long"

# Generating the input fields according to the fields chosen
if "show_inputs" not in st.session_state:
    st.session_state.show_inputs = False    
premium_activated = col2.checkbox('Premium included')
if col1.button('Generate Input Fields'):
    st.session_state.show_inputs = True
if col1.button('Erase Input Fields'):
    st.session_state.show_inputs = False

# Generating payoff graph based on the selection    
if st.session_state.show_inputs:
    # Input in order to calculate the premium of the strategy chosen
    if premium_activated:
        st.write("### Input Datas for Premium calculating")
        col1, col2 = st.columns(2)
        S0 = col1.select_slider('Choose a spot price at t = 0', np.arange(70, 130.5, .5), value = 100)
        t = col2.select_slider('Choose a maturity (in Years)', np.arange(0.25, 10.25, 0.25), value = 1)
        r = col1.select_slider('Choose a risk-free rate (in %)', np.arange(0, 18.5, 0.5), value = 4)/100
        q = col2.select_slider('Choose a dividend yield (in %)', np.arange(0, 15.5, 0.5), value = 2)/100
        sigma = col1.select_slider('Choose an implied volatility (in %)', np.arange(0, 70.5, 0.5), value = 10)/100
    st.write('### Input Datas')
    col1, col2 = st.columns(2)
    
    # Strategies with 1 strike
    if option_strategy == 'Call' or option_strategy == 'Put':
        K = col1.select_slider('Select a strike', np.arange(70, 130.5, 0.5), value = 100)
        if premium_activated:
            premium = bsm_option(S0, K, t, r, q, sigma, option_strategy)
            st.text(f'Premium: {premium:.2f}$')
        else:
            premium = 0
           
        payoff = option_payoff(S, K, premium, option_strategy, position_investor)
        
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(S, payoff, label="Payoff", color="black", linewidth=1.5)
    
    # Strategies with 2 strikes
    if option_strategy == 'Call Spread':
        K1 = col1.select_slider('Select a strike for K1', np.arange(70, 100.5, 0.5), value = 90)
        K2 = col2.select_slider('Select a strike for K2', np.arange(100, 130.5, 0.5), value = 110)
        if premium_activated:
            premium_1 = bsm_option(S0, K1, t, r, q, sigma, 'Call')
            premium_2 = bsm_option(S0, K2, t, r, q, sigma, 'Call')
            st.text(f'Premium: ${premium_1 - premium_2:.2f}')
        else:
            premium_1 = 0
            premium_2 = 0
        call_1 = option_payoff(S, K1, premium_1, 'Call', f'{position_investor}')
        call_2 = option_payoff(S, K2, premium_2, 'Call', f'{opposition_position}')
        payoff = call_1 + call_2
        
        # Creating the plot
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(S, payoff, label="Payoff", color="black", linewidth=1.5)
        ax.plot(S, call_1, '--', label=f"{position_investor} Call", color="blue", alpha=0.6)
        ax.plot(S, call_2, '--', label=f"{opposition_position} Call", color="red", alpha=0.6)
        
    if option_strategy == 'Put Spread':
        K1 = col1.select_slider('Select a strike for K1', np.arange(70, 100.5, 0.5), value = 90)
        K2 = col2.select_slider('Select a strike for K2', np.arange(100, 130.5, 0.5), value = 110)
        if premium_activated:
            premium_1 = bsm_option(S0, K1, t, r, q, sigma, 'Put')
            premium_2 = bsm_option(S0, K2, t, r, q, sigma, 'Put')
            st.text(f'Premium: ${premium_1 - premium_2:.2f}')
        else:
            premium_1 = 0
            premium_2 = 0
        put_1 = option_payoff(S, K1, premium_1, 'Put', f'{position_investor}')
        put_2 = option_payoff(S, K2, premium_2, 'Put', f'{opposition_position}')
        payoff = put_1 + put_2
        
        # Creating the plot
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(S, payoff, label="Payoff", color="black", linewidth=1.5)
        ax.plot(S, put_1, '--', label=f"{position_investor} Put", color="blue", alpha=0.6)
        ax.plot(S, put_2, '--', label=f"{opposition_position} Put", color="red", alpha=0.6)
    
    if option_strategy == 'Straddle':
        K = col1.select_slider('Select a strike for K', np.arange(70, 130.5, 0.5), value = 100)
        if premium_activated:
            premium_1 = bsm_option(S0, K, t, r, q, sigma, 'Call')
            premium_2 = bsm_option(S0, K, t, r, q, sigma, 'Put')
            st.text(f'Premium: ${premium_1 + premium_2:.2f}')
        else:
            premium_1 = 0
            premium_2 = 0
        call = option_payoff(S, K, premium_1, 'Call', f'{position_investor}')
        put = option_payoff(S, K, premium_2, 'Put', f'{position_investor}')
        payoff = call + put
        
        # Creating the plot
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(S, payoff, label="Payoff", color="black", linewidth=1.5)
        ax.plot(S, call, '--', label=f"{position_investor} Call", color="blue", alpha=0.6)
        ax.plot(S, put, '--', label=f"{position_investor} Put", color="red", alpha=0.6)
    
    if option_strategy == 'Strangle':
        K1 = col1.select_slider('Select a strike for K1', np.arange(70, 100, 0.5), value = 90)
        K2 = col2.select_slider('Select a strike for K2', np.arange(100, 130.5, 0.5), value = 110)
        if premium_activated:
            premium_1 = bsm_option(S0, K1, t, r, q, sigma, 'Put')
            premium_2 = bsm_option(S0, K2, t, r, q, sigma, 'Call')
            st.text(f'Premium: ${premium_1 + premium_2:.2f}')  
        else:
            premium_1 = 0
            premium_2 = 0 
        put = option_payoff(S, K1, premium_1, 'Put', f'{position_investor}')
        call = option_payoff(S, K2, premium_2, 'Call', f'{position_investor}')
        payoff = put + call
        
        # Creating the plot
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(S, payoff, label="Payoff", color="black", linewidth=1.5)
        ax.plot(S, call, '--', label=f"{position_investor} Call", color="blue", alpha=0.6)
        ax.plot(S, put, '--', label=f"{position_investor} Put", color="red", alpha=0.6)
        
    if option_strategy == 'Call Backspread':
        K1 = col1.select_slider('Select a strike for K1', np.arange(70, 100, 0.5), value = 90)
        K2 = col2.select_slider('Select a strike for K2', np.arange(100, 130.5, 0.5), value = 110)
        if premium_activated:
            premium_1 = bsm_option(S0, K1, t, r, q, sigma, 'Call')
            premium_2 = bsm_option(S0, K2, t, r, q, sigma, 'Call')
            st.text(f'Premium: ${- premium_1 + 2 * premium_2:.2f}')  
        else:
            premium_1 = 0
            premium_2 = 0 
        call_1 = option_payoff(S, K1, premium_1, 'Call', f'{opposition_position}')
        call_2 = 2 * option_payoff(S, K2, premium_2, 'Call', f'{position_investor}')
        payoff = call_1 + call_2
        
        # Creating the plot
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(S, payoff, label="Payoff", color="black", linewidth=1.5)
        ax.plot(S, call_1, '--', label=f"{opposition_position} Call", color="blue", alpha=0.6)
        ax.plot(S, call_2, '--', label=f"{position_investor} 2 Calls", color="red", alpha=0.6)
        
    if option_strategy == 'Put Backspread':
        K1 = col1.select_slider('Select a strike for K1', np.arange(70, 100, 0.5), value = 90)
        K2 = col2.select_slider('Select a strike for K2', np.arange(100, 130.5, 0.5), value = 110)
        if premium_activated:
            premium_1 = bsm_option(S0, K1, t, r, q, sigma, 'Put')
            premium_2 = bsm_option(S0, K2, t, r, q, sigma, 'Put')
            st.text(f'Premium: ${2 * premium_1 - premium_2:.2f}')  
        else:
            premium_1 = 0
            premium_2 = 0 
        put_1 = 2 * option_payoff(S, K1, premium_1, 'Put', f'{position_investor}')
        put_2 = option_payoff(S, K2, premium_2, 'Put', f'{opposition_position}')
        payoff = put_1 + put_2
        
        # Creating the plot
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(S, payoff, label="Payoff", color="black", linewidth=1.5)
        ax.plot(S, put_1, '--', label=f"{position_investor} 2 Puts", color="blue", alpha=0.6)
        ax.plot(S, put_2, '--', label=f"{opposition_position} Put", color="red", alpha=0.6)
        
    if option_strategy == 'Call Butterfly':
        K1 = col1.select_slider('Select a strike for K1', np.arange(60, 90.5, 0.5), value = 80)
        K3 = col2.select_slider('Select a strike for K2', np.arange(110, 140.5, 0.5), value = 120)
        K2 = (K1 + K3)/2
        col2.metric('The strike for K2 is', value=f"${K2:.2f}")
        if premium_activated:
            premium_1 = bsm_option(S0, K1, t, r, q, sigma, 'Call')
            premium_2 = bsm_option(S0, K2, t, r, q, sigma, 'Call')
            premium_3 = bsm_option(S0, K3, t, r, q, sigma, 'Call')
            st.text(f'Premium: ${-premium_1 + 2 * premium_2 - premium_3:.2f}')  
        else:
            premium_1 = premium_2 = premium_3 = 0 
        call_1 = option_payoff(S, K1, premium_1, 'Call', f'{position_investor}')
        call_2 = 2 * option_payoff(S, K2, premium_2, 'Call', f'{opposition_position}')
        call_3 = option_payoff(S, K3, premium_3, 'Call', f'{position_investor}')
        payoff = call_1 + call_2 + call_3
        
        # Creating the plot
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(S, payoff, label="Payoff", color="black", linewidth=1.5)
        ax.plot(S, call_1, '--', label=f"{position_investor} Call", color="blue", alpha=0.6)
        ax.plot(S, call_2, '--', label=f"{opposition_position} 2 Calls", color="red", alpha=0.6)
        ax.plot(S, call_3, '--', label=f"{position_investor} Call", color="green", alpha=0.6)
    
    if option_strategy == 'Put Butterfly':
        K1 = col1.select_slider('Select a strike for K1', np.arange(60, 90.5, 0.5), value = 80)
        K3 = col2.select_slider('Select a strike for K2', np.arange(110, 140.5, 0.5), value = 120)
        K2 = (K1 + K3)/2
        col2.metric(label="The strike for K2 is", value=f"${K2:.2f}")
        if premium_activated:
            premium_1 = bsm_option(S0, K1, t, r, q, sigma, 'Put')
            premium_2 = bsm_option(S0, K2, t, r, q, sigma, 'Put')
            premium_3 = bsm_option(S0, K3, t, r, q, sigma, 'Put')
            st.text(f'Premium: ${-premium_1 + 2 * premium_2 - premium_3:.2f}')  
        else:
            premium_1 = premium_2 = premium_3 = 0 
        put_1 = option_payoff(S, K1, premium_1, 'Put', f'{position_investor}')
        put_2 = 2 * option_payoff(S, K2, premium_2, 'Put', f'{opposition_position}')
        put_3 = option_payoff(S, K3, premium_3, 'Put', f'{position_investor}')
        payoff = put_1 + put_2 + put_3
        
        # Creating the plot
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(S, payoff, label="Payoff", color="black", linewidth=1.5)
        ax.plot(S, put_1, '--', label=f"{position_investor} Put", color="blue", alpha=0.6)
        ax.plot(S, put_2, '--', label=f"{opposition_position} 2 Puts", color="red", alpha=0.6)
        ax.plot(S, put_3, '--', label=f"{position_investor} Put", color="green", alpha=0.6)
    
    if option_strategy == 'Call Condor':
        K1 = col1.select_slider('Select a strike for K1', np.arange(60, 90.5, 0.5), value = 80)
        K4 = col2.select_slider('Select a strike for K2', np.arange(110, 140.5, 0.5), value = 120)
        diff_strike = col1.select_slider('Select the difference to get K2 and K3', np.arange(1, (K4 - K1)/2, 1), value = 5)
        K2 = K1 + diff_strike
        K3 = K4 - diff_strike
        col2.metric(label="The strike for K2 is ", value=f"${K2:.2f}")
        col2.metric(label="The strike for K3 is ", value=f"${K3:.2f}")
        if premium_activated:
            premium_1 = bsm_option(S0, K1, t, r, q, sigma, 'Call')
            premium_2 = bsm_option(S0, K2, t, r, q, sigma, 'Call')
            premium_3 = bsm_option(S0, K3, t, r, q, sigma, 'Call')
            premium_4 = bsm_option(S0, K4, t, r, q, sigma, 'Call')
            st.text(f'Premium: ${premium_1 - premium_2 - premium_3 + premium_4:.2f}')  
        else:
            premium_1 = premium_2 = premium_3 = premium_4 = 0 
        call_1 = option_payoff(S, K1, premium_1, 'Call', f'{position_investor}')
        call_2 = option_payoff(S, K2, premium_2, 'Call', f'{opposition_position}')
        call_3 = option_payoff(S, K3, premium_3, 'Call', f'{opposition_position}')
        call_4 = option_payoff(S, K4, premium_4, 'Call', f'{position_investor}')
        payoff = call_1 + call_2 + call_3 + call_4
        
        # Creating the plot
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(S, payoff, label="Payoff", color="black", linewidth=1.5)
        ax.plot(S, call_1, '--', label=f"{position_investor} Call", color="blue", alpha=0.6)
        ax.plot(S, call_2, '--', label=f"{opposition_position} Call", color="red", alpha=0.6)
        ax.plot(S, call_3, '--', label=f"{opposition_position} Call", color="green", alpha=0.6)
        ax.plot(S, call_4, '--', label=f"{position_investor} Call", color="orange", alpha=0.6)
        
    if option_strategy == 'Put Condor':
        K1 = col1.select_slider('Select a strike for K1', np.arange(60, 90.5, 0.5), value = 80)
        K4 = col2.select_slider('Select a strike for K2', np.arange(110, 140.5, 0.5), value = 120)
        diff_strike = col1.select_slider('Select the difference to get K2 and K3', np.arange(1, (K4 - K1)/2, 1), value = 5)
        K2 = K1 + diff_strike
        K3 = K4 - diff_strike
        col2.metric(label="The strike for K2 is ", value=f"${K2:.2f}")
        col2.metric(label="The strike for K3 is ", value=f"${K3:.2f}")
        if premium_activated:
            premium_1 = bsm_option(S0, K1, t, r, q, sigma, 'Put')
            premium_2 = bsm_option(S0, K2, t, r, q, sigma, 'Put')
            premium_3 = bsm_option(S0, K3, t, r, q, sigma, 'Put')
            premium_4 = bsm_option(S0, K4, t, r, q, sigma, 'Put')
            st.text(f'Premium: ${premium_1 - premium_2 - premium_3 + premium_4:.2f}')  
        else:
            premium_1 = premium_2 = premium_3 = premium_4 = 0 
        put_1 = option_payoff(S, K1, premium_1, 'Put', f'{position_investor}')
        put_2 = option_payoff(S, K2, premium_2, 'Put', f'{opposition_position}')
        put_3 = option_payoff(S, K3, premium_3, 'Put', f'{opposition_position}')
        put_4 = option_payoff(S, K4, premium_4, 'Put', f'{position_investor}')
        payoff = put_1 + put_2 + put_3 + put_4
        
        # Creating the plot
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(S, payoff, label="Payoff", color="black", linewidth=1.5)
        ax.plot(S, put_1, '--', label=f"{position_investor} Put", color="blue", alpha=0.6)
        ax.plot(S, put_2, '--', label=f"{opposition_position} Put", color="red", alpha=0.6)
        ax.plot(S, put_3, '--', label=f"{opposition_position} Put", color="green", alpha=0.6)
        ax.plot(S, put_4, '--', label=f"{position_investor} Put", color="orange", alpha=0.6)
        
    if option_strategy == 'Call Ladder':
        K1 = col1.select_slider('Select a strike for K1', np.arange(60, 90, 0.5), value = 80)
        K2 = col1.select_slider('Select a strike for K2', np.arange(90, 110, 0.5), value = 100)
        K3 = col1.select_slider('Select a strike for K3', np.arange(110, 130, 0.5), value = 120)
        if premium_activated:
            premium_1 = bsm_option(S0, K1, t, r, q, sigma, 'Call')
            premium_2 = bsm_option(S0, K2, t, r, q, sigma, 'Call')
            premium_3 = bsm_option(S0, K3, t, r, q, sigma, 'Call')
            st.text(f'Premium: {premium_1 - premium_2 - premium_3:.2f}$')  
        else:
            premium_1 = premium_2 = premium_3 = 0 
        call_1 = option_payoff(S, K1, premium_1, 'Call', f'{position_investor}')
        call_2 = option_payoff(S, K2, premium_2, 'Call', f'{opposition_position}')
        call_3 = option_payoff(S, K3, premium_3, 'Call', f'{opposition_position}')
        payoff = call_1 + call_2 + call_3
        
        # Creating the plot
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(S, payoff, label="Payoff", color="black", linewidth=1.5)
        ax.plot(S, call_1, '--', label=f"{position_investor} Call", color="blue", alpha=0.6)
        ax.plot(S, call_2, '--', label=f"{opposition_position} Call", color="red", alpha=0.6)
        ax.plot(S, call_3, '--', label=f"{opposition_position} Call", color="green", alpha=0.6)
        
    if option_strategy == 'Put Ladder':
        K1 = col1.select_slider('Select a strike for K1', np.arange(60, 90, 0.5), value = 80)
        K2 = col1.select_slider('Select a strike for K2', np.arange(90, 110, 0.5), value = 100)
        K3 = col1.select_slider('Select a strike for K3', np.arange(110, 130, 0.5), value = 120)
        if premium_activated:
            premium_1 = bsm_option(S0, K1, t, r, q, sigma, 'Put')
            premium_2 = bsm_option(S0, K2, t, r, q, sigma, 'Put')
            premium_3 = bsm_option(S0, K3, t, r, q, sigma, 'Put')
            st.text(f'Premium: {- premium_1 - premium_2 + premium_3:.2f}$')  
        else:
            premium_1 = premium_2 = premium_3 = 0 
        put_1 = option_payoff(S, K1, premium_1, 'Put', f'{opposition_position}')
        put_2 = option_payoff(S, K2, premium_2, 'Put', f'{opposition_position}')
        put_3 = option_payoff(S, K3, premium_3, 'Put', f'{position_investor}')
        payoff = put_1 + put_2 + put_3
        
        # Creating the plot
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(S, payoff, label="Payoff", color="black", linewidth=1.5)
        ax.plot(S, put_1, '--', label=f"{opposition_position} Put", color="blue", alpha=0.6)
        ax.plot(S, put_2, '--', label=f"{opposition_position} Put", color="red", alpha=0.6)
        ax.plot(S, put_3, '--', label=f"{position_investor} Put", color="green", alpha=0.6)
        
    if option_strategy == 'Strap':
        K = col1.select_slider('Select a strike for K', np.arange(70, 130.5, 0.5), value = 100)
        if premium_activated:
            premium_1 = bsm_option(S0, K, t, r, q, sigma, 'Call')
            premium_2 = bsm_option(S0, K, t, r, q, sigma, 'Put')
            st.text(f'Premium: {2 * premium_1 + premium_2:.2f}$')
        else:
            premium_1 = 0
            premium_2 = 0
        call = 2 * option_payoff(S, K, premium_1, 'Call', f'{position_investor}')
        put = option_payoff(S, K, premium_2, 'Put', f'{position_investor}')
        payoff = call + put
        
        # Creating the plot
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(S, payoff, label="Payoff", color="black", linewidth=1.5)
        ax.plot(S, call, '--', label=f"{position_investor} 2 Calls", color="blue", alpha=0.6)
        ax.plot(S, put, '--', label=f"{position_investor} Put", color="red", alpha=0.6)
        
    if option_strategy == 'Strip':
        K = col1.select_slider('Select a strike for K', np.arange(70, 130.5, 0.5), value = 100)
        if premium_activated:
            premium_1 = bsm_option(S0, K, t, r, q, sigma, 'Call')
            premium_2 = bsm_option(S0, K, t, r, q, sigma, 'Put')
            st.text(f'Premium: {premium_1 + 2 * premium_2:.2f}$')
        else:
            premium_1 = 0
            premium_2 = 0
        call = option_payoff(S, K, premium_1, 'Call', f'{position_investor}')
        put = 2 * option_payoff(S, K, premium_2, 'Put', f'{position_investor}')
        payoff = call + put
        
        # Creating the plot
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(S, payoff, label="Payoff", color="black", linewidth=1.5)
        ax.plot(S, call, '--', label=f"{position_investor} Call", color="blue", alpha=0.6)
        ax.plot(S, put, '--', label=f"{position_investor} 2 Puts", color="red", alpha=0.6)
    
    ax.axhline(0, color='black', linewidth=0.5)
    ax.set_title(f"{position_investor} {option_strategy}")
    ax.set_xlabel("Spot Price ($)")
    ax.set_ylabel("Payoff ($)")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.5)

    st.pyplot(fig)