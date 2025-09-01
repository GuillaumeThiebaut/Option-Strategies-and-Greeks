import streamlit as st
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
import pandas as pd

# Defining the formulas needed
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
    if option_type == 'Call': # Formula for a Call pricing
        option_price = S * np.exp(-q * t) * norm.cdf(d1) - K * np.exp(-r * t) * norm.cdf(d2)
    elif option_type == 'Put': # Formula for a Put pricing
        option_price = K * np.exp(-r * t) * norm.cdf(-d2) - S * np.exp(-q * t) * norm.cdf(-d1)
    return option_price

def returnd1(S, K, t, r, q, sigma): 
    return calculation_d1_d2(S, K, t, r, q, sigma)

# Defining the Black-Scholes Delta formula
def delta(S, K, t, r, q, sigma, option_type):
    d1, _ = calculation_d1_d2(S, K, t, r, q, sigma)
    if option_type == 'Call': 
        return np.exp(-q * t) * norm.cdf(d1)
    elif option_type == 'Put':
        return -np.exp(-q * t) * norm.cdf(-d1)

# Defining the Black-Scholes Gamma formula
def gamma(S, K, t, r, q, sigma): 
    d1, _ = calculation_d1_d2(S, K, t, r, q, sigma)
    return (norm.pdf(d1) * np.exp(-q * t)) / (S * sigma * np.sqrt(t)) # J'ai changé ici

# Defining the Black-Scholes Vega formula
def vega(S, K, t, r, q, sigma): 
    d1, _ = calculation_d1_d2(S, K, t, r, q, sigma)
    return S * np.sqrt(t) * probability_density_function(d1) * np.exp(-q * t)

# Defining the Black-Scholes Theta formula 
def theta(S, K, t, r, q, sigma, option_type): 
    d1, d2 = calculation_d1_d2(S, K, t, r, q, sigma)
    term_1 = -S * probability_density_function(d1) * sigma * np.exp(-q * t) / (2 * np.sqrt(t))
    if option_type == 'Call':
        return term_1 + q * S * np.exp(-q * t) * norm.cdf(d1) - r * K * np.exp(-r * t) * norm.cdf(d2)
    elif option_type == 'Put':
        return term_1 - q * S * np.exp(-q * t) * norm.cdf(-d1) + r * K * np.exp(-r * t * norm.cdf(-d2))

# Defining the Black-Scholes Rho formula
def rho(S, K, t, r, q, sigma, option_type):
    _, d2 = calculation_d1_d2(S, K, t, r, q, sigma)
    if option_type == 'Call':
        return K * t * np.exp(-r * t) * norm.cdf(d2)/100
    elif option_type == 'Put':
        return -K * t * np.exp(-r * t) * norm.cdf(-d2)/100
    
def creating_greek(S, K, t, r, q, sigma, option_type):
    df = pd.DataFrame(index=S)
    df['Delta'] = delta(S, K, t, r, q, sigma, option_type)
    df['Gamma'] = gamma(S, K, t, r, q, sigma)
    df['Vega'] = vega(S, K, t, r, q, sigma)
    df['Theta'] = theta(S, K, t, r, q, sigma, option_type)
    df['Rho'] = rho(S, K, t, r, q, sigma, option_type)
    return df

def creating_greek_array(S, K, t, r, q, sigma, option_type):
    return {
        'Delta': delta(S, K, t, r, q, sigma, option_type),
        'Gamma': gamma(S, K, t, r, q, sigma),
        'Vega': vega(S, K, t, r, q, sigma),
        'Theta': theta(S, K, t, r, q, sigma, option_type),
        'Rho': rho(S, K, t, r, q, sigma, option_type)
    }

    
# Inputs needed for plotting the greeks
S = np.linspace(50, 150, 200)

st.title('📊 Hedge Ratios Analysis')

# Selecting a strategy and position hold
st.write('### Type of Option Strategy')

col1, col2 = st.columns(2)

strategies = ['Put', 'Call', 'Call Spread', 'Put Spread', 'Call Backspread', 'Put Backspread', 'Call Butterfly', 
              'Put Butterfly', 'Straddle', 'Strangle', 'Call Condor', 'Put Condor', 'Call Ladder',
              'Put Ladder', 'Strap', 'Strip']
option_strategy = col1.selectbox('Choose your Strategy:', strategies)

greek = col2.selectbox('Choose a Greek', ['Delta', 'Gamma', 'Vega', 'Theta', 'Rho'])

# Writing the field to enter the inputs needed
st.write('### Input Datas')
col1, col2 = st.columns(2)

t = float(col2.text_input('Choose a maturity (in Years)', value=1))
r = float(col1.text_input('Choose a risk-free rate (in %)', value=4))/100
q = float(col2.text_input('Choose a dividend yield (in %)', value = 2))/100
sigma = float(col1.text_input('Choose an implied volatility (in %)', value = 10))/100

# Plot fields
st.write("### Plot Fields")
col1, col2 = st.columns(2)

plot_2d = col1.checkbox('Show 2D Plot')
plot_3d = col2.checkbox('Show 3D Plot')

if plot_2d:
    # Doing all Delta plots
    if option_strategy == 'Call' or option_strategy=='Put':
        K = float(col1.text_input('Choose a strike for K', value = 100))
        greeks = creating_greek(S, K, t, r, q, sigma, option_strategy)
        # Creating 4 plots 
        fig, ax = plt.subplots(1, figsize = (8,5))
        
        # Plotting Deltas for call options 
        ax.plot(S, greeks[greek], label=f'{greek}')
    
    if option_strategy == 'Call Spread':
        K1 = float(col1.text_input('Choose a strike for K1', value = 90))
        K2 = float(col2.text_input('Choose a strike for K2', value = 110))
        greeks_K1 = creating_greek(S, K1, t, r, q, sigma, 'Call')
        greeks_K2 = - creating_greek(S, K2, t, r, q, sigma, 'Call')
        greeks = greeks_K1 + greeks_K2
        # Creating 4 plots 
        fig, ax = plt.subplots(1, figsize = (8,5))
        
        # Plotting Deltas for call options 
        ax.plot(S, greeks[greek], label=f'{greek}', color='black', linewidth=1.5)
        ax.plot(S, greeks_K1[greek], '--', label=f'{greek} Long Call', color='red', linewidth=0.6)
        ax.plot(S, greeks_K2[greek], '--', label=f'{greek} Short Call', color='blue', linewidth=0.6)
    
    if option_strategy == 'Put Spread':
        K1 = float(col1.text_input('Choose a strike for K1', value = 90))
        K2 = float(col2.text_input('Choose a strike for K2', value = 110))
        greeks_K1 = creating_greek(S, K1, t, r, q, sigma, 'Put')
        greeks_K2 = - creating_greek(S, K2, t, r, q, sigma, 'Put')
        greeks = greeks_K1 + greeks_K2
        # Creating 4 plots 
        fig, ax = plt.subplots(1, figsize = (8,5))
        
        # Plotting Deltas for call options 
        ax.plot(S, greeks[greek], label=f'{greek}', color='black', linewidth=1.5)
        ax.plot(S, greeks_K1[greek], '--', label=f'{greek} Long Put', color='red', linewidth=0.6)
        ax.plot(S, greeks_K2[greek], '--', label=f'{greek} Short Put', color='blue', linewidth=0.6)
    
    if option_strategy == 'Straddle':
        K = float(col1.text_input('Choose a strike for K', value = 100))
        greeks_1 = creating_greek(S, K, t, r, q, sigma, 'Call')
        greeks_2 = creating_greek(S, K, t, r, q, sigma, 'Put')
        greeks = greeks_1 + greeks_2
        # Creating 4 plots 
        fig, ax = plt.subplots(1, figsize = (8,5))
         
        # Plotting Deltas for call options 
        ax.plot(S, greeks[greek], label=f'{greek}', color='black', linewidth=1.5)
        ax.plot(S, greeks_1[greek], '--', label=f'{greek} Long Call', color='red', linewidth=0.6)
        ax.plot(S, greeks_2[greek], '--', label=f'{greek} Long Put', color='blue', linewidth=0.6)
    
    if option_strategy == 'Strangle':
        K1 = float(col1.text_input('Choose a strike for K1', value = 90))
        K2 = float(col2.text_input('Choose a strike for K2', value = 110))
        greeks_1 = creating_greek(S, K1, t, r, q, sigma, 'Put')
        greeks_2 = creating_greek(S, K2, t, r, q, sigma, 'Call')
        greeks = greeks_1 + greeks_2
        # Creating 4 plots 
        fig, ax = plt.subplots(1, figsize = (8,5))
         
        # Plotting Deltas for call options 
        ax.plot(S, greeks[greek], label=f'{greek}', color='black', linewidth=1.5)
        ax.plot(S, greeks_1[greek], '--', label=f'{greek} Long Put', color='red', linewidth=0.6)
        ax.plot(S, greeks_2[greek], '--', label=f'{greek} Long Call', color='blue', linewidth=0.6)
        
    if option_strategy == 'Strip':
        K = float(col1.text_input('Choose a strike for K', value = 100))
        greeks_1 = creating_greek(S, K, t, r, q, sigma, 'Call')
        greeks_2 = 2 * creating_greek(S, K, t, r, q, sigma, 'Put')
        greeks = greeks_1 + greeks_2
        # Creating 4 plots 
        fig, ax = plt.subplots(1, figsize = (8,5))
         
        # Plotting Deltas for call options 
        ax.plot(S, greeks[greek], label=f'{greek}', color='black', linewidth=1.5)
        ax.plot(S, greeks_1[greek], '--', label=f'{greek} Long Call', color='red', linewidth=0.6)
        ax.plot(S, greeks_2[greek], '--', label=f'{greek} Long 2 Puts', color='blue', linewidth=0.6)
        
    if option_strategy == 'Strap':
        K = float(col1.text_input('Choose a strike for K', value = 100))
        greeks_1 = 2 * creating_greek(S, K, t, r, q, sigma, 'Call')
        greeks_2 = creating_greek(S, K, t, r, q, sigma, 'Put')
        greeks = greeks_1 + greeks_2
        # Creating 4 plots 
        fig, ax = plt.subplots(1, figsize = (8,5))
         
        # Plotting Deltas for call options 
        ax.plot(S, greeks[greek], label=f'{greek}', color='black', linewidth=1.5)
        ax.plot(S, greeks_1[greek], '--', label=f'{greek} Long 2 Calls', color='red', linewidth=0.6)
        ax.plot(S, greeks_2[greek], '--', label=f'{greek} Long Put', color='blue', linewidth=0.6)
    
    if option_strategy == 'Call Backspread':
        K1 = float(col1.text_input('Choose a strike for K1', value = 90))
        K2 = float(col2.text_input('Choose a strike for K2', value = 110))
        greeks_1 = - creating_greek(S, K1, t, r, q, sigma, 'Call')
        greeks_2 = 2 * creating_greek(S, K2, t, r, q, sigma, 'Call')
        greeks = greeks_1 + greeks_2
        # Creating 4 plots 
        fig, ax = plt.subplots(1, figsize = (8,5))
         
        # Plotting Deltas for call options 
        ax.plot(S, greeks[greek], label=f'{greek}', color='black', linewidth=1.5)
        ax.plot(S, greeks_1[greek], '--', label=f'{greek} Short Call', color='red', linewidth=0.6)
        ax.plot(S, greeks_2[greek], '--', label=f'{greek} Long 2 Calls', color='blue', linewidth=0.6)
        
    if option_strategy == 'Put Backspread':
        K1 = float(col1.text_input('Choose a strike for K1', value = 90))
        K2 = float(col2.text_input('Choose a strike for K2', value = 110))
        greeks_1 = - 2 * creating_greek(S, K1, t, r, q, sigma, 'Put')
        greeks_2 = creating_greek(S, K2, t, r, q, sigma, 'Put')
        greeks = greeks_1 + greeks_2
        # Creating 4 plots 
        fig, ax = plt.subplots(1, figsize = (8,5))
         
        # Plotting Deltas for call options 
        ax.plot(S, greeks[greek], label=f'{greek}', color='black', linewidth=1.5)
        ax.plot(S, greeks_1[greek], '--', label=f'{greek} Long 2 Puts', color='red', linewidth=0.6)
        ax.plot(S, greeks_2[greek], '--', label=f'{greek} Short Put', color='blue', linewidth=0.6)
        
    if option_strategy == 'Call Butterfly':
        K1 = float(col1.text_input('Choose a strike for K1', value = 90))
        K3 = float(col2.text_input('Choose a strike for K3', value = 110))
        K2 = (K1+K3)/2
        col1.metric('Strike for K2 is', value = f'${K2:.2f}')
        greeks_1 = creating_greek(S, K1, t, r, q, sigma, 'Call')
        greeks_2 = - 2 * creating_greek(S, K2, t, r, q, sigma, 'Call')
        greeks_3 = creating_greek(S, K3, t, r, q, sigma, 'Call')
        greeks = greeks_1 + greeks_2 + greeks_3
        
        
        fig, ax = plt.subplots(1, figsize = (8,5))
        # Plotting the Greeks for the strategy
        ax.plot(S, greeks[greek], label=f'{greek}', color='black', linewidth=1.5)
        ax.plot(S, greeks_1[greek], '--', label=f'{greek} Long Call', color='red', linewidth=0.6)
        ax.plot(S, greeks_2[greek], '--', label=f'{greek} Short 2 Calls', color='blue', linewidth=0.6)
        ax.plot(S, greeks_3[greek], '--', label=f'{greek} Long Call', color='orange', linewidth=0.6)
     
    if option_strategy == 'Put Butterfly':
        K1 = float(col1.text_input('Choose a strike for K1', value = 90))
        K3 = float(col2.text_input('Choose a strike for K3', value = 110))
        K2 = (K1+K3)/2
        col1.metric('Strike for K2 is', value = f'${K2:.2f}')
        greeks_1 = creating_greek(S, K1, t, r, q, sigma, 'Put')
        greeks_2 = - 2 * creating_greek(S, K2, t, r, q, sigma, 'Put')
        greeks_3 = creating_greek(S, K3, t, r, q, sigma, 'Put')
        greeks = greeks_1 + greeks_2 + greeks_3
        
        
        fig, ax = plt.subplots(1, figsize = (8,5))
        # Plotting the Greeks for the strategy
        ax.plot(S, greeks[greek], label=f'{greek}', color='black', linewidth=1.5)
        ax.plot(S, greeks_1[greek], '--', label=f'{greek} Long Put', color='red', linewidth=0.6)
        ax.plot(S, greeks_2[greek], '--', label=f'{greek} Short 2 Puts', color='blue', linewidth=0.6)
        ax.plot(S, greeks_3[greek], '--', label=f'{greek} Long Put', color='orange', linewidth=0.6)
    
    if option_strategy == 'Call Condor':
        K1 = float(col1.text_input('Choose a strike for K1', value = 90))
        K4 = float(col2.text_input('Choose a strike for K4', value = 110))
        space = float(st.text_input('Enter the space value to get K2 and K3', value = 5))
        K2 = K1 + space
        K3 = K4 - space
        col1.metric('Strike for K2 is', value = f'${K2}')
        col2.metric('Strike for K3 is', value = f'${K3}')
        greeks_1 = creating_greek(S, K1, t, r, q, sigma, 'Call')
        greeks_2 = - creating_greek(S, K2, t, r, q, sigma, 'Call')
        greeks_3 = - creating_greek(S, K3, t, r, q, sigma, 'Call')
        greeks_4 = creating_greek(S, K4, t, r, q, sigma, 'Call')
        greeks = greeks_1 + greeks_2 + greeks_3 + greeks_4
        
        
        fig, ax = plt.subplots(1, figsize = (8,5))
        # Plotting the Greeks for the strategy
        ax.plot(S, greeks[greek], label=f'{greek}', color='black', linewidth=1.5)
        ax.plot(S, greeks_1[greek], '--', label=f'{greek} Long Call', color='red', linewidth=0.6)
        ax.plot(S, greeks_2[greek], '--', label=f'{greek} Short Call', color='blue', linewidth=0.6)
        ax.plot(S, greeks_3[greek], '--', label=f'{greek} Short Call', color='orange', linewidth=0.6)
        ax.plot(S, greeks_4[greek], '--', label=f'{greek} Long Call', color='green', linewidth=0.6)
    
    if option_strategy == 'Put Condor':
        K1 = float(col1.text_input('Choose a strike for K1', value = 90))
        K4 = float(col2.text_input('Choose a strike for K4', value = 110))
        space = float(st.text_input('Enter the space value to get K2 and K3', value = 5))
        K2 = K1 + space
        K3 = K4 - space
        col1.metric('Strike for K2 is', value = f'${K2}')
        col2.metric('Strike for K3 is', value = f'${K3}')
        greeks_1 = creating_greek(S, K1, t, r, q, sigma, 'Put')
        greeks_2 = - creating_greek(S, K2, t, r, q, sigma, 'Put')
        greeks_3 = - creating_greek(S, K3, t, r, q, sigma, 'Put')
        greeks_4 = creating_greek(S, K4, t, r, q, sigma, 'Put')
        greeks = greeks_1 + greeks_2 + greeks_3 + greeks_4
        
        
        fig, ax = plt.subplots(1, figsize = (8,5))
        # Plotting the Greeks for the strategy
        ax.plot(S, greeks[greek], label=f'{greek}', color='black', linewidth=1.5)
        ax.plot(S, greeks_1[greek], '--', label=f'{greek} Long Put', color='red', linewidth=0.6)
        ax.plot(S, greeks_2[greek], '--', label=f'{greek} Short Put', color='blue', linewidth=0.6)
        ax.plot(S, greeks_3[greek], '--', label=f'{greek} Short Put', color='orange', linewidth=0.6)
        ax.plot(S, greeks_4[greek], '--', label=f'{greek} Long Put', color='green', linewidth=0.6)
        
    if option_strategy == 'Call Ladder':
        K1 = float(col1.text_input('Choose a strike for K1', value = 90))
        K2 = float(col2.text_input('Choose a strike for K2', value = 100))
        K3 = float(col1.text_input('Choose a strike for K3', value = 110))
        greeks_1 = creating_greek(S, K1, t, r, q, sigma, 'Call')
        greeks_2 = - creating_greek(S, K2, t, r, q, sigma, 'Call')
        greeks_3 = - creating_greek(S, K3, t, r, q, sigma, 'Call')
        greeks = greeks_1 + greeks_2 + greeks_3
        
        
        fig, ax = plt.subplots(1, figsize = (8,5))
        # Plotting the Greeks for the strategy
        ax.plot(S, greeks[greek], label=f'{greek}', color='black', linewidth=1.5)
        ax.plot(S, greeks_1[greek], '--', label=f'{greek} Long Call', color='red', linewidth=0.6)
        ax.plot(S, greeks_2[greek], '--', label=f'{greek} Short Call', color='blue', linewidth=0.6)
        ax.plot(S, greeks_3[greek], '--', label=f'{greek} Short Call', color='orange', linewidth=0.6)

    if option_strategy == 'Put Ladder':
        K1 = float(col1.text_input('Choose a strike for K1', value = 90))
        K2 = float(col2.text_input('Choose a strike for K2', value = 100))
        K3 = float(col1.text_input('Choose a strike for K3', value = 110))
        greeks_1 = - creating_greek(S, K1, t, r, q, sigma, 'Put')
        greeks_2 = - creating_greek(S, K2, t, r, q, sigma, 'Put')
        greeks_3 = creating_greek(S, K3, t, r, q, sigma, 'Put')
        greeks = greeks_1 + greeks_2 + greeks_3
        
        
        fig, ax = plt.subplots(1, figsize = (8,5))
        # Plotting the Greeks for the strategy
        ax.plot(S, greeks[greek], label=f'{greek}', color='black', linewidth=1.5)
        ax.plot(S, greeks_1[greek], '--', label=f'{greek} Short Put', color='red', linewidth=0.6)
        ax.plot(S, greeks_2[greek], '--', label=f'{greek} Short Put', color='blue', linewidth=0.6)
        ax.plot(S, greeks_3[greek], '--', label=f'{greek} Long Put', color='orange', linewidth=0.6)
        
    ax.set_xlabel('Stock Price ($)')
    ax.set_ylabel(f'{greek} for {option_strategy} Strategy')
    ax.set_title(f'{greek} {option_strategy} vs Stock Prices', fontsize=8)
    ax.grid(True)
    ax.legend()
     
    st.pyplot(fig)


# PLUS TARD         
if plot_3d:
    # Setting the different ranges
    t_range = np.linspace(0.25, 5, 150)  # Time to Maturity (range)
    r_range = np.linspace(0, 10, 150) # Free Risk Rate (range)
    dividend_range = np.linspace(0, 8, 150) # Dividend Yield (range)
    sigma_range = np.linspace(5, 50, 150) # Volatility (range)
    
    ranges = ['Time to Maturity (Y)','Volatility (in %)', 'Risk Free Rate (in %)', 'Dividend Yield (in %)']
    range_selected = st.selectbox('Choose a parameter to plot in the 3D plot', ranges)
    
    if option_strategy == 'Call' or option_strategy=='Put':
        if not plot_2d:
            K = float(col1.text_input('Choose a strike for K', value = 100))
        # Creating the meshgrid for stock prices and the second parameter chosen
        if range_selected == 'Time to Maturity (Y)':
            S, second_grid = np.meshgrid(S, t_range)
            # Greeks Calculation
            greeks = creating_greek_array(S, K, second_grid, r, q, sigma, option_strategy)
        elif range_selected == 'Risk Free Rate (in %)':
            S, second_grid = np.meshgrid(S, r_range)
            # Greeks Calculation
            greeks = creating_greek_array(S, K, t, second_grid/100, q, sigma, option_strategy)
        elif range_selected == 'Dividend Yield (in %)':
            S, second_grid = np.meshgrid(S, r_range)
            # Greeks Calculation
            greeks = creating_greek_array(S, K, t, r, second_grid/100, sigma, option_strategy)
        else:
            S, second_grid = np.meshgrid(S, sigma_range)
            # Greeks Calculation
            greeks = creating_greek_array(S, K, t, r, q, second_grid/100, option_strategy)
        
        greeks = greeks[greek]
        
    if option_strategy == 'Call Spread':
        if not plot_2d:
            K1 = float(col1.text_input('Choose a strike for K1', value = 90))
            K2 = float(col2.text_input('Choose a strike for K2', value = 110))
        # Creating the meshgrid for stock prices and the second parameter chosen
        if range_selected == 'Time to Maturity (Y)':
            S, second_grid = np.meshgrid(S, t_range)
            # Greeks Calculation
            greeks_1 = creating_greek_array(S, K1, second_grid, r, q, sigma, 'Call')
            greeks_2 = creating_greek_array(S, K2, second_grid, r, q, sigma, 'Call')
        elif range_selected == 'Risk Free Rate (in %)':
            S, second_grid = np.meshgrid(S, r_range)
            # Greeks Calculation
            greeks_1 = creating_greek_array(S, K1, t, second_grid/100, q, sigma, 'Call')
            greeks_2 = creating_greek_array(S, K2, t, second_grid/100, q, sigma, 'Call')
        elif range_selected == 'Dividend Yield (in %)':
            S, second_grid = np.meshgrid(S, r_range)
            # Greeks Calculation
            greeks_1 = creating_greek_array(S, K1, t, r, second_grid/100, sigma, 'Call')
            greeks_2 = creating_greek_array(S, K2, t, r, second_grid/100, sigma, 'Call')
        else:
            S, second_grid = np.meshgrid(S, sigma_range)
            # Greeks Calculation
            greeks_1 = creating_greek_array(S, K1, t, r, q, second_grid/100, 'Call')
            greeks_2 = creating_greek_array(S, K2, t, r, q, second_grid/100, 'Call')

        # Call spread = Long call - Short call
        greeks = greeks_1[greek] - greeks_2[greek]
    
    if option_strategy == 'Put Spread':
        if not plot_2d:
            K1 = float(col1.text_input('Choose a strike for K1', value = 90))
            K2 = float(col2.text_input('Choose a strike for K2', value = 110))
        # Creating the meshgrid for stock prices and the second parameter chosen
        if range_selected == 'Time to Maturity (Y)':
            S, second_grid = np.meshgrid(S, t_range)
            # Greeks Calculation
            greeks_1 = creating_greek_array(S, K1, second_grid, r, q, sigma, 'Put')
            greeks_2 = creating_greek_array(S, K2, second_grid, r, q, sigma, 'Put')
        elif range_selected == 'Risk Free Rate (in %)':
            S, second_grid = np.meshgrid(S, r_range)
            # Greeks Calculation
            greeks_1 = creating_greek_array(S, K1, t, second_grid/100, q, sigma, 'Put')
            greeks_2 = creating_greek_array(S, K2, t, second_grid/100, q, sigma, 'Put')
        elif range_selected == 'Dividend Yield (in %)':
            S, second_grid = np.meshgrid(S, r_range)
            # Greeks Calculation
            greeks_1 = creating_greek_array(S, K1, t, r, second_grid/100, sigma, 'Put')
            greeks_2 = creating_greek_array(S, K2, t, r, second_grid/100, sigma, 'Put')
        else:
            S, second_grid = np.meshgrid(S, sigma_range)
            # Greeks Calculation
            greeks_1 = creating_greek_array(S, K1, t, r, q, second_grid/100, 'Put')
            greeks_2 = creating_greek_array(S, K2, t, r, q, second_grid/100, 'Put')

        # Call spread = Long call - Short call
        greeks = greeks_1[greek] - greeks_2[greek]
        
    if option_strategy == 'Call Backspread':
        if not plot_2d: 
            K1 = float(col1.text_input('Choose a strike for K1', value = 90))
            K2 = float(col2.text_input('Choose a strike for K2', value = 110))
        # Creating the meshgrid for stock prices and the second parameter chosen
        if range_selected == 'Time to Maturity (Y)':
            S, second_grid = np.meshgrid(S, t_range)
            # Greeks Calculation
            greeks_1 = creating_greek_array(S, K1, second_grid, r, q, sigma, 'Call')
            greeks_2 = creating_greek_array(S, K2, second_grid, r, q, sigma, 'Call')
        elif range_selected == 'Risk Free Rate (in %)':
            S, second_grid = np.meshgrid(S, r_range)
            # Greeks Calculation
            greeks_1 = creating_greek_array(S, K1, t, second_grid/100, q, sigma, 'Call')
            greeks_2 = creating_greek_array(S, K2, t, second_grid/100, q, sigma, 'Call')
        elif range_selected == 'Dividend Yield (in %)':
            S, second_grid = np.meshgrid(S, r_range)
            # Greeks Calculation
            greeks_1 = creating_greek_array(S, K1, t, r, second_grid/100, sigma, 'Call')
            greeks_2 = creating_greek_array(S, K2, t, r, second_grid/100, sigma, 'Call')
        else:
            S, second_grid = np.meshgrid(S, sigma_range)
            # Greeks Calculation
            greeks_1 = creating_greek_array(S, K1, t, r, q, second_grid/100, 'Call')
            greeks_2 = creating_greek_array(S, K2, t, r, q, second_grid/100, 'Call')

        # Cal backspread = - Short Call K1 + 2 Long Calls K2
        greeks = - greeks_1[greek] + 2 * greeks_2[greek]
    
    if option_strategy == 'Put Backspread':
        if not plot_2d: 
            K1 = float(col1.text_input('Choose a strike for K1', value = 90))
            K2 = float(col2.text_input('Choose a strike for K2', value = 110))
        # Creating the meshgrid for stock prices and the second parameter chosen
        if range_selected == 'Time to Maturity (Y)':
            S, second_grid = np.meshgrid(S, t_range)
            # Greeks Calculation
            greeks_1 = creating_greek_array(S, K1, second_grid, r, q, sigma, 'Put')
            greeks_2 = creating_greek_array(S, K2, second_grid, r, q, sigma, 'Put')
        elif range_selected == 'Risk Free Rate (in %)':
            S, second_grid = np.meshgrid(S, r_range)
            # Greeks Calculation
            greeks_1 = creating_greek_array(S, K1, t, second_grid/100, q, sigma, 'Put')
            greeks_2 = creating_greek_array(S, K2, t, second_grid/100, q, sigma, 'Put')
        elif range_selected == 'Dividend Yield (in %)':
            S, second_grid = np.meshgrid(S, r_range)
            # Greeks Calculation
            greeks_1 = creating_greek_array(S, K1, t, r, second_grid/100, sigma, 'Put')
            greeks_2 = creating_greek_array(S, K2, t, r, second_grid/100, sigma, 'Put')
        else:
            S, second_grid = np.meshgrid(S, sigma_range)
            # Greeks Calculation
            greeks_1 = creating_greek_array(S, K1, t, r, q, second_grid/100, 'Put')
            greeks_2 = creating_greek_array(S, K2, t, r, q, second_grid/100, 'Put')

        # Call spread = Long 2 Calls K1 - Short Call K2
        greeks = 2 * greeks_1[greek] - greeks_2[greek]
    
    if option_strategy == 'Call Butterfly':
        if not plot_2d:
            K1 = float(col1.text_input('Choose a strike for K1', value = 90))
            K3 = float(col2.text_input('Choose a strike for K3', value = 110))
            K2 = (K1+K3)/2
            col1.metric('Strike for K2 is', value = f'${K2:.2f}')
        # Creating the meshgrid for stock prices and the second parameter chosen
        if range_selected == 'Time to Maturity (Y)':
            S, second_grid = np.meshgrid(S, t_range)
            # Greeks Calculation
            greeks_1 = creating_greek_array(S, K1, second_grid, r, q, sigma, 'Call')
            greeks_2 = creating_greek_array(S, K2, second_grid, r, q, sigma, 'Call')
            greeks_3 = creating_greek_array(S, K3, second_grid, r, q, sigma, 'Call')
        elif range_selected == 'Risk Free Rate (in %)':
            S, second_grid = np.meshgrid(S, r_range)
            # Greeks Calculation
            greeks_1 = creating_greek_array(S, K1, t, second_grid/100, q, sigma, 'Call')
            greeks_2 = creating_greek_array(S, K2, t, second_grid/100, q, sigma, 'Call')
            greeks_3 = creating_greek_array(S, K3, t, second_grid/100, q, sigma, 'Call')
        elif range_selected == 'Dividend Yield (in %)':
            S, second_grid = np.meshgrid(S, r_range)
            # Greeks Calculation
            greeks_1 = creating_greek_array(S, K1, t, r, second_grid/100, sigma, 'Call')
            greeks_2 = creating_greek_array(S, K2, t, r, second_grid/100, sigma, 'Call')
            greeks_3 = creating_greek_array(S, K3, t, r, second_grid/100, sigma, 'Call')
        else:
            S, second_grid = np.meshgrid(S, sigma_range)
            # Greeks Calculation
            greeks_1 = creating_greek_array(S, K1, t, r, q, second_grid/100, 'Call')
            greeks_2 = creating_greek_array(S, K2, t, r, q, second_grid/100, 'Call')
            greeks_3 = creating_greek_array(S, K3, t, r, q, second_grid/100, 'Call')

        # Call spread = Long call K1 - Short 2 Calls K2 + Long Call K3
        greeks = greeks_1[greek] - 2 * greeks_2[greek] + greeks_3[greek]
        
    if option_strategy == 'Put Butterfly':
        if not plot_2d:
            K1 = float(col1.text_input('Choose a strike for K1', value = 90))
            K3 = float(col2.text_input('Choose a strike for K3', value = 110))
            K2 = (K1+K3)/2
            col1.metric('Strike for K2 is', value = f'${K2:.2f}')
        # Creating the meshgrid for stock prices and the second parameter chosen
        if range_selected == 'Time to Maturity (Y)':
            S, second_grid = np.meshgrid(S, t_range)
            # Greeks Calculation
            greeks_1 = creating_greek_array(S, K1, second_grid, r, q, sigma, 'Put')
            greeks_2 = creating_greek_array(S, K2, second_grid, r, q, sigma, 'Put')
            greeks_3 = creating_greek_array(S, K3, second_grid, r, q, sigma, 'Put')
        elif range_selected == 'Risk Free Rate (in %)':
            S, second_grid = np.meshgrid(S, r_range)
            # Greeks Calculation
            greeks_1 = creating_greek_array(S, K1, t, second_grid/100, q, sigma, 'Put')
            greeks_2 = creating_greek_array(S, K2, t, second_grid/100, q, sigma, 'Put')
            greeks_3 = creating_greek_array(S, K3, t, second_grid/100, q, sigma, 'Put')
        elif range_selected == 'Dividend Yield (in %)':
            S, second_grid = np.meshgrid(S, r_range)
            # Greeks Calculation
            greeks_1 = creating_greek_array(S, K1, t, r, second_grid/100, sigma, 'Put')
            greeks_2 = creating_greek_array(S, K2, t, r, second_grid/100, sigma, 'Put')
            greeks_3 = creating_greek_array(S, K3, t, r, second_grid/100, sigma, 'Put')
        else:
            S, second_grid = np.meshgrid(S, sigma_range)
            # Greeks Calculation
            greeks_1 = creating_greek_array(S, K1, t, r, q, second_grid/100, 'Put')
            greeks_2 = creating_greek_array(S, K2, t, r, q, second_grid/100, 'Put')
            greeks_3 = creating_greek_array(S, K3, t, r, q, second_grid/100, 'Put')

        # Call spread = Long Put K1 - Short 2 Puts K2 + Long Put K3
        greeks = greeks_1[greek] - 2 * greeks_2[greek] + greeks_3[greek]
        
    if option_strategy == 'Straddle':
        if not plot_2d:
            K = float(col1.text_input('Choose a strike for K', value = 100))
        # Creating the meshgrid for stock prices and the second parameter chosen
        if range_selected == 'Time to Maturity (Y)':
            S, second_grid = np.meshgrid(S, t_range)
            # Greeks Calculation
            greeks_1 = creating_greek_array(S, K, second_grid, r, q, sigma, 'Call')
            greeks_2 = creating_greek_array(S, K, second_grid, r, q, sigma, 'Put')
        elif range_selected == 'Risk Free Rate (in %)':
            S, second_grid = np.meshgrid(S, r_range)
            # Greeks Calculation
            greeks_1 = creating_greek_array(S, K, t, second_grid/100, q, sigma, 'Call')
            greeks_2 = creating_greek_array(S, K, t, second_grid/100, q, sigma, 'Put')
        elif range_selected == 'Dividend Yield (in %)':
            S, second_grid = np.meshgrid(S, r_range)
            # Greeks Calculation
            greeks_1 = creating_greek_array(S, K, t, r, second_grid/100, sigma, 'Call')
            greeks_2 = creating_greek_array(S, K, t, r, second_grid/100, sigma, 'Put')
        else:
            S, second_grid = np.meshgrid(S, sigma_range)
            # Greeks Calculation
            greeks_1 = creating_greek_array(S, K, t, r, q, second_grid/100, 'Call')
            greeks_2 = creating_greek_array(S, K, t, r, q, second_grid/100, 'Put')

        # Straddle = Long call + Long Short
        greeks = greeks_1[greek] + greeks_2[greek]
        
    if option_strategy == 'Strangle':
        if not plot_2d: 
            K1 = float(col1.text_input('Choose a strike for K1', value = 90))
            K2 = float(col2.text_input('Choose a strike for K2', value = 110))
        # Creating the meshgrid for stock prices and the second parameter chosen
        if range_selected == 'Time to Maturity (Y)':
            S, second_grid = np.meshgrid(S, t_range)
            # Greeks Calculation
            greeks_1 = creating_greek_array(S, K1, second_grid, r, q, sigma, 'Call')
            greeks_2 = creating_greek_array(S, K2, second_grid, r, q, sigma, 'Put')
        elif range_selected == 'Risk Free Rate (in %)':
            S, second_grid = np.meshgrid(S, r_range)
            # Greeks Calculation
            greeks_1 = creating_greek_array(S, K1, t, second_grid/100, q, sigma, 'Call')
            greeks_2 = creating_greek_array(S, K2, t, second_grid/100, q, sigma, 'Put')
        elif range_selected == 'Dividend Yield (in %)':
            S, second_grid = np.meshgrid(S, r_range)
            # Greeks Calculation
            greeks_1 = creating_greek_array(S, K1, t, r, second_grid/100, sigma, 'Call')
            greeks_2 = creating_greek_array(S, K2, t, r, second_grid/100, sigma, 'Put')
        else:
            S, second_grid = np.meshgrid(S, sigma_range)
            # Greeks Calculation
            greeks_1 = creating_greek_array(S, K1, t, r, q, second_grid/100, 'Call')
            greeks_2 = creating_greek_array(S, K2, t, r, q, second_grid/100, 'Put')

        # Straddle = Long call + Long Short
        greeks = greeks_1[greek] + greeks_2[greek]
        
    if option_strategy == 'Strap':
        if not plot_2d:
            K = float(col1.text_input('Choose a strike for K', value = 100))
        # Creating the meshgrid for stock prices and the second parameter chosen
        if range_selected == 'Time to Maturity (Y)':
            S, second_grid = np.meshgrid(S, t_range)
            # Greeks Calculation
            greeks_1 = creating_greek_array(S, K, second_grid, r, q, sigma, 'Call')
            greeks_2 = creating_greek_array(S, K, second_grid, r, q, sigma, 'Put')
        elif range_selected == 'Risk Free Rate (in %)':
            S, second_grid = np.meshgrid(S, r_range)
            # Greeks Calculation
            greeks_1 = creating_greek_array(S, K, t, second_grid/100, q, sigma, 'Call')
            greeks_2 = creating_greek_array(S, K, t, second_grid/100, q, sigma, 'Put')
        elif range_selected == 'Dividend Yield (in %)':
            S, second_grid = np.meshgrid(S, r_range)
            # Greeks Calculation
            greeks_1 = creating_greek_array(S, K, t, r, second_grid/100, sigma, 'Call')
            greeks_2 = creating_greek_array(S, K, t, r, second_grid/100, sigma, 'Put')
        else:
            S, second_grid = np.meshgrid(S, sigma_range)
            # Greeks Calculation
            greeks_1 = creating_greek_array(S, K, t, r, q, second_grid/100, 'Call')
            greeks_2 = creating_greek_array(S, K, t, r, q, second_grid/100, 'Put')

        # Straddle = Long call + Long Short
        greeks = 2 * greeks_1[greek] + greeks_2[greek]
        
    if option_strategy == 'Strip':
        if not plot_2d:
            K = float(col1.text_input('Choose a strike for K', value = 100))
        # Creating the meshgrid for stock prices and the second parameter chosen
        if range_selected == 'Time to Maturity (Y)':
            S, second_grid = np.meshgrid(S, t_range)
            # Greeks Calculation
            greeks_1 = creating_greek_array(S, K, second_grid, r, q, sigma, 'Call')
            greeks_2 = creating_greek_array(S, K, second_grid, r, q, sigma, 'Put')
        elif range_selected == 'Risk Free Rate (in %)':
            S, second_grid = np.meshgrid(S, r_range)
            # Greeks Calculation
            greeks_1 = creating_greek_array(S, K, t, second_grid/100, q, sigma, 'Call')
            greeks_2 = creating_greek_array(S, K, t, second_grid/100, q, sigma, 'Put')
        elif range_selected == 'Dividend Yield (in %)':
            S, second_grid = np.meshgrid(S, r_range)
            # Greeks Calculation
            greeks_1 = creating_greek_array(S, K, t, r, second_grid/100, sigma, 'Call')
            greeks_2 = creating_greek_array(S, K, t, r, second_grid/100, sigma, 'Put')
        else:
            S, second_grid = np.meshgrid(S, sigma_range)
            # Greeks Calculation
            greeks_1 = creating_greek_array(S, K, t, r, q, second_grid/100, 'Call')
            greeks_2 = creating_greek_array(S, K, t, r, q, second_grid/100, 'Put')

        # Straddle = Long call + Long Short
        greeks = greeks_1[greek] + 2 * greeks_2[greek]
        
    if option_strategy == 'Call Condor':
        if not plot_2d:
            K1 = float(col1.text_input('Choose a strike for K1', value = 90))
            K4 = float(col2.text_input('Choose a strike for K4', value = 110))
            space = float(col1.text_input('Enter the space value to get K2 and K3', value = 5))
            K2 = K1 + space
            K3 = K4 - space
            col1.metric('Strike for K2 is', value = f'${K2}')
            col1.metric('Strike for K3 is', value = f'${K3}')
        # Creating the meshgrid for stock prices and the second parameter chosen
        if range_selected == 'Time to Maturity (Y)':
            S, second_grid = np.meshgrid(S, t_range)
            # Greeks Calculation
            greeks_1 = creating_greek_array(S, K1, second_grid, r, q, sigma, 'Call')
            greeks_2 = creating_greek_array(S, K2, second_grid, r, q, sigma, 'Call')
            greeks_3 = creating_greek_array(S, K3, second_grid, r, q, sigma, 'Call')
            greeks_4 = creating_greek_array(S, K4, second_grid, r, q, sigma, 'Call')
        elif range_selected == 'Risk Free Rate (in %)':
            S, second_grid = np.meshgrid(S, r_range)
            # Greeks Calculation
            greeks_1 = creating_greek_array(S, K1, t, second_grid/100, q, sigma, 'Call')
            greeks_2 = creating_greek_array(S, K2, t, second_grid/100, q, sigma, 'Call')
            greeks_3 = creating_greek_array(S, K3, t, second_grid/100, q, sigma, 'Call')
            greeks_4 = creating_greek_array(S, K4, t, second_grid/100, q, sigma, 'Call')
        elif range_selected == 'Dividend Yield (in %)':
            S, second_grid = np.meshgrid(S, r_range)
            # Greeks Calculation
            greeks_1 = creating_greek_array(S, K1, t, r, second_grid/100, sigma, 'Call')
            greeks_2 = creating_greek_array(S, K2, t, r, second_grid/100, sigma, 'Call')
            greeks_3 = creating_greek_array(S, K3, t, r, second_grid/100, sigma, 'Call')
            greeks_4 = creating_greek_array(S, K4, t, r, second_grid/100, sigma, 'Call')            
        else:
            S, second_grid = np.meshgrid(S, sigma_range)
            # Greeks Calculation
            greeks_1 = creating_greek_array(S, K1, t, r, q, second_grid/100, 'Call')
            greeks_2 = creating_greek_array(S, K2, t, r, q, second_grid/100, 'Call')
            greeks_3 = creating_greek_array(S, K3, t, r, q, second_grid/100, 'Call')
            greeks_4 = creating_greek_array(S, K4, t, r, q, second_grid/100, 'Call')            

        # Call spread = Long Put K1 - Short 2 Puts K2 + Long Put K3
        greeks = greeks_1[greek] - greeks_2[greek] - greeks_3[greek] + greeks_4[greek]
        
    if option_strategy == 'Put Condor':
        if not plot_2d:
            K1 = float(col1.text_input('Choose a strike for K1', value = 90))
            K4 = float(col2.text_input('Choose a strike for K4', value = 110))
            space = float(st.text_input('Enter the space value to get K2 and K3', value = 5))
            K2 = K1 + space
            K3 = K4 - space
            col1.metric('Strike for K2 is', value = f'${K2}')
            col2.metric('Strike for K3 is', value = f'${K3}')
        # Creating the meshgrid for stock prices and the second parameter chosen
        if range_selected == 'Time to Maturity (Y)':
            S, second_grid = np.meshgrid(S, t_range)
            # Greeks Calculation
            greeks_1 = creating_greek_array(S, K1, second_grid, r, q, sigma, 'Put')
            greeks_2 = creating_greek_array(S, K2, second_grid, r, q, sigma, 'Put')
            greeks_3 = creating_greek_array(S, K3, second_grid, r, q, sigma, 'Put')
            greeks_4 = creating_greek_array(S, K4, second_grid, r, q, sigma, 'Put')
        elif range_selected == 'Risk Free Rate (in %)':
            S, second_grid = np.meshgrid(S, r_range)
            # Greeks Calculation
            greeks_1 = creating_greek_array(S, K1, t, second_grid/100, q, sigma, 'Put')
            greeks_2 = creating_greek_array(S, K2, t, second_grid/100, q, sigma, 'Put')
            greeks_3 = creating_greek_array(S, K3, t, second_grid/100, q, sigma, 'Put')
            greeks_4 = creating_greek_array(S, K4, t, second_grid/100, q, sigma, 'Put')
        elif range_selected == 'Dividend Yield (in %)':
            S, second_grid = np.meshgrid(S, r_range)
            # Greeks Calculation
            greeks_1 = creating_greek_array(S, K1, t, r, second_grid/100, sigma, 'Put')
            greeks_2 = creating_greek_array(S, K2, t, r, second_grid/100, sigma, 'Put')
            greeks_3 = creating_greek_array(S, K3, t, r, second_grid/100, sigma, 'Put')
            greeks_4 = creating_greek_array(S, K4, t, r, second_grid/100, sigma, 'Put')            
        else:
            S, second_grid = np.meshgrid(S, sigma_range)
            # Greeks Calculation
            greeks_1 = creating_greek_array(S, K1, t, r, q, second_grid/100, 'Put')
            greeks_2 = creating_greek_array(S, K2, t, r, q, second_grid/100, 'Put')
            greeks_3 = creating_greek_array(S, K3, t, r, q, second_grid/100, 'Put')
            greeks_4 = creating_greek_array(S, K4, t, r, q, second_grid/100, 'Put')            

        # Call spread = Long Put K1 - Short 2 Puts K2 + Long Put K3
        greeks = greeks_1[greek] - greeks_2[greek] - greeks_3[greek] + greeks_4[greek]
        
    if option_strategy == 'Call Ladder':
        if not plot_2d:
            K1 = float(col1.text_input('Choose a strike for K1', value = 90))
            K2 = float(col2.text_input('Choose a strike for K2', value = 100))
            K3 = float(col1.text_input('Choose a strike for K3', value = 110))
        # Creating the meshgrid for stock prices and the second parameter chosen
        if range_selected == 'Time to Maturity (Y)':
            S, second_grid = np.meshgrid(S, t_range)
            # Greeks Calculation
            greeks_1 = creating_greek_array(S, K1, second_grid, r, q, sigma, 'Call')
            greeks_2 = creating_greek_array(S, K2, second_grid, r, q, sigma, 'Call')
            greeks_3 = creating_greek_array(S, K3, second_grid, r, q, sigma, 'Call')
        elif range_selected == 'Risk Free Rate (in %)':
            S, second_grid = np.meshgrid(S, r_range)
            # Greeks Calculation
            greeks_1 = creating_greek_array(S, K1, t, second_grid/100, q, sigma, 'Call')
            greeks_2 = creating_greek_array(S, K2, t, second_grid/100, q, sigma, 'Call')
            greeks_3 = creating_greek_array(S, K3, t, second_grid/100, q, sigma, 'Call')
        elif range_selected == 'Dividend Yield (in %)':
            S, second_grid = np.meshgrid(S, r_range)
            # Greeks Calculation
            greeks_1 = creating_greek_array(S, K1, t, r, second_grid/100, sigma, 'Call')
            greeks_2 = creating_greek_array(S, K2, t, r, second_grid/100, sigma, 'Call')
            greeks_3 = creating_greek_array(S, K3, t, r, second_grid/100, sigma, 'Call')          
        else:
            S, second_grid = np.meshgrid(S, sigma_range)
            # Greeks Calculation
            greeks_1 = creating_greek_array(S, K1, t, r, q, second_grid/100, 'Call')
            greeks_2 = creating_greek_array(S, K2, t, r, q, second_grid/100, 'Call')
            greeks_3 = creating_greek_array(S, K3, t, r, q, second_grid/100, 'Call')           

        # Call spread = Long Cal K1 + Short Call K2 + Short Call K3
        greeks = greeks_1[greek] - greeks_2[greek] - greeks_3[greek]
        
    if option_strategy == 'Put Ladder':
        if not plot_2d:
            K1 = float(col1.text_input('Choose a strike for K1', value = 90))
            K2 = float(col2.text_input('Choose a strike for K2', value = 100))
            K3 = float(col1.text_input('Choose a strike for K3', value = 110))
        # Creating the meshgrid for stock prices and the second parameter chosen
        if range_selected == 'Time to Maturity (Y)':
            S, second_grid = np.meshgrid(S, t_range)
            # Greeks Calculation
            greeks_1 = creating_greek_array(S, K1, second_grid, r, q, sigma, 'Put')
            greeks_2 = creating_greek_array(S, K2, second_grid, r, q, sigma, 'Put')
            greeks_3 = creating_greek_array(S, K3, second_grid, r, q, sigma, 'Put')
        elif range_selected == 'Risk Free Rate (in %)':
            S, second_grid = np.meshgrid(S, r_range)
            # Greeks Calculation
            greeks_1 = creating_greek_array(S, K1, t, second_grid/100, q, sigma, 'Put')
            greeks_2 = creating_greek_array(S, K2, t, second_grid/100, q, sigma, 'Put')
            greeks_3 = creating_greek_array(S, K3, t, second_grid/100, q, sigma, 'Put')
        elif range_selected == 'Dividend Yield (in %)':
            S, second_grid = np.meshgrid(S, r_range)
            # Greeks Calculation
            greeks_1 = creating_greek_array(S, K1, t, r, second_grid/100, sigma, 'Put')
            greeks_2 = creating_greek_array(S, K2, t, r, second_grid/100, sigma, 'Put')
            greeks_3 = creating_greek_array(S, K3, t, r, second_grid/100, sigma, 'Put')          
        else:
            S, second_grid = np.meshgrid(S, sigma_range)
            # Greeks Calculation
            greeks_1 = creating_greek_array(S, K1, t, r, q, second_grid/100, 'Put')
            greeks_2 = creating_greek_array(S, K2, t, r, q, second_grid/100, 'Put')
            greeks_3 = creating_greek_array(S, K3, t, r, q, second_grid/100, 'Put')           

        # Call spread = Short Put K1 + Short Put K2 + Long Put K3
        greeks = - greeks_1[greek] - greeks_2[greek] + greeks_3[greek]
    
    # Plot
    fig = plt.figure(figsize=(12, 7))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(S, second_grid, greeks, cmap='viridis', edgecolor='k', alpha=0.8)
    ax.set_title(f'{greek} of {option_strategy} vs Stock Price ($) & {range_selected}')
    ax.set_xlabel('Stock Price ($)')
    ax.set_ylabel(f'{range_selected}')
    ax.set_zlabel(f'{greek}')
    fig.colorbar(surf, shrink=0.5, aspect=8, label=greek)
    st.pyplot(fig)