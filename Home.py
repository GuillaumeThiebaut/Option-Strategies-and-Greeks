import streamlit as st

# Setting the description and the title
st.title('Option Strategies & Hedge Ratios')
st.text("""Welcome! Here’s what you’ll discover:
        
1. Option Strategies – Learn how to design and implement powerful trading strategies 
using options.""")
st.page_link("pages/1_Option Strategies.py", label="📈 Option Strategies")

st.text("""2. Strategies & Hedge Ratios – Explore how hedge ratios change across different 
strategies and how to apply them effectively.""")
st.page_link("pages/2_Hedge Ratios.py", label="📊 Hedge Ratios")

st.text("""3. Key Formulas – Access the essential formulas that drive the strategies
used in this website.""")
st.page_link("pages/3_Key Formulas.py", label="🧠 Key Formulas")

st.write("### Author of the Website")
st.markdown(
    """
    <div style='margin-bottom: 25px; padding: 15px; background-color: #ebebeb; border-radius: 15px;'>
        <span style='font-weight: bold; color: #0A66C2; font-size: 24px;'>Created by:</span><br>
        <a href='https://www.linkedin.com/in/guillaume-thiebaut-88a137283/' target='_blank' style='text-decoration: none; display: flex; align-items: center; gap: 12px; margin-top: 8px;'>
            <img src='https://cdn-icons-png.flaticon.com/512/174/174857.png' width='32' height='32'/>
            <span style='color: #0A66C2; font-size: 18px; font-weight: bold;'>Guillaume Thiebaut</span>
        </a>
    </div>
    """,
    unsafe_allow_html=True
)