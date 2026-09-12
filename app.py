import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import date
import numpy as np

# Set page config with a dark theme
st.set_page_config(
    page_title="Mortgage Analyzer",
    page_icon="💻",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for a more tech-focused theme
st.markdown("""
    <style>
    .stApp {
        background-color: #0E1117;
        color: #FAFAFA;
    }
    .stButton>button {
        background-color: #1E88E5;
        color: white;
        border-radius: 5px;
        border: none;
        padding: 0.5rem 1rem;
    }
    .stButton>button:hover {
        background-color: #1565C0;
    }
    .css-1d391kg {
        background-color: #1E1E1E;
    }
    .stMetric {
        background-color: #1E1E1E;
        padding: 1rem;
        border-radius: 5px;
    }
    </style>
    """, unsafe_allow_html=True)

def calculate_monthly_payment(principal, annual_rate, years):
    try:
        monthly_rate = annual_rate / 12 / 100
        num_payments = years * 12
        
        # Handle edge cases to prevent overflow
        if monthly_rate == 0:
            return principal / num_payments
        if monthly_rate > 0.99:  # Very high interest rate
            return principal * monthly_rate * 1.1
        
        monthly_payment = principal * (monthly_rate * (1 + monthly_rate)**num_payments) / ((1 + monthly_rate)**num_payments - 1)
        return monthly_payment
    except OverflowError:
        # Fallback calculation for very large numbers
        return principal / num_payments

def generate_amortization_schedule(principal, annual_rate, years):
    monthly_rate = annual_rate / 12 / 100
    num_payments = years * 12
    monthly_payment = calculate_monthly_payment(principal, annual_rate, years)
    
    schedule = []
    balance = principal
    cumulative_principal = 0
    cumulative_interest = 0
    
    for payment in range(1, num_payments + 1):
        interest_payment = balance * monthly_rate
        principal_payment = monthly_payment - interest_payment
        balance = max(0, balance - principal_payment)  # Prevent negative balance
        
        cumulative_principal += principal_payment
        cumulative_interest += interest_payment
        
        schedule.append({
            'Payment #': payment,
            'Date': (date.today().replace(day=1) + pd.DateOffset(months=payment-1)).strftime('%Y-%m-%d'),
            'Payment': f"${monthly_payment:,.2f}",
            'Principal': f"${principal_payment:,.2f}",
            'Interest': f"${interest_payment:,.2f}",
            'Balance': f"${balance:,.2f}",
            'Cumulative Principal': cumulative_principal,
            'Cumulative Interest': cumulative_interest,
            'Cumulative Total': cumulative_principal + cumulative_interest
        })
    
    return pd.DataFrame(schedule)

def generate_early_repayment_schedule(principal, annual_rate, years, early_payments):
    """Generate amortization schedule with early repayment points"""
    monthly_rate = annual_rate / 12 / 100
    num_payments = years * 12
    monthly_payment = calculate_monthly_payment(principal, annual_rate, years)
    
    schedule = []
    balance = principal
    cumulative_principal = 0
    cumulative_interest = 0
    
    # Sort early payments by payment number
    early_payments = sorted(early_payments, key=lambda x: x['payment_number'])
    
    for payment in range(1, num_payments + 1):
        interest_payment = balance * monthly_rate
        principal_payment = monthly_payment - interest_payment
        
        # Check if this is an early repayment point
        extra_payment = 0
        for early_pay in early_payments:
            if early_pay['payment_number'] == payment:
                extra_payment = early_pay['amount']
                break
        
        # Apply extra payment to principal
        principal_payment += extra_payment
        balance = max(0, balance - principal_payment)
        
        cumulative_principal += principal_payment
        cumulative_interest += interest_payment
        
        # If balance is paid off, stop
        if balance <= 0:
            break
        
        schedule.append({
            'Payment #': payment,
            'Date': (date.today().replace(day=1) + pd.DateOffset(months=payment-1)).strftime('%Y-%m-%d'),
            'Payment': f"${monthly_payment + extra_payment:,.2f}",
            'Principal': f"${principal_payment:,.2f}",
            'Interest': f"${interest_payment:,.2f}",
            'Balance': f"${balance:,.2f}",
            'Cumulative Principal': cumulative_principal,
            'Cumulative Interest': cumulative_interest,
            'Cumulative Total': cumulative_principal + cumulative_interest,
            'Extra Payment': f"${extra_payment:,.2f}",
            'Has Early Payment': extra_payment > 0
        })
    
    return pd.DataFrame(schedule)

def calculate_extra_payment_impact(principal, annual_rate, years, extra_payment):
    """Calculate the impact of extra payments"""
    monthly_rate = annual_rate / 12 / 100
    original_payment = calculate_monthly_payment(principal, annual_rate, years)
    total_payment = original_payment + extra_payment
    
    # Calculate new loan term with extra payments
    if monthly_rate > 0:
        new_term = np.log(total_payment / (total_payment - principal * monthly_rate)) / np.log(1 + monthly_rate)
        new_years = new_term / 12
    else:
        new_years = principal / total_payment / 12
    
    interest_saved = (original_payment * years * 12) - (total_payment * new_years * 12)
    
    return {
        'new_years': max(0, new_years),
        'interest_saved': max(0, interest_saved),
        'total_payment': total_payment
    }

def calculate_refinance_comparison(original_principal, original_rate, original_years, new_rate, new_years):
    """Compare original loan with refinance options"""
    original_payment = calculate_monthly_payment(original_principal, original_rate, original_years)
    new_payment = calculate_monthly_payment(original_principal, new_rate, new_years)
    
    original_total = original_payment * original_years * 12
    new_total = new_payment * new_years * 12
    
    monthly_savings = original_payment - new_payment
    total_savings = original_total - new_total
    
    return {
        'original_payment': original_payment,
        'new_payment': new_payment,
        'monthly_savings': monthly_savings,
        'total_savings': total_savings
    }

# Sidebar for inputs
with st.sidebar:
    st.title("💻 Mortgage Parameters")
    
    property_value = st.number_input(
        "Property Value",
        min_value=0,
        value=300000,
        step=10000,
        format="%d"
    )
    
    down_payment = st.number_input(
        "Down Payment",
        min_value=0,
        value=60000,
        step=1000,
        format="%d"
    )
    
    loan_amount = property_value - down_payment
    st.metric("Loan Amount", f"${loan_amount:,.2f}")
    
    interest_rate = st.number_input(
        "Annual Interest Rate (%)",
        min_value=0.0,
        value=5.0,
        step=0.01,
        format="%.2f"
    )
    
    loan_term = st.number_input(
        "Loan Term (Years)",
        min_value=1,
        value=30,
        step=1,
        format="%d"
    )

# Main content area
st.title("🏦 Mortgage Analyzer")

# Create tabs
tab1, tab2, tab3, tab4 = st.tabs(["📊 Basic Analysis", "💰 Advanced Features", "🔄 Refinance Calculator", "⚡ Early Repayment Simulator"])

with tab1:
    st.header("📊 Basic Mortgage Analysis")
    
    # Calculate key metrics
    monthly_payment = calculate_monthly_payment(loan_amount, interest_rate, loan_term)
    total_payments = monthly_payment * loan_term * 12
    total_interest = total_payments - loan_amount

    # Display metrics in a row
    col1, col2, col3, col4, col5 = st.columns(5)  
    with col1:
        st.metric("Monthly Payment", f"${monthly_payment:,.2f}")
    with col2:
        st.metric("Total Payments", f"${total_payments:,.2f}")
    with col3:
        st.metric("Total Interest", f"${total_interest:,.2f}")
    with col4:
        st.metric("Interest Ratio", f"{(total_interest/loan_amount)*100:.1f}%")
    with col5:
        st.metric("Loan to Value", f"{(1-(down_payment/property_value))*100:.1f}%")

    # Early Repayment Section
    st.subheader("⚡ Early Repayment Simulation")
    
    # Initialize session state for early payments
    if 'early_payments' not in st.session_state:
        st.session_state.early_payments = []
    
    # Add new early payment
    col1, col2, col3 = st.columns(3)
    
    with col1:
        payment_number = st.number_input(
            "Payment Number",
            min_value=1,
            max_value=loan_term * 12,
            value=12,
            step=1,
            format="%d"
        )
    
    with col2:
        extra_amount = st.number_input(
            "Extra Amount ($)",
            min_value=0,
            value=5000,
            step=1000,
            format="%d"
        )
    
    with col3:
        if st.button("Add Early Payment", use_container_width=True):
            new_payment = {
                'payment_number': payment_number,
                'amount': extra_amount
            }
            st.session_state.early_payments.append(new_payment)
            st.success(f"Added ${extra_amount:,.2f} at payment #{payment_number}")
    
    # Display current early payments
    if st.session_state.early_payments:
        st.write("**Current Early Payments:**")
        early_payments_df = pd.DataFrame(st.session_state.early_payments)
        early_payments_df['Year'] = early_payments_df['payment_number'] / 12
        early_payments_df['Month'] = early_payments_df['payment_number'] % 12
        early_payments_df['Year'] = early_payments_df['Year'].apply(lambda x: f"Year {int(x) + 1}, Month {int(x % 1 * 12) + 1}")
        
        st.dataframe(
            early_payments_df[['payment_number', 'Year', 'amount']].rename(
                columns={'payment_number': 'Payment #', 'amount': 'Amount ($)'}
            ),
            use_container_width=True,
            height=150
        )
        
        if st.button("Clear All Early Payments"):
            st.session_state.early_payments = []
            st.rerun()
    
    # Generate appropriate schedule
    if st.session_state.early_payments:
        schedule_df = generate_early_repayment_schedule(
            loan_amount, interest_rate, loan_term, st.session_state.early_payments
        )
        
        # Calculate new metrics with early payments
        new_total_interest = sum([float(x.replace('$', '').replace(',', '')) for x in schedule_df['Interest']])
        new_total_payments = sum([float(x.replace('$', '').replace(',', '')) for x in schedule_df['Payment']])
        interest_saved = total_interest - new_total_interest
        
        # Display adjusted metrics
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.metric("Adjusted Monthly Payment", f"${monthly_payment:,.2f}")
        with col2:
            st.metric("New Total Payments", f"${new_total_payments:,.2f}")
        with col3:
            st.metric("New Total Interest", f"${new_total_interest:,.2f}")
        with col4:
            st.metric("Interest Saved", f"${interest_saved:,.2f}")
        with col5:
            st.metric("New Loan Term", f"{len(schedule_df) / 12:.1f} years")
    else:
        schedule_df = generate_amortization_schedule(loan_amount, interest_rate, loan_term)

    # Create a new figure
    fig = go.Figure()

    # Prepare data for tooltips
    residual_principal = [float(x.replace('$', '').replace(',', '')) for x in schedule_df['Balance']]
    ten_percent_residual = [x * 0.1 for x in residual_principal]

    # Add monthly payment traces (lines) with fluorescent colors
    fig.add_trace(go.Scatter(
        x=schedule_df['Payment #'],
        y=[float(x.replace('$', '').replace(',', '')) for x in schedule_df['Principal']],
        name='Monthly Principal',
        line=dict(color='#00FF00', width=3),  # Bright green
        yaxis='y1',
        hovertemplate='<b>Payment #%{x}</b><br>' +
                     'Monthly Principal: $%{y:,.2f}<br>' +
                     'Residual Principal: $%{customdata[0]:,.2f}<br>' +
                     '10% of Residual: $%{customdata[1]:,.2f}<br>' +
                     '<extra></extra>',
        customdata=list(zip(residual_principal, ten_percent_residual))
    ))

    fig.add_trace(go.Scatter(
        x=schedule_df['Payment #'],
        y=[float(x.replace('$', '').replace(',', '')) for x in schedule_df['Interest']],
        name='Monthly Interest',
        line=dict(color='#FF00FF', width=3),  # Bright magenta
        yaxis='y1',
        hovertemplate='<b>Payment #%{x}</b><br>' +
                     'Monthly Interest: $%{y:,.2f}<br>' +
                     'Residual Principal: $%{customdata[0]:,.2f}<br>' +
                     '10% of Residual: $%{customdata[1]:,.2f}<br>' +
                     '<extra></extra>',
        customdata=list(zip(residual_principal, ten_percent_residual))
    ))

    # Add cumulative payment traces (bars)
    fig.add_trace(go.Bar(
        x=schedule_df['Payment #'],
        y=schedule_df['Cumulative Principal'],
        name='Cum. Principal',
        marker_color='rgba(30, 136, 229, 0.6)',
        marker_line=dict(color='#1E88E5', width=1),
        yaxis='y2',
        hovertemplate='<b>Payment #%{x}</b><br>' +
                     'Cumulative Principal: $%{y:,.2f}<br>' +
                     'Residual Principal: $%{customdata[0]:,.2f}<br>' +
                     '10% of Residual: $%{customdata[1]:,.2f}<br>' +
                     '<extra></extra>',
        customdata=list(zip(residual_principal, ten_percent_residual))
    ))

    fig.add_trace(go.Bar(
        x=schedule_df['Payment #'],
        y=schedule_df['Cumulative Interest'],
        name='Cum. Interest',
        marker_color='rgba(229, 57, 53, 0.6)',
        marker_line=dict(color='#E53935', width=1),
        yaxis='y2',
        hovertemplate='<b>Payment #%{x}</b><br>' +
                     'Cumulative Interest: $%{y:,.2f}<br>' +
                     'Residual Principal: $%{customdata[0]:,.2f}<br>' +
                     '10% of Residual: $%{customdata[1]:,.2f}<br>' +
                     '<extra></extra>',
        customdata=list(zip(residual_principal, ten_percent_residual))
    ))

    fig.add_trace(go.Bar(
        x=schedule_df['Payment #'],
        y=schedule_df['Cumulative Total'],
        name='Cum. Total',
        marker_color='rgba(67, 160, 71, 0.6)',
        marker_line=dict(color='#43A047', width=1),
        yaxis='y2',
        hovertemplate='<b>Payment #%{x}</b><br>' +
                     'Cumulative Total: $%{y:,.2f}<br>' +
                     'Residual Principal: $%{customdata[0]:,.2f}<br>' +
                     '10% of Residual: $%{customdata[1]:,.2f}<br>' +
                     '<extra></extra>',
        customdata=list(zip(residual_principal, ten_percent_residual))
    ))

    # Add markers for early payment points if they exist
    if st.session_state.early_payments:
        early_payment_x = []
        early_payment_y = []
        for payment in st.session_state.early_payments:
            payment_num = payment['payment_number']
            if payment_num <= len(schedule_df):
                early_payment_x.append(payment_num)
                balance_at_payment = float(schedule_df.iloc[payment_num-1]['Balance'].replace('$', '').replace(',', ''))
                early_payment_y.append(balance_at_payment)
        
        if early_payment_x:
            fig.add_trace(go.Scatter(
                x=early_payment_x,
                y=early_payment_y,
                mode='markers',
                name='Early Payment Points',
                marker=dict(
                    color='#FFD93D',
                    size=12,
                    symbol='star'
                ),
                yaxis='y2'
            ))

    # Add vertical lines for each year
    for year in range(1, loan_term + 1):
        payment_number = year * 12
        if payment_number <= len(schedule_df):
            # Add vertical line
            fig.add_shape(
                type="line",
                x0=payment_number,
                y0=0,
                x1=payment_number,
                y1=1,
                line=dict(
                    color="rgba(255, 255, 255, 0.3)",
                    width=1,
                    dash="dot"
                ),
                yref='paper'
            )
            
            # Add year label
            fig.add_annotation(
                x=payment_number,
                y=1,
                text=f"Year {year}",
                showarrow=False,
                xanchor='left',
                yanchor='bottom',
                textangle=-90,
                font=dict(size=10, color='rgba(255, 255, 255, 0.7)'),
                yref='paper'
            )

    # Update layout with dual axes
    fig.update_layout(
        title='Mortgage Payment Analysis' + (' (with Early Payments)' if st.session_state.early_payments else ''),
        template='plotly_dark',
        height=600,
        margin=dict(l=20, r=20, t=60, b=100),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.3,
            xanchor="center",
            x=0.5
        ),
        xaxis=dict(
            title='Payment Number',
            showgrid=True
        ),
        yaxis=dict(
            title='Monthly Amount ($)',
            showgrid=True,
            tickformat=',.0f',
            side='left',
            range=[0, max([float(x.replace('$', '').replace(',', '')) for x in schedule_df['Principal']]) * 1.2]
        ),
        yaxis2=dict(
            title='Cumulative Amount ($)',
            showgrid=False,
            tickformat=',.0f',
            side='right',
            overlaying='y',
            range=[0, max(schedule_df['Cumulative Total']) * 1.1]
        ),
        hovermode='x unified',
        barmode='group',
        bargap=0.15,
        bargroupgap=0.1
    )

    # Add the loan amount line
    fig.add_shape(
        type="line",
        x0=0,
        y0=loan_amount,
        x1=len(schedule_df),
        y1=loan_amount,
        line=dict(
            color="rgba(255, 255, 255, 0.5)",
            width=2,
            dash="dash"
        ),
        yref='y2'
    )

    # Display the chart
    st.plotly_chart(fig, use_container_width=True)

    # Display the amortization schedule using Streamlit's dataframe
    st.subheader("Amortization Schedule" + (' (with Early Payments)' if st.session_state.early_payments else ''))
    st.dataframe(
        schedule_df,
        use_container_width=True,
        height=400
    )

    # Download button
    csv = schedule_df.to_csv(index=False)
    filename = "mortgage_schedule_with_early_payments.csv" if st.session_state.early_payments else "mortgage_schedule.csv"
    st.download_button(
        label="📥 Download Schedule",
        data=csv,
        file_name=filename,
        mime="text/csv",
        use_container_width=True
    )

with tab2:
    st.header("💰 Advanced Features")
    
    # Extra payment calculator
    st.subheader("Extra Payment Calculator")
    col1, col2 = st.columns(2)
    
    with col1:
        extra_payment = st.number_input(
            "Extra Monthly Payment ($)",
            min_value=0,
            value=100,
            step=50,
            format="%d"
        )
    
    with col2:
        extra_payment_frequency = st.selectbox(
            "Payment Frequency",
            ["Monthly", "Bi-weekly", "Weekly"],
            index=0
        )
    
    if extra_payment > 0:
        # Calculate impact of extra payments
        impact = calculate_extra_payment_impact(loan_amount, interest_rate, loan_term, extra_payment)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("New Loan Term", f"{impact['new_years']:.1f} years")
        with col2:
            st.metric("Interest Saved", f"${impact['interest_saved']:,.2f}")
        with col3:
            st.metric("New Monthly Payment", f"${impact['total_payment']:,.2f}")
        
        # Show comparison chart
        original_payment = calculate_monthly_payment(loan_amount, interest_rate, loan_term)
        
        fig_comparison = go.Figure()
        
        fig_comparison.add_trace(go.Bar(
            x=['Original', 'With Extra Payment'],
            y=[original_payment, impact['total_payment']],
            name='Monthly Payment',
            marker_color=['#1E88E5', '#4CAF50']
        ))
        
        fig_comparison.update_layout(
            title='Monthly Payment Comparison',
            template='plotly_dark',
            height=400,
            yaxis_title='Monthly Payment ($)',
            showlegend=False
        )
        
        st.plotly_chart(fig_comparison, use_container_width=True)
    
    # Break-even analysis
    st.subheader("Break-even Analysis")
    
    col1, col2 = st.columns(2)
    with col1:
        closing_costs = st.number_input(
            "Closing Costs ($)",
            min_value=0,
            value=3000,
            step=500,
            format="%d"
        )
    
    with col2:
        monthly_savings = st.number_input(
            "Monthly Savings ($)",
            min_value=0,
            value=200,
            step=50,
            format="%d"
        )
    
    if monthly_savings > 0:
        break_even_months = closing_costs / monthly_savings
        st.metric("Break-even Time", f"{break_even_months:.1f} months")

with tab3:
    st.header("🔄 Refinance Calculator")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Current Loan")
        current_balance = st.number_input(
            "Current Loan Balance ($)",
            min_value=0,
            value=loan_amount,
            step=1000,
            format="%d"
        )
        current_rate = st.number_input(
            "Current Interest Rate (%)",
            min_value=0.0,
            value=interest_rate,
            step=0.01,
            format="%.2f"
        )
        current_years = st.number_input(
            "Years Remaining",
            min_value=1,
            value=loan_term,
            step=1,
            format="%d"
        )
    
    with col2:
        st.subheader("New Loan")
        new_rate = st.number_input(
            "New Interest Rate (%)",
            min_value=0.0,
            value=4.5,
            step=0.01,
            format="%.2f"
        )
        new_years = st.number_input(
            "New Loan Term (Years)",
            min_value=1,
            value=30,
            step=1,
            format="%d"
        )
        closing_costs_refi = st.number_input(
            "Refinance Closing Costs ($)",
            min_value=0,
            value=3000,
            step=500,
            format="%d"
        )
    
    # Calculate refinance comparison
    comparison = calculate_refinance_comparison(
        current_balance, current_rate, current_years,
        new_rate, new_years
    )
    
    # Display results
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Current Payment", f"${comparison['original_payment']:,.2f}")
    with col2:
        st.metric("New Payment", f"${comparison['new_payment']:,.2f}")
    with col3:
        st.metric("Monthly Savings", f"${comparison['monthly_savings']:,.2f}")
    with col4:
        st.metric("Total Savings", f"${comparison['total_savings']:,.2f}")
    
    # Break-even analysis for refinance
    if comparison['monthly_savings'] > 0:
        break_even_months = closing_costs_refi / comparison['monthly_savings']
        st.metric("Break-even Time", f"{break_even_months:.1f} months")
    
    # Refinance recommendation
    if comparison['monthly_savings'] > 0 and comparison['total_savings'] > closing_costs_refi:
        st.success("✅ Refinancing is recommended!")
    elif comparison['monthly_savings'] > 0:
        st.warning("⚠️ Refinancing may be beneficial for monthly cash flow")
    else:
        st.error("❌ Refinancing is not recommended")

with tab4:
    st.header("⚡ Early Repayment Simulator")
    
    # Initialize session state for early payments
    if 'early_payments' not in st.session_state:
        st.session_state.early_payments = []
    
    # Add new early payment
    st.subheader("Add Early Repayment Point")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        payment_number = st.number_input(
            "Payment Number",
            min_value=1,
            max_value=loan_term * 12,
            value=12,
            step=1,
            format="%d"
        )
    
    with col2:
        extra_amount = st.number_input(
            "Extra Amount ($)",
            min_value=0,
            value=5000,
            step=1000,
            format="%d"
        )
    
    with col3:
        if st.button("Add Early Payment", use_container_width=True):
            new_payment = {
                'payment_number': payment_number,
                'amount': extra_amount
            }
            st.session_state.early_payments.append(new_payment)
            st.success(f"Added ${extra_amount:,.2f} at payment #{payment_number}")
    
    # Display current early payments
    if st.session_state.early_payments:
        st.subheader("Current Early Payments")
        early_payments_df = pd.DataFrame(st.session_state.early_payments)
        early_payments_df['Year'] = early_payments_df['payment_number'] / 12
        early_payments_df['Month'] = early_payments_df['payment_number'] % 12
        early_payments_df['Year'] = early_payments_df['Year'].apply(lambda x: f"Year {int(x) + 1}, Month {int(x % 1 * 12) + 1}")
        
        st.dataframe(
            early_payments_df[['payment_number', 'Year', 'amount']].rename(
                columns={'payment_number': 'Payment #', 'amount': 'Amount ($)'}
            ),
            use_container_width=True
        )
        
        if st.button("Clear All Early Payments"):
            st.session_state.early_payments = []
            st.rerun()
    
    # Generate schedule with early payments
    if st.session_state.early_payments:
        schedule_with_early = generate_early_repayment_schedule(
            loan_amount, interest_rate, loan_term, st.session_state.early_payments
        )
        
        # Calculate new metrics
        original_schedule = generate_amortization_schedule(loan_amount, interest_rate, loan_term)
        original_total_interest = sum([float(x.replace('$', '').replace(',', '')) for x in original_schedule['Interest']])
        new_total_interest = sum([float(x.replace('$', '').replace(',', '')) for x in schedule_with_early['Interest']])
        interest_saved = original_total_interest - new_total_interest
        
        # Display comparison metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Original Loan Term", f"{loan_term} years")
        with col2:
            st.metric("New Loan Term", f"{len(schedule_with_early) / 12:.1f} years")
        with col3:
            st.metric("Interest Saved", f"${interest_saved:,.2f}")
        with col4:
            st.metric("Years Saved", f"{loan_term - len(schedule_with_early) / 12:.1f} years")
        
        # Create comparison chart
        fig_early = go.Figure()
        
        # Original schedule
        fig_early.add_trace(go.Scatter(
            x=original_schedule['Payment #'],
            y=[float(x.replace('$', '').replace(',', '')) for x in original_schedule['Balance']],
            name='Original Balance',
            line=dict(color='#FF6B6B', width=2),
            mode='lines'
        ))
        
        # New schedule with early payments
        fig_early.add_trace(go.Scatter(
            x=schedule_with_early['Payment #'],
            y=[float(x.replace('$', '').replace(',', '')) for x in schedule_with_early['Balance']],
            name='Balance with Early Payments',
            line=dict(color='#4ECDC4', width=2),
            mode='lines'
        ))
        
        # Add markers for early payment points
        early_payment_x = []
        early_payment_y = []
        for payment in st.session_state.early_payments:
            payment_num = payment['payment_number']
            if payment_num <= len(schedule_with_early):
                early_payment_x.append(payment_num)
                balance_at_payment = float(schedule_with_early.iloc[payment_num-1]['Balance'].replace('$', '').replace(',', ''))
                early_payment_y.append(balance_at_payment)
        
        if early_payment_x:
            fig_early.add_trace(go.Scatter(
                x=early_payment_x,
                y=early_payment_y,
                mode='markers',
                name='Early Payment Points',
                marker=dict(
                    color='#FFD93D',
                    size=10,
                    symbol='star'
                )
            ))
        
        fig_early.update_layout(
            title='Loan Balance Comparison',
            template='plotly_dark',
            height=500,
            xaxis_title='Payment Number',
            yaxis_title='Loan Balance ($)',
            hovermode='x unified'
        )
        
        st.plotly_chart(fig_early, use_container_width=True)
        
        # Display adjusted schedule
        st.subheader("Adjusted Amortization Schedule")
        st.dataframe(
            schedule_with_early,
            use_container_width=True,
            height=400
        )
        
        # Download adjusted schedule
        csv_adjusted = schedule_with_early.to_csv(index=False)
        st.download_button(
            label="📥 Download Adjusted Schedule",
            data=csv_adjusted,
            file_name="mortgage_schedule_with_early_payments.csv",
            mime="text/csv",
            use_container_width=True
        )
    else:
        st.info("Add early repayment points above to see the simulation.") 