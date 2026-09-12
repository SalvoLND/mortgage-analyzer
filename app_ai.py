"""Mortgage Analyzer + AI assistant.

Standalone copy of ``app.py`` (all four original tabs, identical behaviour and
styling) with the five math functions imported from ``mortgage_core`` instead of
being defined inline, plus a DeepSeek-powered "Ask AI" tab and sidebar controls.

``app.py`` is never imported or modified by this module.
"""

import hmac
import os
import time

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import date
import numpy as np

# --- Math engine (owned by mortgage_core.py) --------------------------------
from mortgage_core import (
    calculate_monthly_payment,
    generate_amortization_schedule,
    generate_early_repayment_schedule,
    calculate_extra_payment_impact,
    calculate_refinance_comparison,
)

# --- AI engine (owned by mortgage_agent.py) ---------------------------------
# Imported defensively so that a missing/broken agent module can never take down
# the four calculator tabs, and so the module always imports without an API key.
AI_ENGINE_ERROR = None
try:
    from mortgage_agent import (
        MortgageContext,
        DeepSeekClient,
        DeepSeekError,
        MortgageAgent,
    )
except Exception as _ai_import_error:  # pragma: no cover - defensive
    AI_ENGINE_ERROR = f"{type(_ai_import_error).__name__}: {_ai_import_error}"
    MortgageContext = None
    DeepSeekClient = None
    MortgageAgent = None

    class DeepSeekError(Exception):
        """Fallback so ``except DeepSeekError`` is always valid."""

        kind = "unknown"
        status_code = None


# Set page config with a dark theme
st.set_page_config(
    page_title="Mortgage Analyzer",
    page_icon="💻",
    layout="wide",
    # "auto" keeps the sidebar open on a laptop but collapses it behind the
    # hamburger on narrow screens, where "expanded" covers the whole viewport.
    initial_sidebar_state="auto",
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

    /* --- Phones and small tablets -------------------------------------
       Streamlit keeps st.columns side by side at every viewport width, so a
       five-metric row renders as five unreadable slivers on a phone. Below
       640px let the row wrap and give each column the full width, so columns
       stack vertically instead. The data-testid hooks are Streamlit
       internals and may need revisiting after a major Streamlit upgrade. */
    @media (max-width: 640px) {
        [data-testid="stHorizontalBlock"] {
            flex-wrap: wrap;
            gap: 0.5rem;
        }
        [data-testid="stColumn"] {
            flex: 1 1 100% !important;
            min-width: 100% !important;
        }
        /* Reclaim the wide desktop gutters for content. */
        .block-container {
            padding-left: 0.75rem !important;
            padding-right: 0.75rem !important;
            padding-top: 2.5rem !important;
        }
        .stMetric {
            padding: 0.6rem;
        }
        /* Tab labels are long ("Early Repayment Simulator"); let them scroll
           horizontally rather than wrap into an unreadable stack. */
        [data-testid="stTabs"] [data-baseweb="tab-list"] {
            overflow-x: auto;
        }
    }
    </style>
    """, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Access gate
# ---------------------------------------------------------------------------
def _require_password() -> None:
    """Hold the whole app behind one shared password.

    The password lives in ``st.secrets["APP_PASSWORD"]`` — set in the Streamlit
    Cloud dashboard, never committed to this repo, which is public.

    When no password is configured the gate stays open. That keeps local runs
    and the test suite working without a secrets.toml, but it also means a
    deployment that forgets to set ``APP_PASSWORD`` is publicly reachable.
    """
    try:
        expected = st.secrets["APP_PASSWORD"]
    except Exception:  # no secrets.toml at all raises on some versions
        expected = None
    if not expected or not str(expected).strip():
        return

    if st.session_state.get("_access_granted"):
        return

    st.title("🔒 Mortgage Analyzer")
    st.write("This app is private. Enter the access password to continue.")
    entered = st.text_input("Password", type="password", key="_access_password")
    if entered:
        # compare_digest keeps the check constant-time, so a wrong guess cannot
        # be narrowed down by how long the comparison takes.
        if hmac.compare_digest(str(entered), str(expected).strip()):
            st.session_state["_access_granted"] = True
            st.rerun()
        else:
            st.error("Incorrect password.")
    st.stop()


_require_password()


# ---------------------------------------------------------------------------
# Cached schedule generation (keyed on the numeric inputs only)
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def cached_amortization_schedule(principal, annual_rate, years):
    """Cached wrapper around mortgage_core.generate_amortization_schedule."""
    return generate_amortization_schedule(principal, annual_rate, years)


@st.cache_data(show_spinner=False)
def cached_early_repayment_schedule(principal, annual_rate, years, early_payments_key):
    """Cached wrapper around mortgage_core.generate_early_repayment_schedule.

    ``early_payments_key`` is a tuple of ``(payment_number, amount)`` pairs so the
    cache key stays hashable.
    """
    early_payments = [
        {'payment_number': int(number), 'amount': amount}
        for number, amount in early_payments_key
    ]
    return generate_early_repayment_schedule(principal, annual_rate, years, early_payments)


def early_payments_cache_key(early_payments):
    """Hashable, order-stable representation of the early payment list."""
    return tuple(
        (int(p['payment_number']), float(p['amount']))
        for p in sorted(early_payments, key=lambda x: x['payment_number'])
    )


# ---------------------------------------------------------------------------
# AI helpers
# ---------------------------------------------------------------------------

AI_MODELS = ["deepseek-v4-pro", "deepseek-v4-flash"]
AI_EFFORTS = ["off", "low", "high", "max"]
STARTER_QUESTIONS = [
    "How much interest do I pay in total?",
    "What if I pay 200 extra a month?",
    "When do I cross over to paying more principal than interest?",
    "Is refinancing at 4% worth it?",
]


def mask_api_key(key):
    """Return a masked rendering of an API key. Never shows the full value."""
    if not key:
        return ""
    key = key.strip()
    if len(key) <= 8:
        return "•" * len(key)
    return f"{key[:3]}…{key[-4:]}"


def resolve_api_key(typed_key):
    """Resolve the DeepSeek API key. First hit wins:

    1. the value typed into the sidebar box
    2. ``st.secrets["DEEPSEEK_API_KEY"]`` (guarded: a missing secrets.toml raises
       on some Streamlit versions)
    3. ``os.environ["DEEPSEEK_API_KEY"]``

    Returns ``(key, source_label)``; ``(None, None)`` when nothing is configured.
    """
    if typed_key and typed_key.strip():
        return typed_key.strip(), "the box above"

    try:
        secret_key = st.secrets["DEEPSEEK_API_KEY"]
    except Exception:
        secret_key = None
    if secret_key and str(secret_key).strip():
        return str(secret_key).strip(), "st.secrets"

    env_key = os.environ.get("DEEPSEEK_API_KEY")
    if env_key and env_key.strip():
        return env_key.strip(), "the DEEPSEEK_API_KEY environment variable"

    return None, None


def _show_json(target, payload):
    """Render a tool payload as JSON, degrading gracefully for odd types."""
    try:
        target.json(payload)
    except Exception:
        try:
            target.code(repr(payload))
        except Exception:
            pass


def _usage_caption(usage):
    """Small token-usage caption, or None when the shape is unexpected."""
    if not isinstance(usage, dict):
        return None
    parts = []
    for field, label in (
        ("prompt_tokens", "in"),
        ("completion_tokens", "out"),
        ("total_tokens", "total"),
    ):
        value = usage.get(field)
        if isinstance(value, (int, float)):
            parts.append(f"{label} {int(value):,}")
    if not parts:
        return None
    return "🔢 tokens · " + " · ".join(parts)


def _tool_pair_key(event):
    """Stable identity shared by a ``tool_start``/``tool_end`` pair, or ``None``.

    ``MortgageAgent`` stamps every tool call with a per-exchange ``index`` and the
    DeepSeek ``tool_call_id``. Matching on that is the only way to stay correct when
    the model calls the same tool twice in one round — name and arrival order are both
    ambiguous there. ``None`` means the event predates those fields, and the caller
    falls back to FIFO.
    """
    index = event.get("index")
    if isinstance(index, int) and not isinstance(index, bool):
        return ("index", index)
    call_id = event.get("id")
    if call_id:
        return ("id", str(call_id))
    return None


def _take_pending_tool(pending, key, name):
    """Pop the pending tool entry a ``tool_end`` belongs to.

    Exact key match first, then FIFO among entries for the same tool name, then plain
    FIFO. One lookup drives both the live ``st.status`` panel and the persisted
    transcript record, so the two can never drift apart.
    """
    if key is not None:
        for position, item in enumerate(pending):
            if item["key"] == key:
                return pending.pop(position)
    for position, item in enumerate(pending):
        if item["name"] == name:
            return pending.pop(position)
    if pending:
        return pending.pop(0)
    return None


def _tool_label(name, ms=None, pending=False):
    if pending:
        return f"🔧 calling `{name}` …"
    if isinstance(ms, (int, float)):
        return f"🔧 called `{name}` · {int(ms)} ms"
    return f"🔧 called `{name}`"


def render_ai_message(message):
    """Render one stored transcript entry inside an open chat_message block."""
    if message.get("reasoning"):
        with st.expander("🧠 Reasoning", expanded=False):
            st.markdown(message["reasoning"])

    for tool in message.get("tools") or []:
        with st.expander(_tool_label(tool.get("name", "tool"), tool.get("ms")), expanded=False):
            st.markdown("**Arguments**")
            _show_json(st, tool.get("args") or {})
            st.markdown("**Result**")
            if tool.get("result") is None:
                st.caption("_no result recorded_")
            else:
                _show_json(st, tool.get("result"))

    if message.get("content"):
        st.markdown(message["content"])

    if message.get("error"):
        st.error(message["error"])

    caption = _usage_caption(message.get("usage"))
    if caption:
        st.caption(caption)


# Early payments are read by the sidebar/AI context as well as the tabs, so make
# sure the key exists before anything touches it.
if 'early_payments' not in st.session_state:
    st.session_state.early_payments = []
if 'ai_messages' not in st.session_state:
    st.session_state.ai_messages = []
if 'ai_fingerprint' not in st.session_state:
    st.session_state.ai_fingerprint = None


# Sidebar for inputs
with st.sidebar:
    st.title("💻 Mortgage Parameters")

    property_value = st.number_input(
        "Property Value",
        min_value=0,
        value=None,
        step=10000,
        format="%d",
        placeholder="e.g. 300000",
    )

    down_payment = st.number_input(
        "Down Payment",
        min_value=0,
        value=None,
        step=1000,
        format="%d",
        placeholder="e.g. 60000",
    )

    # Both figures are None until the user fills them in, so the metric shows a
    # placeholder rather than attempting the subtraction.
    if property_value is None or down_payment is None:
        loan_amount = None
        st.metric("Loan Amount", "—")
    else:
        loan_amount = property_value - down_payment
        st.metric("Loan Amount", f"${loan_amount:,.2f}")

    interest_rate = st.number_input(
        "Annual Interest Rate (%)",
        min_value=0.0,
        value=None,
        step=0.01,
        format="%.2f",
        placeholder="e.g. 5.00",
    )

    loan_term = st.number_input(
        "Loan Term (Years)",
        min_value=1,
        value=None,
        step=1,
        format="%d",
        placeholder="e.g. 30",
    )

    # ----------------------------- AI controls -----------------------------
    with st.expander("🤖 AI Assistant", expanded=False):
        typed_api_key = st.text_input(
            "DeepSeek API key",
            type="password",
            key="ai_api_key_input",
            placeholder="sk-…",
            help="Kept in this browser session only — never logged and never written to disk.",
        )

        ai_api_key, ai_api_key_source = resolve_api_key(typed_api_key)

        if ai_api_key:
            st.caption(f"🔑 Using `{mask_api_key(ai_api_key)}` from {ai_api_key_source}.")
        else:
            st.caption(
                "No key found. Paste one above, add `DEEPSEEK_API_KEY` to "
                "`.streamlit/secrets.toml`, or export it as an environment variable."
            )

        ai_model_choice = st.selectbox(
            "Model",
            AI_MODELS,
            index=0,
            key="ai_model_choice",
        )
        ai_model_override = st.text_input(
            "Model override (optional)",
            key="ai_model_override",
            placeholder="leave empty to use the selection above",
        )
        ai_model = (ai_model_override or "").strip() or ai_model_choice

        ai_effort_choice = st.selectbox(
            "Reasoning effort",
            AI_EFFORTS,
            index=0,
            key="ai_effort_choice",
            help="`off` disables thinking mode. Higher effort is slower and costs more.",
        )
        ai_reasoning_effort = None if ai_effort_choice == "off" else ai_effort_choice

        ai_temperature = st.slider(
            "Temperature",
            min_value=0.0,
            max_value=1.0,
            value=0.2,
            step=0.05,
            key="ai_temperature",
            help=(
                "How much randomness the model is allowed. Low = deterministic and "
                "repeatable, which is what you want for financial answers: the model's "
                "job here is to pick the right tool and report its numbers, not to be "
                "creative. Raise it only to explore different phrasings."
            ),
        )

        ai_max_tool_rounds = st.number_input(
            "Max tool rounds",
            min_value=1,
            max_value=12,
            value=6,
            step=1,
            format="%d",
            key="ai_max_tool_rounds",
            help=(
                "How many times the assistant may call tools and think again before it "
                "has to answer. Most questions need 1–2. On the last round the request "
                "is sent without any tools, forcing a prose answer instead of a loop."
            ),
        )

        test_col, clear_col = st.columns(2)
        with test_col:
            test_connection = st.button(
                "🔌 Test connection",
                key="ai_test_connection",
                use_container_width=True,
            )
        with clear_col:
            if st.button("🧹 Clear chat", key="ai_clear_chat", use_container_width=True):
                st.session_state.ai_messages = []
                st.session_state.ai_fingerprint = None
                st.rerun()

        if test_connection:
            if AI_ENGINE_ERROR is not None:
                st.error(f"AI engine unavailable — {AI_ENGINE_ERROR}")
            elif not ai_api_key:
                st.warning("Add an API key first.")
            else:
                with st.spinner("Contacting DeepSeek…"):
                    try:
                        probe = DeepSeekClient(
                            api_key=ai_api_key,
                            model=ai_model,
                            reasoning_effort=ai_reasoning_effort,
                        )
                        ok, message = probe.validate()
                        if ok:
                            st.success(message or "Connection OK.")
                        else:
                            st.error(message or "Connection failed.")
                    except DeepSeekError as exc:
                        st.error(str(exc))
                    except Exception as exc:
                        st.error(f"Unexpected error: {exc}")

        st.markdown(
            "Need a key? Get one at "
            "[platform.deepseek.com](https://platform.deepseek.com)."
        )

# Main content area
st.title("🏦 Mortgage Analyzer bla bla bla")

# The sidebar inputs start empty (value=None) so the app opens as a blank slate
# rather than showing numbers the user never entered. Every tab below divides by
# or iterates over these figures, so stop the script until all four are present.
_required = {
    "Property Value": property_value,
    "Down Payment": down_payment,
    "Annual Interest Rate (%)": interest_rate,
    "Loan Term (Years)": loan_term,
}
_missing = [label for label, value in _required.items() if value is None]
if _missing:
    st.info(
        "👈 Enter your mortgage details in the sidebar to get started.\n\n"
        "Still needed: " + ", ".join(f"**{label}**" for label in _missing)
    )
    st.stop()

# Create tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Basic Analysis",
    "💰 Advanced Features",
    "🔄 Refinance Calculator",
    "⚡ Early Repayment Simulator",
    "🤖 Ask AI",
])

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
        schedule_df = cached_early_repayment_schedule(
            loan_amount, interest_rate, loan_term,
            early_payments_cache_key(st.session_state.early_payments)
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
        schedule_df = cached_amortization_schedule(loan_amount, interest_rate, loan_term)

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
            format="%d",
            key="tab4_payment_number"
        )

    with col2:
        extra_amount = st.number_input(
            "Extra Amount ($)",
            min_value=0,
            value=5000,
            step=1000,
            format="%d",
            key="tab4_extra_amount"
        )

    with col3:
        if st.button("Add Early Payment", use_container_width=True, key="tab4_add_early_payment"):
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

        if st.button("Clear All Early Payments", key="tab4_clear_early_payments"):
            st.session_state.early_payments = []
            st.rerun()

    # Generate schedule with early payments
    if st.session_state.early_payments:
        schedule_with_early = cached_early_repayment_schedule(
            loan_amount, interest_rate, loan_term,
            early_payments_cache_key(st.session_state.early_payments)
        )

        # Calculate new metrics
        original_schedule = cached_amortization_schedule(loan_amount, interest_rate, loan_term)
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

with tab5:
    st.header("🤖 Ask AI")

    if AI_ENGINE_ERROR is not None:
        st.warning(
            "The AI engine (`mortgage_agent.py`) could not be loaded, so the "
            f"assistant is unavailable.\n\n`{AI_ENGINE_ERROR}`"
        )
        ai_ctx = None
        ai_fingerprint = None
    else:
        try:
            ai_ctx = MortgageContext(
                property_value=float(property_value),
                down_payment=float(down_payment),
                loan_amount=float(loan_amount),
                interest_rate=float(interest_rate),
                loan_term=int(loan_term),
                early_payments=list(st.session_state.early_payments),
                currency="$",
            )
        except Exception as exc:
            ai_ctx = None
            st.error(f"Could not build the mortgage context for the assistant: {exc}")

        ai_fingerprint = None
        if ai_ctx is not None:
            try:
                ai_fingerprint = ai_ctx.fingerprint()
            except Exception:
                ai_fingerprint = None

    ai_ready = (
        AI_ENGINE_ERROR is None
        and ai_ctx is not None
        and bool(ai_api_key)
    )

    # Non-destructive banner when the mortgage parameters moved mid-conversation.
    if (
        st.session_state.ai_messages
        and ai_fingerprint is not None
        and st.session_state.ai_fingerprint is not None
        and st.session_state.ai_fingerprint != ai_fingerprint
    ):
        st.info(
            "📌 Mortgage parameters changed — the assistant now sees the new "
            "numbers. Earlier answers in this chat refer to the previous inputs."
        )

    if AI_ENGINE_ERROR is None and not ai_api_key:
        st.info(
            "**Set up the assistant in three steps**\n\n"
            "1. Get a DeepSeek API key at "
            "[platform.deepseek.com](https://platform.deepseek.com).\n"
            "2. Open **🤖 AI Assistant** in the sidebar and paste it in — or add "
            "`DEEPSEEK_API_KEY` to `.streamlit/secrets.toml`, or export it as an "
            "environment variable.\n"
            "3. Hit **🔌 Test connection**, then come back here and ask anything "
            "about this mortgage.\n\n"
            "The key stays in your browser session — it is never logged or written to disk."
        )
    elif ai_ready:
        st.caption(
            f"Model `{ai_model}` · reasoning `{ai_effort_choice}` · "
            f"temperature `{ai_temperature:.2f}` · "
            f"max tool rounds `{int(ai_max_tool_rounds)}` · "
            f"key `{mask_api_key(ai_api_key)}` from {ai_api_key_source}. "
            "Every figure below comes from a deterministic Python tool call — "
            "open the 🔧 panels to audit it."
        )
        if ai_ctx is not None:
            try:
                prompt_block = ai_ctx.to_prompt_block()
            except Exception:
                prompt_block = None
            if prompt_block:
                with st.expander("📋 What the assistant currently sees", expanded=False):
                    st.code(prompt_block)

    # Starter questions — these feed the exact same code path as st.chat_input.
    st.markdown("**Try one of these:**")
    starter_prompt = None
    starter_cols = st.columns(len(STARTER_QUESTIONS))
    for index, question in enumerate(STARTER_QUESTIONS):
        with starter_cols[index]:
            if st.button(
                question,
                key=f"ai_starter_{index}",
                use_container_width=True,
                disabled=not ai_ready,
            ):
                starter_prompt = question

    # Declared here so the exchange renders above the input box.
    transcript_container = st.container()

    typed_prompt = st.chat_input(
        "Ask about your mortgage…" if ai_ready
        else "Add a DeepSeek API key in the sidebar to start chatting",
        key="ai_chat_input",
        disabled=not ai_ready,
    )

    user_prompt = typed_prompt or starter_prompt

    with transcript_container:
        # Always re-render the whole transcript from session state: Streamlit
        # reruns this script top-to-bottom on every widget interaction.
        for stored_message in st.session_state.ai_messages:
            with st.chat_message(stored_message.get("role", "assistant")):
                if stored_message.get("role") == "user":
                    st.markdown(stored_message.get("content", ""))
                else:
                    render_ai_message(stored_message)

        if user_prompt and ai_ready:
            st.session_state.ai_messages.append({"role": "user", "content": user_prompt})
            st.session_state.ai_fingerprint = ai_fingerprint

            # UI-level history (no tool noise, no empty/failed assistant turns).
            history = [
                {"role": message["role"], "content": message["content"]}
                for message in st.session_state.ai_messages
                if message.get("content")
            ]

            # Append the assistant record up front and mutate it in place, so a
            # mid-stream failure can never leave a dangling user message.
            assistant_record = {
                "role": "assistant",
                "content": "",
                "reasoning": "",
                "tools": [],
                "error": None,
                "usage": None,
            }
            st.session_state.ai_messages.append(assistant_record)

            with st.chat_message("user"):
                st.markdown(user_prompt)

            with st.chat_message("assistant"):
                reasoning_slot = st.empty()
                tools_slot = st.container()
                text_slot = st.empty()
                error_slot = st.empty()
                usage_slot = st.empty()

                content_text = ""
                reasoning_text = ""
                error_message = None
                usage_payload = None
                pending_tools = []  # FIFO of {'key', 'name', 'status', 'record'}
                last_reasoning_render = 0.0

                try:
                    client = DeepSeekClient(
                        api_key=ai_api_key,
                        model=ai_model,
                        reasoning_effort=ai_reasoning_effort,
                    )
                    agent = MortgageAgent(
                        ai_ctx,
                        client,
                        max_tool_rounds=int(ai_max_tool_rounds),
                        temperature=float(ai_temperature),
                    )

                    for event in agent.run(history):
                        if not isinstance(event, dict):
                            continue
                        event_type = event.get("type")

                        if event_type == "reasoning":
                            reasoning_text += event.get("text") or ""
                            now = time.monotonic()
                            if now - last_reasoning_render > 0.15:
                                last_reasoning_render = now
                                with reasoning_slot.container():
                                    with st.expander("🧠 Reasoning", expanded=False):
                                        st.markdown(reasoning_text)

                        elif event_type == "text":
                            content_text += event.get("text") or ""
                            text_slot.markdown(content_text + " ▌")

                        elif event_type == "tool_start":
                            tool_name = event.get("name") or "tool"
                            tool_args = event.get("args") or {}
                            status = tools_slot.status(
                                _tool_label(tool_name, pending=True), expanded=False
                            )
                            status.markdown("**Arguments**")
                            _show_json(status, tool_args)
                            tool_record = {
                                "name": tool_name,
                                "args": tool_args,
                                "result": None,
                                "ms": None,
                            }
                            assistant_record["tools"].append(tool_record)
                            pending_tools.append({
                                "key": _tool_pair_key(event),
                                "name": tool_name,
                                "status": status,
                                "record": tool_record,
                            })

                        elif event_type == "tool_end":
                            tool_name = event.get("name") or "tool"
                            tool_result = event.get("result")
                            tool_ms = event.get("ms")

                            pending = _take_pending_tool(
                                pending_tools, _tool_pair_key(event), tool_name
                            )
                            if pending is None:
                                # tool_end without a matching tool_start
                                tool_record = {
                                    "name": tool_name,
                                    "args": {},
                                    "result": None,
                                    "ms": None,
                                }
                                assistant_record["tools"].append(tool_record)
                                pending = {
                                    "status": tools_slot.status(
                                        _tool_label(tool_name, tool_ms), expanded=False
                                    ),
                                    "record": tool_record,
                                }

                            status = pending["status"]
                            status.markdown("**Result**")
                            _show_json(status, tool_result)
                            status.update(
                                label=_tool_label(tool_name, tool_ms),
                                state="complete",
                                expanded=False,
                            )

                            # Same entry that owned the live panel, so the replayed
                            # transcript shows this result next to its own arguments.
                            pending["record"]["result"] = tool_result
                            pending["record"]["ms"] = tool_ms

                        elif event_type == "done":
                            final_content = event.get("content") or ""
                            if len(final_content) > len(content_text):
                                content_text = final_content
                            usage_payload = event.get("usage")

                        elif event_type == "error":
                            error_message = event.get("message") or "The assistant failed."
                            break

                except DeepSeekError as exc:
                    error_message = str(exc) or "The DeepSeek request failed."
                except Exception as exc:
                    error_message = f"Unexpected error: {exc}"

                if not content_text and not error_message:
                    error_message = (
                        "The model returned an empty response. Please try again."
                    )

                # Final, non-throttled render of every slot.
                if reasoning_text:
                    with reasoning_slot.container():
                        with st.expander("🧠 Reasoning", expanded=False):
                            st.markdown(reasoning_text)
                else:
                    reasoning_slot.empty()

                for leftover in pending_tools:
                    leftover["status"].update(state="error")

                if content_text:
                    text_slot.markdown(content_text)
                else:
                    text_slot.empty()

                if error_message:
                    error_slot.error(error_message)

                usage_caption = _usage_caption(usage_payload)
                if usage_caption:
                    usage_slot.caption(usage_caption)

                assistant_record["content"] = content_text
                assistant_record["reasoning"] = reasoning_text
                assistant_record["error"] = error_message
                assistant_record["usage"] = usage_payload
