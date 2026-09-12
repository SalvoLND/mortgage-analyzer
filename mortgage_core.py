"""
mortgage_core.py — pure mortgage math shared by the Streamlit UI and the AI agent.

NO streamlit import, NO I/O, no network. Everything here is deterministic.

The first five functions are clones of the same-named functions in ``app.py`` (including
the overflow guards, the ``monthly_rate > 0.99`` shortcut and the ``max(0, ...)`` balance
clamp), so ``app_ai.py`` can import them as a drop-in replacement. Currency columns stay
pre-formatted ``"$1,234.56"`` strings exactly like app.py.

One deliberate deviation: ``generate_early_repayment_schedule`` keeps the payoff row that
``app.py`` drops, so the UI table and the AI assistant can never report different payoff
months. See that function's docstring. ``app.py`` itself is never modified.

The numeric helpers at the bottom (``amortization_records`` / ``yearly_rollup``) are new.
They return plain floats and are the single numeric source of truth for the agent tools.
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

__all__ = [
    "calculate_monthly_payment",
    "generate_amortization_schedule",
    "generate_early_repayment_schedule",
    "calculate_extra_payment_impact",
    "calculate_refinance_comparison",
    "amortization_records",
    "yearly_rollup",
]


# ---------------------------------------------------------------------------
# 1:1 clones of app.py  (do not "improve" these — the UI depends on them)
# ---------------------------------------------------------------------------

def calculate_monthly_payment(principal, annual_rate, years):
    try:
        monthly_rate = annual_rate / 12 / 100
        num_payments = years * 12

        # Handle edge cases to prevent overflow
        if monthly_rate == 0:
            return principal / num_payments
        if monthly_rate > 0.99:  # Very high interest rate
            return principal * monthly_rate * 1.1

        monthly_payment = principal * (monthly_rate * (1 + monthly_rate) ** num_payments) / ((1 + monthly_rate) ** num_payments - 1)
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
            'Date': (date.today().replace(day=1) + pd.DateOffset(months=payment - 1)).strftime('%Y-%m-%d'),
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
    """Generate amortization schedule with early repayment points.

    Consistent with ``amortization_records()`` below by construction: same balance
    progression, same row count, same interest total. Two things this does that the
    inline copy still living in ``app.py`` does not:

    * The payoff row is **appended before** the loop terminates. ``app.py`` ``break``s
      first, which silently drops the final payment and makes ``len(df)`` under-report
      the payoff month by one — so the "New Loan Term" metric and the chat assistant
      disagreed by a month whenever early payments were configured.
    * Principal is clamped to the outstanding balance, so the last row shows what is
      actually paid rather than a full instalment, and ``Balance`` lands on exactly
      ``$0.00``. On every other row the clamp is a no-op, so those rows are unchanged.

    Several entries on the same ``payment_number`` are summed (``app.py`` applies only
    the first one it finds, even though its own "Current Early Payments" table lists
    them all).

    ``app.py`` is intentionally left alone; only ``app_ai.py`` imports this module.
    """
    monthly_rate = annual_rate / 12 / 100
    num_payments = years * 12
    monthly_payment = calculate_monthly_payment(principal, annual_rate, years)

    schedule = []
    balance = principal
    cumulative_principal = 0
    cumulative_interest = 0

    # Collapse the early payments into {payment_number: total amount}
    extras = {}
    for early_pay in sorted(early_payments or [], key=lambda x: x['payment_number']):
        try:
            number = int(early_pay['payment_number'])
            amount = float(early_pay['amount'])
        except (KeyError, TypeError, ValueError):
            continue
        if number >= 1 and amount:
            extras[number] = extras.get(number, 0.0) + amount

    for payment in range(1, num_payments + 1):
        interest_payment = balance * monthly_rate
        scheduled_principal = monthly_payment - interest_payment
        requested_extra = extras.get(payment, 0)

        # Never collect more principal than is outstanding: scheduled portion first,
        # then whatever of the lump sum still fits.
        principal_paid = min(scheduled_principal, balance)
        extra_payment = min(requested_extra, max(0, balance - principal_paid))
        principal_payment = principal_paid + extra_payment  # 'Principal' includes the extra

        balance = max(0, balance - principal_payment)

        cumulative_principal += principal_payment
        cumulative_interest += interest_payment

        schedule.append({
            'Payment #': payment,
            'Date': (date.today().replace(day=1) + pd.DateOffset(months=payment - 1)).strftime('%Y-%m-%d'),
            'Payment': f"${principal_payment + interest_payment:,.2f}",
            'Principal': f"${principal_payment:,.2f}",
            'Interest': f"${interest_payment:,.2f}",
            'Balance': f"${balance:,.2f}",
            'Cumulative Principal': cumulative_principal,
            'Cumulative Interest': cumulative_interest,
            'Cumulative Total': cumulative_principal + cumulative_interest,
            'Extra Payment': f"${extra_payment:,.2f}",
            'Has Early Payment': extra_payment > 0
        })

        # If balance is paid off, stop — the payoff row above is part of the schedule.
        if balance <= 0:
            break

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


# ---------------------------------------------------------------------------
# Numeric helpers — new, used by the agent tools. Floats, never strings.
# ---------------------------------------------------------------------------

def _month_start(offset_months: int) -> str:
    """First day of the month ``offset_months`` after the current month, ISO format.

    Equivalent to ``date.today().replace(day=1) + pd.DateOffset(months=offset_months)``
    but ~100x cheaper, which matters when building 360-row schedules per tool call.
    """
    base = date.today().replace(day=1)
    total = base.month - 1 + int(offset_months)
    year = base.year + total // 12
    month = total % 12 + 1
    return date(year, month, 1).isoformat()


def amortization_records(principal, annual_rate, years, early_payments=None):
    """Numeric amortization schedule — the single source of truth for the agent tools.

    Returns one dict per payment::

        {'payment_number': int, 'date': 'YYYY-MM-DD', 'payment': float,
         'principal': float, 'interest': float, 'extra': float, 'balance': float,
         'cum_principal': float, 'cum_interest': float}

    Conventions:
      * ``principal`` is the scheduled principal portion only; ``extra`` is any lump sum
        applied on top. ``payment == principal + interest + extra`` always holds, and
        ``cum_principal`` includes the extras (so ``cum_principal + balance == principal``).
      * The loop stops on the payment that clears the balance and **includes** that row.
        ``generate_early_repayment_schedule`` above does the same, so the UI table and the
        agent always report the same payoff month and the same interest total.
      * The final payment is clamped to the outstanding balance, so the schedule never
        over-collects and the principal columns sum exactly to the loan amount.

    ``early_payments`` is the UI shape: ``[{'payment_number': int, 'amount': float}, ...]``.
    Several entries on the same payment number are summed.
    """
    monthly_rate = annual_rate / 12 / 100
    num_payments = int(years) * 12
    monthly_payment = calculate_monthly_payment(principal, annual_rate, years)

    extras: dict[int, float] = {}
    for item in (early_payments or []):
        try:
            n = int(item['payment_number'])
            amount = float(item['amount'])
        except (KeyError, TypeError, ValueError):
            continue
        if n >= 1 and amount:
            extras[n] = extras.get(n, 0.0) + amount

    records: list[dict] = []
    balance = float(principal)
    cum_principal = 0.0
    cum_interest = 0.0

    for payment in range(1, num_payments + 1):
        interest_payment = balance * monthly_rate
        scheduled_principal = monthly_payment - interest_payment
        extra = extras.get(payment, 0.0)

        # Never collect more principal than is outstanding: scheduled portion first,
        # then whatever of the lump sum still fits.
        principal_paid = min(scheduled_principal, balance)
        extra_paid = min(extra, max(0.0, balance - principal_paid))

        balance = max(0.0, balance - principal_paid - extra_paid)
        cum_principal += principal_paid + extra_paid
        cum_interest += interest_payment

        records.append({
            'payment_number': payment,
            'date': _month_start(payment - 1),
            'payment': principal_paid + interest_payment + extra_paid,
            'principal': principal_paid,
            'interest': interest_payment,
            'extra': extra_paid,
            'balance': balance,
            'cum_principal': cum_principal,
            'cum_interest': cum_interest,
        })

        if balance <= 0:
            break

    return records


def yearly_rollup(records):
    """Aggregate ``amortization_records`` output into 1-based loan years (12 payments each).

    Returns::

        {'year': int, 'payments': int, 'principal': float, 'interest': float,
         'extra': float, 'total_paid': float, 'end_balance': float}

    ``principal`` excludes ``extra`` (same convention as the records), so
    ``principal + interest + extra == total_paid``.
    """
    buckets: dict[int, dict] = {}
    order: list[int] = []

    for rec in records:
        year = (int(rec['payment_number']) - 1) // 12 + 1
        bucket = buckets.get(year)
        if bucket is None:
            bucket = {
                'year': year,
                'payments': 0,
                'principal': 0.0,
                'interest': 0.0,
                'extra': 0.0,
                'total_paid': 0.0,
                'end_balance': 0.0,
            }
            buckets[year] = bucket
            order.append(year)

        bucket['payments'] += 1
        bucket['principal'] += float(rec['principal'])
        bucket['interest'] += float(rec['interest'])
        bucket['extra'] += float(rec['extra'])
        bucket['total_paid'] += float(rec['payment'])
        bucket['end_balance'] = float(rec['balance'])  # last row of the year wins

    return [buckets[y] for y in sorted(order)]
