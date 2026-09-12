#!/usr/bin/env python3
"""Offline test suite for the mortgage calculator's AI assistant.

Run it with plain::

    python3 test_ai_assistant.py

No pytest, no API key, no network. Every DeepSeek interaction is mocked at the
``requests.post`` boundary with realistic Server-Sent-Event byte streams, so the suite
is deterministic and safe to run anywhere. If a test ever makes a real HTTP request,
``TestNoRealNetwork`` will catch it: ``requests.post`` is replaced globally for the
whole run and only the explicitly patched tests can reach a fake.

Covers:
  * mortgage_core arithmetic and its edge cases
  * the D2 regression: the UI schedule and the agent's records must agree
  * all 10 agent tools, their schemas, and their bad-input behaviour
  * SSE parsing (fragmented arguments, parallel calls, junk frames, [DONE])
  * HTTP error mapping and the retry / no-retry split
  * the empty-completion workaround
  * the MortgageAgent event loop and its round cap
  * app_ai.py rendering under streamlit.testing.v1.AppTest
"""

from __future__ import annotations

import inspect
import json
import os
import sys
import unittest
from unittest import mock

APP_DIR = os.path.dirname(os.path.abspath(__file__))
if APP_DIR not in sys.path:
    sys.path.insert(0, APP_DIR)

from mortgage_core import (  # noqa: E402
    amortization_records,
    calculate_monthly_payment,
    calculate_refinance_comparison,
    generate_amortization_schedule,
    generate_early_repayment_schedule,
    yearly_rollup,
)
from mortgage_agent import (  # noqa: E402
    DeepSeekClient,
    DeepSeekError,
    MortgageAgent,
    MortgageContext,
    TOOL_REGISTRY,
    execute_tool,
    tool_schemas,
)

# The app's defaults: $300,000 property, $60,000 down, 5% for 30 years.
LOAN_AMOUNT = 240000
RATE = 5.0
TERM = 30
EXPECTED_MONTHLY_PAYMENT = 1288.3718952291354
EXPECTED_TOTAL_INTEREST = 223813.88


def default_context(early_payments=None):
    return MortgageContext(
        property_value=300000.0,
        down_payment=60000.0,
        loan_amount=float(LOAN_AMOUNT),
        interest_rate=RATE,
        loan_term=TERM,
        early_payments=list(early_payments or []),
        currency="$",
    )


def money(formatted):
    """Turn a pre-formatted ``"$1,234.56"`` cell back into a float."""
    return float(str(formatted).replace("$", "").replace(",", ""))


# ---------------------------------------------------------------------------
# Fake transport
# ---------------------------------------------------------------------------

def sse(*frames):
    """Build a realistic SSE byte stream as ``requests`` would hand it to us.

    ``Response.iter_lines()`` yields each line without its newline, blank separator
    lines included (as ``b""``). A dict frame becomes ``data: {json}``; a string is
    emitted verbatim, so tests can inject comments and malformed frames.
    """
    lines = []
    for frame in frames:
        if isinstance(frame, dict):
            lines.append(("data: " + json.dumps(frame, separators=(",", ":"))).encode("utf-8"))
        elif isinstance(frame, bytes):
            lines.append(frame)
        else:
            lines.append(str(frame).encode("utf-8"))
        lines.append(b"")  # SSE frame separator / keep-alive
    return lines


def chunk(delta=None, finish_reason=None, usage=None):
    """One ``chat.completion.chunk`` in DeepSeek's (OpenAI-compatible) shape."""
    payload = {
        "id": "chatcmpl-test-0001",
        "object": "chat.completion.chunk",
        "created": 1785000000,
        "model": "deepseek-v4-pro",
        "system_fingerprint": "fp_test",
        "choices": [],
    }
    if delta is not None or finish_reason is not None:
        payload["choices"] = [{
            "index": 0,
            "delta": delta if delta is not None else {},
            "logprobs": None,
            "finish_reason": finish_reason,
        }]
    if usage is not None:
        payload["usage"] = usage
    return payload


def text_stream(*pieces, usage=None):
    """A plain prose completion, streamed one piece at a time."""
    frames = [chunk({"role": "assistant", "content": ""})]
    frames += [chunk({"content": piece}) for piece in pieces]
    frames.append(chunk({}, finish_reason="stop"))
    if usage is not None:
        frames.append(chunk(usage=usage))
    frames.append("data: [DONE]")
    return sse(*frames)


def tool_call_stream(name, arg_fragments, call_id="call_test_0", index=0):
    """A tool call whose ``arguments`` arrive fragmented across chunks, as they really do."""
    frames = [
        chunk({"role": "assistant", "content": None}),
        chunk({"tool_calls": [{
            "index": index,
            "id": call_id,
            "type": "function",
            "function": {"name": name, "arguments": ""},
        }]}),
    ]
    for fragment in arg_fragments:
        frames.append(chunk({"tool_calls": [{
            "index": index,
            "function": {"arguments": fragment},
        }]}))
    frames.append(chunk({}, finish_reason="tool_calls"))
    frames.append("data: [DONE]")
    return sse(*frames)


class FakeResponse:
    """Stands in for ``requests.Response`` — only what ``DeepSeekClient`` touches."""

    def __init__(self, status_code=200, lines=None, body=None, headers=None, bad_json=False):
        self.status_code = status_code
        self.headers = headers or {}
        self._lines = lines or []
        self._body = body
        self._bad_json = bad_json
        self.closed = False

    def iter_lines(self, *args, **kwargs):
        for line in self._lines:
            yield line

    def json(self):
        if self._bad_json or self._body is None:
            raise ValueError("No JSON object could be decoded")
        return self._body

    def close(self):
        self.closed = True


def completion_body(content=None, tool_calls=None, usage=None):
    """A non-streamed ``chat.completion`` body, used by the empty-completion retry."""
    message = {"role": "assistant", "content": content}
    if tool_calls:
        message["tool_calls"] = tool_calls
    body = {
        "id": "chatcmpl-test-retry",
        "object": "chat.completion",
        "created": 1785000000,
        "model": "deepseek-v4-pro",
        "choices": [{"index": 0, "message": message, "finish_reason": "stop"}],
    }
    if usage is not None:
        body["usage"] = usage
    return body


def error_body(message, err_type="invalid_request_error"):
    return {"error": {"message": message, "type": err_type, "code": None}}


class PostRecorder:
    """``requests.post`` replacement that replays a queue of ``FakeResponse`` objects."""

    def __init__(self, *responses):
        self.responses = list(responses)
        self.calls = []

    def __call__(self, url, **kwargs):
        self.calls.append({"url": url, **kwargs})
        if not self.responses:
            raise AssertionError(
                f"Unexpected extra HTTP call #{len(self.calls)} to {url} — the queue is empty."
            )
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response

    @property
    def call_count(self):
        return len(self.calls)

    def payload(self, index=0):
        return self.calls[index]["json"]


def patch_post(*responses):
    """Patch ``mortgage_agent.requests.post`` with a queue; returns the recorder."""
    recorder = PostRecorder(*responses)
    patcher = mock.patch("mortgage_agent.requests.post", recorder)
    patcher.start()
    return recorder, patcher


def make_client(**kwargs):
    kwargs.setdefault("api_key", "sk-test-key-not-a-real-secret")
    kwargs.setdefault("max_retries", 0)
    return DeepSeekClient(**kwargs)


class ClientTestCase(unittest.TestCase):
    """Base class that wires up the fake transport and kills real sleeping."""

    def transport(self, *responses):
        recorder, patcher = patch_post(*responses)
        self.addCleanup(patcher.stop)
        return recorder

    def no_sleep(self):
        patcher = mock.patch("mortgage_agent.time.sleep", return_value=None)
        patcher.start()
        self.addCleanup(patcher.stop)


# ---------------------------------------------------------------------------
# 1. Math
# ---------------------------------------------------------------------------

class TestMortgageMath(unittest.TestCase):

    def test_monthly_payment_app_defaults(self):
        payment = calculate_monthly_payment(LOAN_AMOUNT, RATE, TERM)
        self.assertAlmostEqual(payment, EXPECTED_MONTHLY_PAYMENT, places=10)

    def test_default_schedule_totals(self):
        records = amortization_records(LOAN_AMOUNT, RATE, TERM)
        self.assertEqual(len(records), 360)
        self.assertAlmostEqual(records[-1]["cum_interest"], EXPECTED_TOTAL_INTEREST, places=2)
        self.assertAlmostEqual(records[-1]["balance"], 0.0, places=2)
        self.assertAlmostEqual(records[-1]["cum_principal"], LOAN_AMOUNT, places=6)

    def test_payment_components_sum_to_the_payment(self):
        for record in amortization_records(LOAN_AMOUNT, RATE, TERM):
            self.assertAlmostEqual(
                record["payment"],
                record["principal"] + record["interest"] + record["extra"],
                places=9,
            )

    def test_balance_plus_cumulative_principal_is_the_loan(self):
        for record in amortization_records(LOAN_AMOUNT, RATE, TERM):
            self.assertAlmostEqual(record["balance"] + record["cum_principal"], LOAN_AMOUNT, places=6)

    def test_dataframe_schedule_matches_records(self):
        frame = generate_amortization_schedule(LOAN_AMOUNT, RATE, TERM)
        records = amortization_records(LOAN_AMOUNT, RATE, TERM)
        self.assertEqual(len(frame), len(records))
        self.assertAlmostEqual(
            frame["Cumulative Interest"].iloc[-1], records[-1]["cum_interest"], places=6
        )
        self.assertEqual(money(frame["Balance"].iloc[-1]), 0.00)

    def test_zero_interest_loan(self):
        payment = calculate_monthly_payment(LOAN_AMOUNT, 0.0, TERM)
        self.assertAlmostEqual(payment, LOAN_AMOUNT / 360.0, places=9)
        records = amortization_records(LOAN_AMOUNT, 0.0, TERM)
        self.assertEqual(len(records), 360)
        self.assertEqual(records[-1]["cum_interest"], 0.0)
        self.assertAlmostEqual(records[-1]["cum_principal"], LOAN_AMOUNT, places=6)
        self.assertAlmostEqual(records[-1]["balance"], 0.0, places=6)

    def test_one_year_term(self):
        records = amortization_records(12000, 6.0, 1)
        self.assertEqual(len(records), 12)
        self.assertAlmostEqual(records[-1]["balance"], 0.0, places=6)
        self.assertAlmostEqual(records[-1]["cum_principal"], 12000, places=6)
        self.assertGreater(records[-1]["cum_interest"], 0.0)
        self.assertEqual(len(yearly_rollup(records)), 1)

    def test_very_large_principal(self):
        principal = 1e12
        payment = calculate_monthly_payment(principal, RATE, TERM)
        self.assertTrue(payment == payment and payment not in (float("inf"), float("-inf")))
        records = amortization_records(principal, RATE, TERM)
        self.assertEqual(len(records), 360)
        self.assertAlmostEqual(records[-1]["balance"], 0.0, places=2)
        self.assertAlmostEqual(records[-1]["cum_principal"] / principal, 1.0, places=9)

    def test_high_rate_guard(self):
        """monthly_rate > 0.99 short-circuits instead of overflowing the compounding term."""
        annual = 1200.0  # monthly_rate == 1.0
        self.assertGreater(annual / 12 / 100, 0.99)
        payment = calculate_monthly_payment(100000, annual, TERM)
        self.assertAlmostEqual(payment, 100000 * 1.0 * 1.1, places=6)

    def test_high_rate_guard_boundary_stays_on_the_formula(self):
        annual = 1180.0  # monthly_rate 0.9833… — just under the guard
        self.assertLess(annual / 12 / 100, 0.99)
        payment = calculate_monthly_payment(100000, annual, 1)
        self.assertNotAlmostEqual(payment, 100000 * (annual / 12 / 100) * 1.1, places=2)
        self.assertGreater(payment, 0.0)

    def test_overflow_fallback(self):
        """A rate under the guard but a term long enough to overflow (1+r)**n."""
        payment = calculate_monthly_payment(100000, 1187.0, 100)
        self.assertAlmostEqual(payment, 100000 / 1200, places=9)

    def test_refinance_comparison_shape(self):
        result = calculate_refinance_comparison(LOAN_AMOUNT, RATE, TERM, 4.0, 30)
        self.assertGreater(result["monthly_savings"], 0)
        self.assertAlmostEqual(
            result["original_payment"] - result["new_payment"], result["monthly_savings"], places=9
        )

    def test_yearly_rollup_conserves_totals(self):
        records = amortization_records(LOAN_AMOUNT, RATE, TERM)
        rollup = yearly_rollup(records)
        self.assertEqual(len(rollup), 30)
        self.assertEqual(sum(y["payments"] for y in rollup), len(records))
        self.assertAlmostEqual(
            sum(y["interest"] for y in rollup), records[-1]["cum_interest"], places=6
        )
        for year in rollup:
            self.assertAlmostEqual(
                year["total_paid"], year["principal"] + year["interest"] + year["extra"], places=6
            )


# ---------------------------------------------------------------------------
# 2. D2 regression — UI schedule vs agent records
# ---------------------------------------------------------------------------

EARLY_PAYMENT_CASES = [
    ("no early payments", []),
    ("default $5,000 at #12", [{"payment_number": 12, "amount": 5000}]),
    ("$200,000 at #6 (clears the loan early)", [{"payment_number": 6, "amount": 200000}]),
    ("$1,000 at the very last payment", [{"payment_number": 360, "amount": 1000}]),
    ("$500,000 at #1 (over-pays the whole loan)", [{"payment_number": 1, "amount": 500000}]),
    ("three lump sums", [
        {"payment_number": 12, "amount": 5000},
        {"payment_number": 60, "amount": 20000},
        {"payment_number": 120, "amount": 10000},
    ]),
    ("two lump sums on the same payment", [
        {"payment_number": 12, "amount": 5000},
        {"payment_number": 12, "amount": 5000},
    ]),
    ("lump sum that lands exactly on payoff", [{"payment_number": 2, "amount": 239000}]),
    ("extra on every payment", [
        {"payment_number": n, "amount": 250} for n in range(1, 361)
    ]),
]


class TestD2Consistency(unittest.TestCase):
    """The UI table and the chat assistant must never report different numbers.

    ``generate_early_repayment_schedule`` used to ``break`` before appending the payoff
    row, so it under-reported the payoff month by one and the assistant contradicted the
    chart beside it.
    """

    def test_row_counts_and_interest_agree(self):
        for label, early in EARLY_PAYMENT_CASES:
            with self.subTest(case=label):
                frame = generate_early_repayment_schedule(LOAN_AMOUNT, RATE, TERM, early)
                records = amortization_records(LOAN_AMOUNT, RATE, TERM, early)
                self.assertEqual(len(frame), len(records), "payoff month must match")
                self.assertAlmostEqual(
                    frame["Cumulative Interest"].iloc[-1],
                    records[-1]["cum_interest"],
                    places=6,
                    msg="total interest must match",
                )
                self.assertAlmostEqual(
                    frame["Cumulative Principal"].iloc[-1],
                    records[-1]["cum_principal"],
                    places=6,
                )

    def test_final_row_is_a_clean_payoff(self):
        for label, early in EARLY_PAYMENT_CASES:
            with self.subTest(case=label):
                frame = generate_early_repayment_schedule(LOAN_AMOUNT, RATE, TERM, early)
                self.assertEqual(money(frame["Balance"].iloc[-1]), 0.00)
                self.assertAlmostEqual(
                    frame["Cumulative Principal"].iloc[-1], LOAN_AMOUNT, places=6,
                    msg="principal collected must equal the loan, never more",
                )

    def test_final_row_amounts_are_clamped(self):
        """You cannot pay more principal than is outstanding."""
        early = [{"payment_number": 1, "amount": 500000}]
        frame = generate_early_repayment_schedule(LOAN_AMOUNT, RATE, TERM, early)
        self.assertEqual(len(frame), 1)
        row = frame.iloc[0]
        self.assertLess(money(row["Extra Payment"]), 500000)
        self.assertAlmostEqual(money(row["Principal"]), LOAN_AMOUNT, places=2)
        self.assertAlmostEqual(
            money(row["Payment"]), money(row["Principal"]) + money(row["Interest"]), places=2
        )
        self.assertEqual(money(row["Balance"]), 0.00)

    def test_rows_before_the_payoff_are_untouched(self):
        """The clamp is a no-op everywhere except the final row."""
        early = [{"payment_number": 12, "amount": 5000}]
        frame = generate_early_repayment_schedule(LOAN_AMOUNT, RATE, TERM, early)
        scheduled = calculate_monthly_payment(LOAN_AMOUNT, RATE, TERM)
        for position in range(len(frame) - 1):
            row = frame.iloc[position]
            expected = scheduled + (5000 if row["Payment #"] == 12 else 0)
            self.assertAlmostEqual(money(row["Payment"]), expected, places=2)

    def test_defaults_still_give_360_payments(self):
        frame = generate_early_repayment_schedule(LOAN_AMOUNT, RATE, TERM, [])
        self.assertEqual(len(frame), 360)
        self.assertAlmostEqual(
            frame["Cumulative Interest"].iloc[-1], EXPECTED_TOTAL_INTEREST, places=2
        )

    def test_context_headline_matches_the_ui_table(self):
        """What the assistant is told in its system prompt is what the table shows."""
        early = [{"payment_number": 12, "amount": 5000}]
        frame = generate_early_repayment_schedule(LOAN_AMOUNT, RATE, TERM, early)
        headline = default_context(early).headline()
        self.assertEqual(headline["payoff_months"], len(frame))
        self.assertAlmostEqual(
            headline["total_interest"], frame["Cumulative Interest"].iloc[-1], places=6
        )
        self.assertAlmostEqual(headline["payoff_years"], len(frame) / 12, places=9)


# ---------------------------------------------------------------------------
# 3. Tools
# ---------------------------------------------------------------------------

VALID_TOOL_ARGS = {
    "get_mortgage_summary": {},
    "get_payment_details": {"payment_number": 12},
    "get_schedule_window": {"start_payment": 1, "end_payment": 12},
    "get_yearly_summary": {"start_year": 1, "end_year": 3},
    "simulate_extra_monthly_payment": {"extra_monthly": 200},
    "simulate_lump_sum": {"payment_number": 36, "amount": 10000},
    "compare_refinance": {"new_rate": 4.0, "new_years": 30, "closing_costs": 3000},
    "what_if": {"interest_rate": 4.25},
    "find_month_when_balance_below": {"target_balance": 100000},
    "get_interest_principal_crossover": {},
}


class TestTools(unittest.TestCase):

    def setUp(self):
        self.ctx = default_context([{"payment_number": 12, "amount": 5000}])

    def test_registry_has_all_ten_tools(self):
        self.assertEqual(len(TOOL_REGISTRY), 10)
        self.assertEqual(set(TOOL_REGISTRY), set(VALID_TOOL_ARGS))
        for name, spec in TOOL_REGISTRY.items():
            self.assertEqual(name, spec.name)

    def test_every_tool_executes_and_is_json_serialisable(self):
        for name, args in VALID_TOOL_ARGS.items():
            with self.subTest(tool=name):
                result = execute_tool(self.ctx, name, dict(args))
                self.assertIsInstance(result, dict)
                self.assertNotIn("error", result, f"{name} returned an error: {result.get('error')}")
                encoded = json.dumps(result)  # no default= fallback: must be natively serialisable
                self.assertEqual(json.loads(encoded), result)

    def test_schema_required_names_match_the_handler_signature(self):
        for name, spec in TOOL_REGISTRY.items():
            with self.subTest(tool=name):
                params = inspect.signature(spec.handler).parameters
                names = [p for p in params if p != "ctx"]
                mandatory = [p for p in names if params[p].default is inspect.Parameter.empty]
                properties = spec.parameters.get("properties", {})
                required = spec.parameters.get("required", [])
                self.assertEqual(spec.parameters.get("type"), "object")
                self.assertEqual(sorted(properties), sorted(names),
                                 "schema properties must mirror the handler arguments")
                self.assertEqual(sorted(required), sorted(mandatory),
                                 "schema 'required' must be exactly the arguments with no default")
                for prop in properties.values():
                    self.assertIn("type", prop)
                    self.assertIn("description", prop)

    def test_tool_schemas_are_wire_ready(self):
        schemas = tool_schemas()
        self.assertEqual(len(schemas), 10)
        for schema in schemas:
            self.assertEqual(schema["type"], "function")
            self.assertIn("name", schema["function"])
            self.assertIn("description", schema["function"])
            self.assertIn("parameters", schema["function"])
        json.dumps(schemas)

    def test_tools_agree_with_the_core_numbers(self):
        ctx = default_context()
        summary = execute_tool(ctx, "get_mortgage_summary", {})
        self.assertEqual(summary["payoff_months"], 360)
        self.assertAlmostEqual(summary["monthly_payment"], EXPECTED_MONTHLY_PAYMENT, places=2)
        self.assertAlmostEqual(summary["total_interest"], EXPECTED_TOTAL_INTEREST, places=2)

        details = execute_tool(ctx, "get_payment_details", {"payment_number": 1})
        self.assertAlmostEqual(details["interest"], LOAN_AMOUNT * RATE / 12 / 100, places=2)

        crossover = execute_tool(ctx, "get_interest_principal_crossover", {})
        self.assertTrue(crossover["found"])
        self.assertGreater(crossover["principal"], crossover["interest"])

    def test_schedule_window_aggregates_above_the_row_cap(self):
        ctx = default_context()
        small = execute_tool(ctx, "get_schedule_window", {"start_payment": 1, "end_payment": 60})
        self.assertFalse(small["aggregated"])
        self.assertEqual(len(small["rows"]), 60)

        big = execute_tool(ctx, "get_schedule_window", {"start_payment": 1, "end_payment": 360})
        self.assertTrue(big["aggregated"])
        self.assertNotIn("rows", big)
        self.assertEqual(big["totals"]["payments"], 360)


class TestBadToolInput(unittest.TestCase):
    """Bad model-supplied input must come back as ``{"error": ...}`` — never an exception."""

    def setUp(self):
        self.ctx = default_context()

    def assertToolError(self, name, args):
        result = execute_tool(self.ctx, name, args)
        self.assertIsInstance(result, dict)
        self.assertIn("error", result, f"expected an error dict for {name}({args!r})")
        self.assertIsInstance(result["error"], str)
        self.assertTrue(result["error"].strip())
        json.dumps(result)
        return result

    def test_unknown_tool(self):
        self.assertToolError("get_mortgage_summry", {})

    def test_arguments_not_an_object(self):
        for args in (["a"], "payment_number=1", 42):
            with self.subTest(args=args):
                result = execute_tool(self.ctx, "get_payment_details", args)
                self.assertIn("error", result)

    def test_unexpected_argument(self):
        self.assertToolError("get_mortgage_summary", {"loan_amount": 1})
        self.assertToolError("get_payment_details", {"payment_number": 1, "colour": "blue"})

    def test_missing_required_argument(self):
        self.assertToolError("get_payment_details", {})
        self.assertToolError("simulate_lump_sum", {"amount": 1000})

    def test_out_of_range_values(self):
        self.assertToolError("get_payment_details", {"payment_number": 0})
        self.assertToolError("get_payment_details", {"payment_number": 9999})
        self.assertToolError("get_schedule_window", {"start_payment": 10, "end_payment": 2})
        self.assertToolError("get_yearly_summary", {"start_year": 99})
        self.assertToolError("simulate_extra_monthly_payment", {"extra_monthly": 0})
        self.assertToolError("simulate_extra_monthly_payment", {"extra_monthly": -50})
        self.assertToolError("simulate_lump_sum", {"payment_number": 400, "amount": 100})
        self.assertToolError("compare_refinance", {"new_rate": -1, "new_years": 30})
        self.assertToolError("compare_refinance", {"new_rate": 4.0, "new_years": 0})
        self.assertToolError("find_month_when_balance_below", {"target_balance": -1})
        self.assertToolError("what_if", {})
        self.assertToolError("what_if", {"loan_term": 0})

    def test_wrong_types(self):
        self.assertToolError("get_payment_details", {"payment_number": "twelve"})
        self.assertToolError("get_payment_details", {"payment_number": 12.5})
        self.assertToolError("get_payment_details", {"payment_number": None})
        self.assertToolError("get_payment_details", {"payment_number": True})
        self.assertToolError("find_month_when_balance_below", {"target_balance": float("nan")})
        self.assertToolError("find_month_when_balance_below", {"target_balance": float("inf")})

    def test_none_arguments_are_treated_as_empty(self):
        result = execute_tool(self.ctx, "get_mortgage_summary", None)
        self.assertNotIn("error", result)

    def test_a_crashing_handler_is_contained(self):
        from mortgage_agent import ToolSpec

        def boom(ctx):
            raise ZeroDivisionError("division by zero")

        TOOL_REGISTRY["_test_boom"] = ToolSpec(
            name="_test_boom", description="test only",
            parameters={"type": "object", "properties": {}, "required": []},
            handler=boom,
        )
        self.addCleanup(TOOL_REGISTRY.pop, "_test_boom", None)
        result = execute_tool(self.ctx, "_test_boom", {})
        self.assertIn("error", result)
        self.assertIn("ZeroDivisionError", result["error"])


# ---------------------------------------------------------------------------
# 4. SSE parsing
# ---------------------------------------------------------------------------

class TestSSEParsing(ClientTestCase):

    def collect(self, lines):
        return list(DeepSeekClient._iter_sse_payloads(lines))

    def test_keepalives_comments_and_junk_are_skipped(self):
        lines = [
            b"",
            b": keep-alive",
            b":",
            b"event: message",
            b"garbage that is not a data frame",
            b'data: {"choices":[{"delta":{"content":"hi"}}]}',
            b"",
            b"data: ",
            b'data: {"broken": ',
            b"data: not json at all",
            b'data: [1,2,3]',  # valid JSON but not an object
            b'data: {"choices":[{"delta":{"content":" there"}}]}',
            b"data: [DONE]",
        ]
        chunks = self.collect(lines)
        self.assertEqual(len(chunks), 2)
        self.assertEqual(chunks[0]["choices"][0]["delta"]["content"], "hi")

    def test_data_after_done_is_ignored(self):
        lines = [
            b'data: {"choices":[{"delta":{"content":"a"}}]}',
            b"data: [DONE]",
            b'data: {"choices":[{"delta":{"content":"SHOULD NOT APPEAR"}}]}',
        ]
        chunks = self.collect(lines)
        self.assertEqual(len(chunks), 1)

    def test_str_lines_and_undecodable_bytes(self):
        lines = [
            'data: {"choices":[{"delta":{"content":"str line"}}]}',
            b"data: {\"choices\":[{\"delta\":{\"content\":\"\xff\xfe\"}}]}",
            None,
            b"data: [DONE]",
        ]
        chunks = self.collect(lines)
        self.assertGreaterEqual(len(chunks), 1)
        self.assertEqual(chunks[0]["choices"][0]["delta"]["content"], "str line")

    def test_content_and_reasoning_events(self):
        recorder = self.transport(FakeResponse(lines=sse(
            chunk({"role": "assistant", "content": ""}),
            chunk({"reasoning_content": "Let me "}),
            chunk({"reasoning_content": "check the schedule."}),
            chunk({"content": "Your monthly payment is "}),
            chunk({"content": "$1,288.37."}),
            chunk({}, finish_reason="stop"),
            chunk(usage={"prompt_tokens": 1834, "completion_tokens": 96, "total_tokens": 1930}),
            "data: [DONE]",
        )))
        events = list(make_client().stream_chat([{"role": "user", "content": "hi"}]))
        kinds = [event["type"] for event in events]
        self.assertEqual(kinds, ["reasoning", "reasoning", "content", "content", "usage"])
        self.assertEqual(
            "".join(e["text"] for e in events if e["type"] == "content"),
            "Your monthly payment is $1,288.37.",
        )
        self.assertEqual(events[-1]["usage"]["total_tokens"], 1930)
        self.assertEqual(recorder.call_count, 1)
        self.assertIs(recorder.payload(0)["stream"], True)

    def test_fragmented_tool_call_arguments(self):
        self.transport(FakeResponse(lines=tool_call_stream(
            "simulate_extra_monthly_payment",
            ['{"', "extra", "_mont", 'hly": ', "200}"],
            call_id="call_abc123",
        )))
        events = list(make_client().stream_chat([{"role": "user", "content": "hi"}], tools=tool_schemas()))
        self.assertEqual([e["type"] for e in events], ["tool_calls"])
        calls = events[0]["tool_calls"]
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["id"], "call_abc123")
        self.assertEqual(calls[0]["function"]["name"], "simulate_extra_monthly_payment")
        self.assertEqual(
            json.loads(calls[0]["function"]["arguments"]), {"extra_monthly": 200}
        )

    def test_fragmented_tool_name(self):
        frames = [
            chunk({"tool_calls": [{"index": 0, "id": "call_1", "type": "function",
                                   "function": {"name": "get_", "arguments": ""}}]}),
            chunk({"tool_calls": [{"index": 0, "function": {"name": "yearly_summary"}}]}),
            chunk({"tool_calls": [{"index": 0, "function": {"arguments": "{}"}}]}),
            chunk({}, finish_reason="tool_calls"),
            "data: [DONE]",
        ]
        self.transport(FakeResponse(lines=sse(*frames)))
        events = list(make_client().stream_chat([{"role": "user", "content": "hi"}]))
        self.assertEqual(events[0]["tool_calls"][0]["function"]["name"], "get_yearly_summary")

    def test_repeated_whole_tool_name_is_not_doubled(self):
        frames = [
            chunk({"tool_calls": [{"index": 0, "id": "call_1", "type": "function",
                                   "function": {"name": "what_if", "arguments": ""}}]}),
            chunk({"tool_calls": [{"index": 0, "function": {"name": "what_if",
                                                            "arguments": '{"loan_term":15}'}}]}),
            chunk({}, finish_reason="tool_calls"),
            "data: [DONE]",
        ]
        self.transport(FakeResponse(lines=sse(*frames)))
        events = list(make_client().stream_chat([{"role": "user", "content": "hi"}]))
        self.assertEqual(events[0]["tool_calls"][0]["function"]["name"], "what_if")

    def test_parallel_tool_calls_at_two_indices(self):
        frames = [
            chunk({"tool_calls": [
                {"index": 0, "id": "call_a", "type": "function",
                 "function": {"name": "get_payment_details", "arguments": ""}},
                {"index": 1, "id": "call_b", "type": "function",
                 "function": {"name": "get_payment_details", "arguments": ""}},
            ]}),
            chunk({"tool_calls": [{"index": 0, "function": {"arguments": '{"payment_'}}]}),
            chunk({"tool_calls": [{"index": 1, "function": {"arguments": '{"payment_'}}]}),
            chunk({"tool_calls": [{"index": 0, "function": {"arguments": 'number": 12}'}}]}),
            chunk({"tool_calls": [{"index": 1, "function": {"arguments": 'number": 24}'}}]}),
            chunk({}, finish_reason="tool_calls"),
            "data: [DONE]",
        ]
        self.transport(FakeResponse(lines=sse(*frames)))
        events = list(make_client().stream_chat([{"role": "user", "content": "hi"}]))
        calls = events[0]["tool_calls"]
        self.assertEqual(len(calls), 2)
        self.assertEqual([c["id"] for c in calls], ["call_a", "call_b"])
        self.assertEqual(json.loads(calls[0]["function"]["arguments"]), {"payment_number": 12})
        self.assertEqual(json.loads(calls[1]["function"]["arguments"]), {"payment_number": 24})

    def test_tool_call_without_an_id_gets_a_synthetic_one(self):
        frames = [
            chunk({"tool_calls": [{"index": 0, "function": {"name": "get_mortgage_summary",
                                                            "arguments": "{}"}}]}),
            "data: [DONE]",
        ]
        self.transport(FakeResponse(lines=sse(*frames)))
        events = list(make_client().stream_chat([{"role": "user", "content": "hi"}]))
        self.assertEqual(events[0]["tool_calls"][0]["id"], "call_0")

    def test_malformed_frames_do_not_kill_the_stream(self):
        lines = sse(chunk({"content": "before "}))
        lines += [b'data: {"choices": [', b"", b": still alive", b""]
        lines += sse(chunk({"content": "after"}), "data: [DONE]")
        self.transport(FakeResponse(lines=lines))
        events = list(make_client().stream_chat([{"role": "user", "content": "hi"}]))
        self.assertEqual("".join(e["text"] for e in events), "before after")


# ---------------------------------------------------------------------------
# 5. HTTP error mapping and retries
# ---------------------------------------------------------------------------

class TestErrorMapping(ClientTestCase):

    NON_RETRYABLE = [
        (400, "bad_request"),
        (401, "auth"),
        (402, "balance"),
        (422, "bad_request"),
    ]
    RETRYABLE = [
        (429, "rate_limit"),
        (500, "server"),
        (503, "server"),
    ]

    def test_non_retryable_statuses_raise_immediately(self):
        for status, kind in self.NON_RETRYABLE:
            with self.subTest(status=status):
                self.no_sleep()
                recorder = self.transport(
                    FakeResponse(status_code=status, body=error_body(f"boom {status}"))
                )
                client = make_client(max_retries=3)
                with self.assertRaises(DeepSeekError) as caught:
                    list(client.stream_chat([{"role": "user", "content": "hi"}]))
                self.assertEqual(caught.exception.kind, kind)
                self.assertEqual(caught.exception.status_code, status)
                self.assertTrue(str(caught.exception).strip())
                self.assertEqual(recorder.call_count, 1,
                                 f"{status} must be attempted exactly once, never retried")

    def test_retryable_statuses_retry_then_give_up(self):
        for status, kind in self.RETRYABLE:
            with self.subTest(status=status):
                self.no_sleep()
                recorder = self.transport(*[
                    FakeResponse(status_code=status, body=error_body(f"boom {status}"))
                    for _ in range(3)
                ])
                client = make_client(max_retries=2)
                with self.assertRaises(DeepSeekError) as caught:
                    list(client.stream_chat([{"role": "user", "content": "hi"}]))
                self.assertEqual(caught.exception.kind, kind)
                self.assertEqual(recorder.call_count, 3, "1 attempt + 2 retries")

    def test_retry_then_success(self):
        self.no_sleep()
        recorder = self.transport(
            FakeResponse(status_code=503, body=error_body("overloaded")),
            FakeResponse(lines=text_stream("recovered")),
        )
        events = list(make_client(max_retries=2).stream_chat([{"role": "user", "content": "hi"}]))
        self.assertEqual(recorder.call_count, 2)
        self.assertEqual("".join(e["text"] for e in events), "recovered")

    def test_retry_after_header_is_honoured(self):
        patcher = mock.patch("mortgage_agent.time.sleep")
        sleeper = patcher.start()
        self.addCleanup(patcher.stop)
        self.transport(
            FakeResponse(status_code=429, body=error_body("slow down"), headers={"Retry-After": "2"}),
            FakeResponse(lines=text_stream("ok")),
        )
        list(make_client(max_retries=1).stream_chat([{"role": "user", "content": "hi"}]))
        sleeper.assert_called_once()
        self.assertAlmostEqual(sleeper.call_args[0][0], 2.0, places=6)

    def test_error_detail_is_folded_into_the_message(self):
        self.transport(FakeResponse(status_code=401, body=error_body("Authentication Fails")))
        with self.assertRaises(DeepSeekError) as caught:
            list(make_client().stream_chat([{"role": "user", "content": "hi"}]))
        self.assertIn("Authentication Fails", str(caught.exception))

    def test_timeout_and_network_errors(self):
        import requests as _requests
        for exc, kind in (
            (_requests.exceptions.Timeout("timed out"), "timeout"),
            (_requests.exceptions.ConnectionError("no route"), "network"),
        ):
            with self.subTest(kind=kind):
                self.transport(exc)
                with self.assertRaises(DeepSeekError) as caught:
                    list(make_client().stream_chat([{"role": "user", "content": "hi"}]))
                self.assertEqual(caught.exception.kind, kind)

    def test_missing_api_key_never_reaches_the_network(self):
        recorder = self.transport()
        client = DeepSeekClient(api_key="")
        with self.assertRaises(DeepSeekError) as caught:
            list(client.stream_chat([{"role": "user", "content": "hi"}]))
        self.assertEqual(caught.exception.kind, "auth")
        self.assertEqual(recorder.call_count, 0)

    def test_validate_success_and_failure(self):
        self.transport(FakeResponse(body={"model": "deepseek-v4-pro", "choices": []}))
        ok, message = make_client().validate()
        self.assertTrue(ok)
        self.assertIn("deepseek-v4-pro", message)

        self.transport(FakeResponse(status_code=401, body=error_body("bad key")))
        ok, message = make_client().validate()
        self.assertFalse(ok)
        self.assertIn("Invalid DeepSeek API key", message)

        ok, message = DeepSeekClient(api_key="").validate()
        self.assertFalse(ok)
        self.assertEqual(message, "No API key provided.")


# ---------------------------------------------------------------------------
# 6. Empty-completion workaround
# ---------------------------------------------------------------------------

class TestEmptyCompletionGuard(ClientTestCase):
    """``deepseek-v4-pro`` can return a 0-token completion after tool results."""

    def test_empty_stream_retries_once_non_streamed(self):
        recorder = self.transport(
            FakeResponse(lines=sse("data: [DONE]")),
            FakeResponse(body=completion_body(
                content="Recovered on the non-streamed retry.",
                usage={"prompt_tokens": 10, "completion_tokens": 7, "total_tokens": 17},
            )),
        )
        events = list(make_client().stream_chat([{"role": "user", "content": "hi"}]))
        self.assertEqual(recorder.call_count, 2, "exactly one retry")
        self.assertIs(recorder.payload(0)["stream"], True)
        self.assertIs(recorder.payload(1)["stream"], False)
        self.assertEqual(
            [e["type"] for e in events], ["content", "usage"]
        )
        self.assertEqual(events[0]["text"], "Recovered on the non-streamed retry.")

    def test_usage_only_stream_still_counts_as_empty(self):
        recorder = self.transport(
            FakeResponse(lines=sse(
                chunk(usage={"prompt_tokens": 5, "completion_tokens": 0, "total_tokens": 5}),
                "data: [DONE]",
            )),
            FakeResponse(body=completion_body(content="second try")),
        )
        events = list(make_client().stream_chat([{"role": "user", "content": "hi"}]))
        self.assertEqual(recorder.call_count, 2)
        self.assertIn("second try", "".join(e.get("text", "") for e in events))

    def test_empty_twice_raises_empty_response(self):
        recorder = self.transport(
            FakeResponse(lines=sse("data: [DONE]")),
            FakeResponse(body=completion_body(content="")),
        )
        with self.assertRaises(DeepSeekError) as caught:
            list(make_client().stream_chat([{"role": "user", "content": "hi"}]))
        self.assertEqual(caught.exception.kind, "empty_response")
        self.assertEqual(recorder.call_count, 2, "one retry, then give up")

    def test_retry_may_come_back_with_tool_calls(self):
        self.transport(
            FakeResponse(lines=sse("data: [DONE]")),
            FakeResponse(body=completion_body(tool_calls=[{
                "id": "call_retry", "type": "function",
                "function": {"name": "get_mortgage_summary", "arguments": "{}"},
            }])),
        )
        events = list(make_client().stream_chat([{"role": "user", "content": "hi"}]))
        self.assertEqual(events[0]["type"], "tool_calls")
        self.assertEqual(events[0]["tool_calls"][0]["id"], "call_retry")

    def test_unreadable_retry_body_is_a_server_error(self):
        self.transport(
            FakeResponse(lines=sse("data: [DONE]")),
            FakeResponse(bad_json=True),
        )
        with self.assertRaises(DeepSeekError) as caught:
            list(make_client().stream_chat([{"role": "user", "content": "hi"}]))
        self.assertEqual(caught.exception.kind, "server")

    def test_a_stream_with_content_is_never_retried(self):
        recorder = self.transport(FakeResponse(lines=text_stream("all good")))
        list(make_client().stream_chat([{"role": "user", "content": "hi"}]))
        self.assertEqual(recorder.call_count, 1)


# ---------------------------------------------------------------------------
# 7. Agent loop
# ---------------------------------------------------------------------------

class TestAgentLoop(ClientTestCase):

    def setUp(self):
        self.ctx = default_context()

    def test_no_tool_call_is_a_single_round(self):
        recorder = self.transport(FakeResponse(lines=text_stream(
            "Your monthly payment is ", "$1,288.37.",
            usage={"prompt_tokens": 900, "completion_tokens": 12, "total_tokens": 912},
        )))
        agent = MortgageAgent(self.ctx, make_client())
        events = list(agent.run([{"role": "user", "content": "What is my monthly payment?"}]))
        self.assertEqual([e["type"] for e in events], ["text", "text", "done"])
        self.assertEqual(events[-1]["content"], "Your monthly payment is $1,288.37.")
        self.assertEqual(events[-1]["tool_calls"], [])
        self.assertEqual(events[-1]["usage"]["total_tokens"], 912)
        self.assertEqual(recorder.call_count, 1)

    def test_full_two_round_tool_calling_conversation(self):
        recorder = self.transport(
            FakeResponse(lines=tool_call_stream(
                "simulate_extra_monthly_payment",
                ['{"', "extra", "_mont", 'hly": ', "200}"],
                call_id="call_round1",
            )),
            FakeResponse(lines=text_stream(
                "You would save ", "about 6 years.",
                usage={"prompt_tokens": 2100, "completion_tokens": 40, "total_tokens": 2140},
            )),
        )
        agent = MortgageAgent(self.ctx, make_client())
        events = list(agent.run([{"role": "user", "content": "What if I pay 200 extra a month?"}]))

        self.assertEqual(
            [e["type"] for e in events],
            ["tool_start", "tool_end", "text", "text", "done"],
        )
        start, end, _, _, done = events
        self.assertEqual(start["name"], "simulate_extra_monthly_payment")
        self.assertEqual(start["args"], {"extra_monthly": 200})
        self.assertEqual(start["id"], "call_round1")
        self.assertEqual(start["index"], 0)
        self.assertEqual(end["id"], start["id"])
        self.assertEqual(end["index"], start["index"])
        self.assertNotIn("error", end["result"])
        self.assertGreater(end["result"]["months_saved"], 0)
        self.assertIsInstance(end["ms"], int)
        self.assertEqual(done["content"], "You would save about 6 years.")
        self.assertEqual(len(done["tool_calls"]), 1)
        self.assertEqual(done["tool_calls"][0]["name"], "simulate_extra_monthly_payment")
        self.assertEqual(done["usage"]["total_tokens"], 2140)
        self.assertEqual(recorder.call_count, 2)

        # Round 2 must carry the assistant tool_calls message and the tool result.
        second = recorder.payload(1)["messages"]
        self.assertEqual(second[0]["role"], "system")
        self.assertEqual(second[-2]["role"], "assistant")
        self.assertEqual(second[-2]["tool_calls"][0]["id"], "call_round1")
        self.assertEqual(second[-1]["role"], "tool")
        self.assertEqual(second[-1]["tool_call_id"], "call_round1")
        json.loads(second[-1]["content"])  # the tool result must be valid JSON

    def test_same_tool_twice_in_one_round_gets_distinct_indices(self):
        """The D3 pairing fix depends on these being unique."""
        frames = [
            chunk({"tool_calls": [
                {"index": 0, "id": "call_x", "type": "function",
                 "function": {"name": "get_payment_details", "arguments": '{"payment_number": 12}'}},
                {"index": 1, "id": "call_y", "type": "function",
                 "function": {"name": "get_payment_details", "arguments": '{"payment_number": 240}'}},
            ]}),
            chunk({}, finish_reason="tool_calls"),
            "data: [DONE]",
        ]
        self.transport(
            FakeResponse(lines=sse(*frames)),
            FakeResponse(lines=text_stream("Both payments checked.")),
        )
        agent = MortgageAgent(self.ctx, make_client())
        events = list(agent.run([{"role": "user", "content": "compare payment 12 and 240"}]))

        starts = [e for e in events if e["type"] == "tool_start"]
        ends = [e for e in events if e["type"] == "tool_end"]
        self.assertEqual(len(starts), 2)
        self.assertEqual([s["index"] for s in starts], [0, 1])
        self.assertEqual([s["id"] for s in starts], ["call_x", "call_y"])
        self.assertEqual([e["index"] for e in ends], [0, 1])
        # Pairing by index must line up arguments with the matching result.
        by_index = {e["index"]: e for e in ends}
        self.assertEqual(by_index[0]["result"]["payment_number"], 12)
        self.assertEqual(by_index[1]["result"]["payment_number"], 240)

    def test_round_cap_forces_a_final_tools_omitted_call(self):
        tool_round = lambda n: FakeResponse(lines=tool_call_stream(  # noqa: E731
            "get_mortgage_summary", ["{}"], call_id=f"call_{n}"
        ))
        recorder = self.transport(
            tool_round(1),
            tool_round(2),
            FakeResponse(lines=text_stream("Final answer without tools.")),
        )
        agent = MortgageAgent(self.ctx, make_client(), max_tool_rounds=2)
        events = list(agent.run([{"role": "user", "content": "loop forever please"}]))

        self.assertEqual(recorder.call_count, 3)
        self.assertIn("tools", recorder.payload(0))
        self.assertIn("tools", recorder.payload(1))
        self.assertNotIn("tools", recorder.payload(2),
                         "the capped round must omit tools entirely")
        self.assertNotIn("tool_choice", recorder.payload(2))
        self.assertEqual(events[-1]["type"], "done")
        self.assertEqual(events[-1]["content"], "Final answer without tools.")
        self.assertEqual(len(events[-1]["tool_calls"]), 2)

    def test_tool_call_indices_keep_counting_across_rounds(self):
        self.transport(
            FakeResponse(lines=tool_call_stream("get_mortgage_summary", ["{}"], call_id="c1")),
            FakeResponse(lines=tool_call_stream("get_mortgage_summary", ["{}"], call_id="c2")),
            FakeResponse(lines=text_stream("done")),
        )
        agent = MortgageAgent(self.ctx, make_client(), max_tool_rounds=3)
        events = list(agent.run([{"role": "user", "content": "hi"}]))
        indices = [e["index"] for e in events if e["type"] == "tool_start"]
        self.assertEqual(indices, [0, 1])

    def test_bad_tool_arguments_are_fed_back_not_raised(self):
        self.transport(
            FakeResponse(lines=tool_call_stream(
                "get_payment_details", ['{"payment_number": 99999}'], call_id="call_bad"
            )),
            FakeResponse(lines=text_stream("Sorry, that payment does not exist.")),
        )
        agent = MortgageAgent(self.ctx, make_client())
        events = list(agent.run([{"role": "user", "content": "show payment 99999"}]))
        end = next(e for e in events if e["type"] == "tool_end")
        self.assertIn("error", end["result"])
        self.assertEqual(events[-1]["type"], "done")

    def test_unparseable_tool_arguments_become_an_error_result(self):
        self.transport(
            FakeResponse(lines=tool_call_stream(
                "get_payment_details", ['{"payment_number": '], call_id="call_trunc"
            )),
            FakeResponse(lines=text_stream("Let me try that again.")),
        )
        agent = MortgageAgent(self.ctx, make_client())
        events = list(agent.run([{"role": "user", "content": "hi"}]))
        end = next(e for e in events if e["type"] == "tool_end")
        self.assertIn("error", end["result"])
        self.assertIn("JSON", end["result"]["error"])

    def test_api_failure_becomes_a_terminal_error_event(self):
        self.no_sleep()
        self.transport(FakeResponse(status_code=402, body=error_body("Insufficient Balance")))
        agent = MortgageAgent(self.ctx, make_client())
        events = list(agent.run([{"role": "user", "content": "hi"}]))
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["type"], "error")
        self.assertEqual(events[0]["kind"], "balance")
        self.assertIn("platform.deepseek.com", events[0]["message"])

    def test_temperature_is_threaded_through(self):
        recorder = self.transport(FakeResponse(lines=text_stream("ok")))
        agent = MortgageAgent(self.ctx, make_client(), temperature=0.85)
        list(agent.run([{"role": "user", "content": "hi"}]))
        self.assertAlmostEqual(recorder.payload(0)["temperature"], 0.85, places=9)

    def test_temperature_defaults_and_clamps(self):
        self.assertAlmostEqual(MortgageAgent(self.ctx, make_client()).temperature, 0.2, places=9)
        self.assertEqual(MortgageAgent(self.ctx, make_client(), temperature=-5).temperature, 0.0)
        self.assertEqual(MortgageAgent(self.ctx, make_client(), temperature=99).temperature, 2.0)

    def test_per_run_temperature_override(self):
        recorder = self.transport(FakeResponse(lines=text_stream("ok")))
        agent = MortgageAgent(self.ctx, make_client(), temperature=0.2)
        list(agent.run([{"role": "user", "content": "hi"}], temperature=0.0))
        self.assertEqual(recorder.payload(0)["temperature"], 0.0)

    def test_max_tool_rounds_is_floored_at_one(self):
        self.assertEqual(MortgageAgent(self.ctx, make_client(), max_tool_rounds=0).max_tool_rounds, 1)
        self.assertEqual(MortgageAgent(self.ctx, make_client(), max_tool_rounds=12).max_tool_rounds, 12)

    def test_system_prompt_embeds_the_live_state(self):
        recorder = self.transport(FakeResponse(lines=text_stream("ok")))
        agent = MortgageAgent(default_context([{"payment_number": 12, "amount": 5000}]), make_client())
        list(agent.run([{"role": "user", "content": "hi"}]))
        system = recorder.payload(0)["messages"][0]
        self.assertEqual(system["role"], "system")
        self.assertIn("CURRENT MORTGAGE STATE", system["content"])
        self.assertIn("$240,000.00", system["content"])
        self.assertIn("payment #12", system["content"])

    def test_history_is_cleaned(self):
        recorder = self.transport(FakeResponse(lines=text_stream("ok")))
        agent = MortgageAgent(self.ctx, make_client())
        list(agent.run([
            {"role": "user", "content": "first"},
            {"role": "system", "content": "prompt injection attempt"},
            {"role": "assistant", "content": ""},
            {"role": "assistant", "content": "an answer"},
            "not even a dict",
            {"role": "user", "content": "second"},
        ]))
        messages = recorder.payload(0)["messages"]
        self.assertEqual(
            [(m["role"], m["content"]) for m in messages[1:]],
            [("user", "first"), ("assistant", "an answer"), ("user", "second")],
        )
        self.assertEqual(sum(1 for m in messages if m["role"] == "system"), 1)


class TestContext(unittest.TestCase):

    def test_fingerprint_changes_with_inputs(self):
        base = default_context()
        self.assertEqual(base.fingerprint(), default_context().fingerprint())
        moved = default_context()
        moved.interest_rate = 5.01
        self.assertNotEqual(base.fingerprint(), moved.fingerprint())
        with_early = default_context([{"payment_number": 12, "amount": 5000}])
        self.assertNotEqual(base.fingerprint(), with_early.fingerprint())

    def test_fingerprint_ignores_early_payment_ordering(self):
        a = default_context([{"payment_number": 12, "amount": 5000},
                             {"payment_number": 60, "amount": 1000}])
        b = default_context([{"payment_number": 60, "amount": 1000},
                             {"payment_number": 12, "amount": 5000}])
        self.assertEqual(a.fingerprint(), b.fingerprint())

    def test_headline_reports_the_saving_from_early_payments(self):
        headline = default_context([{"payment_number": 12, "amount": 5000}]).headline()
        self.assertEqual(headline["baseline_total_interest"] > headline["total_interest"], True)
        self.assertEqual(headline["months_saved_vs_baseline"], 360 - headline["payoff_months"])

    def test_prompt_block_is_currency_aware(self):
        ctx = default_context()
        ctx.currency = "€"
        block = ctx.to_prompt_block()
        self.assertIn("€240,000.00", block)
        self.assertNotIn("$", block)


# ---------------------------------------------------------------------------
# 8. UI — app_ai.py under AppTest
# ---------------------------------------------------------------------------

EXPECTED_TABS = [
    "📊 Basic Analysis",
    "💰 Advanced Features",
    "🔄 Refinance Calculator",
    "⚡ Early Repayment Simulator",
    "🤖 Ask AI",
]


class TestStreamlitApp(unittest.TestCase):
    """``app_ai.py`` must render every tab cleanly, with and without a key or extra payments."""

    APP = os.path.join(APP_DIR, "app_ai.py")

    def build(self, env=None):
        from streamlit.testing.v1 import AppTest

        overrides = {"DEEPSEEK_API_KEY": ""}
        overrides.update(env or {})
        patcher = mock.patch.dict(os.environ, overrides)
        patcher.start()
        self.addCleanup(patcher.stop)
        return AppTest.from_file(self.APP, default_timeout=120)

    def assertClean(self, at):
        self.assertEqual(
            list(at.exception), [],
            "app_ai.py raised: " + "; ".join(str(e.value) for e in at.exception),
        )
        self.assertEqual([tab.label for tab in at.tabs], EXPECTED_TABS)

    def test_renders_with_no_api_key(self):
        at = self.build()
        at.run()
        self.assertClean(at)
        self.assertTrue(at.chat_input[0].disabled, "chat must be disabled without a key")
        setup = " ".join(info.value for info in at.info)
        self.assertIn("platform.deepseek.com", setup)

    def test_renders_with_early_payments(self):
        at = self.build()
        at.run()
        self.assertClean(at)
        at.session_state["early_payments"] = [
            {"payment_number": 12, "amount": 5000},
            {"payment_number": 60, "amount": 20000},
        ]
        at.run()
        self.assertClean(at)
        labels = [metric.label for metric in at.metric]
        self.assertIn("New Loan Term", labels)
        self.assertIn("Interest Saved", labels)

    def test_early_payment_metric_matches_the_core_math(self):
        at = self.build()
        at.run()
        at.session_state["early_payments"] = [{"payment_number": 12, "amount": 5000}]
        at.run()
        self.assertClean(at)
        records = amortization_records(LOAN_AMOUNT, RATE, TERM,
                                       [{"payment_number": 12, "amount": 5000}])
        expected = f"{len(records) / 12:.1f} years"
        shown = [m.value for m in at.metric if m.label == "New Loan Term"]
        self.assertIn(expected, shown, "the UI term must match what the assistant would say")

    def test_renders_with_an_api_key_configured(self):
        at = self.build(env={"DEEPSEEK_API_KEY": "sk-test-000000000000000000004f2a"})
        at.run()
        self.assertClean(at)
        self.assertFalse(at.chat_input[0].disabled)
        captions = " ".join(caption.value for caption in at.caption)
        self.assertIn("sk-…4f2a", captions, "the key must be masked, never printed in full")
        self.assertNotIn("sk-test-000000000000000000004f2a", captions)

    def test_new_ai_controls_exist_with_the_documented_defaults(self):
        at = self.build()
        at.run()
        self.assertClean(at)
        temperature = next(s for s in at.slider if s.label == "Temperature")
        self.assertEqual(temperature.value, 0.2)
        rounds = next(n for n in at.number_input if n.label == "Max tool rounds")
        self.assertEqual(rounds.value, 6)
        self.assertEqual(at.session_state["ai_temperature"], 0.2)
        self.assertEqual(at.session_state["ai_max_tool_rounds"], 6)

    def test_ai_controls_are_interactive(self):
        at = self.build(env={"DEEPSEEK_API_KEY": "sk-test-000000000000000000004f2a"})
        at.run()
        next(s for s in at.slider if s.label == "Temperature").set_value(0.7).run()
        self.assertClean(at)
        self.assertEqual(at.session_state["ai_temperature"], 0.7)
        next(n for n in at.number_input if n.label == "Max tool rounds").set_value(11).run()
        self.assertClean(at)
        self.assertEqual(at.session_state["ai_max_tool_rounds"], 11)


class TestToolPairingHelpers(unittest.TestCase):
    """``app_ai.py``'s D3 pairing helpers, exercised without a Streamlit runtime."""

    @classmethod
    def setUpClass(cls):
        import importlib.util

        # Import the module's helper functions without executing the Streamlit script:
        # read the source, and exec only the two pure functions we need.
        cls.source = open(os.path.join(APP_DIR, "app_ai.py"), encoding="utf-8").read()
        namespace = {}
        tree_src = []
        import ast

        module = ast.parse(cls.source)
        for node in module.body:
            if isinstance(node, ast.FunctionDef) and node.name in (
                "_tool_pair_key", "_take_pending_tool", "mask_api_key"
            ):
                tree_src.append(ast.get_source_segment(cls.source, node))
        exec("\n\n".join(tree_src), namespace)
        cls.ns = namespace
        assert importlib.util  # silence linters

    def test_pair_key_prefers_index(self):
        key = self.ns["_tool_pair_key"]
        self.assertEqual(key({"index": 3, "id": "abc"}), ("index", 3))
        self.assertEqual(key({"id": "abc"}), ("id", "abc"))
        self.assertIsNone(key({"name": "get_mortgage_summary"}))
        self.assertEqual(key({"index": True}), None)  # bools are not indices

    def test_same_tool_twice_pairs_by_index_not_order(self):
        take = self.ns["_take_pending_tool"]
        pending = [
            {"key": ("index", 0), "name": "get_yearly_summary", "record": {"args": "first"}},
            {"key": ("index", 1), "name": "get_yearly_summary", "record": {"args": "second"}},
        ]
        # A tool_end for call #1 arriving first must still find call #1's record.
        second = take(pending, ("index", 1), "get_yearly_summary")
        self.assertEqual(second["record"]["args"], "second")
        first = take(pending, ("index", 0), "get_yearly_summary")
        self.assertEqual(first["record"]["args"], "first")
        self.assertEqual(pending, [])

    def test_falls_back_to_fifo_without_a_key(self):
        take = self.ns["_take_pending_tool"]
        pending = [
            {"key": None, "name": "get_yearly_summary", "record": {"args": "first"}},
            {"key": None, "name": "get_yearly_summary", "record": {"args": "second"}},
        ]
        self.assertEqual(take(pending, None, "get_yearly_summary")["record"]["args"], "first")
        self.assertEqual(take(pending, None, "get_yearly_summary")["record"]["args"], "second")
        self.assertIsNone(take(pending, None, "get_yearly_summary"))

    def test_unknown_key_falls_back_to_the_same_named_tool(self):
        take = self.ns["_take_pending_tool"]
        pending = [
            {"key": ("index", 0), "name": "get_mortgage_summary", "record": {"args": "a"}},
            {"key": ("index", 1), "name": "what_if", "record": {"args": "b"}},
        ]
        picked = take(pending, ("index", 99), "what_if")
        self.assertEqual(picked["record"]["args"], "b")

    def test_api_keys_are_masked(self):
        mask = self.ns["mask_api_key"]
        self.assertEqual(mask("sk-abcdefghijklmnop4f2a"), "sk-…4f2a")
        self.assertEqual(mask(""), "")
        self.assertNotIn("abcdefgh", mask("sk-abcdefghijklmnop4f2a"))


# ---------------------------------------------------------------------------
# 9. Offline guarantee
# ---------------------------------------------------------------------------

class TestNoRealNetwork(unittest.TestCase):
    """The suite must never touch the network, even if a mock is forgotten."""

    def test_requests_post_is_blocked_globally(self):
        import requests
        with self.assertRaises(AssertionError):
            requests.post("https://api.deepseek.com/chat/completions", json={})

    def test_no_api_key_in_the_environment_is_required(self):
        client = DeepSeekClient(api_key="")
        with self.assertRaises(DeepSeekError):
            list(client.stream_chat([{"role": "user", "content": "hi"}]))


def _block_network():
    """Replace ``requests.post`` for the whole run so a missed mock fails loudly."""
    import requests

    def blocked(*args, **kwargs):
        raise AssertionError(
            "A test tried to make a real HTTP request. This suite must run fully offline — "
            "patch mortgage_agent.requests.post instead."
        )

    requests.post = blocked
    requests.Session.request = lambda *a, **k: blocked()


def main():
    _block_network()
    os.environ.pop("DEEPSEEK_API_KEY", None)

    loader = unittest.TestLoader()
    suite = loader.loadTestsFromModule(sys.modules[__name__])
    result = unittest.TextTestRunner(verbosity=2, stream=sys.stdout).run(suite)

    total = result.testsRun
    failed = len(result.failures)
    errored = len(result.errors)
    skipped = len(result.skipped)
    passed = total - failed - errored - skipped

    print()
    print("=" * 70)
    print("  MORTGAGE AI ASSISTANT — OFFLINE TEST SUITE")
    print("=" * 70)
    print(f"  ran      {total}")
    print(f"  passed   {passed}")
    print(f"  failed   {failed}")
    print(f"  errors   {errored}")
    print(f"  skipped  {skipped}")
    print("-" * 70)
    if failed or errored:
        for case, _ in result.failures:
            print(f"  FAIL  {case.id()}")
        for case, _ in result.errors:
            print(f"  ERROR {case.id()}")
        print("-" * 70)
        print("  RESULT: FAILED")
        print("=" * 70)
        return 1
    print("  RESULT: ALL TESTS PASSED  (0 network calls, no API key needed)")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
