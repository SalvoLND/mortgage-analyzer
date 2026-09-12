"""
mortgage_agent.py — DeepSeek-backed mortgage analyst attached to a live calculation.

Dependencies: ``requests`` only (the ``openai`` SDK is intentionally NOT used) plus
``mortgage_core`` for every number. The LLM never does arithmetic itself: each figure it
reports comes from a deterministic tool call backed by
``mortgage_core.amortization_records``.

Public surface (see CONTRACT.md):
    MortgageContext   — the live mortgage state + headline()/fingerprint()/to_prompt_block()
    DeepSeekError     — one exception type, ``.kind`` + ``.status_code``, user-facing str()
    DeepSeekClient    — hand-rolled SSE streaming chat client with retry/backoff
    TOOL_REGISTRY     — dict[str, ToolSpec] of the 10 deterministic tools
    MortgageAgent     — run(history) -> Iterator[event dict]
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Iterator, Optional

import requests

try:  # normal case: both modules sit side by side in the app directory
    from mortgage_core import (
        amortization_records,
        calculate_monthly_payment,
        calculate_refinance_comparison,
        yearly_rollup,
    )
except ImportError:  # pragma: no cover - only when imported as part of a package
    from .mortgage_core import (  # type: ignore
        amortization_records,
        calculate_monthly_payment,
        calculate_refinance_comparison,
        yearly_rollup,
    )

__all__ = [
    "MortgageContext",
    "DeepSeekError",
    "DeepSeekClient",
    "ToolSpec",
    "ToolInputError",
    "TOOL_REGISTRY",
    "tool_schemas",
    "execute_tool",
    "MortgageAgent",
    "DEFAULT_MODEL",
    "AVAILABLE_MODELS",
]

DEFAULT_MODEL = "deepseek-v4-pro"
AVAILABLE_MODELS = ["deepseek-v4-pro", "deepseek-v4-flash"]

_MAX_WINDOW_ROWS = 60


def _r(value: Any, digits: int = 2) -> Any:
    """Round for reporting; passes None/NaN/inf through untouched (JSON-safe check upstream)."""
    if value is None:
        return None
    try:
        f = float(value)
    except (TypeError, ValueError):
        return value
    if math.isnan(f) or math.isinf(f):
        return None
    return round(f, digits)


# ---------------------------------------------------------------------------
# Mortgage context
# ---------------------------------------------------------------------------

@dataclass
class MortgageContext:
    """The live calculation the assistant is attached to.

    ``early_payments`` uses the UI shape: ``[{'payment_number': int, 'amount': float}, ...]``
    """

    property_value: float
    down_payment: float
    loan_amount: float
    interest_rate: float          # annual %, e.g. 5.0
    loan_term: int                # years
    early_payments: list = field(default_factory=list)
    currency: str = "$"

    _cache: dict = field(default_factory=dict, init=False, repr=False, compare=False)

    # -- numbers -----------------------------------------------------------

    def records(self, early_payments: Optional[list] = None) -> list[dict]:
        """Numeric schedule for this context (cached on the fingerprint)."""
        if early_payments is not None:
            return amortization_records(
                self.loan_amount, self.interest_rate, self.loan_term, early_payments
            )
        key = self.fingerprint()
        cached = self._cache.get(key)
        if cached is None:
            cached = amortization_records(
                self.loan_amount, self.interest_rate, self.loan_term, self.early_payments
            )
            self._cache.clear()
            self._cache[key] = cached
        return cached

    def monthly_payment(self) -> float:
        return calculate_monthly_payment(self.loan_amount, self.interest_rate, self.loan_term)

    def headline(self) -> dict:
        """Every top-line figure, straight from ``amortization_records``."""
        records = self.records()
        baseline = (
            self.records(early_payments=[]) if self.early_payments else records
        )

        total_interest = records[-1]["cum_interest"] if records else 0.0
        total_extra = sum(r["extra"] for r in records)
        total_principal = records[-1]["cum_principal"] if records else 0.0
        total_paid = total_principal + total_interest
        payoff_months = len(records)

        baseline_interest = baseline[-1]["cum_interest"] if baseline else 0.0
        baseline_months = len(baseline)

        return {
            "currency": self.currency,
            "property_value": self.property_value,
            "down_payment": self.down_payment,
            "loan_amount": self.loan_amount,
            "interest_rate_pct": self.interest_rate,
            "loan_term_years": self.loan_term,
            "scheduled_payments": int(self.loan_term) * 12,
            "monthly_payment": self.monthly_payment(),
            "total_paid": total_paid,
            "total_principal": total_principal,
            "total_interest": total_interest,
            "total_extra_payments": total_extra,
            "interest_ratio_pct": (total_interest / self.loan_amount * 100.0) if self.loan_amount else None,
            "ltv_pct": ((1 - (self.down_payment / self.property_value)) * 100.0) if self.property_value else None,
            "payoff_months": payoff_months,
            "payoff_years": payoff_months / 12.0,
            "first_payment_date": records[0]["date"] if records else None,
            "payoff_date": records[-1]["date"] if records else None,
            "early_payment_count": len(self.early_payments or []),
            "baseline_total_interest": baseline_interest,
            "interest_saved_vs_baseline": baseline_interest - total_interest,
            "months_saved_vs_baseline": baseline_months - payoff_months,
        }

    # -- identity ----------------------------------------------------------

    def fingerprint(self) -> str:
        """Stable short hash of the inputs; the UI uses it to detect parameter changes."""
        payload = {
            "property_value": round(float(self.property_value), 6),
            "down_payment": round(float(self.down_payment), 6),
            "loan_amount": round(float(self.loan_amount), 6),
            "interest_rate": round(float(self.interest_rate), 6),
            "loan_term": int(self.loan_term),
            "currency": self.currency,
            "early_payments": sorted(
                (int(p["payment_number"]), round(float(p["amount"]), 6))
                for p in (self.early_payments or [])
            ),
        }
        blob = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]

    # -- prompt ------------------------------------------------------------

    def money(self, value: Optional[float]) -> str:
        if value is None:
            return "n/a"
        return f"{self.currency}{value:,.2f}"

    def to_prompt_block(self) -> str:
        """Compact human-readable state block embedded in the system prompt."""
        h = self.headline()
        lines = [
            "CURRENT MORTGAGE STATE (live values from the calculator):",
            f"- Currency: {self.currency}",
            f"- Property value: {self.money(h['property_value'])}",
            f"- Down payment: {self.money(h['down_payment'])}"
            + (f" (LTV {h['ltv_pct']:.1f}%)" if h["ltv_pct"] is not None else ""),
            f"- Loan amount: {self.money(h['loan_amount'])}",
            f"- Annual interest rate: {h['interest_rate_pct']:.2f}%",
            f"- Term: {h['loan_term_years']} years ({h['scheduled_payments']} monthly payments)",
            f"- Monthly payment: {self.money(h['monthly_payment'])}",
            f"- Total paid over the loan: {self.money(h['total_paid'])}",
            f"- Total interest: {self.money(h['total_interest'])}"
            + (f" ({h['interest_ratio_pct']:.1f}% of the loan amount)" if h["interest_ratio_pct"] is not None else ""),
            f"- First payment: {h['first_payment_date']}, final payment: {h['payoff_date']}"
            f" (payment #{h['payoff_months']})",
        ]

        if self.early_payments:
            lines.append(f"- Early/extra payments currently configured ({len(self.early_payments)}):")
            for p in sorted(self.early_payments, key=lambda x: x["payment_number"])[:12]:
                lines.append(
                    f"    * {self.money(float(p['amount']))} at payment #{int(p['payment_number'])}"
                )
            if len(self.early_payments) > 12:
                lines.append(f"    * ... and {len(self.early_payments) - 12} more")
            lines.append(
                f"- Effect of those extra payments: interest saved "
                f"{self.money(h['interest_saved_vs_baseline'])}, "
                f"{h['months_saved_vs_baseline']} months earlier payoff "
                f"(without them: {self.money(h['baseline_total_interest'])} interest)."
            )
        else:
            lines.append("- Early/extra payments currently configured: none")

        return "\n".join(lines)


# ---------------------------------------------------------------------------
# DeepSeek client
# ---------------------------------------------------------------------------

class DeepSeekError(Exception):
    """One error type for everything that can go wrong talking to DeepSeek.

    ``str(e)`` is always a short, actionable, user-facing sentence — safe to drop
    straight into ``st.error()``.
    """

    def __init__(self, message: str, kind: str = "server", status_code: Optional[int] = None):
        super().__init__(message)
        self.kind = kind              # auth|balance|rate_limit|server|network|bad_request|empty_response|timeout
        self.status_code = status_code


_RETRYABLE_STATUS = {429, 500, 503}

_STATUS_MAP: dict[int, tuple[str, str]] = {
    400: ("bad_request", "DeepSeek rejected the request (400) — try clearing the chat and asking again."),
    401: ("auth", "Invalid DeepSeek API key — check the key in the sidebar."),
    402: ("balance", "Insufficient DeepSeek balance — top up at platform.deepseek.com"),
    403: ("auth", "This DeepSeek API key is not allowed to use that model — check your account."),
    404: ("bad_request", "DeepSeek endpoint or model not found (404) — check the model name."),
    413: ("bad_request", "The conversation is too large for DeepSeek — clear the chat and try again."),
    422: ("bad_request", "DeepSeek rejected the request parameters (422) — try another model or turn reasoning off."),
    429: ("rate_limit", "DeepSeek rate limit reached — wait a few seconds and try again."),
    500: ("server", "DeepSeek had a server error — please retry in a moment."),
    502: ("server", "DeepSeek is unreachable right now (502) — please retry in a moment."),
    503: ("server", "DeepSeek is temporarily overloaded — please retry in a moment."),
    504: ("timeout", "DeepSeek timed out (504) — please retry in a moment."),
}


class DeepSeekClient:
    """Minimal OpenAI-compatible chat client for api.deepseek.com built on ``requests``."""

    def __init__(
        self,
        api_key: str,
        model: str = DEFAULT_MODEL,
        base_url: str = "https://api.deepseek.com",
        timeout: int = 120,
        max_retries: int = 3,
        reasoning_effort: Optional[str] = None,  # None => thinking disabled
    ):
        self.api_key = (api_key or "").strip()
        self.model = model or DEFAULT_MODEL
        self.base_url = (base_url or "https://api.deepseek.com").rstrip("/")
        self.timeout = timeout
        self.max_retries = max(0, int(max_retries))
        self.reasoning_effort = reasoning_effort if reasoning_effort not in ("", "off", "none") else None

    # -- plumbing ----------------------------------------------------------

    @property
    def endpoint(self) -> str:
        return f"{self.base_url}/chat/completions"

    def _headers(self, stream: bool) -> dict:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Accept": "text/event-stream" if stream else "application/json",
        }

    def _thinking(self) -> dict:
        if self.reasoning_effort:
            return {"type": "enabled", "reasoning_effort": self.reasoning_effort}
        return {"type": "disabled"}

    def _payload(self, messages, tools, temperature, stream: bool) -> dict:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "stream": bool(stream),
            "temperature": temperature,
            "thinking": self._thinking(),
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"
        return payload

    @staticmethod
    def _error_from_response(response) -> DeepSeekError:
        kind, message = _STATUS_MAP.get(
            response.status_code,
            ("server", f"DeepSeek returned an unexpected error ({response.status_code})."),
        )
        detail = ""
        try:
            body = response.json()
            if isinstance(body, dict):
                err = body.get("error")
                if isinstance(err, dict):
                    detail = str(err.get("message") or "")
                elif isinstance(err, str):
                    detail = err
                elif body.get("message"):
                    detail = str(body["message"])
        except Exception:
            detail = ""
        if detail:
            detail = detail.strip().replace("\n", " ")
            if len(detail) > 160:
                detail = detail[:157] + "..."
            message = f"{message} ({detail})"
        return DeepSeekError(message, kind=kind, status_code=response.status_code)

    def _sleep_for(self, attempt: int, response=None) -> float:
        """Exponential backoff with jitter; honours Retry-After when the API sends one."""
        if response is not None:
            retry_after = response.headers.get("Retry-After")
            if retry_after:
                try:
                    return min(30.0, max(0.0, float(retry_after)))
                except (TypeError, ValueError):
                    pass
        return min(8.0, 0.75 * (2 ** attempt)) + random.uniform(0.0, 0.4)

    def _post(self, payload: dict, stream: bool):
        if not self.api_key:
            raise DeepSeekError("No DeepSeek API key set — add one in the sidebar.", kind="auth")

        attempt = 0
        while True:
            try:
                response = requests.post(
                    self.endpoint,
                    headers=self._headers(stream),
                    json=payload,
                    stream=stream,
                    timeout=self.timeout,
                )
            except requests.exceptions.Timeout as exc:
                raise DeepSeekError(
                    "DeepSeek took too long to respond — try again, or lower the reasoning effort.",
                    kind="timeout",
                ) from exc
            except requests.exceptions.RequestException as exc:
                raise DeepSeekError(
                    "Could not reach DeepSeek — check your internet connection.",
                    kind="network",
                ) from exc

            if response.status_code < 400:
                return response

            # Retry 429/500/503 only. 400/401/402/422 are never retried.
            if response.status_code in _RETRYABLE_STATUS and attempt < self.max_retries:
                delay = self._sleep_for(attempt, response)
                response.close()
                time.sleep(delay)
                attempt += 1
                continue

            error = self._error_from_response(response)
            response.close()
            raise error

    # -- SSE ---------------------------------------------------------------

    @staticmethod
    def _iter_sse_payloads(lines) -> Iterator[dict]:
        """Hand-rolled SSE reader.

        Tolerates keep-alive blank lines, ``:`` comment lines, non-JSON junk and any
        line that is not a ``data:`` frame. Stops at ``data: [DONE]``.
        """
        for raw in lines:
            if raw is None:
                continue
            if isinstance(raw, (bytes, bytearray)):
                try:
                    line = raw.decode("utf-8")
                except UnicodeDecodeError:
                    line = raw.decode("utf-8", errors="replace")
            else:
                line = str(raw)

            line = line.strip()
            if not line:            # keep-alive / frame separator
                continue
            if line.startswith(":"):  # SSE comment, also used as keep-alive
                continue
            if not line.startswith("data:"):
                continue

            data = line[5:].strip()
            if not data:
                continue
            if data == "[DONE]":
                return
            try:
                chunk = json.loads(data)
            except (json.JSONDecodeError, ValueError):
                continue            # malformed frame: skip, never crash the stream
            if isinstance(chunk, dict):
                yield chunk

    @staticmethod
    def _merge_tool_call_delta(acc: dict[int, dict], delta_calls) -> None:
        """Assemble streamed tool calls by ``index``.

        ``arguments`` arrives as fragmented JSON string chunks and MUST be concatenated
        before ``json.loads``. ``name`` normally arrives once, but some providers repeat
        it whole on every delta and others fragment it — concatenate unless the fragment
        is an exact repeat of what we already have.
        """
        for call in delta_calls or []:
            if not isinstance(call, dict):
                continue
            try:
                index = int(call.get("index", 0) or 0)
            except (TypeError, ValueError):
                index = 0

            slot = acc.get(index)
            if slot is None:
                slot = {"id": "", "type": "function", "index": index,
                        "function": {"name": "", "arguments": ""}}
                acc[index] = slot

            if call.get("id"):
                slot["id"] = call["id"]
            if call.get("type"):
                slot["type"] = call["type"]

            fn = call.get("function") or {}
            name_fragment = fn.get("name")
            if name_fragment:
                current = slot["function"]["name"]
                if not current:
                    slot["function"]["name"] = name_fragment
                elif current != name_fragment:
                    slot["function"]["name"] = current + name_fragment
            args_fragment = fn.get("arguments")
            if args_fragment:
                slot["function"]["arguments"] += args_fragment

    @staticmethod
    def _finalise_tool_calls(acc: dict[int, dict]) -> list[dict]:
        calls = []
        for index in sorted(acc):
            slot = acc[index]
            calls.append({
                "id": slot.get("id") or f"call_{index}",
                "type": slot.get("type") or "function",
                "function": {
                    "name": slot["function"]["name"],
                    "arguments": slot["function"]["arguments"],
                },
            })
        return calls

    # -- public API --------------------------------------------------------

    def stream_chat(self, messages, tools=None, temperature: float = 0.2) -> Iterator[dict]:
        """Stream one completion.

        Yields ``{'type':'reasoning'|'content','text':str}`` as they arrive,
        ``{'type':'tool_calls','tool_calls':[...]}`` once at the end (fully assembled),
        and ``{'type':'usage','usage':{...}}`` if the API reports usage.

        Implements the documented ``deepseek-v4-pro`` empty-completion bug workaround:
        a stream that produced no content, no reasoning and no tool calls is retried once
        non-streamed before ``DeepSeekError(kind='empty_response')`` is raised.
        """
        payload = self._payload(messages, tools, temperature, stream=True)
        response = self._post(payload, stream=True)

        produced = False
        tool_acc: dict[int, dict] = {}
        usage: Optional[dict] = None

        try:
            for chunk in self._iter_sse_payloads(response.iter_lines()):
                if isinstance(chunk.get("usage"), dict):
                    usage = chunk["usage"]

                choices = chunk.get("choices") or []
                if not choices:
                    continue
                choice = choices[0] or {}
                delta = choice.get("delta") or choice.get("message") or {}
                if not isinstance(delta, dict):
                    continue

                reasoning = delta.get("reasoning_content") or delta.get("reasoning")
                if reasoning:
                    produced = True
                    yield {"type": "reasoning", "text": str(reasoning)}

                content = delta.get("content")
                if content:
                    produced = True
                    yield {"type": "content", "text": str(content)}

                if delta.get("tool_calls"):
                    self._merge_tool_call_delta(tool_acc, delta["tool_calls"])
        finally:
            try:
                response.close()
            except Exception:
                pass

        if tool_acc:
            produced = True
            yield {"type": "tool_calls", "tool_calls": self._finalise_tool_calls(tool_acc)}

        if usage:
            yield {"type": "usage", "usage": usage}

        if not produced:
            # Known deepseek-v4-pro bug: empty completion after tool results.
            yield from self._retry_non_streamed(payload)

    def _retry_non_streamed(self, payload: dict) -> Iterator[dict]:
        retry_payload = dict(payload)
        retry_payload["stream"] = False
        response = self._post(retry_payload, stream=False)
        try:
            body = response.json()
        except ValueError as exc:
            raise DeepSeekError(
                "DeepSeek returned a response that could not be read — please retry.",
                kind="server",
                status_code=response.status_code,
            ) from exc
        finally:
            try:
                response.close()
            except Exception:
                pass

        message = {}
        choices = body.get("choices") or []
        if choices and isinstance(choices[0], dict):
            message = choices[0].get("message") or {}

        produced = False
        reasoning = message.get("reasoning_content") or message.get("reasoning")
        if reasoning:
            produced = True
            yield {"type": "reasoning", "text": str(reasoning)}
        content = message.get("content")
        if content:
            produced = True
            yield {"type": "content", "text": str(content)}
        if message.get("tool_calls"):
            acc: dict[int, dict] = {}
            for i, call in enumerate(message["tool_calls"]):
                call = dict(call)
                call.setdefault("index", i)
                self._merge_tool_call_delta(acc, [call])
            calls = self._finalise_tool_calls(acc)
            if calls:
                produced = True
                yield {"type": "tool_calls", "tool_calls": calls}
        if isinstance(body.get("usage"), dict):
            yield {"type": "usage", "usage": body["usage"]}

        if not produced:
            raise DeepSeekError(
                "DeepSeek returned an empty response — please ask again.",
                kind="empty_response",
            )

    def validate(self) -> tuple[bool, str]:
        """Cheap 1-token call to check key + model. Returns ``(ok, message)``."""
        if not self.api_key:
            return False, "No API key provided."
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": "ping"}],
            "stream": False,
            "max_tokens": 1,
            "temperature": 0,
            "thinking": {"type": "disabled"},
        }
        try:
            response = self._post(payload, stream=False)
        except DeepSeekError as exc:
            return False, str(exc)
        try:
            body = response.json()
        except ValueError:
            return False, "DeepSeek replied with something unreadable — try again."
        finally:
            try:
                response.close()
            except Exception:
                pass
        model = body.get("model") or self.model
        return True, f"Connection OK — model '{model}' responded."


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

class ToolInputError(ValueError):
    """Raised inside a tool for bad model-supplied input; converted to {'error': ...}."""


@dataclass
class ToolSpec:
    name: str
    description: str
    parameters: dict
    handler: Callable[..., dict]

    @property
    def schema(self) -> dict:
        """OpenAI-style function schema sent in the ``tools`` array."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


def _as_float(value, name: str) -> float:
    if isinstance(value, bool) or value is None:
        raise ToolInputError(f"'{name}' must be a number.")
    try:
        out = float(value)
    except (TypeError, ValueError):
        raise ToolInputError(f"'{name}' must be a number, got {value!r}.") from None
    if math.isnan(out) or math.isinf(out):
        raise ToolInputError(f"'{name}' must be a finite number.")
    return out


def _as_int(value, name: str) -> int:
    f = _as_float(value, name)
    if abs(f - round(f)) > 1e-9:
        raise ToolInputError(f"'{name}' must be a whole number, got {value!r}.")
    return int(round(f))


def _record_view(rec: dict) -> dict:
    """One ``amortization_records`` row, rounded for reporting."""
    return {
        "payment_number": rec["payment_number"],
        "date": rec["date"],
        "payment": _r(rec["payment"]),
        "principal": _r(rec["principal"]),
        "interest": _r(rec["interest"]),
        "extra": _r(rec["extra"]),
        "balance": _r(rec["balance"]),
        "cum_principal": _r(rec["cum_principal"]),
        "cum_interest": _r(rec["cum_interest"]),
    }


# --- 1. summary -------------------------------------------------------------

def _tool_get_mortgage_summary(ctx: MortgageContext) -> dict:
    h = ctx.headline()
    out = {k: (_r(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else v)
           for k, v in h.items()}
    out["loan_term_years"] = int(ctx.loan_term)
    out["payoff_months"] = int(h["payoff_months"])
    out["scheduled_payments"] = int(h["scheduled_payments"])
    out["early_payment_count"] = int(h["early_payment_count"])
    out["months_saved_vs_baseline"] = int(h["months_saved_vs_baseline"])
    out["active_early_payments"] = [
        {"payment_number": int(p["payment_number"]), "amount": _r(float(p["amount"]))}
        for p in sorted(ctx.early_payments or [], key=lambda x: x["payment_number"])
    ]
    return out


# --- 2. single payment ------------------------------------------------------

def _tool_get_payment_details(ctx: MortgageContext, payment_number) -> dict:
    n = _as_int(payment_number, "payment_number")
    records = ctx.records()
    if n < 1:
        raise ToolInputError("'payment_number' must be 1 or greater.")
    if n > len(records):
        raise ToolInputError(
            f"There is no payment #{n}: this loan is fully repaid after {len(records)} payments."
        )
    rec = records[n - 1]
    view = _record_view(rec)
    view.update({
        "loan_year": (n - 1) // 12 + 1,
        "month_of_year": (n - 1) % 12 + 1,
        "total_principal_applied": _r(rec["principal"] + rec["extra"]),
        "interest_share_of_payment_pct": _r(rec["interest"] / rec["payment"] * 100.0) if rec["payment"] else None,
        "remaining_payments": len(records) - n,
        "currency": ctx.currency,
    })
    return view


# --- 3. window --------------------------------------------------------------

def _tool_get_schedule_window(ctx: MortgageContext, start_payment, end_payment) -> dict:
    start = _as_int(start_payment, "start_payment")
    end = _as_int(end_payment, "end_payment")
    records = ctx.records()
    if start < 1:
        raise ToolInputError("'start_payment' must be 1 or greater.")
    if end < start:
        raise ToolInputError("'end_payment' must be greater than or equal to 'start_payment'.")
    if start > len(records):
        raise ToolInputError(
            f"'start_payment' {start} is past the end of the loan ({len(records)} payments)."
        )
    end = min(end, len(records))
    window = records[start - 1:end]

    if len(window) > _MAX_WINDOW_ROWS:
        return {
            "aggregated": True,
            "reason": (
                f"The requested window is {len(window)} payments, more than the {_MAX_WINDOW_ROWS}-row "
                "limit, so a per-year aggregate is returned instead of individual rows."
            ),
            "start_payment": start,
            "end_payment": end,
            "start_date": window[0]["date"],
            "end_date": window[-1]["date"],
            "totals": {
                "payments": len(window),
                "principal": _r(sum(r["principal"] for r in window)),
                "interest": _r(sum(r["interest"] for r in window)),
                "extra": _r(sum(r["extra"] for r in window)),
                "total_paid": _r(sum(r["payment"] for r in window)),
                "balance_at_start": _r(window[0]["balance"] + window[0]["principal"] + window[0]["extra"]),
                "balance_at_end": _r(window[-1]["balance"]),
            },
            "by_year": [
                {k: (int(v) if k in ("year", "payments") else _r(v)) for k, v in y.items()}
                for y in yearly_rollup(window)
            ],
            "currency": ctx.currency,
        }

    return {
        "aggregated": False,
        "start_payment": start,
        "end_payment": end,
        "rows": [_record_view(r) for r in window],
        "totals": {
            "principal": _r(sum(r["principal"] for r in window)),
            "interest": _r(sum(r["interest"] for r in window)),
            "extra": _r(sum(r["extra"] for r in window)),
            "total_paid": _r(sum(r["payment"] for r in window)),
        },
        "currency": ctx.currency,
    }


# --- 4. yearly --------------------------------------------------------------

def _tool_get_yearly_summary(ctx: MortgageContext, start_year=1, end_year=None) -> dict:
    rollup = yearly_rollup(ctx.records())
    if not rollup:
        raise ToolInputError("This loan has no payments to summarise.")
    last_year = rollup[-1]["year"]

    start = 1 if start_year is None else _as_int(start_year, "start_year")
    end = last_year if end_year is None else _as_int(end_year, "end_year")
    if start < 1:
        raise ToolInputError("'start_year' must be 1 or greater.")
    if start > last_year:
        raise ToolInputError(f"'start_year' {start} is past the end of the loan (year {last_year}).")
    if end < start:
        raise ToolInputError("'end_year' must be greater than or equal to 'start_year'.")
    end = min(end, last_year)

    rows = [
        {
            "year": int(y["year"]),
            "payments": int(y["payments"]),
            "principal": _r(y["principal"]),
            "interest": _r(y["interest"]),
            "extra": _r(y["extra"]),
            "total_paid": _r(y["total_paid"]),
            "end_balance": _r(y["end_balance"]),
        }
        for y in rollup if start <= y["year"] <= end
    ]
    return {
        "start_year": start,
        "end_year": end,
        "loan_years": last_year,
        "years": rows,
        "totals": {
            "principal": _r(sum(r["principal"] for r in rows)),
            "interest": _r(sum(r["interest"] for r in rows)),
            "extra": _r(sum(r["extra"] for r in rows)),
            "total_paid": _r(sum(r["total_paid"] for r in rows)),
        },
        "currency": ctx.currency,
    }


# --- 5/6. simulations -------------------------------------------------------

def _compare_scenarios(ctx: MortgageContext, new_records: list[dict], label: str) -> dict:
    current = ctx.records()
    cur_interest = current[-1]["cum_interest"] if current else 0.0
    new_interest = new_records[-1]["cum_interest"] if new_records else 0.0
    return {
        "scenario": label,
        "currency": ctx.currency,
        "current_payoff_months": len(current),
        "new_payoff_months": len(new_records),
        "months_saved": len(current) - len(new_records),
        "years_saved": _r((len(current) - len(new_records)) / 12.0),
        "current_payoff_date": current[-1]["date"] if current else None,
        "new_payoff_date": new_records[-1]["date"] if new_records else None,
        "current_total_interest": _r(cur_interest),
        "new_total_interest": _r(new_interest),
        "interest_saved": _r(cur_interest - new_interest),
        "current_total_paid": _r((current[-1]["cum_principal"] + cur_interest) if current else 0.0),
        "new_total_paid": _r((new_records[-1]["cum_principal"] + new_interest) if new_records else 0.0),
        "extra_paid_in_scenario": _r(sum(r["extra"] for r in new_records)),
    }


def _tool_simulate_extra_monthly_payment(ctx: MortgageContext, extra_monthly) -> dict:
    extra = _as_float(extra_monthly, "extra_monthly")
    if extra <= 0:
        raise ToolInputError("'extra_monthly' must be greater than 0.")

    months = int(ctx.loan_term) * 12
    extras = list(ctx.early_payments or []) + [
        {"payment_number": n, "amount": extra} for n in range(1, months + 1)
    ]
    new_records = ctx.records(early_payments=extras)

    out = _compare_scenarios(ctx, new_records, f"extra {ctx.currency}{extra:,.2f} every month")
    out["extra_monthly"] = _r(extra)
    out["current_monthly_payment"] = _r(ctx.monthly_payment())
    out["new_monthly_outlay"] = _r(ctx.monthly_payment() + extra)
    out["note"] = (
        "The extra amount is added to every scheduled payment on top of any early payments "
        "already configured in the calculator."
    )
    return out


def _tool_simulate_lump_sum(ctx: MortgageContext, payment_number, amount) -> dict:
    n = _as_int(payment_number, "payment_number")
    value = _as_float(amount, "amount")
    if n < 1:
        raise ToolInputError("'payment_number' must be 1 or greater.")
    if n > int(ctx.loan_term) * 12:
        raise ToolInputError(
            f"'payment_number' must be within the {int(ctx.loan_term) * 12}-payment term."
        )
    if value <= 0:
        raise ToolInputError("'amount' must be greater than 0.")

    current = ctx.records()
    if n > len(current):
        raise ToolInputError(
            f"The loan is already repaid after {len(current)} payments, so a lump sum at "
            f"payment #{n} would have no effect."
        )

    extras = list(ctx.early_payments or []) + [{"payment_number": n, "amount": value}]
    new_records = ctx.records(early_payments=extras)

    out = _compare_scenarios(
        ctx, new_records, f"one-off {ctx.currency}{value:,.2f} at payment #{n}"
    )
    out["lump_sum_payment_number"] = n
    out["lump_sum_amount"] = _r(value)
    out["lump_sum_date"] = current[n - 1]["date"]
    out["balance_before_lump_sum"] = _r(current[n - 1]["balance"] + current[n - 1]["principal"] + current[n - 1]["extra"])
    out["interest_saved_per_unit_paid"] = _r((out["interest_saved"] or 0.0) / value, 4)
    return out


# --- 7. refinance -----------------------------------------------------------

def _tool_compare_refinance(ctx: MortgageContext, new_rate, new_years, closing_costs=0) -> dict:
    rate = _as_float(new_rate, "new_rate")
    years = _as_int(new_years, "new_years")
    costs = _as_float(closing_costs if closing_costs is not None else 0, "closing_costs")
    if rate < 0:
        raise ToolInputError("'new_rate' cannot be negative.")
    if years < 1:
        raise ToolInputError("'new_years' must be at least 1.")
    if costs < 0:
        raise ToolInputError("'closing_costs' cannot be negative.")

    comparison = calculate_refinance_comparison(
        ctx.loan_amount, ctx.interest_rate, int(ctx.loan_term), rate, years
    )
    new_records = amortization_records(ctx.loan_amount, rate, years)
    current_records = ctx.records(early_payments=[])  # like-for-like: no extra payments

    monthly_savings = comparison["monthly_savings"]
    break_even_month = (
        int(math.ceil(costs / monthly_savings)) if monthly_savings > 0 and costs > 0 else (0 if monthly_savings > 0 else None)
    )
    total_savings = comparison["total_savings"]

    if monthly_savings > 0 and total_savings > costs:
        recommendation = "Refinancing looks worthwhile on these numbers."
    elif monthly_savings > 0:
        recommendation = "Refinancing lowers the monthly payment but may not pay for its closing costs."
    else:
        recommendation = "Refinancing does not lower the monthly payment on these numbers."

    return {
        "currency": ctx.currency,
        "refinanced_amount": _r(ctx.loan_amount),
        "current_rate_pct": _r(ctx.interest_rate),
        "current_term_years": int(ctx.loan_term),
        "new_rate_pct": _r(rate),
        "new_term_years": years,
        "closing_costs": _r(costs),
        "current_monthly_payment": _r(comparison["original_payment"]),
        "new_monthly_payment": _r(comparison["new_payment"]),
        "monthly_payment_delta": _r(-monthly_savings),
        "monthly_savings": _r(monthly_savings),
        "current_total_paid": _r(comparison["original_payment"] * int(ctx.loan_term) * 12),
        "new_total_paid": _r(comparison["new_payment"] * years * 12),
        "total_savings": _r(total_savings),
        "net_savings_after_costs": _r(total_savings - costs),
        "current_total_interest": _r(current_records[-1]["cum_interest"] if current_records else 0.0),
        "new_total_interest": _r(new_records[-1]["cum_interest"] if new_records else 0.0),
        "interest_saved": _r(
            (current_records[-1]["cum_interest"] if current_records else 0.0)
            - (new_records[-1]["cum_interest"] if new_records else 0.0)
        ),
        "break_even_month": break_even_month,
        "recommendation": recommendation,
        "note": (
            "Compares the full current loan amount refinanced at the new rate/term, ignoring any "
            "early payments. Real offers also depend on fees, jurisdiction and personal circumstances."
        ),
    }


# --- 8. what-if -------------------------------------------------------------

def _tool_what_if(ctx: MortgageContext, property_value=None, down_payment=None,
                  interest_rate=None, loan_term=None) -> dict:
    overrides = {}
    pv = ctx.property_value if property_value is None else _as_float(property_value, "property_value")
    dp = ctx.down_payment if down_payment is None else _as_float(down_payment, "down_payment")
    rate = ctx.interest_rate if interest_rate is None else _as_float(interest_rate, "interest_rate")
    term = int(ctx.loan_term) if loan_term is None else _as_int(loan_term, "loan_term")

    if property_value is not None:
        overrides["property_value"] = _r(pv)
    if down_payment is not None:
        overrides["down_payment"] = _r(dp)
    if interest_rate is not None:
        overrides["interest_rate"] = _r(rate)
    if loan_term is not None:
        overrides["loan_term"] = term
    if not overrides:
        raise ToolInputError(
            "Provide at least one of property_value, down_payment, interest_rate or loan_term."
        )

    if pv < 0 or dp < 0:
        raise ToolInputError("'property_value' and 'down_payment' cannot be negative.")
    if rate < 0:
        raise ToolInputError("'interest_rate' cannot be negative.")
    if term < 1:
        raise ToolInputError("'loan_term' must be at least 1 year.")

    # Loan amount follows the calculator: property value minus down payment.
    if property_value is None and down_payment is None:
        loan_amount = ctx.loan_amount
    else:
        loan_amount = pv - dp
    if loan_amount <= 0:
        raise ToolInputError(
            "That combination gives a loan amount of zero or less — nothing to borrow."
        )

    scenario = MortgageContext(
        property_value=pv,
        down_payment=dp,
        loan_amount=loan_amount,
        interest_rate=rate,
        loan_term=term,
        early_payments=list(ctx.early_payments or []),
        currency=ctx.currency,
    )
    current = ctx.headline()
    new = scenario.headline()

    def num(value):
        return _r(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else value

    return {
        "overrides_applied": overrides,
        "currency": ctx.currency,
        "current": {k: num(v) for k, v in current.items()},
        "scenario": {k: num(v) for k, v in new.items()},
        "deltas": {
            "monthly_payment": _r(new["monthly_payment"] - current["monthly_payment"]),
            "total_interest": _r(new["total_interest"] - current["total_interest"]),
            "total_paid": _r(new["total_paid"] - current["total_paid"]),
            "payoff_months": int(new["payoff_months"] - current["payoff_months"]),
            "loan_amount": _r(new["loan_amount"] - current["loan_amount"]),
        },
        "note": "Existing early payments are carried into the scenario.",
    }


# --- 9. balance threshold ---------------------------------------------------

def _tool_find_month_when_balance_below(ctx: MortgageContext, target_balance) -> dict:
    target = _as_float(target_balance, "target_balance")
    if target < 0:
        raise ToolInputError("'target_balance' cannot be negative.")

    records = ctx.records()
    if not records:
        raise ToolInputError("This loan has no payments.")

    if ctx.loan_amount <= target:
        return {
            "found": True,
            "payment_number": 0,
            "date": records[0]["date"],
            "balance": _r(ctx.loan_amount),
            "target_balance": _r(target),
            "currency": ctx.currency,
            "note": "The starting balance is already at or below the target.",
        }

    for rec in records:
        if rec["balance"] <= target:
            return {
                "found": True,
                "payment_number": rec["payment_number"],
                "date": rec["date"],
                "balance": _r(rec["balance"]),
                "target_balance": _r(target),
                "loan_year": (rec["payment_number"] - 1) // 12 + 1,
                "months_from_start": rec["payment_number"],
                "years_from_start": _r(rec["payment_number"] / 12.0),
                "remaining_payments": len(records) - rec["payment_number"],
                "cum_interest_paid_by_then": _r(rec["cum_interest"]),
                "currency": ctx.currency,
            }

    return {
        "found": False,
        "target_balance": _r(target),
        "final_balance": _r(records[-1]["balance"]),
        "payments": len(records),
        "currency": ctx.currency,
        "note": "The balance never falls to that level within the loan term.",
    }


# --- 10. crossover ----------------------------------------------------------

def _tool_get_interest_principal_crossover(ctx: MortgageContext) -> dict:
    records = ctx.records()
    if not records:
        raise ToolInputError("This loan has no payments.")

    for rec in records:
        # Scheduled principal only — a one-off lump sum should not be mistaken for the
        # structural crossover point of the amortization curve.
        if rec["principal"] > rec["interest"]:
            return {
                "found": True,
                "payment_number": rec["payment_number"],
                "date": rec["date"],
                "principal": _r(rec["principal"]),
                "interest": _r(rec["interest"]),
                "balance": _r(rec["balance"]),
                "loan_year": (rec["payment_number"] - 1) // 12 + 1,
                "months_from_start": rec["payment_number"],
                "years_from_start": _r(rec["payment_number"] / 12.0),
                "at_first_payment": rec["payment_number"] == 1,
                "interest_paid_up_to_crossover": _r(rec["cum_interest"]),
                "share_of_term_pct": _r(rec["payment_number"] / len(records) * 100.0),
                "currency": ctx.currency,
                "note": "First payment where the scheduled principal portion exceeds the interest portion.",
            }

    return {
        "found": False,
        "payments": len(records),
        "currency": ctx.currency,
        "note": "The principal portion never exceeds the interest portion within this term.",
    }


# --- registry ---------------------------------------------------------------

TOOL_REGISTRY: dict[str, ToolSpec] = {
    "get_mortgage_summary": ToolSpec(
        name="get_mortgage_summary",
        description=(
            "Return every headline figure for the mortgage currently loaded in the calculator: "
            "loan amount, monthly payment, total paid, total interest, interest ratio, LTV, payoff "
            "date and the effect of any early payments already configured. Call this first when the "
            "user asks a general question about their mortgage."
        ),
        parameters={"type": "object", "properties": {}, "required": []},
        handler=_tool_get_mortgage_summary,
    ),
    "get_payment_details": ToolSpec(
        name="get_payment_details",
        description=(
            "Return the exact figures for one single payment of the amortization schedule: "
            "date, amount, principal portion, interest portion, any extra payment and the balance "
            "remaining afterwards. Use it for questions like 'what does payment 37 look like?' or "
            "'how much interest do I pay in month 12?'."
        ),
        parameters={
            "type": "object",
            "properties": {
                "payment_number": {
                    "type": "integer",
                    "description": "1-based payment number, where 1 is the first monthly payment.",
                    "minimum": 1,
                },
            },
            "required": ["payment_number"],
        },
        handler=_tool_get_payment_details,
    ),
    "get_schedule_window": ToolSpec(
        name="get_schedule_window",
        description=(
            "Return a contiguous slice of the amortization schedule, one row per payment. "
            "Windows of more than 60 payments are returned as a per-year aggregate instead of "
            "individual rows, so prefer get_yearly_summary for long spans."
        ),
        parameters={
            "type": "object",
            "properties": {
                "start_payment": {
                    "type": "integer",
                    "description": "First payment number of the window (1-based, inclusive).",
                    "minimum": 1,
                },
                "end_payment": {
                    "type": "integer",
                    "description": "Last payment number of the window (inclusive). Clipped to the payoff month.",
                    "minimum": 1,
                },
            },
            "required": ["start_payment", "end_payment"],
        },
        handler=_tool_get_schedule_window,
    ),
    "get_yearly_summary": ToolSpec(
        name="get_yearly_summary",
        description=(
            "Return a per-year rollup of the schedule: payments made, principal, interest, extra "
            "payments, total paid and the balance at the end of each loan year. Best tool for "
            "'how much interest do I pay in the first 5 years?' style questions."
        ),
        parameters={
            "type": "object",
            "properties": {
                "start_year": {
                    "type": "integer",
                    "description": "First loan year to include, 1-based (year 1 = payments 1-12). Defaults to 1.",
                    "minimum": 1,
                },
                "end_year": {
                    "type": "integer",
                    "description": "Last loan year to include, inclusive. Defaults to the final year of the loan.",
                    "minimum": 1,
                },
            },
            "required": [],
        },
        handler=_tool_get_yearly_summary,
    ),
    "simulate_extra_monthly_payment": ToolSpec(
        name="simulate_extra_monthly_payment",
        description=(
            "Simulate paying a fixed extra amount on top of EVERY monthly payment. Returns the new "
            "payoff month and date, months and years saved, and the interest saved versus the "
            "current setup. Use for 'what if I pay X more each month?'."
        ),
        parameters={
            "type": "object",
            "properties": {
                "extra_monthly": {
                    "type": "number",
                    "description": "Extra amount added to every monthly payment, in the loan currency. Must be greater than 0.",
                    "exclusiveMinimum": 0,
                },
            },
            "required": ["extra_monthly"],
        },
        handler=_tool_simulate_extra_monthly_payment,
    ),
    "simulate_lump_sum": ToolSpec(
        name="simulate_lump_sum",
        description=(
            "Simulate a single one-off overpayment at a specific payment number, on top of anything "
            "already configured. Returns the new payoff month and date, months saved and interest "
            "saved. Use for 'what if I put 10,000 in after 3 years?'."
        ),
        parameters={
            "type": "object",
            "properties": {
                "payment_number": {
                    "type": "integer",
                    "description": "Payment number the lump sum is applied at (1-based, e.g. 36 for after 3 years).",
                    "minimum": 1,
                },
                "amount": {
                    "type": "number",
                    "description": "Size of the one-off overpayment in the loan currency. Must be greater than 0.",
                    "exclusiveMinimum": 0,
                },
            },
            "required": ["payment_number", "amount"],
        },
        handler=_tool_simulate_lump_sum,
    ),
    "compare_refinance": ToolSpec(
        name="compare_refinance",
        description=(
            "Compare the current loan against refinancing the same amount at a new rate and term. "
            "Returns the payment delta, total cost delta, interest saved and the break-even month "
            "for the closing costs."
        ),
        parameters={
            "type": "object",
            "properties": {
                "new_rate": {
                    "type": "number",
                    "description": "New annual interest rate as a percentage, e.g. 4.5 for 4.5%.",
                    "minimum": 0,
                },
                "new_years": {
                    "type": "integer",
                    "description": "Term of the new loan in years, e.g. 30.",
                    "minimum": 1,
                },
                "closing_costs": {
                    "type": "number",
                    "description": "One-off cost of refinancing in the loan currency. Defaults to 0 when unknown.",
                    "minimum": 0,
                },
            },
            "required": ["new_rate", "new_years"],
        },
        handler=_tool_compare_refinance,
    ),
    "what_if": ToolSpec(
        name="what_if",
        description=(
            "Recompute all headline figures with one or more inputs changed. Any argument left out "
            "keeps the value currently in the calculator. Returns the current figures, the scenario "
            "figures and the deltas between them."
        ),
        parameters={
            "type": "object",
            "properties": {
                "property_value": {
                    "type": "number",
                    "description": "Alternative property value. Loan amount becomes property_value minus down_payment.",
                    "minimum": 0,
                },
                "down_payment": {
                    "type": "number",
                    "description": "Alternative down payment. Loan amount becomes property_value minus down_payment.",
                    "minimum": 0,
                },
                "interest_rate": {
                    "type": "number",
                    "description": "Alternative annual interest rate as a percentage, e.g. 4.25.",
                    "minimum": 0,
                },
                "loan_term": {
                    "type": "integer",
                    "description": "Alternative loan term in whole years, e.g. 15.",
                    "minimum": 1,
                },
            },
            "required": [],
        },
        handler=_tool_what_if,
    ),
    "find_month_when_balance_below": ToolSpec(
        name="find_month_when_balance_below",
        description=(
            "Find the first payment at which the outstanding balance drops to or below a target "
            "amount. Returns the payment number, its date and the exact balance. Use for questions "
            "like 'when will I owe less than 100,000?' or 'when am I half way?'."
        ),
        parameters={
            "type": "object",
            "properties": {
                "target_balance": {
                    "type": "number",
                    "description": "Balance threshold in the loan currency, e.g. 100000. Must be 0 or more.",
                    "minimum": 0,
                },
            },
            "required": ["target_balance"],
        },
        handler=_tool_find_month_when_balance_below,
    ),
    "get_interest_principal_crossover": ToolSpec(
        name="get_interest_principal_crossover",
        description=(
            "Find the first payment where the principal portion becomes larger than the interest "
            "portion — the tipping point of the amortization curve. Returns the payment number, "
            "date, both amounts and how far into the term it happens."
        ),
        parameters={"type": "object", "properties": {}, "required": []},
        handler=_tool_get_interest_principal_crossover,
    ),
}


def tool_schemas() -> list[dict]:
    """The ``tools`` array sent to DeepSeek."""
    return [spec.schema for spec in TOOL_REGISTRY.values()]


def execute_tool(ctx: MortgageContext, name: str, args: Optional[dict]) -> dict:
    """Run one tool against ``ctx``. Never raises: bad input becomes ``{'error': ...}``."""
    spec = TOOL_REGISTRY.get(name)
    if spec is None:
        return {"error": f"Unknown tool '{name}'. Available tools: {', '.join(sorted(TOOL_REGISTRY))}."}
    if args is None:
        args = {}
    if not isinstance(args, dict):
        return {"error": f"Arguments for '{name}' must be a JSON object, got {type(args).__name__}."}

    allowed = set(spec.parameters.get("properties", {}))
    unexpected = [k for k in args if k not in allowed]
    if unexpected:
        return {
            "error": (
                f"Unexpected argument(s) for '{name}': {', '.join(sorted(unexpected))}. "
                f"Accepted arguments: {', '.join(sorted(allowed)) or 'none'}."
            )
        }

    try:
        return spec.handler(ctx, **args)
    except ToolInputError as exc:
        return {"error": str(exc)}
    except TypeError as exc:
        detail = str(exc).replace(spec.handler.__name__, name)  # hide the private handler name
        return {
            "error": (
                f"Bad arguments for '{name}': {detail}. Required: "
                f"{', '.join(spec.parameters.get('required', [])) or 'none'}."
            )
        }
    except Exception as exc:  # never let a tool crash the chat
        return {"error": f"'{name}' failed: {type(exc).__name__}: {exc}"}


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

SYSTEM_PROMPT_TEMPLATE = """You are the mortgage analyst built into this mortgage calculator app. \
You are attached to ONE specific, live calculation — the one the user currently has on screen, \
described in the state block below. Everything you say is about THIS mortgage unless the user \
explicitly asks about a hypothetical.

{state_block}

HOW TO WORK
- The state block above is live. Answer trivial questions about it directly, without a tool call.
- For ANY figure that is not literally in the state block, you MUST call a tool. Never estimate, \
never do the arithmetic yourself, never reuse a number from earlier in the conversation if the \
parameters may have changed. The tools are the only source of truth.
- Tools are deterministic Python running on the user's own schedule, so their output is exact. \
If a tool returns {{"error": ...}}, read the message, fix your arguments and call it again.
- Chain tools when a question needs several figures. Prefer get_yearly_summary over long \
schedule windows.

HOW TO ANSWER
- Reply in the same language the user wrote in.
- Be concise. Lead with the key number, then a short explanation of where it comes from and what \
it means. Use a small markdown table only when comparing several scenarios.
- All amounts are in {currency}. Use that symbol exactly and never convert to another currency.
- Round money to whole units or 2 decimals as appropriate, and say which payment number or date a \
figure refers to.
- You are a calculator, not a licensed financial adviser. When an answer depends on jurisdiction, \
tax treatment, lender fees, prepayment penalties, insurance or someone's personal circumstances, \
say so in one short sentence and recommend checking with the lender or a qualified adviser. Do not \
pad every answer with disclaimers — one line, only when it is genuinely relevant.
"""


class MortgageAgent:
    """Drives the tool-calling loop between DeepSeek and the deterministic mortgage tools."""

    def __init__(
        self,
        ctx: MortgageContext,
        client: DeepSeekClient,
        max_tool_rounds: int = 6,
        temperature: float = 0.2,
    ):
        self.ctx = ctx
        self.client = client
        self.max_tool_rounds = max(1, int(max_tool_rounds))
        # 0.2 keeps the routing deterministic; the API rejects anything outside 0..2.
        self.temperature = min(2.0, max(0.0, float(temperature)))

    # -- prompt ------------------------------------------------------------

    def system_prompt(self) -> str:
        return SYSTEM_PROMPT_TEMPLATE.format(
            state_block=self.ctx.to_prompt_block(),
            currency=self.ctx.currency,
        )

    @staticmethod
    def _clean_history(history) -> list[dict]:
        cleaned = []
        for message in history or []:
            if not isinstance(message, dict):
                continue
            role = message.get("role")
            content = message.get("content")
            if role not in ("user", "assistant"):
                continue
            if not isinstance(content, str) or not content.strip():
                continue
            cleaned.append({"role": role, "content": content})
        return cleaned

    # -- main loop ---------------------------------------------------------

    def run(self, history: list[dict], temperature: Optional[float] = None) -> Iterator[dict]:
        """Run one exchange, yielding UI events (see CONTRACT.md).

        ``temperature`` overrides ``self.temperature`` for this exchange only; leave it
        as ``None`` to use the value the agent was constructed with.

        ``tool_start`` / ``tool_end`` events additionally carry ``id`` (the DeepSeek
        ``tool_call_id``) and ``index`` (a 0-based counter over every tool call made in
        this exchange, unique even when the same tool is called twice). The UI pairs the
        two events on ``index`` instead of guessing from arrival order.
        """
        messages: list[dict] = [{"role": "system", "content": self.system_prompt()}]
        messages.extend(self._clean_history(history))

        temp = self.temperature if temperature is None else min(2.0, max(0.0, float(temperature)))

        executed: list[dict] = []
        usage: dict = {}
        rounds = 0
        call_index = 0

        while True:
            final_round = rounds >= self.max_tool_rounds
            tools = None if final_round else tool_schemas()

            content_parts: list[str] = []
            tool_calls: list[dict] = []

            try:
                for event in self.client.stream_chat(messages, tools=tools, temperature=temp):
                    kind = event.get("type")
                    if kind == "content":
                        content_parts.append(event["text"])
                        yield {"type": "text", "text": event["text"]}
                    elif kind == "reasoning":
                        yield {"type": "reasoning", "text": event["text"]}
                    elif kind == "tool_calls":
                        tool_calls = event["tool_calls"]
                    elif kind == "usage":
                        usage = event["usage"]
            except DeepSeekError as exc:
                yield {"type": "error", "message": str(exc), "kind": exc.kind}
                return
            except Exception as exc:  # defensive: never leak a traceback into the UI
                yield {
                    "type": "error",
                    "message": f"Unexpected problem talking to DeepSeek: {exc}",
                    "kind": "server",
                }
                return

            content = "".join(content_parts)

            # No tool calls (or we asked for a final answer without tools): we're done.
            if not tool_calls or final_round:
                yield {
                    "type": "done",
                    "content": content,
                    "tool_calls": executed,
                    "usage": usage,
                }
                return

            messages.append({
                "role": "assistant",
                "content": content,
                "tool_calls": [
                    {"id": tc["id"], "type": "function", "function": tc["function"]}
                    for tc in tool_calls
                ],
            })

            for call in tool_calls:
                name = (call.get("function") or {}).get("name") or ""
                raw_args = (call.get("function") or {}).get("arguments") or ""
                call_id = call.get("id") or f"call_{call_index}"
                index = call_index
                call_index += 1

                args: dict = {}
                parse_error: Optional[str] = None
                if raw_args.strip():
                    try:
                        parsed = json.loads(raw_args)
                    except (json.JSONDecodeError, ValueError):
                        parse_error = (
                            f"Could not parse the arguments for '{name}' as JSON. "
                            f"Received: {raw_args[:200]}. Send a valid JSON object."
                        )
                    else:
                        if isinstance(parsed, dict):
                            args = parsed
                        else:
                            parse_error = (
                                f"Arguments for '{name}' must be a JSON object, "
                                f"got {type(parsed).__name__}."
                            )

                yield {
                    "type": "tool_start",
                    "name": name,
                    "args": args,
                    "id": call_id,
                    "index": index,
                }

                started = time.time()
                if parse_error:
                    result: dict = {"error": parse_error}
                else:
                    result = execute_tool(self.ctx, name, args)
                elapsed_ms = int((time.time() - started) * 1000)

                yield {
                    "type": "tool_end",
                    "name": name,
                    "result": result,
                    "ms": elapsed_ms,
                    "id": call_id,
                    "index": index,
                }
                executed.append({
                    "name": name,
                    "args": args,
                    "result": result,
                    "ms": elapsed_ms,
                    "id": call_id,
                    "index": index,
                })

                try:
                    payload = json.dumps(result, default=str)
                except (TypeError, ValueError):
                    payload = json.dumps({"error": "Tool result could not be serialised."})

                messages.append({
                    "role": "tool",
                    "tool_call_id": call.get("id") or "",
                    "content": payload,
                })

            rounds += 1
