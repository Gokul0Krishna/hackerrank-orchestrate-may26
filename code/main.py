import csv
import json
import logging
import pandas as pd
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv(Path(__file__).parent.parent / '.env')

from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.progress import (
    BarColumn, MofNCompleteColumn, Progress,
    SpinnerColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn,
)
from rich.table import Table
from rich.text import Text
from rich import box

sys.path.insert(0, str(Path(__file__).parent))
from agent import triage

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

DEFAULT_INPUT  = Path(__file__).parent.parent / 'support_tickets' / 'support_tickets.csv'
DEFAULT_OUTPUT = Path(__file__).parent.parent / 'support_tickets' / 'output.csv'
OUTPUT_FIELDS  = [
    'status', 'product_area', 'response', 'justification', 'request_type'
]

LOG_DIR  = Path.home() / 'hackerrank_orchestrate'
LOG_FILE = LOG_DIR / 'log.txt'
LOG_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Two loggers:
#   `log`       — structured DEBUG lines, goes only to log.txt
#   `audit_log` — human-readable narrative, goes to a separate audit.log
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s | %(levelname)-8s | %(message)s',
    handlers=[logging.FileHandler(LOG_FILE, encoding='utf-8')]
)
log = logging.getLogger('agent')

audit_handler = logging.FileHandler(LOG_DIR / 'audit.log', encoding='utf-8')
audit_handler.setLevel(logging.INFO)
audit_handler.setFormatter(logging.Formatter('%(message)s'))   # plain text only
audit_log = logging.getLogger('audit')
audit_log.addHandler(audit_handler)
audit_log.propagate = False

console = Console()


# ---------------------------------------------------------------------------
# Audit log writer — this is the "human reading later" log
# ---------------------------------------------------------------------------

def write_audit_entry(row_num: int, total: int, result: dict):
    """
    Write a fully self-contained narrative entry for one ticket.
    When a human reads audit.log later they should understand exactly
    what happened and why, without needing to look at any other file.
    """
    trace   = result.get('_trace', {})
    inp     = trace.get('input', {})
    issue   = inp.get('issue', '')[:200]
    subject = inp.get('subject', '') or '(none)'
    company = inp.get('company', '')

    urgency  = trace.get('urgency', 'unknown')
    signals  = trace.get('matched_signals', [])
    chunks   = trace.get('retrieved_chunks', [])
    ev       = trace.get('evaluator', {})
    gen      = trace.get('generator', {})
    overrides = trace.get('overrides_applied', [])
    decision  = trace.get('final_decision', result)
    eval_usage = ev.get('usage', {})
    gen_usage  = gen.get('usage', {})

    sep = "=" * 72

    lines = [
        "",
        sep,
        f"TICKET {row_num}/{total}",
        f"Timestamp : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Company   : {company}",
        f"Subject   : {subject}",
        f"Issue     : {issue}{'...' if len(inp.get('issue','')) > 200 else ''}",
        "",
        "── STEP 1: URGENCY ASSESSMENT ──────────────────────────────────────",
        f"  Result   : {urgency.upper()}",
    ]

    if signals:
        lines.append(f"  Triggers : {', '.join(signals)}")
    else:
        lines.append("  Triggers : none")

    if trace.get('pre_escalated'):
        lines += [
            "",
            "── DECISION: PRE-ESCALATED (no API calls made) ─────────────────────",
            f"  Reason   : Urgency level '{urgency}' is non-negotiable escalation.",
            f"  Signals  : {', '.join(signals)}",
            "  API cost : $0.00 (blocked before retrieval)",
        ]
    else:
        # Retrieval
        lines += [
            "",
            "── STEP 2: RETRIEVAL ────────────────────────────────────────────────",
            f"  Chunks found : {len(chunks)}",
        ]
        if chunks:
            for c in chunks:
                lines.append(f"    [{c['source'].upper()}] {c['title']} (score={c['score']})")
        else:
            lines.append("    (no chunks returned — corpus may not cover this topic)")

        # Evaluator
        lines += [
            "",
            "── STEP 3: RETRIEVAL QUALITY EVALUATION ────────────────────────────",
            f"  Verdict  : {ev.get('verdict', 'N/A')}",
        ]
        _fmt_usage(lines, eval_usage, "Evaluator")

        if ev.get('retry_attempted'):
            lines += [
                f"  Verdict was AMBIGUOUS — retried with rewritten query:",
                f"    Rewritten query : {ev.get('rewritten_query', '')}",
                f"    Verdict after  : {ev.get('verdict_after_retry', 'N/A')}",
            ]

        # Generator
        lines += [
            "",
            "── STEP 4: RESPONSE GENERATION ─────────────────────────────────────",
        ]
        if gen.get('called'):
            lines.append("  Generator was called.")
            _fmt_usage(lines, gen_usage, "Generator")
        else:
            verdict = ev.get('verdict')
            if verdict == 'INSUFFICIENT':
                lines.append(
                    "  Generator was NOT called — retrieval was INSUFFICIENT.\n"
                    "  Reason: Calling the generator with no grounding would risk hallucination.\n"
                    "  Action: Ticket escalated without any LLM-generated content."
                )
            else:
                lines.append("  Generator was NOT called.")

    if overrides:
        lines += [
            "",
            "── OVERRIDES APPLIED ────────────────────────────────────────────────",
        ]
        for o in overrides:
            lines.append(f"  - {o}")

    lines += [
        "",
        "── FINAL DECISION ───────────────────────────────────────────────────",
        f"  status       : {decision.get('status', '')}",
        f"  product_area : {decision.get('product_area', '')}",
        f"  request_type : {decision.get('request_type', '')}",
        f"  justification: {decision.get('justification', '')}",
        "",
        f"  USER RESPONSE:",
        f"  {decision.get('response', '')}",
        sep,
    ]

    for line in lines:
        audit_log.info(line)


def _fmt_usage(lines: list, usage: dict, label: str):
    if not usage:
        return
    total_in  = usage.get('input_tokens', 0)
    total_out = usage.get('output_tokens', 0)
    cache_cr  = usage.get('cache_creation_tokens', 0)
    cache_rd  = usage.get('cache_read_tokens', 0)
    lines.append(
        f"  Tokens   : in={total_in} out={total_out} "
        f"cache_write={cache_cr} cache_read={cache_rd}"
    )
    if cache_rd > 0:
        lines.append(
            f"  ✓ Prompt cache HIT on {label} system prompt "
            f"({cache_rd} tokens read from cache — ~90% discount applied)"
        )
    elif cache_cr > 0:
        lines.append(
            f"  Prompt cache WRITE on {label} system prompt "
            f"({cache_cr} tokens written — subsequent calls will hit cache)"
        )


# ---------------------------------------------------------------------------
# Rich terminal helpers
# ---------------------------------------------------------------------------

STATUS_STYLE = {'replied': 'bold green', 'escalated': 'bold red'}
URGENCY_STYLE = {
    'critical': 'bold red', 'high': 'red',
    'medium': 'yellow',     'low': 'dim',
}


def make_progress() -> Progress:
    return Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(bar_width=30),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
        transient=False,
    )


def make_live_table(rows_done: list[dict]) -> Table:
    t = Table(
        box=box.SIMPLE_HEAVY,
        show_header=True,
        header_style="bold cyan",
        min_width=90,
    )
    t.add_column("#",            width=4,  justify="right")
    t.add_column("Company",      width=12)
    t.add_column("Subject",      width=22, no_wrap=True)
    t.add_column("Urgency",      width=9)
    t.add_column("Verdict",      width=12)
    t.add_column("Status",       width=10)
    t.add_column("Product Area", width=18)

    for r in rows_done[-15:]:   # show last 15 rows so table doesn't overflow
        trace   = r.get('_trace', {})
        urgency = trace.get('urgency', '?')
        verdict = trace.get('evaluator', {}).get('verdict') or (
            'PRE-ESC' if trace.get('pre_escalated') else '—'
        )
        t.add_row(
            str(r['_row']),
            r.get('company', ''),
            (r.get('subject') or r.get('issue', ''))[:22],
            Text(urgency, style=URGENCY_STYLE.get(urgency, '')),
            verdict,
            Text(r.get('status', ''), style=STATUS_STYLE.get(r.get('status', ''), '')),
            r.get('product_area', ''),
        )
    return t


# ---------------------------------------------------------------------------
# Main run
# ---------------------------------------------------------------------------

def run(input_path: Path, output_path: Path):
    start_time = time.time()

    log.info("=" * 60)
    log.info(f"RUN STARTED  {datetime.now().isoformat()}")
    log.info(f"Input  : {input_path}")
    log.info(f"Output : {output_path}")
    log.info("=" * 60)

    audit_log.info("")
    audit_log.info("=" * 72)
    audit_log.info(f"SUPPORT TRIAGE AGENT — RUN STARTED")
    audit_log.info(f"Timestamp : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    audit_log.info(f"Input     : {input_path}")
    audit_log.info(f"Output    : {output_path}")
    audit_log.info("=" * 72)

    df = pd.read_csv(input_path).fillna('')
    rows = df.to_dict('records')

    console.rule("[bold cyan]Support Triage Agent[/bold cyan]")
    console.print(
        f"  [dim]Input :[/dim]  {input_path}\n"
        f"  [dim]Output:[/dim]  {output_path}\n"
        f"  [dim]Tickets:[/dim] {len(rows)}\n"
        f"  [dim]Log   :[/dim]  {LOG_FILE}\n"
        f"  [dim]Audit :[/dim]  {LOG_DIR / 'audit.log'}\n"
    )

    results     = []
    rows_done   = []
    error_count = 0
    token_totals = {
        'input': 0, 'output': 0,
        'cache_creation': 0, 'cache_read': 0
    }

    progress = make_progress()
    task_id  = progress.add_task("Processing tickets...", total=len(rows))

    with Live(console=console, refresh_per_second=4) as live:
        for i, row in enumerate(rows, 1):
            # Using user-provided logic for structured extraction
            issue   = str(row.get('Issue') or row.get('issue') or '').strip()
            subject = str(row.get('Subject') or row.get('subject') or '').strip()
            company = str(row.get('Company') or row.get('company') or 'None').strip() or 'None'

            progress.update(
                task_id,
                description=f"[{i}/{len(rows)}] {company} — {(subject or issue)[:40]}...",
                completed=i - 1,
            )
            live.update(
                Panel(
                    progress,
                    title="[bold]Processing[/bold]",
                    subtitle=f"Errors so far: {error_count}",
                    border_style="cyan",
                )
            )

            log.info(f"Row {i}: company={company} subject={subject!r}")

            try:
                result = triage(issue=issue, subject=subject, company=company)
            except Exception as e:
                err_msg = str(e)
                if "429" in err_msg:
                    log.error(f"Row {i}: Rate limit exceeded after attempting key rotation.")
                    justification = "Rate limit exceeded on all available API keys."
                else:
                    log.error(f"Row {i}: unhandled exception: {e}", exc_info=True)
                    justification = f"Unhandled agent error: {e}"
                
                error_count += 1
                result = {
                    'status':       'escalated',
                    'product_area': 'unknown',
                    'response':     'We could not process your request. A human agent will follow up.',
                    'justification': justification,
                    'request_type': 'product_issue',
                    '_trace': {'input': {'issue': issue, 'subject': subject, 'company': company}},
                }

            result['_row']    = i
            result['issue']   = issue
            result['subject'] = subject
            result['company'] = company

            # Accumulate token usage
            trace = result.get('_trace', {})
            for step_key in ('evaluator', 'generator'):
                u = trace.get(step_key, {}).get('usage', {})
                token_totals['input']          += u.get('input_tokens', 0)
                token_totals['output']         += u.get('output_tokens', 0)
                token_totals['cache_creation'] += u.get('cache_creation_tokens', 0)
                token_totals['cache_read']     += u.get('cache_read_tokens', 0)

            write_audit_entry(i, len(rows), result)
            rows_done.append(result)

            log.info(
                f"Row {i}: status={result['status']} "
                f"area={result['product_area']} type={result['request_type']}"
            )

            # Update live display every 5 rows to avoid flicker
            if i % 5 == 0 or i == len(rows):
                live.update(
                    Panel(
                        make_live_table(rows_done),
                        title=f"[bold]Results so far ({i}/{len(rows)})[/bold]",
                        border_style="green",
                    )
                )
                time.sleep(0.1)

            results.append(result)
            progress.update(task_id, completed=i)
            time.sleep(0.3)   # light rate-limit buffer

    # Write output CSV (exclude internal keys)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(results)[OUTPUT_FIELDS].to_csv(output_path, index=False)

    elapsed = time.time() - start_time
    _print_final_summary(results, elapsed, token_totals, error_count, output_path)
    _write_audit_summary(results, elapsed, token_totals, error_count)

    log.info(f"RUN COMPLETE in {elapsed:.1f}s — {len(results)} rows written to {output_path}")


def _print_final_summary(results, elapsed, token_totals, error_count, output_path):
    replied   = sum(1 for r in results if r.get('status') == 'replied')
    escalated = len(results) - replied

    type_counts = {}
    for r in results:
        k = r.get('request_type', 'unknown')
        type_counts[k] = type_counts.get(k, 0) + 1

    area_counts = {}
    for r in results:
        k = r.get('product_area', 'unknown')
        area_counts[k] = area_counts.get(k, 0) + 1

    t = Table(title="Final Summary", box=box.ROUNDED, header_style="bold cyan")
    t.add_column("Metric", style="bold", width=28)
    t.add_column("Value",  width=20)

    t.add_row("Total tickets",       str(len(results)))
    t.add_row("Replied",             f"[green]{replied}[/green]")
    t.add_row("Escalated",           f"[red]{escalated}[/red]")
    t.add_row("Errors",              f"[yellow]{error_count}[/yellow]")
    t.add_row("Elapsed",             f"{elapsed:.1f}s")
    t.add_row("Avg per ticket",      f"{elapsed / len(results):.2f}s")
    t.add_row("─" * 20,             "─" * 12)
    t.add_row("Input tokens",        str(token_totals['input']))
    t.add_row("Output tokens",       str(token_totals['output']))
    t.add_row("Cache writes",        str(token_totals['cache_creation']))
    t.add_row("Cache reads (saved)", f"[green]{token_totals['cache_read']}[/green]")
    t.add_row("─" * 20,             "─" * 12)

    for k, v in sorted(type_counts.items()):
        t.add_row(f"  {k}", str(v))

    console.print()
    console.print(t)
    console.print(f"\n[bold green]✓ Output:[/bold green] {output_path}")
    console.print(
        f"[bold green]✓ Audit log:[/bold green] "
        f"{Path.home() / 'hackerrank_orchestrate' / 'audit.log'}"
    )


def _write_audit_summary(results, elapsed, token_totals, error_count):
    replied   = sum(1 for r in results if r.get('status') == 'replied')
    escalated = len(results) - replied

    lines = [
        "",
        "=" * 72,
        "RUN SUMMARY",
        f"Completed  : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Duration   : {elapsed:.1f}s ({elapsed / len(results):.2f}s per ticket)",
        "",
        f"Total      : {len(results)}",
        f"Replied    : {replied}",
        f"Escalated  : {escalated}",
        f"Errors     : {error_count}",
        "",
        "Token usage:",
        f"  Input tokens        : {token_totals['input']}",
        f"  Output tokens       : {token_totals['output']}",
        f"  Cache writes        : {token_totals['cache_creation']}",
        f"  Cache reads (saved) : {token_totals['cache_read']}",
        "=" * 72,
    ]
    for line in lines:
        audit_log.info(line)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    inp = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_INPUT
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_OUTPUT
    run(inp, out)