import csv
import os
import sys
import time
from agent import triage

DEFAULT_INPUT = os.path.join(
    os.path.dirname(__file__), '..', 'support_tickets', 'support_tickets.csv'
)
DEFAULT_OUTPUT = os.path.join(
    os.path.dirname(__file__), '..', 'support_tickets', 'output.csv'
)

OUTPUT_FIELDS = ['issue', 'subject', 'company', 'status', 'product_area', 'response', 'justification', 'request_type']


def run(input_path: str, output_path: str):
    with open(input_path, newline='', encoding='utf-8') as f:
        rows = list(csv.DictReader(f))

    print(f"Processing {len(rows)} tickets from: {input_path}")
    print(f"Output will be written to: {output_path}\n")

    results = []

    for i, row in enumerate(rows, 1):
        issue   = row.get('issue', '').strip()
        subject = row.get('subject', '').strip()
        company = row.get('company', 'None').strip() or 'None'

        preview = (subject or issue)[:60]
        print(f"[{i:>3}/{len(rows)}] {company} | {preview}...")

        try:
            result = triage(issue=issue, subject=subject, company=company)
        except Exception as e:
            print(f"         ERROR: {e}")
            result = {
                'status': 'escalated',
                'product_area': 'unknown',
                'response': 'We could not process your request automatically. A human agent will follow up shortly.',
                'justification': f'Agent error during processing: {str(e)}',
                'request_type': 'product_issue'
            }

        print(f"         -> {result['status']} | {result['request_type']} | {result['product_area']}")

        results.append({
            'issue':        issue,
            'subject':      subject,
            'company':      company,
            'status':       result['status'],
            'product_area': result['product_area'],
            'response':     result['response'],
            'justification':result['justification'],
            'request_type': result['request_type']
        })

        time.sleep(0.4)  # light rate-limit buffer

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()
        writer.writerows(results)

    print(f"\nDone. {len(results)} rows written to {output_path}")


if __name__ == '__main__':
    input_path  = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_INPUT
    output_path = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_OUTPUT
    run(input_path, output_path)