Claude-Sonnet-5 PDF-QA migration package

Target:
  models/Claude-Sonnet-5/

Files:
  run_pdfqa.py
  parse_pdfqa_results.py
  run.sh

Runtime:
  python
  anthropic 0.121.0
  model alias: claude-sonnet-5
  endpoint: https://api.aicodemirror.ai/api/claudecode

Examples:
  export ANTHROPIC_API_KEY='...'
  bash run.sh ordinary 1
  bash run.sh unanswerable 1
  bash run.sh reasoning 1
  bash run.sh cross_pdf 1
  bash run.sh all

Notes:
  - 144 DPI, whole PDF, no page dropping.
  - shared pdfqa-prompt-v2-strict-min.
  - no model-based format repair.
  - parse_failed is treated as attempted for resume; API/runtime errors are retryable.
  - thinking defaults to provider/model default. Set CLAUDE_DISABLE_THINKING=1 only if explicitly desired.
