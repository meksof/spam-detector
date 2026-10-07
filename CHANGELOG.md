# Changelog
  
  All notable changes to this project are documented here.
  
  ---
  
  ## [4.0.0] — 2026-10-07
  
  ### Added
  - `encoder_latency_ms` column in `metrics.csv` to track encoder inference time per borderline email
  - Latency printed to stdout alongside verdict and confidence in the encoder output line
  
  ---
  
  ## [3.0.0] — 2026-09-24
  
  ### Changed
  - Replaced the generic local model classifier with a fine-tuned MiniLM encoder exported to ONNX INT8
  - Metrics schema V3: renamed `verdict` → `encoder_verdict`, `used_model` → `encoder_used_model`; added `typesafe_signals_fired` and `encoder_confidence` columns
  - Encoder is loaded once as a singleton; zero reload overhead on subsequent calls
  - Encoder model path resolved via `ENCODER_MODEL_DIR` env var with a sensible default fallback
  
  ### Added
  - `encoder_classifier.py` — dedicated module for ONNX encoder inference
  - Encoder fine-tuning workspace extracted to sibling project `spam-encoder`
  
  ---
  
  ## [2.0.0] — 2026-09-23
  
  ### Added
  - Borderline case detection: when `spam_signals_fired == SPAM_SIGNAL_THRESHOLD`, a local model is consulted for a second opinion
  - `metrics.py` — CSV logger for borderline case outcomes (`metrics.csv`)
  - `metrics.csv` schema V1/V2: `timestamp`, `filename`, `verdict`, `used_model`
  
  ---
  
  ## [1.0.0] — 2026-09-21
  
  ### Added
  - `parse_emails.py` — converts `.eml` files to structured JSON (subject, sender, body, links)
  - `classify_emails.py` — classifies emails via TypeSafe System One API using `questions.json`
  - Six Noul signal questions: `requests_credentials`, `offers_unexpected_reward`, `creates_time_pressure`, `sender_identity_mismatch`, `link_domain_mismatch`,
  `disguises_link_destination`
  - Configurable `SPAM_SIGNAL_THRESHOLD` (default: 2)
