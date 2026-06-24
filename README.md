---
title: DHVANI Inference
emoji: 🛡️
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
license: mit
---

# DHVANI Inference Backend

Free CPU inference for the DHVANI voice authenticity checker.
Used by the Vercel frontend via `DHVANI_BACKEND_URL`.

Endpoints:
- `GET /health`
- `POST /analyze` (multipart field: `audio`)