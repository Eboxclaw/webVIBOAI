# webVIBOAI

Landing page for SoberanoAI with a Cloudflare Worker-powered waitlist endpoint.

## Architecture

- `public/index.html`: static landing page frontend.
- `worker.js`: serves static assets and handles `POST /api/waitlist`.
- `wrangler.toml`: Cloudflare Worker + assets configuration.

## Cloudflare Worker setup

1. Install Wrangler:
   ```bash
   npm install -g wrangler
   ```
2. Authenticate Wrangler:
   ```bash
   wrangler auth login
   ```
3. (Recommended) Create a KV namespace for durable waitlist storage:
   ```bash
   wrangler kv namespace create WAITLIST_KV
   wrangler kv namespace create WAITLIST_KV --preview
   ```
4. If using KV, uncomment `[[kv_namespaces]]` in `wrangler.toml` and paste your IDs.
5. (Recommended) Set allowed frontend origins in `wrangler.toml`:
   ```toml
   ALLOWED_ORIGINS = "https://your-domain.com,https://www.your-domain.com"
   ```
6. Deploy the worker + static frontend:
   ```bash
   wrangler deploy
   ```
7. Open your worker URL (for example, `https://webvibo.workers.dev`) to view the landing page.
8. This frontend defaults to `https://webvibo.workers.dev/api/waitlist` unless overridden by `window.WAITLIST_ENDPOINT` or `?waitlistEndpoint=`.



### Deploy with API token (non-interactive)

If you already have a Cloudflare API token, deploy with:
```bash
export CLOUDFLARE_API_TOKEN="<your_api_token>"
wrangler deploy
```

This repository is configured for account `836ade685abebb6150aacf0420286683` and worker name `webvibo`, which maps to `https://webvibo.workers.dev`.

## Deploy troubleshooting

If deploy fails with `Invalid TOML document: trying to redefine an already defined table or value` on `assets`:
- Keep **only one** assets declaration in `wrangler.toml`.
- This repo uses the inline form: `assets = { directory = "./public" }`.
- Do **not** add a second `[assets]` table.

## Local preview

Run worker with static assets locally:
```bash
wrangler dev
```

## Waitlist API

- `POST /api/waitlist`
- Payload:
  ```json
  { "email": "user@example.com", "source": "webvibo-landing" }
  ```
- Responses:
  - `200` with `ok: true` when joined
  - `200` with `alreadyJoined: true` for duplicate submissions
  - `400` for invalid payload/email
  - `500` for storage/runtime failures
