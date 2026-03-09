export default {
  async fetch(request, env) {
    const url = new URL(request.url);

    if (url.pathname === '/api/waitlist') {
      const corsHeaders = buildCorsHeaders(request, env);

      if (request.method === 'OPTIONS') {
        return new Response(null, { status: 204, headers: corsHeaders });
      }

      if (request.method !== 'POST') {
        return json({ error: 'Method not allowed' }, 405, corsHeaders);
      }

      let body;
      try {
        body = await request.json();
      } catch {
        return json({ error: 'Invalid JSON payload' }, 400, corsHeaders);
      }

      const email = String(body?.email || '').trim().toLowerCase();
      if (!isValidEmail(email)) {
        return json({ error: 'Please provide a valid email address.' }, 400, corsHeaders);
      }

      const createdAt = new Date().toISOString();
      const record = { email, createdAt, source: sanitizeSource(body?.source) };

      try {
        if (env.WAITLIST_KV) {
          const key = `waitlist:${email}`;
          const existing = await env.WAITLIST_KV.get(key);
          if (existing) {
            return json({ ok: true, alreadyJoined: true, message: 'Email already on waitlist.' }, 200, corsHeaders);
          }
          await env.WAITLIST_KV.put(key, JSON.stringify(record));
        }

        if (env.WAITLIST_WEBHOOK_URL) {
          const webhookRes = await fetch(env.WAITLIST_WEBHOOK_URL, {
            method: 'POST',
            headers: { 'content-type': 'application/json' },
            body: JSON.stringify(record)
          });

          if (!webhookRes.ok) {
            console.warn(`WAITLIST_WEBHOOK_URL returned ${webhookRes.status}`);
          }
        }
      } catch (error) {
        console.error('waitlist_store_error', error);
        return json({ error: 'Unable to process waitlist request.' }, 500, corsHeaders);
      }

      return json({ ok: true, message: 'Joined waitlist successfully.' }, 200, corsHeaders);
    }

    if (env.ASSETS) {
      return env.ASSETS.fetch(request);
    }

    return new Response('Assets binding not configured.', { status: 500 });
  }
};

function json(payload, status, extraHeaders = {}) {
  return new Response(JSON.stringify(payload), {
    status,
    headers: {
      'content-type': 'application/json; charset=utf-8',
      ...extraHeaders
    }
  });
}

function buildCorsHeaders(request, env) {
  const requestOrigin = request.headers.get('Origin');
  const allowlist = parseAllowedOrigins(env.ALLOWED_ORIGINS);

  const allowOrigin =
    allowlist.length === 0
      ? (requestOrigin || '*')
      : (requestOrigin && allowlist.includes(requestOrigin) ? requestOrigin : allowlist[0]);

  return {
    'Access-Control-Allow-Origin': allowOrigin,
    'Access-Control-Allow-Methods': 'POST, OPTIONS',
    'Access-Control-Allow-Headers': 'Content-Type',
    Vary: 'Origin'
  };
}

function parseAllowedOrigins(value) {
  if (!value) return [];
  return String(value)
    .split(',')
    .map((origin) => origin.trim())
    .filter(Boolean);
}

function sanitizeSource(value) {
  const source = String(value || 'soberanoai-landing').trim();
  return source.slice(0, 120);
}

function isValidEmail(value) {
  return /^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(value);
}
