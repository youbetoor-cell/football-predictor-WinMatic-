export async function onRequest(context) {
  const ORIGIN = "https://football-predictor-winmatic.onrender.com";

  const req = context.request;
  const url = new URL(req.url);

  // Strip "/api" prefix
  let path = url.pathname;
  if (path.startsWith("/api/")) path = path.slice(4);     // "/api" removed, keep "/..."
  else if (path === "/api") path = "/";

  const upstreamUrl = new URL(ORIGIN);
  upstreamUrl.pathname = path;
  upstreamUrl.search = url.search;

  // Avoid accidental loops
  if (upstreamUrl.origin === url.origin) {
    return new Response("Proxy misconfigured (origin loop).", { status: 500 });
  }

  // Clone headers but drop hop-by-hop + host
  const headers = new Headers(req.headers);
  headers.delete("host");
  headers.delete("connection");
  headers.delete("content-length");

  const init = {
    method: req.method,
    headers,
    redirect: "manual",
  };

  // Only forward body when it exists
  if (req.method !== "GET" && req.method !== "HEAD") {
    init.body = req.body;
  }

  const upstream = await fetch(upstreamUrl.toString(), init);

  // Copy response headers, add CORS for browser, and disable caching for API
  const respHeaders = new Headers(upstream.headers);
  respHeaders.set("access-control-allow-origin", "*");
  respHeaders.set("access-control-allow-headers", "*");
  respHeaders.set("access-control-allow-methods", "GET,POST,PUT,PATCH,DELETE,OPTIONS");
  respHeaders.set("cache-control", "no-store");
  respHeaders.set("x-wm-pages-proxy", "1");

  // Handle preflight fast
  if (req.method === "OPTIONS") {
    return new Response(null, { status: 204, headers: respHeaders });
  }

  return new Response(upstream.body, {
    status: upstream.status,
    headers: respHeaders,
  });
}
