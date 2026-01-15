// Cloudflare Pages Function: proxy /api/* -> Render API
// Route: /api/<anything>  =>  https://football-predictor-winmatic.onrender.com/<anything>

const ORIGIN = "https://football-predictor-winmatic.onrender.com";

export async function onRequest(context) {
  const req = context.request;

  // Build target URL
  const url = new URL(req.url);
  const parts = context.params.catchall || [];
  const path = Array.isArray(parts) ? parts.join("/") : String(parts || "");
  const target = new URL(ORIGIN.replace(/\/$/, "") + "/" + path.replace(/^\//, ""));
  target.search = url.search;

  // Clone headers safely
  const headers = new Headers(req.headers);
  headers.delete("host");
  headers.delete("cf-connecting-ip");
  headers.delete("cf-ipcountry");
  headers.delete("cf-ray");
  headers.delete("x-forwarded-proto");
  headers.delete("x-forwarded-for");

  // Create proxied request
  const init = {
    method: req.method,
    headers,
    redirect: "manual",
  };

  // Only attach body for non-GET/HEAD
  if (req.method !== "GET" && req.method !== "HEAD") {
    init.body = await req.arrayBuffer();
  }

  const upstream = await fetch(target.toString(), init);

  // Pass response back
  const respHeaders = new Headers(upstream.headers);
  respHeaders.set("cache-control", "no-store");

  return new Response(upstream.body, {
    status: upstream.status,
    headers: respHeaders,
  });
}
