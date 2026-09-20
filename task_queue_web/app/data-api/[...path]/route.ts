const DATA_API = "http://127.0.0.1:8770";

export const dynamic = "force-dynamic";

type RouteContext = {
  params: Promise<{ path: string[] }>;
};

async function forward(request: Request, context: RouteContext): Promise<Response> {
  const { path } = await context.params;
  const incoming = new URL(request.url);
  const target = new URL(`/api/${path.join("/")}${incoming.search}`, DATA_API);
  const headers = new Headers();
  for (const name of [
    "content-type",
    "range",
    "if-none-match",
    "if-modified-since",
    "cookie",
    "user-agent",
    "x-forwarded-for",
  ]) {
    const value = request.headers.get(name);
    if (value) headers.set(name, value);
  }
  headers.set(
    "x-forwarded-proto",
    request.headers.get("x-forwarded-proto") || incoming.protocol.replace(":", ""),
  );
  if (path.join("/") === "auth/perimeter") {
    return Response.json(
      { error: "External identity exchange is disabled. Sign in with an application account." },
      { status: 404, headers: { "cache-control": "no-store" } },
    );
  }
  try {
    const response = await fetch(target, {
      method: request.method,
      headers,
      body: ["GET", "HEAD"].includes(request.method) ? undefined : await request.arrayBuffer(),
      cache: "no-store",
    });
    const outgoing = new Headers();
    for (const name of [
      "content-type",
      "content-length",
      "content-range",
      "accept-ranges",
      "etag",
      "last-modified",
      "cache-control",
      "set-cookie",
    ]) {
      const value = response.headers.get(name);
      if (value) outgoing.set(name, value);
    }
    if (!outgoing.has("cache-control")) outgoing.set("cache-control", "no-store");
    return new Response(response.body, {
      status: response.status,
      headers: outgoing,
    });
  } catch (cause) {
    return Response.json(
      {
        error: "The data review API is temporarily unavailable. This page will recover automatically.",
        detail: cause instanceof Error ? cause.message : String(cause),
      },
      { status: 503, headers: { "cache-control": "no-store" } },
    );
  }
}

export const GET = forward;
export const POST = forward;
export const PATCH = forward;
export const DELETE = forward;
export const OPTIONS = forward;
