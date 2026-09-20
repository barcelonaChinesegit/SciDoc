const QUEUE_API = "http://127.0.0.1:8765";
const DATA_API = "http://127.0.0.1:8770";

export const dynamic = "force-dynamic";

type RouteContext = {
  params: Promise<{ path: string[] }>;
};

async function forward(request: Request, context: RouteContext): Promise<Response> {
  const { path } = await context.params;
  const incoming = new URL(request.url);
  const target = new URL(`/api/${path.join("/")}${incoming.search}`, QUEUE_API);
  const headers = new Headers();
  const contentType = request.headers.get("content-type");
  if (contentType) headers.set("content-type", contentType);

  const cookie = request.headers.get("cookie") || "";
  let user: {
    user_id: string;
    username: string;
    role: "admin" | "reviewer";
    roles?: Array<"admin" | "reviewer">;
  };
  try {
    const authResponse = await fetch(`${DATA_API}/api/auth/me`, {
      headers: { cookie },
      cache: "no-store",
    });
    const authBody = await authResponse.json();
    if (!authResponse.ok) {
      return Response.json(
        { error: authBody.error || "Please sign in first" },
        { status: 401, headers: { "cache-control": "no-store" } },
      );
    }
    user = authBody.user;
  } catch (cause) {
    return Response.json(
      { error: "The authentication service is temporarily unavailable", detail: cause instanceof Error ? cause.message : String(cause) },
      { status: 503, headers: { "cache-control": "no-store" } },
    );
  }

  const isMutation = !["GET", "HEAD", "OPTIONS"].includes(request.method);
  const isAdminRead = path[0] === "audit";
  if ((isMutation || isAdminRead) && !(user.roles?.includes("admin") || user.role === "admin")) {
    return Response.json(
      { error: "Task queue write operations are restricted to administrators" },
      { status: 403, headers: { "cache-control": "no-store" } },
    );
  }
  headers.set("x-pku-actor-user-id", user.user_id);
  headers.set("x-pku-actor-username", user.username);

  try {
    const response = await fetch(target, {
      method: request.method,
      headers,
      body: ["GET", "HEAD"].includes(request.method)
        ? undefined
        : await request.arrayBuffer(),
      cache: "no-store",
    });
    const outgoingHeaders = new Headers();
    outgoingHeaders.set(
      "content-type",
      response.headers.get("content-type") || "application/json; charset=utf-8",
    );
    outgoingHeaders.set("cache-control", "no-store");
    return new Response(response.body, {
      status: response.status,
      headers: outgoingHeaders,
    });
  } catch (cause) {
    return Response.json(
      {
        error: "The task API is temporarily unavailable. This page will retry automatically.",
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
