"use client";

import { useCallback, useEffect, useState } from "react";

export type CurrentUser = {
  user_id: string;
  username: string;
  display_name: string;
  role: "admin" | "reviewer";
  roles: Array<"admin" | "reviewer">;
  email: string;
  email_verified: boolean;
  disabled: boolean;
  password_configured: boolean;
};

export function hasRole(user: CurrentUser | null | undefined, role: "admin" | "reviewer") {
  return !!user && (user.roles?.includes(role) || user.role === role);
}

export function englishText(value: unknown, fallback: string) {
  if (typeof value !== "string" || !value.trim()) return fallback;
  return /[\u3400-\u9fff\uf900-\ufaff]/u.test(value) ? fallback : value;
}

export async function consoleRequest(path: string, init?: RequestInit) {
  const response = await fetch(path, {
    ...init,
    cache: "no-store",
    headers: {
      "Content-Type": "application/json",
      ...(init?.headers || {}),
    },
  });
  const body = await response.json().catch(() => ({}));
  if (!response.ok) {
    const apiError = typeof body.error === "string" && !/[\u3400-\u9fff\uf900-\ufaff]/u.test(body.error)
      ? body.error
      : `The request could not be completed (HTTP ${response.status}).`;
    const error = new Error(apiError) as Error & {
      status?: number;
    };
    error.status = response.status;
    throw error;
  }
  return body;
}

export function useCurrentUser(required = true) {
  const [user, setUser] = useState<CurrentUser | null>(null);
  const [loading, setLoading] = useState(true);

  const refresh = useCallback(async () => {
    try {
      const body = await consoleRequest("/data-api/auth/me");
      setUser(body.user);
    } catch (cause) {
      const status = (cause as Error & { status?: number }).status;
      setUser(null);
      if (required && status === 401 && typeof window !== "undefined") {
        const next = encodeURIComponent(window.location.pathname + window.location.search);
        window.location.replace(`/login?next=${next}`);
      }
    } finally {
      setLoading(false);
    }
  }, [required]);

  useEffect(() => {
    const timer = window.setTimeout(refresh, 0);
    return () => window.clearTimeout(timer);
  }, [refresh]);

  return { user, loading, refresh };
}

export async function logout() {
  await consoleRequest("/data-api/auth/logout", {
    method: "POST",
    body: "{}",
  });
  window.location.replace("/login");
}
