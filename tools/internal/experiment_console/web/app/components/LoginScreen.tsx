"use client";

import { FormEvent, useEffect, useMemo, useState } from "react";
import { ArrowRight, CheckCircle2, KeyRound, LockKeyhole, Mail, ShieldCheck, Sparkles, UserRoundPlus } from "lucide-react";
import { consoleRequest } from "../lib/auth";

type AuthMode = "login" | "register";

function destination() {
  if (typeof window === "undefined") return "/";
  const candidate = new URLSearchParams(window.location.search).get("next") || "/";
  return candidate.startsWith("/") && !candidate.startsWith("//") ? candidate : "/";
}

export default function LoginScreen({ checkingOnly = false, initialMode = "login" }: { checkingOnly?: boolean; initialMode?: AuthMode }) {
  const [mode, setMode] = useState<AuthMode>(initialMode);
  const [username, setUsername] = useState("");
  const [displayName, setDisplayName] = useState("");
  const [email, setEmail] = useState("");
  const [emailCode, setEmailCode] = useState("");
  const [password, setPassword] = useState("");
  const [confirmPassword, setConfirmPassword] = useState("");
  const [checking, setChecking] = useState(true);
  const [submitting, setSubmitting] = useState(false);
  const [sendingCode, setSendingCode] = useState(false);
  const [emailAvailable, setEmailAvailable] = useState(true);
  const [codeCooldown, setCodeCooldown] = useState(0);
  const [message, setMessage] = useState("");
  const [error, setError] = useState("");

  useEffect(() => {
    if (checkingOnly) return;
    let active = true;
    async function bootstrap() {
      try { await consoleRequest("/data-api/auth/me"); window.location.replace(destination()); return; } catch { /* show auth landing */ }
      try { const body = await consoleRequest("/data-api/auth/email-config"); if (active) setEmailAvailable(!!body.available); } catch { if (active) setEmailAvailable(false); }
      if (active) setChecking(false);
    }
    bootstrap();
    return () => { active = false; };
  }, [checkingOnly]);

  useEffect(() => {
    if (codeCooldown <= 0) return;
    const timer = window.setInterval(() => setCodeCooldown((value) => Math.max(0, value - 1)), 1000);
    return () => window.clearInterval(timer);
  }, [codeCooldown]);

  const heading = useMemo(() => mode === "login" ? ["WELCOME BACK", "Sign in to the review console", "Use your personal account to continue your assigned review work."] : ["CREATE ACCOUNT", "Create a reviewer account", "Enter your real name and verify your email address. New accounts receive the reviewer role and wait for an administrator to assign a QA range."], [mode]);
  function switchMode(next: AuthMode) { setMode(next); setError(""); setMessage(""); }

  async function sendCode() {
    setSendingCode(true); setError(""); setMessage("");
    try {
      await consoleRequest("/data-api/auth/email-code", { method: "POST", body: JSON.stringify({ email, purpose: "register" }) });
      setCodeCooldown(60); setMessage("Verification code sent. Check your inbox and spam folder. The code is valid for 10 minutes.");
    } catch (cause) { setError(cause instanceof Error ? cause.message : String(cause)); }
    finally { setSendingCode(false); }
  }

  async function submit(event: FormEvent) {
    event.preventDefault(); setSubmitting(true); setError(""); setMessage("");
    try {
      if (mode === "register") {
        if (password !== confirmPassword) throw new Error("The passwords do not match");
        await consoleRequest("/data-api/auth/register", { method: "POST", body: JSON.stringify({ username, display_name: displayName, email, email_code: emailCode, password }) });
      } else {
        await consoleRequest("/data-api/auth/login", { method: "POST", body: JSON.stringify({ username, password }) });
      }
      window.location.replace(destination());
    } catch (cause) { setError(cause instanceof Error ? cause.message : String(cause)); setSubmitting(false); }
  }

  return <main className="login-page">
    <section className="login-shell" aria-label="PKU QA sign in and registration">
      <div className="login-identity">
        <div className="login-brand-mark"><Sparkles size={23} /></div>
        <p className="eyebrow">PKU QA · PUBLICATION GATE</p>
        <h1>Review every item<br />with a complete audit trail.</h1>
        <p className="login-identity-copy">The collaborative workspace for the final 2,200 QA items. Questions, answers, evidence pages, assignments, and every action share one audit trail.</p>
        <div className="login-trust-list"><span><CheckCircle2 size={16} />Work only on stable QA items assigned to you</span><span><ShieldCheck size={16} />Accounts may hold both administrator and reviewer roles</span><span><KeyRound size={16} />Individual accounts, email verification, and auditable actions</span></div>
        <div className="login-identity-footer"><span>2,200 QA · 4 RELEASE FILES</span><span>HTTPS · AUDITED SESSIONS</span></div>
      </div>
      <div className="login-form-region"><div className="login-form-inner">
        <div className="auth-mode-switch" aria-label="Choose sign in or registration"><button className={mode === "login" ? "active" : ""} type="button" onClick={() => switchMode("login")}>Sign in</button><button className={mode === "register" ? "active" : ""} type="button" onClick={() => switchMode("register")}>Register</button></div>
        <div className="login-form-heading"><p className="eyebrow">{heading[0]}</p><h2>{heading[1]}</h2><p>{heading[2]}</p></div>
        {checking ? <div className="login-checking" role="status"><span className="login-spinner" aria-hidden="true" />Checking your current session...</div> : <form onSubmit={submit}>
          {mode === "register" && <><label>Full name<input autoComplete="name" minLength={2} maxLength={80} required value={displayName} onChange={(event) => setDisplayName(event.target.value)} placeholder="Enter your real name" /></label><label>Email address<input autoComplete="email" inputMode="email" required type="email" value={email} onChange={(event) => setEmail(event.target.value)} placeholder="name@example.com" /></label><label>Email verification code<span className="email-code-field"><input autoComplete="one-time-code" inputMode="numeric" maxLength={6} minLength={6} pattern="[0-9]{6}" required value={emailCode} onChange={(event) => setEmailCode(event.target.value.replace(/\D/g, ""))} placeholder="6-digit code" /><button disabled={!emailAvailable || !email || sendingCode || codeCooldown > 0} type="button" onClick={sendCode}><Mail size={15} />{sendingCode ? "Sending" : codeCooldown > 0 ? `${codeCooldown}s` : "Send code"}</button></span></label></>}
          <label>Username<input autoComplete="username" autoFocus={mode === "login"} minLength={3} maxLength={32} required value={username} onChange={(event) => setUsername(event.target.value)} placeholder="Start with a letter or number, 3-32 characters" /></label>
          <label>Password<input autoComplete={mode === "login" ? "current-password" : "new-password"} minLength={10} required type="password" value={password} onChange={(event) => setPassword(event.target.value)} placeholder="At least 10 characters" /></label>
          {mode === "register" && <label>Confirm password<input autoComplete="new-password" minLength={10} required type="password" value={confirmPassword} onChange={(event) => setConfirmPassword(event.target.value)} /></label>}
          {!emailAvailable && mode === "register" && <div className="form-error" role="alert">Email verification is not configured, so registration is currently unavailable. Contact an administrator.</div>}
          {message && <div className="form-success" role="status">{message}</div>}{error && <div className="form-error" role="alert">{error}</div>}
          <button className="primary login-submit" disabled={submitting || (mode === "register" && !emailAvailable)} type="submit">{submitting ? (mode === "login" ? "Signing in..." : "Creating account...") : (mode === "login" ? "Sign in and continue" : "Verify and create account")}{mode === "login" ? <ArrowRight size={17} /> : <UserRoundPlus size={17} />}</button>
        </form>}
        <p className="login-form-note">New accounts start with the reviewer role and no editable range. You can edit assigned items after an administrator creates your QA assignment; all other content remains viewable.</p><p className="login-security-note"><LockKeyhole size={13} />Passwords are stored only as strong hashes. Email is used only for account verification and address changes.</p>
      </div></div>
    </section>
  </main>;
}
