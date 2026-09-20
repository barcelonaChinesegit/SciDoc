"use client";

import { FormEvent, useEffect, useState } from "react";
import { CheckCircle2, KeyRound, Mail, Save, ShieldCheck, UserCircle } from "lucide-react";
import ConsoleNav from "../components/ConsoleNav";
import { consoleRequest, englishText, useCurrentUser } from "../lib/auth";

type Assignment = {
  assignment_id: string;
  dataset_id: string;
  start_index: number;
  end_index: number;
  first_index: number;
  assigned_count: number;
  reviewed_count: number;
  selection_mode: "range" | "stable_members";
};

export default function ProfilePage() {
  const { user, loading, refresh } = useCurrentUser();
  const [username, setUsername] = useState("");
  const [displayName, setDisplayName] = useState("");
  const [email, setEmail] = useState("");
  const [emailCode, setEmailCode] = useState("");
  const [currentPassword, setCurrentPassword] = useState("");
  const [newPassword, setNewPassword] = useState("");
  const [confirmPassword, setConfirmPassword] = useState("");
  const [cooldown, setCooldown] = useState(0);
  const [busy, setBusy] = useState(false);
  const [message, setMessage] = useState("");
  const [error, setError] = useState("");
  const [assignments, setAssignments] = useState<Assignment[]>([]);

  useEffect(() => {
    if (!user) return;
    const timer = window.setTimeout(() => {
      setUsername(user.username);
      setDisplayName(englishText(user.display_name, user.username));
      setEmail(user.email || "");
    }, 0);
    return () => window.clearTimeout(timer);
  }, [user]);

  useEffect(() => {
    if (!user) return;
    let cancelled = false;
    consoleRequest("/data-api/assignments?scope=mine")
      .then((body) => { if (!cancelled) setAssignments(body.assignments || []); })
      .catch(() => { if (!cancelled) setAssignments([]); });
    return () => { cancelled = true; };
  }, [user]);

  useEffect(() => {
    if (cooldown <= 0) return;
    const timer = window.setInterval(() => setCooldown((value) => Math.max(0, value - 1)), 1000);
    return () => window.clearInterval(timer);
  }, [cooldown]);

  if (loading || !user) return <main className="auth-loading">Opening your profile...</main>;
  const emailChanged = email.trim().toLowerCase() !== (user.email || "").toLowerCase();

  async function sendEmailCode() {
    setBusy(true); setError(""); setMessage("");
    try {
      await consoleRequest("/data-api/auth/email-code", { method: "POST", body: JSON.stringify({ email, purpose: "profile" }) });
      setCooldown(60); setMessage("Verification code sent. It is valid for 10 minutes.");
    } catch (cause) { setError(cause instanceof Error ? cause.message : String(cause)); }
    finally { setBusy(false); }
  }

  async function saveProfile(event: FormEvent) {
    event.preventDefault(); setBusy(true); setError(""); setMessage("");
    try {
      await consoleRequest("/data-api/profile", { method: "POST", body: JSON.stringify({ username, display_name: displayName, email: email.trim() ? email : undefined, email_code: emailChanged && email.trim() ? emailCode : undefined }) });
      await refresh(); setEmailCode(""); setMessage("Your profile has been updated and recorded in the audit log.");
    } catch (cause) { setError(cause instanceof Error ? cause.message : String(cause)); }
    finally { setBusy(false); }
  }

  async function changePassword(event: FormEvent) {
    event.preventDefault(); setBusy(true); setError(""); setMessage("");
    try {
      if (newPassword !== confirmPassword) throw new Error("The new passwords do not match");
      await consoleRequest("/data-api/profile/password", { method: "POST", body: JSON.stringify({ current_password: currentPassword, new_password: newPassword }) });
      setCurrentPassword(""); setNewPassword(""); setConfirmPassword(""); setMessage("Your password has been updated. Sessions on other devices have been signed out.");
    } catch (cause) { setError(cause instanceof Error ? cause.message : String(cause)); }
    finally { setBusy(false); }
  }

  return <main className="profile-page">
    <ConsoleNav user={user} active="profile" />
    <header className="profile-header"><div><p className="eyebrow">MY ACCOUNT · SECURITY</p><h1>Profile</h1><p>Manage your full name, username, verified email, and password. Only administrators can change roles and QA assignments.</p></div><span className="profile-avatar"><UserCircle size={38} /></span></header>
    {error && <div className="error-banner">{error}</div>}{message && <div className="success-banner">{message}</div>}
    <section className="profile-summary">
      <div><span>Current account</span><strong>{user.username}</strong></div>
      <div><span>Roles</span><strong>{user.roles.map((role) => role === "admin" ? "Administrator" : "Reviewer").join(" + ")}</strong></div>
      <div><span>Email status</span><strong>{user.email_verified ? "Verified" : "Not linked"}</strong></div>
    </section>
    <section className="profile-card profile-assignment-card">
      <div className="profile-card-heading"><span><CheckCircle2 size={20} /></span><div><h2>My review assignments</h2><p>Your assigned JSON files and QA item numbers appear here.</p></div></div>
      {assignments.length ? <div className="profile-assignment-list">{assignments.map((assignment) => <div className="profile-assignment-row" key={assignment.assignment_id}><strong>{assignment.dataset_id}</strong><span>{assignment.selection_mode === "stable_members" ? `Stable item set, ${assignment.assigned_count} items` : `Items ${assignment.start_index}-${assignment.end_index}, ${assignment.assigned_count} items`}; {assignment.reviewed_count} completed</span></div>)}</div> : <p className="profile-info">No range has been assigned yet. You may still view the data review page, but you cannot edit unassigned items.</p>}
    </section>
    <div className="profile-grid">
      <form className="profile-card" onSubmit={saveProfile}>
        <div className="profile-card-heading"><span><UserCircle size={20} /></span><div><h2>Account details</h2><p>Use your real name so assignments and actions remain attributable.</p></div></div>
        <label>Full name<input required minLength={2} maxLength={80} value={displayName} onChange={(event) => setDisplayName(event.target.value)} /></label>
        <label>Username<input required minLength={3} maxLength={32} value={username} onChange={(event) => setUsername(event.target.value)} disabled={user.username === "czj-web"} /></label>
        <label>Verified email<input type="email" value={email} onChange={(event) => setEmail(event.target.value)} placeholder="Add an address if none is linked" /></label>
        {emailChanged && email.trim() && <label>New email verification code<span className="email-code-field"><input required inputMode="numeric" minLength={6} maxLength={6} pattern="[0-9]{6}" value={emailCode} onChange={(event) => setEmailCode(event.target.value.replace(/\D/g, ""))} /><button disabled={busy || !email || cooldown > 0} type="button" onClick={sendEmailCode}><Mail size={15} />{cooldown ? `${cooldown}s` : "Send code"}</button></span></label>}
        <button className="primary icon-text-button" disabled={busy} type="submit"><Save size={16} />Save profile</button>
      </form>
      <form className="profile-card" onSubmit={changePassword}>
        <div className="profile-card-heading"><span><KeyRound size={20} /></span><div><h2>Change password</h2><p>Your current device stays signed in; sessions on other devices are revoked.</p></div></div>
        {user.password_configured ? <><label>Current password<input autoComplete="current-password" required type="password" value={currentPassword} onChange={(event) => setCurrentPassword(event.target.value)} /></label><label>New password<input autoComplete="new-password" required minLength={10} type="password" value={newPassword} onChange={(event) => setNewPassword(event.target.value)} /></label><label>Confirm new password<input autoComplete="new-password" required minLength={10} type="password" value={confirmPassword} onChange={(event) => setConfirmPassword(event.target.value)} /></label><button className="primary icon-text-button" disabled={busy} type="submit"><KeyRound size={16} />Update password</button></> : <div className="profile-info"><ShieldCheck size={18} /><p>This account currently uses an external administrator identity and has no application password. An administrator can set one on the User Management page.</p></div>}
      </form>
    </div>
    <aside className="profile-audit-note"><CheckCircle2 size={18} /><p>Changes to usernames, full names, email addresses, and passwords are audited. Plaintext passwords and email verification codes are never stored in pages or logs.</p></aside>
  </main>;
}
