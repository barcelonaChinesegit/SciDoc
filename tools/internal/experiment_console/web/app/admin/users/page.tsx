"use client";

import { FormEvent, useCallback, useEffect, useMemo, useState } from "react";
import {
  ClipboardList,
  KeyRound,
  Save,
  ShieldCheck,
  Trash2,
  UserPlus,
  UserRoundCheck,
  UserRoundX,
} from "lucide-react";
import ConsoleNav from "../../components/ConsoleNav";
import { consoleRequest, CurrentUser, englishText, hasRole, useCurrentUser } from "../../lib/auth";

type ManagedUser = CurrentUser & {
  created_at: number;
  updated_at: number;
};

type AuditEvent = {
  audit_id: string;
  actor_username: string;
  action: string;
  target_type?: string;
  target_id?: string;
  task_id?: string;
  details: Record<string, unknown>;
  created_at: number;
  source?: string;
};

type DatasetOption = {
  id: string;
  qas: number;
  profile: { title: string; collection_id: string; relative_path?: string; path?: string };
};

type ReviewAssignment = {
  assignment_id: string;
  user_id: string;
  username: string;
  display_name: string;
  dataset_id: string;
  start_index: number;
  end_index: number;
  assigned_count: number;
  selection_mode: "range" | "stable_members";
  reviewed_count: number;
  kept_count: number;
  edited_count: number;
  deleted_count: number;
};

type ReviewerProgress = {
  user_id: string;
  username: string;
  display_name: string;
  disabled: boolean;
  assigned_count: number;
  reviewed_count: number;
  kept_count: number;
  edited_count: number;
  deleted_count: number;
  last_reviewed_at: number | null;
};

export default function UserManagementPage() {
  const { user, loading } = useCurrentUser();
  const [users, setUsers] = useState<ManagedUser[]>([]);
  const [events, setEvents] = useState<AuditEvent[]>([]);
  const [datasets, setDatasets] = useState<DatasetOption[]>([]);
  const [assignments, setAssignments] = useState<ReviewAssignment[]>([]);
  const [reviewerProgress, setReviewerProgress] = useState<ReviewerProgress[]>([]);
  const [error, setError] = useState("");
  const [showCreate, setShowCreate] = useState(false);
  const [resetUser, setResetUser] = useState<ManagedUser | null>(null);
  const [pendingUsers, setPendingUsers] = useState<Set<string>>(() => new Set());

  const refreshAccounts = useCallback(async () => {
    const [userBody, reviewAudit, queueAudit, progressBody] = await Promise.all([
      consoleRequest("/data-api/users"),
      consoleRequest("/data-api/audit?limit=250"),
      consoleRequest("/api/audit?limit=250"),
      consoleRequest("/data-api/reviewer-progress"),
    ]);
    setUsers(userBody.users || []);
    setReviewerProgress(progressBody.progress || []);
    setEvents([
      ...(reviewAudit.events || []).map((event: AuditEvent) => ({ ...event, source: "QA Review" })),
      ...(queueAudit.events || []).map((event: AuditEvent) => ({ ...event, source: "Task Queue" })),
    ].sort((left, right) => right.created_at - left.created_at));
    setError("");
  }, []);

  const refreshAssignments = useCallback(async () => {
    const assignmentBody = await consoleRequest("/data-api/assignments");
    setAssignments(assignmentBody.assignments || []);
  }, []);

  const refreshAssignmentState = useCallback(async () => {
    await Promise.all([refreshAssignments(), refreshAccounts()]);
  }, [refreshAccounts, refreshAssignments]);

  const refresh = useCallback(async () => {
    try {
      const [userBody, reviewAudit, queueAudit, datasetBody, assignmentBody, progressBody] = await Promise.all([
        consoleRequest("/data-api/users"),
        consoleRequest("/data-api/audit?limit=250"),
        consoleRequest("/api/audit?limit=250"),
        consoleRequest("/data-api/datasets?collection=final_2200&detail=summary"),
        consoleRequest("/data-api/assignments"),
        consoleRequest("/data-api/reviewer-progress"),
      ]);
      setUsers(userBody.users || []);
      setDatasets(datasetBody.datasets || []);
      setAssignments(assignmentBody.assignments || []);
      setReviewerProgress(progressBody.progress || []);
      setEvents([
        ...(reviewAudit.events || []).map((event: AuditEvent) => ({ ...event, source: "QA Review" })),
        ...(queueAudit.events || []).map((event: AuditEvent) => ({ ...event, source: "Task Queue" })),
      ].sort((left, right) => right.created_at - left.created_at));
      setError("");
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : String(cause));
    }
  }, []);

  useEffect(() => {
    if (!hasRole(user, "admin")) return;
    const timer = window.setTimeout(refresh, 0);
    return () => window.clearTimeout(timer);
  }, [refresh, user]);

  const activeCount = useMemo(
    () => users.filter((account) => !account.disabled).length,
    [users],
  );

  if (loading || !user) return <main className="auth-loading">Verifying administrator access...</main>;
  if (!hasRole(user, "admin")) {
    return (
      <main className="auth-loading">
        This account cannot manage users. <a href="/data">Return to QA Review</a>
      </main>
    );
  }

  async function updateAccount(account: ManagedUser, changes: Record<string, unknown>) {
    const previous = users;
    setUsers((current) => current.map((item) => item.user_id === account.user_id ? {
      ...item,
      ...(Array.isArray(changes.roles) ? { roles: changes.roles as ManagedUser["roles"], role: (changes.roles as string[]).includes("admin") ? "admin" : "reviewer" } : {}),
      ...(typeof changes.disabled === "boolean" ? { disabled: changes.disabled } : {}),
    } : item));
    setPendingUsers((current) => new Set(current).add(account.user_id));
    try {
      await consoleRequest(`/data-api/users/${account.user_id}`, {
        method: "PATCH",
        body: JSON.stringify(changes),
      });
      await refreshAccounts();
    } catch (cause) {
      setUsers(previous);
      setError(cause instanceof Error ? cause.message : String(cause));
    } finally {
      setPendingUsers((current) => { const next = new Set(current); next.delete(account.user_id); return next; });
    }
  }

  async function toggleRole(account: ManagedUser, role: "admin" | "reviewer") {
    const current = account.roles || [account.role];
    const next = current.includes(role)
      ? current.filter((value) => value !== role)
      : [...current, role];
    if (!next.length) {
      setError("An account must retain at least one role");
      return;
    }
    await updateAccount(account, { roles: next });
  }

  return (
    <main className="admin-page">
      <ConsoleNav user={user} active="users" />
      <header className="admin-header">
        <div>
          <p className="eyebrow">ACCESS & TRACEABILITY</p>
          <h1>Users and Audit Trail</h1>
          <p>Use one account per reviewer. Disabling an account immediately revokes all of its sessions.</p>
        </div>
        <button className="primary icon-text-button" onClick={() => setShowCreate(true)}>
          <UserPlus size={17} />Create user
        </button>
      </header>
      {error && <div className="error-banner">{error}</div>}

      <section className="admin-summary">
        <div><strong>{users.length}</strong><span>Total accounts</span></div>
        <div><strong>{activeCount}</strong><span>Active accounts</span></div>
        <div><strong>{users.filter((account) => hasRole(account, "admin")).length}</strong><span>Administrators</span></div>
        <div><strong>{events.length}</strong><span>Recent audit events</span></div>
      </section>

      <section className="assignment-panel">
        <div className="section-heading">
          <div><p className="eyebrow">QA OWNERSHIP</p><h2>Review Assignments</h2></div>
          <span>Assign by JSON file and 1-based QA index. Stable assignments do not shift when earlier items are deleted.</span>
        </div>
        <AssignmentForm users={users} datasets={datasets} onSaved={refreshAssignmentState} />
        <div className="assignment-table">
          {assignments.length ? assignments.map((assignment) => (
            <AssignmentRow
              assignment={assignment}
              datasets={datasets}
              key={assignment.assignment_id}
              onSaved={refreshAssignmentState}
              users={users}
            />
          )) : <div className="empty assignment-empty">No QA items are assigned. Reviewers may view all content but cannot modify any item.</div>}
        </div>
      </section>

      <section className="progress-panel">
        <div className="section-heading">
          <div><p className="eyebrow">REVIEWER PACE</p><h2>Reviewer Progress</h2></div>
          <span>Counts use stable administrator assignments. Unassigned items are excluded from every reviewer&apos;s progress.</span>
        </div>
        <div className="progress-table">
          {reviewerProgress.length ? reviewerProgress.map((progress) => {
            const completion = progress.assigned_count
              ? Math.min(100, progress.reviewed_count / progress.assigned_count * 100)
              : 0;
            return (
              <article className={`progress-row ${progress.disabled ? "disabled" : ""}`} key={progress.user_id}>
                <div className="progress-person"><strong>{englishText(progress.display_name, progress.username)}</strong><small>{progress.username}{progress.disabled ? " · Disabled" : ""}</small></div>
                <div className="progress-numbers"><b>{progress.reviewed_count} / {progress.assigned_count}</b><span>Completed / Assigned</span></div>
                <div className="progress-bar" aria-label={`${englishText(progress.display_name, progress.username)} completion ${completion.toFixed(1)}%`}><i style={{ width: `${completion}%` }} /></div>
                <div className="progress-breakdown"><span>Kept {progress.kept_count}</span><span>Edited {progress.edited_count}</span><span>Deleted {progress.deleted_count}</span></div>
                <time dateTime={progress.last_reviewed_at ? new Date(progress.last_reviewed_at * 1000).toISOString() : undefined}>{progress.last_reviewed_at ? `Last action ${new Date(progress.last_reviewed_at * 1000).toLocaleString()}` : "No completed items"}</time>
              </article>
            );
          }) : <div className="empty">No account currently has the reviewer role.</div>}
        </div>
      </section>

      <section className="admin-grid">
        <section className="user-panel">
          <div className="section-heading">
            <div><p className="eyebrow">ACCOUNTS</p><h2>User Accounts</h2></div>
            <span>Passwords are stored only as strong hashes and cannot be viewed</span>
          </div>
          <div className="role-legend" role="note"><span className="role-legend-swatch reviewer-swatch" />Green Reviewer = assigned role <span className="role-legend-swatch admin-swatch" />Blue Administrator = assigned role; gray buttons indicate roles the account does not have. A selection changes color immediately, then saves to the server and audit log.</div>
          <div className="user-table">
            {users.map((account) => (
              <article className={`user-row ${account.disabled ? "disabled" : ""}`} key={account.user_id}>
                <span className="user-avatar">{englishText(account.display_name, account.username).slice(0, 1).toUpperCase()}</span>
                <div className="user-copy">
                  <strong>{englishText(account.display_name, account.username)}</strong>
                  <small>{account.username} · {account.password_configured ? "Password configured" : "No application password"}</small>
                </div>
                <div className="role-toggles" aria-label={`Roles for ${account.username}`}>
                  <button
                    type="button"
                    aria-pressed={hasRole(account, "reviewer")}
                    className={hasRole(account, "reviewer") ? "role-active" : ""}
                    disabled={account.disabled || pendingUsers.has(account.user_id)}
                    title={hasRole(account, "reviewer") ? "Reviewer role assigned; click to remove" : "Reviewer role not assigned; click to add"}
                    onClick={() => toggleRole(account, "reviewer")}
                  >{pendingUsers.has(account.user_id) ? "Saving..." : "Reviewer"}</button>
                  <button
                    type="button"
                    aria-pressed={hasRole(account, "admin")}
                    className={hasRole(account, "admin") ? "role-active admin-role" : ""}
                    disabled={account.username === "czj-web" || account.disabled || pendingUsers.has(account.user_id)}
                    title={account.username === "czj-web" ? "The czj-web administrator role cannot be removed" : hasRole(account, "admin") ? "Administrator role assigned; click to remove" : "Administrator role not assigned; click to add"}
                    onClick={() => toggleRole(account, "admin")}
                  >{pendingUsers.has(account.user_id) ? "Saving..." : "Administrator"}</button>
                </div>
                <button title="Reset password" aria-label={`Reset password for ${account.username}`} onClick={() => setResetUser(account)}>
                  <KeyRound size={16} />
                </button>
                <button
                  disabled={account.username === "czj-web"}
                  title={account.disabled ? "Enable account" : "Disable account"}
                  aria-label={account.disabled ? `Enable ${account.username}` : `Disable ${account.username}`}
                  onClick={() => updateAccount(account, { disabled: !account.disabled })}
                >
                  {account.disabled ? <UserRoundCheck size={16} /> : <UserRoundX size={16} />}
                </button>
              </article>
            ))}
          </div>
        </section>

        <section className="audit-panel">
          <div className="section-heading">
            <div><p className="eyebrow">IMMUTABLE AUDIT</p><h2>Recent Actions</h2></div>
            <span>Review and queue events in one feed</span>
          </div>
          <div className="audit-feed">
            {events.map((event) => (
              <article key={event.audit_id}>
                <span className="audit-icon"><ShieldCheck size={15} /></span>
                <div>
                  <strong>{event.actor_username}</strong>
                  <p>{actionLabel(event.action)}</p>
                  <small>
                    {event.source} · {new Date(event.created_at * 1000).toLocaleString()}
                    {(event.target_id || event.task_id) ? ` · ${event.target_id || event.task_id}` : ""}
                  </small>
                </div>
              </article>
            ))}
          </div>
        </section>
      </section>

      {showCreate && (
        <CreateUserModal
          onClose={() => setShowCreate(false)}
          onCreated={async () => { setShowCreate(false); await refreshAccounts(); }}
        />
      )}
      {resetUser && (
        <PasswordModal
          account={resetUser}
          currentUserId={user.user_id}
          onClose={() => setResetUser(null)}
          onSaved={async () => { setResetUser(null); await refreshAccounts(); }}
        />
      )}
    </main>
  );
}

function CreateUserModal({ onClose, onCreated }: { onClose: () => void; onCreated: () => void }) {
  const [username, setUsername] = useState("");
  const [displayName, setDisplayName] = useState("");
  const [password, setPassword] = useState("");
  const [roles, setRoles] = useState<Array<"admin" | "reviewer">>(["reviewer"]);
  const [error, setError] = useState("");

  async function submit(event: FormEvent) {
    event.preventDefault();
    try {
      await consoleRequest("/data-api/users", {
        method: "POST",
        body: JSON.stringify({ username, display_name: displayName, password, roles }),
      });
      onCreated();
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : String(cause));
    }
  }

  return (
    <div className="modal-backdrop" onMouseDown={onClose}>
      <form className="modal account-modal" onSubmit={submit} onMouseDown={(event) => event.stopPropagation()}>
        <div className="section-heading"><div><p className="eyebrow">NEW ACCOUNT</p><h2>Create Review Account</h2></div><button type="button" onClick={onClose}>x</button></div>
        <label>Username<input required minLength={3} maxLength={32} value={username} onChange={(event) => setUsername(event.target.value)} placeholder="For example: student-li" /></label>
        <label>Display name<input required value={displayName} onChange={(event) => setDisplayName(event.target.value)} placeholder="For example: Alex Li" /></label>
        <label>Initial password<input required minLength={10} type="password" value={password} onChange={(event) => setPassword(event.target.value)} /></label>
        <fieldset className="account-role-picker">
          <legend>Initial roles</legend>
          <label><input type="checkbox" checked={roles.includes("reviewer")} onChange={(event) => setRoles((current) => event.target.checked ? [...new Set([...current, "reviewer"])] : current.filter((role) => role !== "reviewer"))} />Reviewer</label>
          <label><input type="checkbox" checked={roles.includes("admin")} onChange={(event) => setRoles((current) => event.target.checked ? [...new Set([...current, "admin"])] : current.filter((role) => role !== "admin"))} />Administrator</label>
        </fieldset>
        {error && <div className="form-error">{error}</div>}
        <div className="modal-actions"><button type="button" onClick={onClose}>Cancel</button><button className="primary icon-text-button" type="submit"><UserPlus size={16} />Create account</button></div>
      </form>
    </div>
  );
}

function AssignmentForm({
  users,
  datasets,
  onSaved,
}: {
  users: ManagedUser[];
  datasets: DatasetOption[];
  onSaved: () => Promise<void>;
}) {
  const reviewers = users.filter((account) => hasRole(account, "reviewer") && !account.disabled);
  const orderedDatasets = [...datasets].sort((left, right) => (
    Number(right.profile.collection_id === "final_2200") - Number(left.profile.collection_id === "final_2200")
  ));
  const [userId, setUserId] = useState("");
  const [datasetId, setDatasetId] = useState("");
  const [startIndex, setStartIndex] = useState("1");
  const [endIndex, setEndIndex] = useState("");
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);
  const effectiveUserId = userId || reviewers[0]?.user_id || "";
  const effectiveDatasetId = datasetId || orderedDatasets[0]?.id || "";
  const selectedDataset = orderedDatasets.find((dataset) => dataset.id === effectiveDatasetId);
  const effectiveEndIndex = endIndex || String(selectedDataset?.qas || 1);

  async function submit(event: FormEvent) {
    event.preventDefault();
    setBusy(true);
    try {
      await consoleRequest("/data-api/assignments", {
        method: "POST",
        body: JSON.stringify({
          user_id: effectiveUserId,
          dataset_id: effectiveDatasetId,
          start_index: Number(startIndex),
          end_index: Number(effectiveEndIndex),
        }),
      });
      setError("");
      await onSaved();
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : String(cause));
    } finally {
      setBusy(false);
    }
  }

  return (
    <form className="assignment-create" onSubmit={submit}>
        <label>Reviewer<select required value={effectiveUserId} onChange={(event) => setUserId(event.target.value)}>{reviewers.map((account) => <option key={account.user_id} value={account.user_id}>{englishText(account.display_name, account.username)} · {account.username}</option>)}</select></label>
      <label>JSON file<select required value={effectiveDatasetId} onChange={(event) => { const next = orderedDatasets.find((dataset) => dataset.id === event.target.value); setDatasetId(event.target.value); setStartIndex("1"); setEndIndex(String(next?.qas || 1)); }}>{orderedDatasets.map((dataset) => <option key={dataset.id} value={dataset.id}>{datasetLabel(dataset)} · {dataset.qas} QA</option>)}</select></label>
      <label>Start index<input required min={1} max={selectedDataset?.qas || 1} type="number" value={startIndex} onChange={(event) => setStartIndex(event.target.value)} /></label>
      <label>End index<input required min={1} max={selectedDataset?.qas || 1} type="number" value={effectiveEndIndex} onChange={(event) => setEndIndex(event.target.value)} /></label>
      <button className="primary icon-text-button" disabled={busy || !reviewers.length || !orderedDatasets.length} type="submit"><ClipboardList size={16} />Add assignment</button>
      {error && <div className="form-error assignment-form-error">{error}</div>}
    </form>
  );
}

function AssignmentRow({
  assignment,
  users,
  datasets,
  onSaved,
}: {
  assignment: ReviewAssignment;
  users: ManagedUser[];
  datasets: DatasetOption[];
  onSaved: () => Promise<void>;
}) {
  const reviewers = users.filter((account) => hasRole(account, "reviewer") && !account.disabled);
  const [userId, setUserId] = useState(assignment.user_id);
  const [datasetId, setDatasetId] = useState(assignment.dataset_id);
  const [startIndex, setStartIndex] = useState(String(assignment.start_index));
  const [endIndex, setEndIndex] = useState(String(assignment.end_index));
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);
  const selectedDataset = datasets.find((dataset) => dataset.id === datasetId);
  const stableMembers = assignment.selection_mode === "stable_members";

  async function save() {
    setBusy(true);
    try {
      await consoleRequest(`/data-api/assignments/${assignment.assignment_id}`, {
        method: "PATCH",
        body: JSON.stringify({ user_id: userId, dataset_id: datasetId, start_index: Number(startIndex), end_index: Number(endIndex) }),
      });
      setError("");
      await onSaved();
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : String(cause));
    } finally {
      setBusy(false);
    }
  }

  async function remove() {
    if (!window.confirm(`Delete this QA assignment for ${englishText(assignment.display_name, assignment.username)}?`)) return;
    setBusy(true);
    try {
      await consoleRequest(`/data-api/assignments/${assignment.assignment_id}`, { method: "DELETE" });
      await onSaved();
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : String(cause));
      setBusy(false);
    }
  }

  return (
    <article className="assignment-row">
      <select aria-label="Assigned reviewer" disabled={stableMembers} value={userId} onChange={(event) => setUserId(event.target.value)}>{reviewers.map((account) => <option key={account.user_id} value={account.user_id}>{englishText(account.display_name, account.username)} · {account.username}</option>)}</select>
      <select aria-label="Assigned JSON file" disabled={stableMembers} value={datasetId} onChange={(event) => setDatasetId(event.target.value)}>{datasets.map((dataset) => <option key={dataset.id} value={dataset.id}>{datasetLabel(dataset)}</option>)}</select>
      <input aria-label="Starting QA index" disabled={stableMembers} min={1} max={selectedDataset?.qas || 1} type="number" value={startIndex} onChange={(event) => setStartIndex(event.target.value)} />
      <span>to</span>
      <input aria-label="Ending QA index" disabled={stableMembers} min={1} max={selectedDataset?.qas || 1} type="number" value={endIndex} onChange={(event) => setEndIndex(event.target.value)} />
      <small>{stableMembers ? `${assignment.assigned_count} migrated stable members` : `${assignment.assigned_count} stable QA items`}</small>
      <button title={stableMembers ? "Migrated assignments must be deleted and recreated" : "Save assignment"} aria-label="Save assignment" disabled={busy || stableMembers} onClick={save}><Save size={15} /></button>
      <button title="Delete assignment" aria-label="Delete assignment" disabled={busy} onClick={remove}><Trash2 size={15} /></button>
      {error && <div className="form-error assignment-row-error">{error}</div>}
    </article>
  );
}

function datasetLabel(dataset: DatasetOption) {
  const filename = (dataset.profile.relative_path || dataset.id).split("/").pop() || dataset.id;
  const names: Record<string, string> = {
    "ordinary_qa.json": "Final Ordinary QA 1000",
    "unanswerable_qa.json": "Final Unanswerable QA 200",
    "reasoning_qa.json": "Final Reasoning QA 200",
    "cross_pdf_qa.json": "Final Cross-PDF QA 800",
  };
  return `${names[filename] || "QA dataset"} · ${filename}`;
}

function PasswordModal({
  account,
  currentUserId,
  onClose,
  onSaved,
}: {
  account: ManagedUser;
  currentUserId: string;
  onClose: () => void;
  onSaved: () => void;
}) {
  const [password, setPassword] = useState("");
  const [error, setError] = useState("");

  async function submit(event: FormEvent) {
    event.preventDefault();
    try {
      await consoleRequest(`/data-api/users/${account.user_id}/password`, {
        method: "POST",
        body: JSON.stringify({ password }),
      });
      if (account.user_id === currentUserId) {
        window.location.replace("/login");
        return;
      }
      onSaved();
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : String(cause));
    }
  }

  return (
    <div className="modal-backdrop" onMouseDown={onClose}>
      <form className="modal account-modal" onSubmit={submit} onMouseDown={(event) => event.stopPropagation()}>
        <div className="section-heading"><div><p className="eyebrow">RESET PASSWORD</p><h2>Reset password for {account.username}</h2></div><button type="button" onClick={onClose}>x</button></div>
        <p className="modal-help">Saving revokes every existing session for this account. The user must sign in again with the new password.</p>
        <label>New password<input autoFocus required minLength={10} type="password" value={password} onChange={(event) => setPassword(event.target.value)} /></label>
        {error && <div className="form-error">{error}</div>}
        <div className="modal-actions"><button type="button" onClick={onClose}>Cancel</button><button className="primary icon-text-button" type="submit"><KeyRound size={16} />Save new password</button></div>
      </form>
    </div>
  );
}

function actionLabel(action: string) {
  const labels: Record<string, string> = {
    "auth.login": "Signed in",
    "auth.logout": "Signed out",
    "auth.login_failed": "Sign-in failed",
    "user.create": "Created user",
    "user.update": "Changed user roles or status",
    "user.password_reset": "Reset user password",
    "assignment.create": "Created QA review assignment",
    "assignment.update": "Changed QA review assignment",
    "assignment.delete": "Deleted QA review assignment",
    "review.keep": "Kept one QA item",
    "review.delete": "Deleted one QA item and saved a snapshot",
    "review.edit": "Edited one QA item and saved a snapshot",
    "review.undo": "Undid the latest review action",
    "task.create": "Created task",
    "task.update": "Updated task",
    "task.pause": "Paused task",
    "task.resume": "Resumed task",
    "task.retry": "Retried task",
    "task.move": "Changed task order",
    "task.delete": "Deleted task record",
    "task.delete_results": "Deleted task results",
    "task.restore": "Restored task record",
  };
  return labels[action] || action;
}
