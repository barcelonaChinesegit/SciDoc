"use client";

import { FormEvent, useCallback, useEffect, useMemo, useRef, useState } from "react";
import ConsoleNav from "../components/ConsoleNav";
import { englishText, hasRole, useCurrentUser } from "../lib/auth";

type Progress = {
  status?: string;
  completed?: number;
  total?: number;
  percent?: number;
  updated_at?: number;
  stage?: string;
  stage_name?: string;
  stage_percent?: number;
  resource_waiting?: boolean;
  work_state?: string;
};

type Stage = {
  id: string;
  name: string;
  status: string;
  completed?: number;
  total?: number;
  percent: number;
  workers?: Record<string, { pid?: number }>;
  log_count?: number;
  updated_at?: number;
};

type Task = {
  id: string;
  name: string;
  description: string;
  command: string[];
  cwd: string;
  status: string;
  kind: string;
  position: number;
  priority: number;
  enabled: boolean;
  adopted: boolean;
  alive: boolean;
  restart_count: number;
  max_restarts: number;
  codex_attempts: number;
  max_codex_attempts: number;
  auto_codex: boolean;
  depends_on?: string[];
  started_at?: number;
  last_error?: string;
  progress?: Progress;
  stages?: Stage[];
};

type EtaSample = {
  at: number;
  percent: number;
  startedAt: number;
};

type EtaEstimate = {
  label: string;
  detail: string;
};

type Gpu = {
  index?: number;
  name?: string;
  memory_total_mib?: number;
  memory_used_mib?: number;
  utilization_gpu_percent?: number;
  error?: string;
};

type LogSource = {
  path: string;
  name: string;
  stage: string;
  size_bytes: number;
  updated_at: number;
  current: boolean;
};

type DeletionRecord = {
  id: number;
  task_id: string;
  task_name: string;
  deleted_at: number;
  restored_at?: number;
  restored_task_id?: string;
  task_snapshot?: { status?: string };
};

type ResultDeletionRecord = {
  id: number;
  task_id: string;
  task_name: string;
  deleted_at: number;
  targets?: string[];
  errors?: { path: string; error: string }[];
};

const statusLabel: Record<string, string> = {
  queued: "Queued",
  starting: "Starting",
  running: "Running",
  waiting: "Waiting",
  recovering: "Recovering",
  paused: "Paused",
  failed: "Failed",
  blocked: "Needs attention",
  succeeded: "Completed",
  cancelled: "Cancelled",
};

const ETA_SAMPLE_LIMIT = 8;

function formatEtaDuration(seconds: number) {
  if (seconds < 60) return "less than 1 minute";
  if (seconds < 3600) return `${Math.max(1, Math.round(seconds / 60))} minutes`;
  if (seconds < 86400) {
    const hours = Math.floor(seconds / 3600);
    const minutes = Math.round((seconds % 3600) / 60);
    return minutes ? `${hours} hours ${minutes} minutes` : `${hours} hours`;
  }
  const days = Math.floor(seconds / 86400);
  const hours = Math.round((seconds % 86400) / 3600);
  return hours ? `${days} days ${hours} hours` : `${days} days`;
}

function estimateTaskEta(task: Task, samples: EtaSample[]): EtaEstimate {
  if (task.status === "succeeded") return { label: "Completed", detail: "The task has completed" };
  if (task.status === "paused") return { label: "Paused", detail: "Recalculated after the task resumes" };
  if (["failed", "blocked", "cancelled"].includes(task.status)) {
    return { label: "Unavailable", detail: "The task must be repaired or restarted" };
  }
  if (task.kind === "service") return { label: "Continuous", detail: "A service has no fixed completion time" };
  if (["queued", "waiting"].includes(task.status)) {
    return task.depends_on?.length
      ? { label: "Waiting for dependency", detail: "Timing starts after dependencies complete" }
      : { label: "Waiting for scheduling", detail: "Estimated after valid progress begins" };
  }
  if (task.progress?.resource_waiting || task.progress?.work_state === "waiting_for_free_a800") {
    return { label: "Waiting for GPU/resources", detail: "Resource wait time is excluded from processing speed" };
  }

  const percent = Number(task.progress?.percent);
  if (Number.isFinite(percent) && percent >= 100) {
    return { label: "Finishing", detail: "The computation stage has reached 100%" };
  }
  const rates = samples.slice(1).flatMap((sample, index) => {
    const previous = samples[index];
    const elapsed = sample.at - previous.at;
    const advanced = sample.percent - previous.percent;
    return elapsed > 0 && advanced > 0 ? [advanced / elapsed] : [];
  });
  if (!Number.isFinite(percent) || rates.length < 2) {
    return { label: "Estimating", detail: "At least three recent valid progress samples are required" };
  }
  const sortedRates = [...rates].sort((a, b) => a - b);
  const middle = Math.floor(sortedRates.length / 2);
  const rate = sortedRates.length % 2
    ? sortedRates[middle]
    : (sortedRates[middle - 1] + sortedRates[middle]) / 2;
  const seconds = (100 - percent) / rate;
  if (!Number.isFinite(seconds) || seconds <= 0) {
    return { label: "Finishing", detail: "Final completion steps are running" };
  }
  return {
    label: `About ${formatEtaDuration(seconds)}`,
    detail: `Estimated from the median rate of the latest ${Math.min(samples.length, ETA_SAMPLE_LIMIT)} valid progress samples`,
  };
}

function formatCommand(command: string[]) {
  return command.map((part) => (part.includes(" ") ? JSON.stringify(part) : part)).join(" ");
}

async function request(path: string, init?: RequestInit) {
  let response: Response;
  try {
    response = await fetch(path, {
      ...init,
      cache: "no-store",
      headers: { "Content-Type": "application/json", ...(init?.headers || {}) },
    });
  } catch {
    throw new Error("Cannot connect to the task service. Retrying automatically...");
  }
  const body = await response.json().catch(() => ({}));
  if (!response.ok) {
    const apiError = typeof body.error === "string" && !/[\u3400-\u9fff\uf900-\ufaff]/u.test(body.error)
      ? body.error
      : `The request could not be completed (HTTP ${response.status}).`;
    throw new Error(apiError);
  }
  return body;
}

export default function Home() {
  const { user, loading: authLoading } = useCurrentUser();
  const [tasks, setTasks] = useState<Task[]>([]);
  const [gpus, setGpus] = useState<Gpu[]>([]);
  const [deletions, setDeletions] = useState<DeletionRecord[]>([]);
  const [resultDeletions, setResultDeletions] = useState<ResultDeletionRecord[]>([]);
  const [selectedTaskId, setSelectedTaskId] = useState<string | null>(null);
  const [selectedStage, setSelectedStage] = useState<string | null>(null);
  const [expandedTask, setExpandedTask] = useState<string | null>(null);
  const [log, setLog] = useState("");
  const [logSources, setLogSources] = useState<LogSource[]>([]);
  const [followingLog, setFollowingLog] = useState(true);
  const [showForm, setShowForm] = useState(false);
  const [descriptionTaskId, setDescriptionTaskId] = useState<string | null>(null);
  const [error, setError] = useState("");
  const [hasLoaded, setHasLoaded] = useState(false);
  const [lastUpdated, setLastUpdated] = useState<Date | null>(null);
  const logRef = useRef<HTMLPreElement>(null);
  const [etaSamples, setEtaSamples] = useState<Record<string, EtaSample[]>>({});

  const recordEtaSamples = useCallback((nextTasks: Task[]) => {
    const now = Date.now() / 1000;
    setEtaSamples((currentSamples) => {
      let nextSamples = currentSamples;
      for (const task of nextTasks) {
        if (!["starting", "running", "recovering"].includes(task.status)) continue;
        if (task.progress?.resource_waiting || task.progress?.work_state === "waiting_for_free_a800") continue;
        const percent = Number(task.progress?.percent);
        if (!Number.isFinite(percent)) continue;
        const startedAt = Number(task.started_at || 0);
        let samples = currentSamples[task.id] || [];
        const last = samples.at(-1);
        if (last && (last.startedAt !== startedAt || percent < last.percent)) samples = [];
        const latest = samples.at(-1);
        if (!latest || percent > latest.percent) {
          if (nextSamples === currentSamples) nextSamples = { ...currentSamples };
          nextSamples[task.id] = [
            ...samples,
            { at: now, percent, startedAt },
          ].slice(-ETA_SAMPLE_LIMIT);
        }
      }
      return nextSamples;
    });
  }, []);

  const refresh = useCallback(async () => {
    try {
      const [taskData, systemData, deletionData] = await Promise.all([
        request("/api/tasks"),
        request("/api/system"),
        request("/api/deletions"),
      ]);
      recordEtaSamples(taskData.tasks);
      setTasks(taskData.tasks);
      setGpus(systemData.gpus);
      setDeletions(deletionData.deletions || []);
      setResultDeletions(deletionData.result_deletions || []);
      setHasLoaded(true);
      setLastUpdated(new Date());
      setError("");
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : String(cause));
    }
  }, [recordEtaSamples]);

  useEffect(() => {
    const initial = window.setTimeout(refresh, 0);
    const timer = window.setInterval(refresh, 3000);
    return () => {
      window.clearTimeout(initial);
      window.clearInterval(timer);
    };
  }, [refresh]);

  const counts = useMemo(() => {
    const running = tasks.filter((task) =>
      ["starting", "running", "recovering"].includes(task.status),
    ).length;
    return {
      running,
      queued: tasks.filter((task) => ["queued", "waiting"].includes(task.status)).length,
      failed: tasks.filter((task) => ["failed", "blocked"].includes(task.status)).length,
      completed: tasks.filter((task) => task.status === "succeeded").length,
    };
  }, [tasks]);
  const activeTasks = useMemo(
    () => tasks.filter((task) => task.status !== "succeeded"),
    [tasks],
  );
  const completedTasks = useMemo(
    () => tasks.filter((task) => task.status === "succeeded"),
    [tasks],
  );
  const selected = useMemo(
    () => tasks.find((task) => task.id === selectedTaskId) || null,
    [tasks, selectedTaskId],
  );
  const descriptionTask = useMemo(
    () => tasks.find((task) => task.id === descriptionTaskId) || null,
    [tasks, descriptionTaskId],
  );

  async function action(task: Task, name: string, body: Record<string, unknown> = {}) {
    try {
      await request(`/api/tasks/${task.id}/actions/${name}`, {
        method: "POST",
        body: JSON.stringify(body),
      });
      await refresh();
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : String(cause));
    }
  }

  async function saveDescription(task: Task, description: string) {
    try {
      await request(`/api/tasks/${task.id}`, {
        method: "PATCH",
        body: JSON.stringify({ description }),
      });
      setDescriptionTaskId(null);
      await refresh();
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : String(cause));
    }
  }

  async function openTask(task: Task) {
    if (selectedTaskId !== task.id) {
      setFollowingLog(true);
      setLog("Loading the current worker log...");
      setLogSources([]);
    }
    setSelectedTaskId(task.id);
    setExpandedTask((current) => current === task.id ? null : task.id);
    const currentStage = task.progress?.stage || task.stages?.find(
      (stage) => ["running", "recovering", "starting"].includes(stage.status),
    )?.id;
    setSelectedStage(currentStage || task.stages?.[0]?.id || null);
  }

  async function openStage(task: Task, stage: Stage) {
    setSelectedTaskId(task.id);
    setExpandedTask(task.id);
    setSelectedStage(stage.id);
    setFollowingLog(true);
    setLog(`Loading the log for "${englishText(stage.name, `Stage ${stage.id}`)}"...`);
  }

  const refreshLog = useCallback(async (taskId: string, stageId?: string | null) => {
    const query = new URLSearchParams({ bytes: "100000" });
    if (stageId) query.set("stage", stageId);
    return request(`/api/tasks/${taskId}/logs?${query.toString()}`);
  }, []);

  useEffect(() => {
    if (!selectedTaskId) return;
    let active = true;
    const updateLog = async () => {
      try {
        const body = await refreshLog(selectedTaskId, selectedStage);
        if (!active) return;
        setLog(body.current_log || body.log || "No log available");
        setLogSources(body.sources || []);
      } catch (cause) {
        if (!active) return;
        setLog(cause instanceof Error ? cause.message : String(cause));
        setLogSources([]);
      }
    };
    updateLog();
    const timer = window.setInterval(
      updateLog,
      2000,
    );
    return () => {
      active = false;
      window.clearInterval(timer);
    };
  }, [selectedTaskId, selectedStage, refreshLog]);

  useEffect(() => {
    if (!followingLog || !logRef.current) return;
    const frame = window.requestAnimationFrame(() => {
      if (logRef.current) logRef.current.scrollTop = logRef.current.scrollHeight;
    });
    return () => window.cancelAnimationFrame(frame);
  }, [log, followingLog, selectedTaskId]);

  function trackLogScroll() {
    const viewport = logRef.current;
    if (!viewport) return;
    const distanceFromBottom =
      viewport.scrollHeight - viewport.scrollTop - viewport.clientHeight;
    setFollowingLog(distanceFromBottom < 80);
  }

  function jumpToLatestLog() {
    setFollowingLog(true);
    const viewport = logRef.current;
    if (viewport) viewport.scrollTop = viewport.scrollHeight;
  }

  function renderTaskCard(
    task: Task,
    index: number,
    collection: Task[],
    completed = false,
  ) {
    const progress = task.progress;
    const percent = progress?.percent;
    const eta = estimateTaskEta(task, etaSamples[task.id] || []);
    const previous = collection[index - 1];
    const next = collection[index + 1];
    const previousGlobalIndex = previous
      ? tasks.findIndex((item) => item.id === previous.id)
      : -1;
    const nextGlobalIndex = next
      ? tasks.findIndex((item) => item.id === next.id)
      : -1;
    return (
      <article
        className={`task-card ${completed ? "completed" : ""} ${selectedTaskId === task.id ? "selected" : ""}`}
        key={task.id}
      >
        <button className="task-main" onClick={() => openTask(task)}>
          <span className="order">
            {completed ? "✓" : String(index + 1).padStart(2, "0")}
          </span>
          <span className={`status-dot ${task.status}`} />
          <span className="task-copy">
            <span className="task-title">
              {englishText(task.name, `Task ${task.id}`)}
              {task.adopted && <em>Adopted</em>}
            </span>
            <span className="task-meta">
              {statusLabel[task.status] || task.status} · {task.kind} · restarts {task.restart_count}/{task.max_restarts}
            </span>
            <span className="task-eta" title={eta.detail}>
              Estimated time remaining: <strong>{eta.label}</strong>
            </span>
            <span className={`task-description ${task.description ? "" : "missing"}`}>
              {englishText(task.description, "No English task description")}
            </span>
            <span className="progress-track">
              <i style={{ width: `${Math.max(0, Math.min(100, percent || 0))}%` }} />
            </span>
          </span>
          <span className="progress-value">
            {percent !== undefined && percent !== null ? `${percent.toFixed(1)}%` : "—"}
          </span>
          <span className="expand-mark">
            {expandedTask === task.id ? "−" : "+"}
          </span>
        </button>
        {expandedTask === task.id && task.stages && (
          <div className="stage-list">
            {task.stages.map((stage, stageIndex) => (
              <button
                className={`stage-row ${selectedStage === stage.id ? "active" : ""}`}
                key={stage.id}
                onClick={() => openStage(task, stage)}
              >
                <span className="stage-index">{stageIndex + 1}</span>
                <span className={`status-dot ${stage.status}`} />
                <span className="stage-copy">
                  <strong>{englishText(stage.name, `Stage ${stageIndex + 1}`)}</strong>
                  <small>
                    {statusLabel[stage.status] || stage.status}
                    {stage.total ? ` · ${stage.completed || 0}/${stage.total}` : ""}
                    {stage.workers ? ` · ${Object.keys(stage.workers).length} worker` : ""}
                    {stage.log_count !== undefined ? ` · ${stage.log_count} logs` : ""}
                  </small>
                  <span className="progress-track">
                    <i style={{ width: `${Math.max(0, Math.min(100, stage.percent || 0))}%` }} />
                  </span>
                </span>
                <b>{stage.percent.toFixed(1)}%</b>
              </button>
            ))}
          </div>
        )}
        {hasRole(user, "admin") && <div className="card-actions">
          {!completed && (
            <>
              <button
                disabled={!previous}
                onClick={() => action(task, "move", { index: previousGlobalIndex })}
              >
                ↑
              </button>
              <button
                disabled={!next}
                onClick={() => action(task, "move", { index: nextGlobalIndex })}
              >
                ↓
              </button>
            </>
          )}
          {completed
            ? <button onClick={() => action(task, "resume")}>Run again</button>
            : ["running", "starting", "waiting", "recovering"].includes(task.status)
              ? <button onClick={() => action(task, "pause")}>Pause</button>
              : <button onClick={() => action(task, task.status === "failed" ? "retry" : "resume")}>Run</button>}
          {completed && (
            <button
              className="danger secondary-danger"
              title="Delete model outputs, checkpoint files, and logs while retaining the task record"
              onClick={() => deleteResults(task)}
            >
              Delete task results
            </button>
          )}
          <button
            className="danger"
            title="Delete only the task queue database record; retain outputs, checkpoints, and logs"
            onClick={() => removeTask(task)}
          >
            Delete
          </button>
        </div>}
      </article>
    );
  }

  async function removeTask(task: Task) {
    if (!window.confirm(
      `Delete the task record "${englishText(task.name, `Task ${task.id}`)}"?\n\n` +
      "This removes only the task queue database record. Model outputs, checkpoint files, and logs are retained. " +
      "The deletion is recorded and can be restored from this page.",
    )) return;
    try {
      await request(`/api/tasks/${task.id}`, { method: "DELETE" });
      if (selectedTaskId === task.id) setSelectedTaskId(null);
      await refresh();
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : String(cause));
    }
  }

  async function restoreDeletion(record: DeletionRecord) {
    try {
      await request(`/api/deletions/${record.id}/restore`, {
        method: "POST",
        body: "{}",
      });
      await refresh();
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : String(cause));
    }
  }

  async function deleteResults(task: Task) {
    try {
      const preview = await request(`/api/tasks/${task.id}/result-targets`);
      const targets: string[] = preview.targets || [];
      if (targets.length === 0) {
        window.alert("No explicitly registered output, checkpoint, or log was found for this task.");
        return;
      }
      const sample = targets.slice(0, 6).join("\n");
      if (!window.confirm(
        `Delete results for "${englishText(task.name, `Task ${task.id}`)}"?\n\n` +
        `This will delete ${targets.length} verified targets:\n${sample}` +
        `${targets.length > 6 ? "\n…" : ""}\n\n` +
        "The task queue record will be retained. This file operation cannot be undone with Restore record.",
      )) return;
      const confirmation = window.prompt(
        `This file operation is irreversible. Enter the task ID to confirm:\n${task.id}`,
      );
      if (confirmation !== task.id) {
        setError("The task ID did not match. Task-result deletion was cancelled.");
        return;
      }
      const result = await request(
        `/api/tasks/${task.id}/actions/delete-results`,
        {
          method: "POST",
          body: JSON.stringify({ confirm_task_id: task.id }),
        },
      );
      window.alert(
        `Deleted ${result.deleted_targets?.length || 0} task-result targets.` +
        `${result.errors?.length ? ` ${result.errors.length} targets failed; see deletion records.` : ""}`,
      );
      await refresh();
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : String(cause));
    }
  }

  if (authLoading || !user) {
    return <main className="auth-loading">Verifying task queue access...</main>;
  }

  return (
    <main className="queue-page">
      <ConsoleNav user={user} active="queue" />
      <header className="topbar">
        <div>
          <p className="eyebrow">LOCAL EXPERIMENT CONTROL</p>
          <h1>Experiment Task Queue</h1>
        </div>
        <div className="top-actions">
          <span className={`connection ${error ? "offline" : ""}`}>
            <i /> {error ? "Connection error" : "Scheduler online"}
          </span>
          <a className="secondary-link" href="/data">QA Review</a>
          {hasRole(user, "admin") && <button className="primary" onClick={() => setShowForm(true)}>+ New task</button>}
        </div>
      </header>

      {error && <div className="error-banner">{error}</div>}

      <section className="summary-grid">
        <article><span>Running</span><strong>{hasLoaded ? counts.running : "—"}</strong><small>Includes startup and automatic recovery</small></article>
        <article><span>Queued</span><strong>{hasLoaded ? counts.queued : "—"}</strong><small>Includes dependency and resource waits</small></article>
        <article><span>Failed</span><strong>{hasLoaded ? counts.failed : "—"}</strong><small>Automatic diagnosis or user action required</small></article>
        <article><span>Completed</span><strong>{hasLoaded ? counts.completed : "—"}</strong><small>Logs and checkpoints retained</small></article>
      </section>

      <section className="workspace">
        <div className="queue-column">
          <div className="queue-panel">
            <div className="section-heading">
              <div><p className="eyebrow">QUEUE</p><h2>Task Order</h2></div>
              <span>{lastUpdated ? `Updated ${lastUpdated.toLocaleTimeString()}` : "Connecting..."}</span>
            </div>
            <div className="task-list">
              {!hasLoaded && <div className="empty">Loading the task queue...</div>}
              {hasLoaded && activeTasks.length === 0 && <div className="empty">No tasks are queued or running.</div>}
              {activeTasks.map((task, index) => renderTaskCard(task, index, activeTasks))}
            </div>
          </div>

          <div className="queue-panel completed-panel">
            <div className="section-heading">
              <div><p className="eyebrow">COMPLETED</p><h2>Completed Tasks</h2></div>
              <span>{completedTasks.length} tasks</span>
            </div>
            <div className="task-list completed-list">
              {!hasLoaded && <div className="empty compact">Loading completed tasks...</div>}
              {hasLoaded && completedTasks.length === 0 && <div className="empty compact">No completed tasks.</div>}
              {completedTasks.map((task, index) => renderTaskCard(task, index, completedTasks, true))}
            </div>
            <details className="deletion-history">
              <summary>
                Deletion records
                <span>{deletions.length + resultDeletions.length}</span>
              </summary>
              <p>
                Delete removes only the queue database record. Delete task results removes outputs, checkpoints, and logs.
              </p>
              <div className="deletion-list">
                {deletions.length === 0 && resultDeletions.length === 0 && (
                  <div className="empty compact">No deletion records.</div>
                )}
                {deletions.map((record) => (
                  <article className="deletion-row" key={`task-${record.id}`}>
                    <div>
                      <strong>{englishText(record.task_name, `Task ${record.task_id}`)}</strong>
                      <small>
                        Task record deleted · {new Date(record.deleted_at * 1000).toLocaleString()}
                      </small>
                      <code>{record.task_id}</code>
                    </div>
                    {record.restored_at
                      ? <span className="restored">Restored</span>
                      : hasRole(user, "admin") ? <button onClick={() => restoreDeletion(record)}>Restore record</button> : <span>Only administrators can restore</span>}
                  </article>
                ))}
                {resultDeletions.map((record) => (
                  <article className="deletion-row result" key={`result-${record.id}`}>
                    <div>
                      <strong>{englishText(record.task_name, `Task ${record.task_id}`)}</strong>
                      <small>
                        Task results deleted · {new Date(record.deleted_at * 1000).toLocaleString()}
                      </small>
                      <code>
                        Deleted {record.targets?.length || 0} targets
                        {record.errors?.length ? ` · ${record.errors.length} failed` : ""}
                      </code>
                    </div>
                    <span>Operation log</span>
                  </article>
                ))}
              </div>
            </details>
          </div>
        </div>

        <aside>
          <section className="gpu-panel">
            <div className="section-heading"><div><p className="eyebrow">RESOURCES</p><h2>Live GPU Status</h2></div></div>
            {gpus.map((gpu, index) => {
              if (gpu.error) return <p key={index}>{englishText(gpu.error, "GPU status unavailable")}</p>;
              const used = gpu.memory_used_mib || 0;
              const total = gpu.memory_total_mib || 1;
              const ratio = used / total * 100;
              const isA800 = gpu.name?.includes("A800");
              return (
                <div className="gpu-row" key={gpu.index}>
                  <div><strong>GPU {gpu.index}</strong><span>{englishText(gpu.name, "Unknown GPU")}</span></div>
                  <div className="gpu-stats"><b>{gpu.utilization_gpu_percent}%</b><span>{(used / 1024).toFixed(1)} / {(total / 1024).toFixed(0)} GB</span></div>
                  <div className="memory-track"><i className={isA800 ? "a800" : "a40"} style={{ width: `${ratio}%` }} /></div>
                </div>
              );
            })}
          </section>

          <section className="detail-panel">
            <div className="section-heading">
              <div>
                <p className="eyebrow">DETAIL</p>
                <h2>{selected ? englishText(selected.name, `Task ${selected.id}`) : "Task Details"}</h2>
                {selectedStage && selected?.stages && (
                  <span className="detail-stage">
                    {selected.stages.find((stage) => stage.id === selectedStage)?.name}
                  </span>
                )}
              </div>
              {selected && hasRole(user, "admin") && (
                <button onClick={() => setDescriptionTaskId(selected.id)}>
                  Edit description
                </button>
              )}
            </div>
            {selected ? (
              <>
                <section className="task-explanation">
                  <p className="eyebrow">TASK DESCRIPTION</p>
                  <p>{englishText(selected.description, "No English description. Select Edit description to add the task purpose, steps, and expected output.")}</p>
                </section>
                <dl>
                  <div><dt>ID</dt><dd>{selected.id}</dd></div>
                  <div><dt>Status</dt><dd>{statusLabel[selected.status] || selected.status}</dd></div>
                  <div title={estimateTaskEta(selected, etaSamples[selected.id] || []).detail}>
                    <dt>Estimated time remaining</dt>
                    <dd>{estimateTaskEta(selected, etaSamples[selected.id] || []).label}</dd>
                  </div>
                  <div><dt>Automatic recovery</dt><dd>{selected.auto_codex ? `${selected.codex_attempts}/${selected.max_codex_attempts || "∞"}` : "Off"}</dd></div>
                  <div><dt>Working directory</dt><dd>{selected.cwd}</dd></div>
                  <div className="wide"><dt>Command</dt><dd>{formatCommand(selected.command)}</dd></div>
                </dl>
                {selected.last_error && <div className="last-error">{englishText(selected.last_error.slice(0, 1200), "The latest task error contains non-English text.")}</div>}
                {logSources.length > 0 && (
                  <div className="log-source">
                    <div className="log-source-copy">
                      <span>Current active log</span>
                      <strong>{logSources[0].stage}/{logSources[0].name}</strong>
                      <small>
                        Last written {new Date(logSources[0].updated_at * 1000).toLocaleTimeString()}
                        {" · "}{logSources.length} log files found, refreshed every 2 seconds
                      </small>
                    </div>
                    <button
                      className={`log-follow ${followingLog ? "active" : ""}`}
                      onClick={jumpToLatestLog}
                    >
                      {followingLog ? "Following latest" : "Jump to latest"}
                    </button>
                  </div>
                )}
                <pre
                  ref={logRef}
                  onScroll={trackLogScroll}
                  aria-live="polite"
                  aria-label="Live log for the current task"
                >
                  {englishText(log, log ? "Log output contains non-English text." : "Select a task to display its log.")}
                </pre>
              </>
            ) : <div className="empty compact">Select a task on the left to view its command, checkpoints, and logs.</div>}
          </section>
        </aside>
      </section>

      {showForm && hasRole(user, "admin") && <TaskForm onClose={() => setShowForm(false)} onCreated={() => { setShowForm(false); refresh(); }} />}
      {descriptionTask && (
        <DescriptionForm
          task={descriptionTask}
          onClose={() => setDescriptionTaskId(null)}
          onSave={(description) => saveDescription(descriptionTask, description)}
        />
      )}
    </main>
  );
}

function TaskForm({ onClose, onCreated }: { onClose: () => void; onCreated: () => void }) {
  const [name, setName] = useState("");
  const [description, setDescription] = useState("");
  const [command, setCommand] = useState("");
  const [cwd, setCwd] = useState("/data/czj/SciDoc");
  const [priority, setPriority] = useState(0);
  const [progressPath, setProgressPath] = useState("");
  const [autoRecovery, setAutoRecovery] = useState(true);
  const [saving, setSaving] = useState(false);

  async function submit(event: FormEvent) {
    event.preventDefault();
    setSaving(true);
    try {
      await request("/api/tasks", {
        method: "POST",
        body: JSON.stringify({
          name,
          description,
          command,
          cwd,
          priority,
          progress_path: progressPath || null,
          auto_codex: autoRecovery,
          kind: "batch",
        }),
      });
      onCreated();
    } finally {
      setSaving(false);
    }
  }

  return (
    <div className="modal-backdrop" onMouseDown={onClose}>
      <form className="modal" onSubmit={submit} onMouseDown={(event) => event.stopPropagation()}>
        <div className="section-heading"><div><p className="eyebrow">NEW TASK</p><h2>Add to Task Queue</h2></div><button type="button" onClick={onClose}>x</button></div>
        <label>Task name<input required value={name} onChange={(event) => setName(event.target.value)} placeholder="For example: three-page evidence ablation" /></label>
        <label>
          Task description
          <textarea
            required
            rows={4}
            maxLength={4000}
            value={description}
            onChange={(event) => setDescription(event.target.value)}
            placeholder="Describe the task purpose, input data, main steps, and expected output"
          />
        </label>
        <label>Command<textarea required rows={4} value={command} onChange={(event) => setCommand(event.target.value)} placeholder="python run_hard_eval.py ..." /></label>
        <label>Working directory<input required value={cwd} onChange={(event) => setCwd(event.target.value)} /></label>
        <div className="form-row">
          <label>Priority<input type="number" value={priority} onChange={(event) => setPriority(Number(event.target.value))} /></label>
          <label>Progress JSON<input value={progressPath} onChange={(event) => setProgressPath(event.target.value)} placeholder="Optional" /></label>
        </div>
        <label className="check"><input type="checkbox" checked={autoRecovery} onChange={(event) => setAutoRecovery(event.target.checked)} /> Allow automatic recovery when no built-in strategy applies</label>
        <div className="modal-actions"><button type="button" onClick={onClose}>Cancel</button><button className="primary" disabled={saving}>{saving ? "Saving..." : "Add to queue"}</button></div>
      </form>
    </div>
  );
}

function DescriptionForm({
  task,
  onClose,
  onSave,
}: {
  task: Task;
  onClose: () => void;
  onSave: (description: string) => Promise<void>;
}) {
  const [description, setDescription] = useState(englishText(task.description, ""));
  const [saving, setSaving] = useState(false);

  async function submit(event: FormEvent) {
    event.preventDefault();
    setSaving(true);
    try {
      await onSave(description.trim());
    } finally {
      setSaving(false);
    }
  }

  return (
    <div className="modal-backdrop" onMouseDown={onClose}>
      <form className="modal description-modal" onSubmit={submit} onMouseDown={(event) => event.stopPropagation()}>
        <div className="section-heading">
          <div><p className="eyebrow">TASK PURPOSE</p><h2>Edit Task Description</h2></div>
          <button type="button" onClick={onClose}>×</button>
        </div>
        <p className="description-task-name">{englishText(task.name, `Task ${task.id}`)}</p>
        <label>
          Purpose, steps, and expected output
          <textarea
            required
            autoFocus
            rows={8}
            maxLength={4000}
            value={description}
            onChange={(event) => setDescription(event.target.value)}
            placeholder="Explain why this task runs, what it does, and where it writes its output."
          />
        </label>
        <div className="modal-actions">
          <button type="button" onClick={onClose}>Cancel</button>
          <button className="primary" disabled={saving || !description.trim()}>
            {saving ? "Saving..." : "Save description"}
          </button>
        </div>
      </form>
    </div>
  );
}
