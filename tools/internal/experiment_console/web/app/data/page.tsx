"use client";

import { FormEvent, useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  CornerDownRight,
  ChevronDown,
  ChevronUp,
  LockKeyhole,
  Pencil,
  Plus,
  Save,
  X,
} from "lucide-react";
import ConsoleNav from "../components/ConsoleNav";
import { useCurrentUser } from "../lib/auth";

type Dataset = {
  id: string;
  papers: number;
  qas: number;
  size_bytes: number;
  updated_at: number;
  sha256: string;
  profile: {
    collection_id: string;
    collection_title: string;
    title: string;
    status: string;
    role: string;
    description: string;
    lineage: string;
    caution: string;
    source_summary: string;
    acquisition: string;
    intended_use: string;
    artifact_class: string;
    reviewable: boolean;
    read_only_reason?: string | null;
    path: string;
    relative_path: string;
    page_numbering: string;
    statistics: {
      avg_qas_per_paper: number;
      short_answer: number;
      with_evidence_pages: number;
      multi_page_evidence: number;
      unanswerable: number;
      with_evidence_items: number;
      question_types: Record<string, number>;
      modalities: Record<string, number>;
    };
    paper_fields: DatasetField[];
    qa_fields: DatasetField[];
  };
};

type MyAssignment = {
  assignment_id: string;
  dataset_id: string;
  start_index: number;
  end_index: number;
  first_index: number;
  assigned_count: number;
  selection_mode: "range" | "stable_members";
  reviewed_count: number;
  kept_count: number;
  deleted_count: number;
  edited_count: number;
};

type DatasetField = {
  name: string;
  description: string;
  present: number;
  total: number;
  coverage_percent: number;
  value_types: string[];
};

type QaItem = {
  paper_id: string;
  qa_id: string;
  qa: {
    question?: string;
    answer?: string;
    evidence_pages?: number[];
    evidence_items?: Array<{
      physical_pdf_page?: number;
      source_doc_number?: number;
      source_page?: number;
      supported_fact?: string;
    }>;
    modal_types?: string[];
    evidence_hops?: number;
    review_status?: string;
  };
  paper: Record<string, unknown>;
  review_status: string;
  pdf_available: boolean;
};

type ItemResponse = {
  dataset_id: string;
  dataset_sha256: string;
  index: number;
  total: number;
  reviewed: number;
  kept: number;
  deleted: number;
  edited: number;
  can_modify: boolean;
  assigned_to_current_user: boolean;
  assignment?: {
    assignment_id: string;
    start_index: number;
    end_index: number;
    assigned_count: number;
    selection_mode: "range" | "stable_members";
  } | null;
  item: QaItem;
};

type AuditEvent = {
  event_id: string;
  action: string;
  paper_id: string;
  qa_id: string;
  created_at: number;
  undone_at?: number;
  actor_username?: string;
  undone_by_username?: string;
  can_undo?: boolean;
};

type EvidencePageDraft = {
  id: string;
  value: string;
  originalPage: number | null;
};

async function dataRequest(path: string, init?: RequestInit) {
  const response = await fetch(path, {
    ...init,
    cache: "no-store",
    headers: { "Content-Type": "application/json", ...(init?.headers || {}) },
  });
  const body = await response.json().catch(() => ({}));
  if (!response.ok) {
    const apiError = typeof body.error === "string" && !/[\u3400-\u9fff\uf900-\ufaff]/u.test(body.error)
      ? body.error
      : `The request could not be completed (HTTP ${response.status}).`;
    throw new Error(apiError);
  }
  return body;
}

const finalDatasetNames: Record<string, { title: string; role: string; description: string; caution: string }> = {
  "ordinary_qa.json": { title: "Final Ordinary QA 1000", role: "Answerable single-paper QA", description: "The 1,000 ordinary answerable single-paper items in the final 2,200-item release.", caution: "This is a canonical final_2200 file. Edits update the source JSON and collection manifest." },
  "unanswerable_qa.json": { title: "Final Unanswerable QA 200", role: "Unanswerable QA", description: "The 200 single-paper items whose answer is exactly Unanswerable and whose evidence list is empty.", caution: "The answer must be exactly Unanswerable and evidence_pages must be empty." },
  "reasoning_qa.json": { title: "Final Reasoning QA 200", role: "Single-paper multi-step reasoning", description: "The 200 single-paper multi-step reasoning items in the final release.", caution: "Each item must preserve a necessary reasoning chain and use 1-based physical PDF pages." },
  "cross_pdf_qa.json": { title: "Final Cross-PDF QA 800", role: "Cross-paper reasoning", description: "The 800 cross-paper items in the final release, using canonical z_cross_ PDF identifiers.", caution: "Each item must require multiple documents and use 1-based physical pages in the merged PDF." },
};

const fieldDescriptions: Record<string, string> = {
  paper: "Identifier for the paper or merged PDF.", QA: "QA items keyed by stable QA ID.",
  question: "Question presented to the evaluated model.", answer: "Gold answer used for scoring.",
  evidence_pages: "1-based physical PDF pages supporting the answer.", modal_types: "Modalities required to answer the question.",
  evidence_items: "Source mapping and supported fact for each evidence page.", evidence_hops: "Necessary evidence sources or reasoning steps.",
  answer_format: "Expected answer data format.", answer_aliases: "Accepted equivalent answer expressions.",
  answer_unit: "Unit for a numeric answer.", numeric_tolerance: "Allowed numeric scoring tolerance.",
  annotation_provenance: "Generation, review, correction, and release provenance.", review_status: "Model or human review status.",
};

function englishDataset(dataset: Dataset): Dataset {
  const filename = dataset.profile.relative_path?.split("/").pop() || dataset.id.split("/").pop() || dataset.id;
  const finalProfile = finalDatasetNames[filename];
  const family = /cross[_-]?pdf/i.test(filename) ? "Cross-PDF" : /reasoning/i.test(filename) ? "Reasoning" : /unanswerable/i.test(filename) ? "Unanswerable" : "QA";
  const fallbackTitle = `${family} dataset · ${filename}`;
  const localizeFields = (fields: DatasetField[]) => fields.map((field) => ({
    ...field,
    description: fieldDescriptions[field.name] || `Dataset field: ${field.name}.`,
  }));
  return {
    ...dataset,
    profile: {
      ...dataset.profile,
      collection_title: dataset.profile.collection_id === "final_2200" ? "Final Publication Review (2,200 items)" : "Other data and historical versions",
      title: finalProfile?.title || fallbackTitle,
      status: finalProfile ? "Final publication review" : "Reference dataset",
      role: finalProfile?.role || `${family} reference data`,
      description: finalProfile?.description || "Project dataset available for review and traceability.",
      lineage: finalProfile ? "Assembled from fully reviewed release components with unified global QA IDs." : "Discovered from the project data directory; consult its path and annotation_provenance for lineage.",
      caution: finalProfile?.caution || (dataset.profile.reviewable ? "Edits write to the source JSON and create an audit snapshot." : "This process artifact is read-only."),
      source_summary: "Project-managed QA data with stable paper and QA identifiers.",
      acquisition: "Produced by the project's cleaning, generation, and review workflows.",
      intended_use: finalProfile ? "Canonical input for final human review and evaluation." : "Reference, audit, or intermediate workflow data.",
      artifact_class: finalProfile ? "Canonical QA dataset" : "QA reference dataset",
      read_only_reason: dataset.profile.reviewable ? null : "This process artifact is available only for viewing and traceability.",
      page_numbering: "1-based physical PDF pages; printed paper page numbers are not used for scoring.",
      paper_fields: localizeFields(dataset.profile.paper_fields || []),
      qa_fields: localizeFields(dataset.profile.qa_fields || []),
    },
  };
}

function humanSize(value: number) {
  if (value < 1024 * 1024) return `${(value / 1024).toFixed(0)} KB`;
  return `${(value / 1024 / 1024).toFixed(1)} MB`;
}

export default function DataReviewPage() {
  const { user, loading: authLoading } = useCurrentUser();
  const [datasets, setDatasets] = useState<Dataset[]>([]);
  const [myAssignments, setMyAssignments] = useState<MyAssignment[]>([]);
  const [showOtherDatasets, setShowOtherDatasets] = useState(false);
  const [otherDatasetsLoaded, setOtherDatasetsLoaded] = useState(false);
  const [otherDatasetsLoading, setOtherDatasetsLoading] = useState(false);
  const [selectedDataset, setSelectedDataset] = useState<Dataset | null>(null);
  const [datasetId, setDatasetId] = useState("");
  const [index, setIndex] = useState(0);
  const [current, setCurrent] = useState<ItemResponse | null>(null);
  const [events, setEvents] = useState<AuditEvent[]>([]);
  const [pdfPage, setPdfPage] = useState(1);
  const [pdfNavigationNonce, setPdfNavigationNonce] = useState(0);
  const [message, setMessage] = useState("");
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);
  const [jumpIndex, setJumpIndex] = useState("1");
  const [editing, setEditing] = useState(false);
  const [editTarget, setEditTarget] = useState<"question" | "answer" | "evidence">("question");
  const [draftQuestion, setDraftQuestion] = useState("");
  const [draftAnswer, setDraftAnswer] = useState("");
  const [draftEvidencePages, setDraftEvidencePages] = useState<EvidencePageDraft[]>([]);
  const [editingEvidenceId, setEditingEvidenceId] = useState<string | null>(null);
  const itemCache = useRef(new Map<string, ItemResponse>());
  const itemRequest = useRef(0);

  const loadDatasets = useCallback(async () => {
    try {
      const body = await dataRequest("/data-api/datasets?collection=final_2200&detail=summary");
      setDatasets((body.datasets || []).map(englishDataset));
      setDatasetId((selected) => selected || (
        body.datasets?.find((dataset: Dataset) => dataset.profile.collection_id === "final_2200")?.id
        || body.datasets?.[0]?.id
        || ""
      ));
      setError("");
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : String(cause));
    }
  }, []);

  const loadMyAssignments = useCallback(async () => {
    try {
      const body = await dataRequest("/data-api/assignments?scope=mine");
      setMyAssignments(body.assignments || []);
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : String(cause));
    }
  }, []);

  const loadOtherDatasets = useCallback(async () => {
    if (otherDatasetsLoaded || otherDatasetsLoading) return;
    setOtherDatasetsLoading(true);
    try {
      const body = await dataRequest("/data-api/datasets?collection=other&detail=summary");
      setDatasets((current) => {
        const known = new Set(current.map((dataset) => dataset.id));
        return [...current, ...(body.datasets || []).filter((dataset: Dataset) => !known.has(dataset.id)).map(englishDataset)];
      });
      setOtherDatasetsLoaded(true);
      setShowOtherDatasets(true);
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : String(cause));
    } finally {
      setOtherDatasetsLoading(false);
    }
  }, [otherDatasetsLoaded, otherDatasetsLoading]);

  const loadSelectedDataset = useCallback(async (selected: string) => {
    if (!selected) {
      setSelectedDataset(null);
      return;
    }
    try {
      const body = await dataRequest(`/data-api/datasets/${encodeURIComponent(selected)}`);
      setSelectedDataset(body.dataset ? englishDataset(body.dataset) : null);
    } catch (cause) {
      setSelectedDataset(null);
      setError(cause instanceof Error ? cause.message : String(cause));
    }
  }, []);

  const applyItem = useCallback((itemBody: ItemResponse) => {
    setCurrent(itemBody);
    setIndex(itemBody.index);
    setJumpIndex(String(itemBody.index + 1));
    setPdfPage(itemBody.item.qa.evidence_pages?.[0] || 1);
    setPdfNavigationNonce((value) => value + 1);
    setDraftQuestion(itemBody.item.qa.question || "");
    setDraftAnswer(itemBody.item.qa.answer || "");
    setDraftEvidencePages((itemBody.item.qa.evidence_pages || []).map((page: number, pageIndex: number) => ({
      id: `${page}-${pageIndex}`,
      value: String(page),
      originalPage: page,
    })));
    setEditing(false);
    setEditingEvidenceId(null);
    setError("");
  }, []);

  const cacheKey = useCallback((selected: string, position: number) => `${selected}:${position}`, []);

  const prefetchAdjacentItems = useCallback(async (selected: string, position: number, total: number) => {
    const positions = [position - 1, position + 1].filter((candidate) => candidate >= 0 && candidate < total);
    await Promise.all(positions.map(async (candidate) => {
      const key = cacheKey(selected, candidate);
      if (itemCache.current.has(key)) return;
      try {
        const body = await dataRequest(`/data-api/datasets/${encodeURIComponent(selected)}/item?index=${candidate}`);
        if (itemCache.current.size >= 16) itemCache.current.delete(itemCache.current.keys().next().value);
        itemCache.current.set(cacheKey(selected, body.index), body);
      } catch {
        // Prefetch is optional; normal navigation still displays API errors.
      }
    }));
  }, [cacheKey]);

  const loadItem = useCallback(async (selected: string, position: number) => {
    if (!selected) return;
    const key = cacheKey(selected, position);
    const cached = itemCache.current.get(key);
    if (cached) {
      applyItem(cached);
      void prefetchAdjacentItems(selected, cached.index, cached.total);
      return;
    }
    const requestId = ++itemRequest.current;
    try {
      const itemBody = await dataRequest(`/data-api/datasets/${encodeURIComponent(selected)}/item?index=${position}`);
      if (itemCache.current.size >= 16) itemCache.current.delete(itemCache.current.keys().next().value);
      itemCache.current.set(cacheKey(selected, itemBody.index), itemBody);
      if (requestId !== itemRequest.current) return;
      applyItem(itemBody);
      void prefetchAdjacentItems(selected, itemBody.index, itemBody.total);
    } catch (cause) {
      if (requestId !== itemRequest.current) return;
      setError(cause instanceof Error ? cause.message : String(cause));
    }
  }, [applyItem, cacheKey, prefetchAdjacentItems]);

  const loadEvents = useCallback(async (selected: string) => {
    if (!selected) return;
    try {
      const eventBody = await dataRequest(`/data-api/events?dataset=${encodeURIComponent(selected)}&limit=30`);
      setEvents(eventBody.events || []);
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : String(cause));
    }
  }, []);

  useEffect(() => {
    if (!user) return;
    const timer = window.setTimeout(loadDatasets, 0);
    return () => window.clearTimeout(timer);
  }, [loadDatasets, user]);
  useEffect(() => {
    if (!user) return;
    const timer = window.setTimeout(loadMyAssignments, 0);
    return () => window.clearTimeout(timer);
  }, [loadMyAssignments, user]);
  useEffect(() => {
    if (!user) return;
    const timer = window.setTimeout(() => loadItem(datasetId, index), 0);
    return () => window.clearTimeout(timer);
  }, [datasetId, index, loadItem, user]);
  useEffect(() => {
    if (!user) return;
    const timer = window.setTimeout(() => loadEvents(datasetId), 0);
    return () => window.clearTimeout(timer);
  }, [datasetId, loadEvents, user]);
  useEffect(() => {
    if (!user) return;
    const timer = window.setTimeout(() => loadSelectedDataset(datasetId), 0);
    return () => window.clearTimeout(timer);
  }, [datasetId, loadSelectedDataset, user]);

  function adjustDatasetQaCount(delta: number) {
    setDatasets((rows) => rows.map((dataset) => dataset.id === datasetId
      ? { ...dataset, qas: Math.max(0, dataset.qas + delta) }
      : dataset));
    setSelectedDataset((dataset) => dataset && dataset.id === datasetId
      ? { ...dataset, qas: Math.max(0, dataset.qas + delta) }
      : dataset);
  }

  async function review(action: "keep" | "delete") {
    if (!current) return;
    if (!current.can_modify) {
      setError("This QA item is not assigned to the current account. It is view-only.");
      return;
    }
    if (action === "delete" && !window.confirm("Delete this QA item? This writes to the JSON, but the action can be undone.")) return;
    setBusy(true);
    try {
      const result = await dataRequest(`/data-api/datasets/${encodeURIComponent(datasetId)}/review`, {
        method: "POST",
        body: JSON.stringify({
          paper_id: current.item.paper_id,
          qa_id: current.item.qa_id,
          action,
          expected_sha256: current.dataset_sha256,
        }),
      });
      setMessage(action === "keep" ? "Item kept and recorded in the review log." : "Item deleted from the JSON. A complete reversible snapshot was saved.");
      itemCache.current.clear();
      if (result.action === "delete") adjustDatasetQaCount(-1);
      await loadMyAssignments();
      await loadEvents(datasetId);
      await loadItem(datasetId, action === "delete" ? index : Math.min(index + 1, current.total - 1));
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : String(cause));
    } finally {
      setBusy(false);
    }
  }

  function beginEdit(target: "question" | "answer" | "evidence") {
    if (!current) return;
    if (!current.can_modify) {
      setError("This QA item is not assigned to the current account. It is view-only.");
      return;
    }
    setEditTarget(target);
    setEditing(true);
    setError("");
  }

  function sortEvidenceDrafts(drafts: EvidencePageDraft[]) {
    return [...drafts].sort((left, right) => {
      const leftPage = Number(left.value);
      const rightPage = Number(right.value);
      if (!Number.isFinite(leftPage)) return 1;
      if (!Number.isFinite(rightPage)) return -1;
      return leftPage - rightPage;
    });
  }

  function addEvidencePage() {
    const id = `new-${Date.now()}`;
    setDraftEvidencePages((pages) => [...pages, { id, value: "", originalPage: null }]);
    setEditingEvidenceId(id);
  }

  function removeEvidencePage(id: string) {
    setDraftEvidencePages((pages) => pages.filter((page) => page.id !== id));
    if (editingEvidenceId === id) setEditingEvidenceId(null);
  }

  function finishEvidencePageEdit() {
    setDraftEvidencePages((pages) => sortEvidenceDrafts(pages));
    setEditingEvidenceId(null);
  }

  async function saveEdit() {
    if (!current) return;
    const parsedPages = draftEvidencePages.map((page) => Number(page.value));
    if (parsedPages.some((page) => !Number.isInteger(page) || page < 1)) {
      setError("Physical evidence pages must be integers starting at 1.");
      return;
    }
    setBusy(true);
    try {
      await dataRequest(`/data-api/datasets/${encodeURIComponent(datasetId)}/edit`, {
        method: "POST",
        body: JSON.stringify({
          paper_id: current.item.paper_id,
          qa_id: current.item.qa_id,
          question: draftQuestion,
          answer: draftAnswer,
          evidence_pages: parsedPages,
          evidence_page_changes: draftEvidencePages.map((page) => ({
            from: page.originalPage,
            to: Number(page.value),
          })),
          expected_sha256: current.dataset_sha256,
        }),
      });
      setMessage("Changes saved to the JSON. The complete pre-edit snapshot and audit entry were saved.");
      setEditing(false);
      itemCache.current.clear();
      await loadMyAssignments();
      await loadEvents(datasetId);
      await loadItem(datasetId, index);
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : String(cause));
    } finally {
      setBusy(false);
    }
  }

  function cancelEdit() {
    if (!current) return;
    setDraftQuestion(current.item.qa.question || "");
    setDraftAnswer(current.item.qa.answer || "");
    setDraftEvidencePages((current.item.qa.evidence_pages || []).map((page, pageIndex) => ({
      id: `${page}-${pageIndex}`,
      value: String(page),
      originalPage: page,
    })));
    setEditing(false);
    setEditingEvidenceId(null);
    setError("");
  }

  async function undo() {
    if (!datasetId) return;
    const event = events.find((candidate) => candidate.can_undo);
    if (!event) {
      setError("The current account has no review action to undo.");
      return;
    }
    setBusy(true);
    try {
      await dataRequest(`/data-api/datasets/${encodeURIComponent(datasetId)}/undo`, {
        method: "POST",
        body: JSON.stringify({ event_id: event.event_id }),
      });
      setMessage("The latest review action for this dataset was undone.");
      itemCache.current.clear();
      if (event.action === "delete") adjustDatasetQaCount(1);
      await loadMyAssignments();
      await loadEvents(datasetId);
      await loadItem(datasetId, index);
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : String(cause));
    } finally {
      setBusy(false);
    }
  }

  function jumpToQa(event: FormEvent) {
    event.preventDefault();
    if (!current) return;
    const target = Number(jumpIndex);
    if (!Number.isInteger(target) || target < 1 || target > current.total) {
      setError(`Enter a QA index from 1 to ${current.total}.`);
      return;
    }
    setError("");
    setIndex(target - 1);
  }

  function jumpToPdfPage(page: number) {
    if (!Number.isInteger(page) || page < 1) return;
    setPdfPage(page);
    // Chromium's built-in PDF viewer does not reliably react when an existing
    // iframe changes only its URL fragment. Remount it for every explicit
    // navigation, including a second click on the currently selected page.
    setPdfNavigationNonce((value) => value + 1);
  }

  const pdfBaseUrl = useMemo(() => {
    if (!current?.item.pdf_available) return "";
    return `/data-api/pdf?dataset=${encodeURIComponent(datasetId)}&paper=${encodeURIComponent(current.item.paper_id)}`;
  }, [current?.item.pdf_available, current?.item.paper_id, datasetId]);
  const pdfUrl = pdfBaseUrl ? `${pdfBaseUrl}#page=${pdfPage}&navpanes=0` : "";
  const pdfNavigationKey = `${pdfBaseUrl}:${pdfPage}:${pdfNavigationNonce}`;

  const progress = current?.total ? current.reviewed / current.total * 100 : 0;
  const evidencePages = current?.item.qa.evidence_pages || [];
  const datasetStats = selectedDataset?.profile.statistics;
  const datasetCollections = useMemo(() => {
    const grouped = new Map<string, { title: string; datasets: Dataset[] }>();
    datasets.forEach((dataset) => {
      const id = dataset.profile.collection_id || "other";
      const group = grouped.get(id) || {
        title: dataset.profile.collection_title || "Other data and historical versions",
        datasets: [],
      };
      group.datasets.push(dataset);
      grouped.set(id, group);
    });
    return [...grouped.entries()]
      .sort(([left], [right]) => Number(right === "final_2200") - Number(left === "final_2200"))
      .map(([id, group]) => ({
        id,
        ...group,
        qas: group.datasets.reduce((total, dataset) => total + dataset.qas, 0),
      }));
  }, [datasets]);

  if (authLoading || !user) {
    return <main className="auth-loading">Verifying reviewer access...</main>;
  }

  return (
    <main className="data-page">
      <ConsoleNav user={user} active="review" />
      <header className="data-header">
        <div>
          <p className="eyebrow">LOCAL DATA CONTROL · REVERSIBLE REVIEW</p>
          <h1>Datasets and QA Review</h1>
        </div>
        <div className="data-header-actions">
          <span className={`connection ${error ? "offline" : ""}`}><i />{error ? "Data service error" : "Review service online"}</span>
          <a className="secondary-link" href="/guide">Reviewer Guide</a>
          <a className="secondary-link" href="/queue">Task Queue</a>
          <button disabled={busy || !events.some((event) => event.can_undo)} onClick={undo}>Undo last action</button>
        </div>
      </header>

      {error && <div className="error-banner">{error}</div>}
      {message && <div className="review-message">{message}</div>}

      <section className="my-assignment-panel">
        <div className="section-heading">
          <div><p className="eyebrow">MY QA OWNERSHIP</p><h2>My Review Assignments</h2></div>
          <span>{myAssignments.length ? "An administrator assigned the ranges below. Unassigned items remain view-only." : "No QA range has been assigned to this account."}</span>
        </div>
        {myAssignments.length ? (
          <div className="my-assignment-list">
            {myAssignments.map((assignment) => {
              const dataset = datasets.find((candidate) => candidate.id === assignment.dataset_id);
              const progress = assignment.assigned_count ? assignment.reviewed_count / assignment.assigned_count * 100 : 0;
              return (
                <button
                  className={`my-assignment-card ${assignment.dataset_id === datasetId ? "active" : ""}`}
                  key={assignment.assignment_id}
                  type="button"
                  disabled={editing}
                  onClick={() => { setSelectedDataset(null); setDatasetId(assignment.dataset_id); setIndex(Math.max(0, assignment.first_index - 1)); setMessage(""); }}
                >
                  <strong>{dataset?.profile.title || assignment.dataset_id}</strong>
                  <span>{assignment.selection_mode === "stable_members" ? `Stable item set · ${assignment.assigned_count} items` : `Items ${assignment.start_index}-${assignment.end_index} · ${assignment.assigned_count} items`}</span>
                  <span>{assignment.reviewed_count} completed (kept {assignment.kept_count} · edited {assignment.edited_count} · deleted {assignment.deleted_count})</span>
                  <i><em style={{ width: `${Math.min(100, progress)}%` }} /></i>
                </button>
              );
            })}
          </div>
        ) : <div className="my-assignment-empty">Ask an administrator to assign a JSON file and index range in User Management.</div>}
      </section>

      {selectedDataset && (
        <section className="dataset-guide">
          <div className="dataset-guide-copy">
            <div className="dataset-guide-title">
              <div>
                <p className="eyebrow">SELECTED DATASET · SUMMARY</p>
                <h2>{selectedDataset.profile.title}</h2>
              </div>
              <div className="dataset-badges">
                <span>{selectedDataset.profile.status}</span>
                <span>{selectedDataset.profile.role}</span>
              </div>
            </div>
            <p className="dataset-description">{selectedDataset.profile.description}</p>
            <div className="dataset-path-line"><code>{selectedDataset.profile.relative_path}</code><span className={selectedDataset.profile.reviewable ? "editable-badge" : "readonly-badge"}>{selectedDataset.profile.reviewable ? "Editable in review" : "Read-only artifact"}</span></div>
            <div className="dataset-file-address"><span>File path</span><code>{selectedDataset.profile.path}</code></div>
            <dl className="dataset-notes">
              <div>
                <dt>Source and version</dt>
                <dd>{selectedDataset.profile.source_summary || selectedDataset.profile.lineage}</dd>
              </div>
              <div>
                <dt>Acquisition method</dt>
                <dd>{selectedDataset.profile.acquisition}</dd>
              </div>
              <div>
                <dt>Purpose</dt>
                <dd>{selectedDataset.profile.intended_use}</dd>
              </div>
              <div>
                <dt>Usage notes</dt>
                <dd>{selectedDataset.profile.caution}</dd>
              </div>
              <div className="wide">
                <dt>Evidence page convention</dt>
                <dd>{selectedDataset.profile.page_numbering}</dd>
              </div>
              {!selectedDataset.profile.reviewable && (
                <div className="wide">
                  <dt>Edit permission</dt>
                  <dd>{selectedDataset.profile.read_only_reason || "This item is available only for viewing and traceability."}</dd>
                </div>
              )}
            </dl>
          </div>
          <div className="dataset-facts">
            <div><strong>{selectedDataset.papers}</strong><span>PDFs / records</span></div>
            <div><strong>{selectedDataset.qas}</strong><span>Total QA</span></div>
            <div><strong>{datasetStats?.avg_qas_per_paper}</strong><span>Average QA / PDF</span></div>
            <div><strong>{datasetStats?.short_answer}</strong><span>Short-answer items</span></div>
            <div><strong>{datasetStats?.multi_page_evidence}</strong><span>Multi-page evidence</span></div>
            <div><strong>{datasetStats?.unanswerable}</strong><span>Unanswerable items</span></div>
            <div><strong>{datasetStats?.with_evidence_items}</strong><span>Per-page evidence facts</span></div>
          </div>
          <details className="field-dictionary">
            <summary>View field definitions and coverage</summary>
            <div className="field-groups">
              <FieldTable title="PDF / paper fields" fields={selectedDataset.profile.paper_fields} />
              <FieldTable title="QA fields" fields={selectedDataset.profile.qa_fields} />
            </div>
          </details>
        </section>
      )}

      <section className="data-shell">
        <aside className="dataset-sidebar">
            <p className="eyebrow">DATASETS</p>
            <h2>Local QA Data</h2>
            <p className="dataset-catalog-summary">{datasets.length} QA datasets synchronized · {datasets.reduce((total, dataset) => total + dataset.qas, 0)} QA items total</p>
          <div className="dataset-list">
            {datasetCollections.map((collection) => (
              collection.id === "other" && !showOtherDatasets ? null : (
              <section className={`dataset-collection ${collection.id === "final_2200" ? "primary" : ""}`} key={collection.id}>
                <div className="dataset-collection-heading">
                  <strong>{collection.title}</strong>
                  <span>{collection.datasets.length} JSON files · {collection.qas} QA</span>
                </div>
                {collection.datasets.map((dataset) => (
                  <button
                    className={`dataset-choice ${dataset.id === datasetId ? "active" : ""}`}
                    key={dataset.id}
                    disabled={editing}
                    onClick={() => { setSelectedDataset(null); setDatasetId(dataset.id); setIndex(0); setMessage(""); }}
                  >
                    <strong title={dataset.id}>{dataset.profile.title}</strong>
                    <span>{dataset.papers} PDF · {dataset.qas} QA · {humanSize(dataset.size_bytes)} · {dataset.profile.reviewable ? "Editable" : "Read only"}</span>
                  </button>
                ))}
              </section>)
            ))}
            <button
              className="dataset-history-toggle"
              type="button"
              onClick={() => { if (!otherDatasetsLoaded) void loadOtherDatasets(); else setShowOtherDatasets((visible) => !visible); }}
              disabled={otherDatasetsLoading}
              aria-expanded={showOtherDatasets}
            >
              {showOtherDatasets ? <ChevronUp size={16} /> : <ChevronDown size={16} />}
              <span>{otherDatasetsLoading ? "Loading other data..." : showOtherDatasets ? "Hide other data and historical versions" : "Show other data and historical versions"}</span>
              {!otherDatasetsLoaded && <small>Historical JSON files are scanned only after expansion</small>}
            </button>
          </div>
        </aside>

        <article className="review-card">
          <div className="review-toolbar">
            <button disabled={!current || editing || index <= 0} onClick={() => setIndex((value) => Math.max(0, value - 1))}>← Previous</button>
            <form className="qa-jump" onSubmit={jumpToQa}>
              <input
                aria-label="Jump to QA index"
                disabled={!current || editing}
                inputMode="numeric"
                min={1}
                max={current?.total || 1}
                type="number"
                value={jumpIndex}
                onChange={(event) => setJumpIndex(event.target.value)}
              />
              <span>/ {current?.total || "—"}</span>
              <button title="Jump to QA item" aria-label="Jump to QA item" type="submit" disabled={!current || editing}>
                <CornerDownRight size={15} />
              </button>
            </form>
            <button disabled={!current || editing || index >= current.total - 1} onClick={() => setIndex((value) => value + 1)}>Next →</button>
          </div>
          <div className="review-status-line">
            <span>Reviewed {current?.reviewed || 0}</span>
            <span>Kept {current?.kept || 0}</span>
            <span>Deleted {current?.deleted || 0}</span>
            <span>Edited {current?.edited || 0}</span>
            <span>Current: {current?.item.review_status || "—"}</span>
          </div>
          <div className="review-progress"><i style={{ width: `${Math.min(100, progress)}%` }} /></div>

          {current && !current.can_modify && (
            <div className="permission-warning">
              <LockKeyhole size={17} />
              <div><strong>View only</strong><span>This QA item is not assigned to the current account. Keep, edit, delete, and undo operations will be rejected.</span></div>
            </div>
          )}
          {current?.assignment && (
            <div className="assignment-note">
              {current.assignment.selection_mode === "stable_members"
                ? `This account owns ${current.assignment.assigned_count} migrated stable items. Access is locked by global QA ID.`
                : `This account owns indexes ${current.assignment.start_index}-${current.assignment.end_index} in this file, ${current.assignment.assigned_count} items total.`}
            </div>
          )}

          {current && (
            <div className="qa-scroll">
              <div className="qa-identity">{current.item.paper_id} / {current.item.qa_id}</div>
              <section className="qa-block">
                <div className="qa-block-heading">
                  <h3>Question</h3>
                  <button title="Edit question" aria-label="Edit question" onClick={() => beginEdit("question")}><Pencil size={14} /></button>
                </div>
                {editing ? (
                  <textarea autoFocus={editTarget === "question"} className="qa-edit-textarea" value={draftQuestion} onChange={(event) => setDraftQuestion(event.target.value)} />
                ) : <p>{current.item.qa.question}</p>}
              </section>
              <section className="qa-block">
                <div className="qa-block-heading">
                  <h3>Gold Answer</h3>
                  <button title="Edit gold answer" aria-label="Edit gold answer" onClick={() => beginEdit("answer")}><Pencil size={14} /></button>
                </div>
                {editing ? (
                  <textarea autoFocus={editTarget === "answer"} className="qa-edit-textarea answer" value={draftAnswer} onChange={(event) => setDraftAnswer(event.target.value)} />
                ) : <p>{current.item.qa.answer}</p>}
              </section>
              <section className="qa-block">
                <div className="qa-block-heading">
                  <h3>Physical Evidence Pages</h3>
                  <button title="Edit physical evidence pages" aria-label="Edit physical evidence pages" onClick={() => beginEdit("evidence")}><Pencil size={14} /></button>
                </div>
                {editing ? (
                  <div className="evidence-page-editor">
                    {draftEvidencePages.map((page) => (
                      <div className="evidence-edit-chip" key={page.id} onDoubleClick={() => setEditingEvidenceId(page.id)}>
                        <button className="evidence-remove" title="Remove evidence page" aria-label="Remove evidence page" onClick={() => removeEvidencePage(page.id)}><X size={11} /></button>
                        {editingEvidenceId === page.id ? (
                          <input
                            autoFocus
                            aria-label="Physical evidence page number"
                            min={1}
                            type="number"
                            value={page.value}
                            onBlur={finishEvidencePageEdit}
                            onChange={(event) => setDraftEvidencePages((pages) => pages.map((candidate) => candidate.id === page.id ? { ...candidate, value: event.target.value } : candidate))}
                            onKeyDown={(event) => { if (event.key === "Enter") finishEvidencePageEdit(); }}
                          />
                        ) : (
                          <button className="evidence-page-value" title="Click to open the PDF page; double-click to edit the number" onClick={() => jumpToPdfPage(Number(page.value))}>Page {page.value || "—"}</button>
                        )}
                      </div>
                    ))}
                    <button className="evidence-add" title="Add evidence page" aria-label="Add evidence page" onClick={addEvidencePage}><Plus size={15} /></button>
                  </div>
                ) : (
                  <div className="evidence-pages">
                    {evidencePages.map((page) => <button key={page} onClick={() => jumpToPdfPage(page)}>Page {page}</button>)}
                  </div>
                )}
                <div className="qa-meta">
                  <div><span>Page convention</span><strong>1-based physical PDF pages</strong></div>
                  <div><span>Modalities</span><strong>{current.item.qa.modal_types?.join(", ") || "Not specified"}</strong></div>
                  <div><span>Evidence hops</span><strong>{current.item.qa.evidence_hops || evidencePages.length}</strong></div>
                  <div><span>PDF</span><strong>{current.item.pdf_available ? `Showing page ${pdfPage}` : "No matching file found"}</strong></div>
                </div>
              </section>
              {current.item.qa.evidence_items?.length ? (
                <section className="qa-block">
                  <h3>Evidence Facts</h3>
                  {current.item.qa.evidence_items.map((evidence, position) => (
                    <p key={position}>
                      Doc {evidence.source_doc_number} · physical page {evidence.physical_pdf_page}
                      {evidence.source_page ? ` · source page ${evidence.source_page}` : ""}: {evidence.supported_fact}
                    </p>
                  ))}
                </section>
              ) : null}
              <div className="audit-drawer">
                <p className="eyebrow">RECENT ACTION LOG</p>
                {events.map((event) => (
                  <div className="audit-row" key={event.event_id}>
                    <b>{event.action === "keep" ? "Kept" : event.action === "edit" ? "Edited" : "Deleted"}</b>
                    <span>{event.actor_username || "Historical / system"} · {event.paper_id}/{event.qa_id}</span>
                    <span>{event.undone_at ? `Undone by ${event.undone_by_username || "system"}` : new Date(event.created_at * 1000).toLocaleTimeString()}</span>
                  </div>
                ))}
              </div>
            </div>
          )}

          <div className="review-actions">
            {editing ? (
              <>
                <button disabled={busy} onClick={cancelEdit}><X size={16} />Cancel edit</button>
                <button className="save-edit" disabled={busy} onClick={saveEdit}><Save size={16} />Save changes</button>
              </>
            ) : (
              <>
                <button className="keep" aria-disabled={!!current && !current.can_modify} disabled={!current || busy} onClick={() => review("keep")}>Keep and next</button>
                <button className="delete" aria-disabled={!!current && !current.can_modify} disabled={!current || busy} onClick={() => review("delete")}>Delete and save snapshot</button>
              </>
            )}
          </div>
        </article>

        <section className="pdf-card">
          <div className="pdf-heading">
            <div><p className="eyebrow">SOURCE PDF</p><h2>{current?.item.paper_id || "PDF Reference"}</h2></div>
            <span>Physical page {pdfPage} · click an evidence page to navigate</span>
          </div>
          {pdfUrl ? <iframe key={pdfNavigationKey} loading="lazy" title="PDF for the current QA item" src={pdfUrl} /> : <div className="empty">No PDF was found for the current item.</div>}
        </section>
      </section>
    </main>
  );
}

function FieldTable({ title, fields }: { title: string; fields: DatasetField[] }) {
  return (
    <section className="field-group">
      <h3>{title}</h3>
      <div className="field-table">
        {fields.map((field) => (
          <div className="field-row" key={field.name}>
            <code>{field.name}</code>
            <p>{field.description}</p>
            <span>{field.value_types.join(" / ")}</span>
            <b>{field.present} / {field.total} · {field.coverage_percent}%</b>
          </div>
        ))}
      </div>
    </section>
  );
}
