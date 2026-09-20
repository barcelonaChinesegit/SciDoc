"use client";

import { AlertTriangle, ArrowLeftRight, BookOpenCheck, Check, FileJson, Pencil, ShieldCheck, Trash2, Undo2 } from "lucide-react";
import ConsoleNav from "../components/ConsoleNav";
import { useCurrentUser } from "../lib/auth";

const datasets = [
  ["Final ordinary QA", "data/qa/7.final_2200/ordinary_qa.json", "1000", "Answerable single-paper short-answer QA."],
  ["Final unanswerable QA", "data/qa/7.final_2200/unanswerable_qa.json", "200", "The answer is exactly Unanswerable and evidence pages are empty."],
  ["Final reasoning QA", "data/qa/7.final_2200/reasoning_qa.json", "200", "Multi-step reasoning within one paper."],
  ["Final cross-PDF QA", "data/qa/7.final_2200/cross_pdf_qa.json", "800", "Cross-paper QA using a merged z_cross_* PDF."],
];

const fields = [
  ["paper / paper_id", "Stable identifier for a source paper or merged PDF. It must match the PDF shown in the workspace."],
  ["QA / qa_id", "Stable QA mapping and item ID. Include dataset, paper_id, and qa_id when reporting an issue."],
  ["question", "The final question shown to a model. It must be complete, unambiguous, and free of answer leakage."],
  ["answer", "The single gold answer. Unanswerable items must use the exact spelling Unanswerable."],
  ["answer_format", "Integer, Float, String, List, or Unanswerable. It must match the actual answer type."],
  ["answer_aliases", "Equivalent accepted expressions. Aliases must never conceal an incorrect primary answer."],
  ["answer_unit", "Unit for a numeric answer. It must agree with both the question and the source."],
  ["numeric_tolerance", "Scoring tolerance used only when the source and task justify a precision range."],
  ["evidence_pages", "1-based physical pages in the PDF that support the answer, not printed footer page numbers."],
  ["evidence_items", "Per-fact cross-PDF evidence with a physical page, source document, and supported fact."],
  ["source_documents", "Source-paper metadata and page ranges inside a merged PDF."],
  ["source_paper_ids", "Unique source-paper IDs actually used by a cross-PDF item."],
  ["evidence_source_docs", "Source document numbers covered by gold evidence."],
  ["evidence_hops", "Necessary evidence or reasoning steps. Reasoning and cross-PDF items must not collapse to lookup."],
  ["modal_types", "Required modalities in canonical order: text, image, table, formula. They must match the evidence."],
  ["question_type / question_category", "Analysis labels that must agree with the question's actual logic."],
  ["annotation_provenance", "Generation, review, revision, and release history used for traceability."],
];

const topics = [
  ["task", "1. Current task"], ["boundaries", "2. Access boundaries"], ["files", "3. Release files"],
  ["assignment", "4. Your assignment"], ["screen", "5. Workspace map"], ["fields", "6. Field reference"],
  ["workflow", "7. Review workflow"], ["decisions", "8. Decisions"], ["special", "9. QA-specific rules"],
  ["reporting", "10. Escalation"], ["admin", "11. Administrator workflow"],
  ["troubleshooting", "12. Troubleshooting"], ["finish", "13. End-of-shift checks"],
];

export default function GuidePage() {
  const { user, loading } = useCurrentUser();
  if (loading || !user) return <main className="auth-loading">Opening the reviewer guide...</main>;
  return <main className="guide-page">
    <ConsoleNav user={user} active="guide" />
    <header className="guide-header"><div><p className="eyebrow">MANUAL REVIEW HANDBOOK · PUBLICATION GATE</p><h1>QA Reviewer Guide</h1><p>Read this guide before your first review session. Leave uncertain items unreviewed and report them.</p></div><a className="primary guide-start" href="/data"><BookOpenCheck size={17} />Start QA review</a></header>
    <aside className="guide-warning"><AlertTriangle size={20} /><div><strong>This review is the final publication gate</strong><p>The release contains exactly 2,200 short-answer items and no options. Editing and deletion change the canonical JSON after saving a complete reversible snapshot.</p></div></aside>
    <div className="guide-layout">
      <nav className="guide-toc"><p className="eyebrow">CONTENTS</p>{topics.map(([id, label]) => <a href={`#${id}`} key={id}>{label}</a>)}</nav>
      <article className="guide-content">
        <section id="task"><span className="guide-number">01</span><h2>Current Task</h2>
          <p>Your formal task is to review only the 2,200 items in the Final Publication Review collection. Historical datasets, candidate pools, model outputs, checkpoints, and task queue records are available for traceability, but they are outside this review.</p>
          <p>For every assigned item, verify the question, gold answer, physical evidence pages, and its ordinary, unanswerable, reasoning, or cross-PDF structure. Treat the PDF as the source of truth and the central QA panel as the proposed publication record.</p>
          <div className="guide-rule-grid"><div><Check size={17} /><strong>Correct</strong><span>Question, answer, evidence, and metadata agree</span></div><div><Check size={17} /><strong>Verifiable</strong><span>The gold answer follows from the current PDF</span></div><div><Check size={17} /><strong>Necessary</strong><span>Every reasoning step or source document is required</span></div><div><ShieldCheck size={17} /><strong>Auditable</strong><span>Every action is tied to your account and time</span></div></div>
        </section>
        <section id="boundaries"><span className="guide-number">02</span><h2>What You Can View and Change</h2>
          <div className="guide-callouts"><div><strong>You must review</strong><p>Your assigned stable QA members in the four final_2200 JSON files.</p></div><div><strong>You may view</strong><p>Other final items, historical datasets, and PDFs. Viewing never changes status.</p></div><div><strong>You may edit</strong><p>Only items assigned to your account. Stable membership does not shift after deletions.</p></div><div><strong>You may not edit</strong><p>Unassigned items, read-only process artifacts, or another reviewer&apos;s assignment.</p></div></div>
          <p>A visible item is not necessarily writable. Stop if the panel says View only. Record the dataset, paper_id, and qa_id if an assignment appears wrong, then ask an administrator to correct it. Never use another person&apos;s account.</p>
        </section>
        <section id="files"><span className="guide-number">03</span><h2>The Four Canonical JSON Files</h2>
          <div className="dataset-manual-table">{datasets.map(([title, name, count, purpose]) => <div key={name}><FileJson size={17} /><div><b>{title}</b><code>{name}</code></div><strong>{count} QA</strong><span>{purpose}</span></div>)}<div className="dataset-total"><span>Total</span><strong>2,200 QA</strong></div></div>
          <p>The files contain 1,000 ordinary, 200 unanswerable, 200 reasoning, and 800 cross-PDF items. Global IDs run continuously from QA0001 through QA2200. Stop and report the issue if these counts differ.</p>
        </section>
        <section id="assignment"><span className="guide-number">04</span><h2>Find Your Assignment</h2>
          <ol className="guide-long-steps"><li><strong>Confirm the file.</strong><p>Select the dataset named in your assignment and verify its relative path and QA count in the summary.</p></li><li><strong>Confirm the boundaries.</strong><p>Assignments use either a 1-based start and end index or a stable member set. Follow the range displayed by the application.</p></li><li><strong>Jump to the first item.</strong><p>Enter the QA list index, not a PDF page or qa_id, in the jump field.</p></li><li><strong>Check write access.</strong><p>Your first assigned item must not show View only. Refresh once, then report the stable identity if access is still missing.</p></li><li><strong>Stop at the boundary.</strong><p>Do not continue into another reviewer&apos;s range after completing your assignment.</p></li></ol>
        </section>
        <section id="screen"><span className="guide-number">05</span><h2>Workspace Map</h2>
          <div className="screen-map"><div><b>Navigation</b><p>Home, QA Review, Reviewer Guide, Task Queue, Profile, and User Management for administrators.</p></div><div><b>Dataset summary</b><p>Canonical title, paths, purpose, counts, editability, and field coverage for the selected JSON.</p></div><div><b>Dataset list</b><p>The final collection appears first. Expand Other data and historical versions only for reference.</p></div><div><b>Review controls</b><p>Previous, 1-based jump, next, status totals, assignment notice, and stable QA identity.</p></div><div><b>QA content</b><p>Question, gold answer, physical evidence pages, and metadata. Pencil buttons enter edit mode.</p></div><div><b>PDF viewer</b><p>Use evidence-page buttons to navigate. Always verify the displayed paper before judging evidence.</p></div></div>
        </section>
        <section id="fields"><span className="guide-number">06</span><h2>Field Reference</h2><ul className="guide-checklist">{fields.map(([name, meaning]) => <li key={name}><b>{name}</b><span>{meaning}</span></li>)}</ul></section>
        <section id="workflow"><span className="guide-number">07</span><h2>Four-Step Review Workflow</h2>
          <ol className="guide-long-steps"><li><strong>Read the question and answer.</strong><p>Restate the requested object, operation, conditions, dataset, metric, unit, and output form. Check pronouns, ambiguity, missing constraints, answer leakage, signs, decimal places, abbreviations, and whether every requested part is answered.</p></li><li><strong>Verify every evidence page.</strong><p>Open each physical page and read enough context to confirm entities, conditions, negation, table rows and columns, footnotes, figure legends and axes, or formula definitions. State exactly which answer fact each page supports.</p></li><li><strong>Test the claimed dependency.</strong><p>Remove each page mentally. For reasoning items, every hop must remain necessary. For cross-PDF items, no single source paper may answer the question alone. Do not bridge missing facts with outside knowledge.</p></li><li><strong>Choose and verify an action.</strong><p>Keep a fully correct item, edit a uniquely repairable item, delete an invalid item, or leave an uncertain item unreviewed and report it. Wait for the success message and new status before continuing.</p></li></ol>
          <aside className="guide-inline-warning"><AlertTriangle size={18} /><p>Before submitting, complete this sentence with evidence: “I am keeping, editing, or deleting this item because...” If you cannot, continue reviewing or escalate it.</p></aside>
        </section>
        <section id="decisions"><span className="guide-number">08</span><h2>Decision Controls</h2>
          <div className="decision-grid"><div className="decision keep"><Check size={20} /><h3>Keep and next</h3><p>Use when the question, answer, evidence, and dependency all pass.</p><small>Records kept status and audit metadata without changing QA content.</small></div><div className="decision edit"><Pencil size={20} /><h3>Edit and save</h3><p>Use for a uniquely repairable question, answer, unit, or physical page.</p><small>Saves a complete pre-edit snapshot and records edited status.</small></div><div className="decision delete"><Trash2 size={20} /><h3>Delete and save snapshot</h3><p>Use when the item cannot be repaired from the PDF without changing its core meaning.</p><small>Snapshots the complete JSON before removing the QA item.</small></div><div className="decision skip"><ArrowLeftRight size={20} /><h3>Previous, next, or jump</h3><p>Browse without changing review status. Use this path for uncertain items.</p><small>Report dataset, paper_id, and qa_id to the project owner.</small></div></div>
          <p>In edit mode, change question and answer text directly. Remove an evidence page with its corner control, edit its number, or add a page. Recheck the PDF before saving. Answerable items require evidence; Unanswerable requires an empty evidence list.</p><p><Undo2 size={15} /> Undo last action is a controlled recovery operation. It succeeds only when the current file hash still matches your latest write, preventing an undo from overwriting later work.</p>
        </section>
        <section id="special"><span className="guide-number">09</span><h2>Rules by QA Family</h2>
          <div className="guide-scenarios"><div><b>Ordinary answerable</b><p>The marked pages directly and uniquely support the complete answer. Check comparison direction, metric, experimental setting, and units.</p></div><div><b>Unanswerable</b><p>Search the likely sections and confirm the PDF lacks a necessary fact. The answer must be exactly Unanswerable and evidence_pages must be empty.</p></div><div><b>Reasoning</b><p>Write each evidence fact as a hop. Every hop must be necessary, and no single page may directly reveal the final answer.</p></div><div><b>Cross-PDF</b><p>At least two source papers must contribute indispensable facts connected by a valid relation. Parallel copying from unrelated papers does not qualify.</p></div></div>
          <p>Figures can support trends or approximate ranges, but they cannot justify unsupported precision. Tables require the correct title, row, column, split, and footnote. Formula evidence requires variable definitions, constraints, signs, and direction. Similar names or numbers elsewhere in a paper are not sufficient evidence.</p>
        </section>
        <section id="reporting"><span className="guide-number">10</span><h2>Skip and Escalate</h2><p>Leave an item unreviewed when the PDF fails to load, the source mapping is unclear, two repairs are plausible, another user changed the file, or you cannot establish the evidence chain. Send one compact report containing dataset ID, visible index, paper_id, qa_id, suspected issue, pages checked, and your proposed next step.</p><p>Do not mark uncertainty as Keep. Do not submit on behalf of another reviewer. Do not change gold answers to influence model accuracy.</p></section>
        <section id="admin"><span className="guide-number">11</span><h2>Administrator Workflow</h2>
          <ol className="guide-long-steps"><li><strong>Create individual accounts.</strong><p>Use real display names, unique usernames, strong initial passwords, and only the roles each person needs.</p></li><li><strong>Create non-overlapping assignments.</strong><p>Select the reviewer, canonical JSON file, and inclusive 1-based range. Verify the resulting stable member count.</p></li><li><strong>Monitor progress.</strong><p>Use Completed / Assigned and the kept, edited, and deleted breakdown. Review recent immutable audit events.</p></li><li><strong>Resolve access issues.</strong><p>Adjust assignments in User Management. Reset passwords rather than requesting or viewing an old password.</p></li></ol>
        </section>
        <section id="troubleshooting"><span className="guide-number">12</span><h2>Troubleshooting</h2>
          <ul className="guide-list"><li>A browser-native Basic Auth prompt means the public proxy is misconfigured. Report it instead of entering application credentials.</li><li>View only means the item is unassigned or the artifact is read-only. Confirm the stable identity and assignment.</li><li>If the PDF does not jump, click the evidence page again, then refresh and verify the PDF URL.</li><li>A file-change or hash-conflict message means another write occurred. Reload before making a new decision.</li><li>Do not repeatedly click a decision while the network is slow. Wait for the success message or reload the current state.</li><li>If Undo is rejected, a later write has changed the file. Escalate instead of trying to reconstruct it manually.</li></ul>
        </section>
        <section id="finish"><span className="guide-number">13</span><h2>End-of-Shift Checks</h2>
          <ul className="guide-checklist"><li><b>Coverage</b><span>Every assigned stable member is kept, edited, deleted, or explicitly reported as unreviewed.</span></li><li><b>Identity</b><span>Your account name and role were correct for the full session.</span></li><li><b>Counts</b><span>Your profile and administrator progress totals agree with the work you completed.</span></li><li><b>Escalations</b><span>Each unresolved item includes dataset, paper_id, qa_id, and the evidence checked.</span></li><li><b>Session</b><span>Sign out on any shared machine after finishing.</span></li></ul>
        </section>
      </article>
    </div>
  </main>;
}
