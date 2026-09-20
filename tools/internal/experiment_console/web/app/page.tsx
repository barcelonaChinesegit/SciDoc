"use client";

import {
  ArrowRight,
  BookOpen,
  CheckCircle2,
  ClipboardCheck,
  ListChecks,
  ShieldCheck,
  Users,
} from "lucide-react";
import ConsoleNav from "./components/ConsoleNav";
import LoginScreen from "./components/LoginScreen";
import { hasRole, useCurrentUser } from "./lib/auth";

export default function PortalPage() {
  const { user, loading } = useCurrentUser(false);
  if (loading) return <LoginScreen checkingOnly />;
  if (!user) return <LoginScreen />;

  return (
    <main className="portal-page">
      <ConsoleNav user={user} active="home" />
      <section className="portal-hero">
        <div className="portal-hero-copy">
          <p className="eyebrow">PUBLICATION FINAL CHECK · 2,200 QA</p>
          <h1>Publication Review Console</h1>
          <p>
            Four final JSON files form one release batch. Review every question, answer,
            physical evidence page, and PDF. Every keep, delete, undo, and administrative
            action is attributed to the current account.
          </p>
          <div className="portal-state">
            <span><CheckCircle2 size={16} />4 dataset files</span>
            <span><ShieldCheck size={16} />Fully auditable</span>
            <span><Users size={16} />Individual reviewer accounts</span>
          </div>
        </div>
      </section>

      <section className="portal-destinations">
        <a className="destination primary-destination" href="/data">
          <span className="destination-icon"><ClipboardCheck size={25} /></span>
          <div>
            <p className="eyebrow">PRIMARY WORKSPACE</p>
            <h2>Open QA Review</h2>
            <p>Review all 2,200 final QA items with their PDFs, evidence pages, review status, and reversible snapshots.</p>
          </div>
          <ArrowRight size={22} />
        </a>
        <a className="destination" href="/guide">
          <span className="destination-icon"><BookOpen size={23} /></span>
          <div>
            <p className="eyebrow">START HERE</p>
            <h2>Read the Reviewer Guide</h2>
            <p>Files, fields, review criteria, controls, and complete workflow examples.</p>
          </div>
          <ArrowRight size={20} />
        </a>
        {hasRole(user, "admin") && <a className="destination" href="/queue">
          <span className="destination-icon"><ListChecks size={23} /></span>
          <div>
            <p className="eyebrow">ADMIN OPERATIONS</p>
            <h2>View the Task Queue</h2>
            <p>Internal experiment monitoring for maintainers.</p>
          </div>
          <ArrowRight size={20} />
        </a>}
      </section>
    </main>
  );
}
