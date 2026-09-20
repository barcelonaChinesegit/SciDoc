"use client";

import {
  BookOpen,
  ClipboardCheck,
  Home,
  ListChecks,
  LogOut,
  UserCircle,
  Users,
} from "lucide-react";
import Link from "next/link";
import { CurrentUser, englishText, hasRole, logout } from "../lib/auth";

const entries = [
  { href: "/", label: "Home", icon: Home, id: "home" },
  { href: "/data", label: "QA Review", icon: ClipboardCheck, id: "review" },
  { href: "/guide", label: "Reviewer Guide", icon: BookOpen, id: "guide" },
  { href: "/queue", label: "Task Queue", icon: ListChecks, id: "queue" },
];

export default function ConsoleNav({
  user,
  active,
}: {
  user: CurrentUser;
  active: string;
}) {
  return (
    <nav className="console-nav" aria-label="Console navigation">
      <Link className="console-brand" href="/">
        <span>PKU</span>
        <strong>QA Final Review</strong>
      </Link>
      <div className="console-nav-links">
        {entries.map(({ href, label, icon: Icon, id }) => (
          <Link className={active === id ? "active" : ""} href={href} key={id}>
            <Icon size={16} strokeWidth={1.8} />
            <span>{label}</span>
          </Link>
        ))}
        {hasRole(user, "admin") && (
          <Link className={active === "users" ? "active" : ""} href="/admin/users">
            <Users size={16} strokeWidth={1.8} />
            <span>User Management</span>
          </Link>
        )}
        <Link className={active === "profile" ? "active" : ""} href="/profile">
          <UserCircle size={16} strokeWidth={1.8} />
          <span>Profile</span>
        </Link>
      </div>
      <div className="console-account">
        <span>
          <strong>{englishText(user.display_name, user.username)}</strong>
          <small>{user.username} · {user.roles.map((role) => role === "admin" ? "Administrator" : "Reviewer").join(" + ")}</small>
        </span>
        <button aria-label="Sign out" title="Sign out" onClick={logout}>
          <LogOut size={16} />
        </button>
      </div>
    </nav>
  );
}
