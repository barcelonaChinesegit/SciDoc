import assert from "node:assert/strict";
import { access, mkdir, mkdtemp, readFile, readdir, rename, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import test from "node:test";
import { rebaseFontCache } from "../build/font-cache.mjs";

const HAN = /[\u3400-\u9fff\uf900-\ufaff]/u;

async function render(path = "/") {
  const workerUrl = new URL("../dist/server/index.js", import.meta.url);
  workerUrl.searchParams.set("test", `${process.pid}-${Date.now()}`);
  const { default: worker } = await import(workerUrl.href);
  return worker.fetch(new Request(`http://localhost${path}`, { headers: { accept: "text/html" } }), {
    ASSETS: { fetch: async () => new Response("Not found", { status: 404 }) },
  }, { waitUntil() {}, passThroughOnException() {} });
}

async function sourceFiles(directory) {
  const entries = await readdir(directory, { withFileTypes: true });
  const files = await Promise.all(entries.map((entry) => {
    const url = new URL(`${entry.name}${entry.isDirectory() ? "/" : ""}`, directory);
    return entry.isDirectory() ? sourceFiles(url) : [url];
  }));
  return files.flat();
}

test("server-renders the English publication review shell", async () => {
  const response = await render();
  assert.equal(response.status, 200);
  assert.match(response.headers.get("content-type") ?? "", /^text\/html\b/i);
  const html = await response.text();
  assert.match(html, /<title>PKU QA Publication Review Console<\/title>/i);
  assert.match(html, /Checking your current session/);
  assert.match(html, /Sign in to the review console/);
  assert.doesNotMatch(html, HAN);
  assert.doesNotMatch(html, /Your site is taking shape/);
});

test("serves font URLs from the build instead of filesystem paths", async () => {
  const html = await (await render()).text();
  const fonts = [...html.matchAll(/url\(["']?([^\s)"']+\.woff2)["']?\)/g)];
  assert.ok(fonts.length > 0, "expected self-hosted fonts in rendered HTML");
  for (const [, url] of fonts) {
    assert.match(url, /^\/assets\/_vinext_fonts\//);
    await access(new URL(`../dist/client${url}`, import.meta.url));
  }
});

test("rebases a copied font cache after a project directory move", async () => {
  const temporary = await mkdtemp(join(tmpdir(), "scidoc-fonts-"));
  try {
    const oldRoot = join(temporary, "original");
    const newRoot = join(temporary, "relocated");
    const relative = ".vinext/fonts/geist-example";
    await mkdir(join(oldRoot, relative), { recursive: true });
    await writeFile(join(oldRoot, relative, "font.woff2"), "font fixture");
    await writeFile(join(oldRoot, relative, "style.css"),
      `@font-face { src: url(${join(oldRoot, relative, "font.woff2")}); }`);
    await rename(oldRoot, newRoot);
    await rebaseFontCache(newRoot);
    const css = await readFile(join(newRoot, relative, "style.css"), "utf8");
    assert.ok(!css.includes(oldRoot));
    assert.ok(css.includes(join(newRoot, relative, "font.woff2")));
    await rebaseFontCache(newRoot);
    assert.equal(await readFile(join(newRoot, relative, "style.css"), "utf8"), css);
    await rebaseFontCache(join(temporary, "fresh-clone"));
  } finally {
    await rm(temporary, { recursive: true, force: true });
  }
});

test("does not expose the removed paper learning route", async () => {
  const response = await render("/paper-learning");
  assert.equal(response.status, 404);
});

test("keeps all application source free of Chinese characters", async () => {
  const files = await sourceFiles(new URL("../app/", import.meta.url));
  for (const file of files.filter((url) => /\.(?:ts|tsx|css|mjs)$/.test(url.pathname))) {
    const source = await readFile(file, "utf8");
    assert.doesNotMatch(source, HAN, `${file.pathname} contains Chinese characters`);
  }
});

test("provides English account, review, guide, and administration flows", async () => {
  const [page, login, register, loginScreen, profile, users, guide, nav, auth, layout, styles] = await Promise.all([
    readFile(new URL("../app/page.tsx", import.meta.url), "utf8"),
    readFile(new URL("../app/login/page.tsx", import.meta.url), "utf8"),
    readFile(new URL("../app/register/page.tsx", import.meta.url), "utf8"),
    readFile(new URL("../app/components/LoginScreen.tsx", import.meta.url), "utf8"),
    readFile(new URL("../app/profile/page.tsx", import.meta.url), "utf8"),
    readFile(new URL("../app/admin/users/page.tsx", import.meta.url), "utf8"),
    readFile(new URL("../app/guide/page.tsx", import.meta.url), "utf8"),
    readFile(new URL("../app/components/ConsoleNav.tsx", import.meta.url), "utf8"),
    readFile(new URL("../app/lib/auth.ts", import.meta.url), "utf8"),
    readFile(new URL("../app/layout.tsx", import.meta.url), "utf8"),
    readFile(new URL("../app/globals.css", import.meta.url), "utf8"),
  ]);
  assert.match(page, /Publication Review Console/);
  assert.match(page, /Four final JSON files/);
  assert.match(loginScreen, /2,200 QA · 4 RELEASE FILES/);
  assert.match(page, /Open QA Review/);
  assert.match(page, /Read the Reviewer Guide/);
  assert.ok(page.indexOf("Open QA Review") < page.indexOf("View the Task Queue"));
  assert.match(page, /useCurrentUser\(false\)/);
  assert.match(page, /<LoginScreen checkingOnly \/>/);
  assert.match(login, /<LoginScreen \/>/);
  assert.match(register, /initialMode="register"/);
  assert.match(loginScreen, /auth\/login/);
  assert.match(loginScreen, /auth\/register/);
  assert.match(loginScreen, /auth\/email-code/);
  assert.match(loginScreen, /Full name/);
  assert.match(loginScreen, /Verify and create account/);
  assert.match(users, /Review Assignments/);
  assert.match(users, /Reviewer Progress/);
  assert.match(users, /reviewer-progress/);
  assert.match(users, /\/data-api\/assignments/);
  assert.match(users, /Reset password/);
  assert.match(users, /role-toggles/);
  assert.match(users, /datasets\?collection=final_2200&detail=summary/);
  assert.match(profile, /Profile/);
  assert.match(profile, /\/data-api\/profile/);
  assert.match(profile, /New email verification code/);
  assert.match(guide, /QA Reviewer Guide/);
  assert.match(guide, /data\/qa\/7\.final_2200\/ordinary_qa\.json/);
  assert.match(guide, /Keep and next/);
  assert.match(guide, /Delete and save snapshot/);
  assert.match(guide, /Unanswerable/);
  assert.match(nav, /User Management/);
  assert.match(nav, /Profile/);
  assert.match(auth, /\/data-api\/auth\/me/);
  assert.match(layout, /title:\s*"PKU QA Publication Review Console"/);
  assert.match(layout, /<html lang="en">/);
  assert.match(styles, /\.portal-hero/);
  assert.match(styles, /\.guide-layout/);
});

test("preserves queue controls and protects the queue proxy", async () => {
  const [page, proxy, packageJson, styles] = await Promise.all([
    readFile(new URL("../app/queue/page.tsx", import.meta.url), "utf8"),
    readFile(new URL("../app/api/[...path]/route.ts", import.meta.url), "utf8"),
    readFile(new URL("../package.json", import.meta.url), "utf8"),
    readFile(new URL("../app/globals.css", import.meta.url), "utf8"),
  ]);
  assert.doesNotMatch(page, /http:\/\/127\.0\.0\.1:8765/);
  assert.match(page, /\/api\/tasks/);
  assert.match(page, /action\(task,\s*"move"/);
  assert.match(page, /task\.status === "failed" \? "retry" : "resume"/);
  assert.match(page, /method:\s*"DELETE"/);
  assert.match(page, /Completed Tasks/);
  assert.match(page, /Delete task results/);
  assert.match(page, /Restore record/);
  assert.match(page, /estimateTaskEta/);
  assert.match(page, /ETA_SAMPLE_LIMIT = 8/);
  assert.match(proxy, /http:\/\/127\.0\.0\.1:8765/);
  assert.match(proxy, /http:\/\/127\.0\.0\.1:8770/);
  assert.match(proxy, /x-pku-actor-username/);
  assert.match(proxy, /user\.roles\?\.includes\("admin"\)/);
  assert.match(styles, /\.task-eta/);
  assert.doesNotMatch(packageJson, /react-loading-skeleton/);
  await assert.rejects(access(new URL("../app/_sites-preview/SkeletonPreview.tsx", import.meta.url)));
});

test("preserves reversible dataset review with PDF evidence", async () => {
  const [page, proxy] = await Promise.all([
    readFile(new URL("../app/data/page.tsx", import.meta.url), "utf8"),
    readFile(new URL("../app/data-api/[...path]/route.ts", import.meta.url), "utf8"),
  ]);
  assert.match(page, /Datasets and QA Review/);
  assert.match(page, /1-based physical PDF pages/);
  assert.match(page, /Keep and next/);
  assert.match(page, /Delete and save snapshot/);
  assert.match(page, /Undo last action/);
  assert.match(page, /Edit question/);
  assert.match(page, /Save changes/);
  assert.match(page, /evidence_page_changes/);
  assert.match(page, /\/edit/);
  assert.match(page, /Jump to QA index/);
  assert.match(page, /View only/);
  assert.match(page, /current\.can_modify/);
  assert.match(page, /event\.can_undo/);
  assert.doesNotMatch(page, /with_options/);
  assert.match(page, /SELECTED DATASET · SUMMARY/);
  assert.match(page, /collection_id === "final_2200"/);
  assert.match(page, /function jumpToPdfPage\(page: number\)/);
  assert.match(page, /setPdfNavigationNonce\(\(value\) => value \+ 1\)/);
  assert.match(page, /key=\{pdfNavigationKey\}/);
  assert.match(page, /loading="lazy"/);
  assert.match(proxy, /http:\/\/127\.0\.0\.1:8770/);
  assert.match(proxy, /"content-range"/);
  assert.match(proxy, /if-none-match/);
  assert.match(proxy, /"set-cookie"/);
  assert.match(proxy, /External identity exchange is disabled/);
  assert.match(proxy, /export const DELETE = forward/);
});
