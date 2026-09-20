import { access, readFile, readdir, writeFile } from "node:fs/promises";
import { join, resolve } from "node:path";

// vinext caches absolute disk paths in font CSS. Rebase a moved cache before
// vinext converts those paths into public /assets/_vinext_fonts/ URLs.
export async function rebaseFontCache(root) {
  const cache = resolve(root, ".vinext/fonts");
  let entries;
  try {
    entries = await readdir(cache, { withFileTypes: true });
  } catch (error) {
    if (error.code === "ENOENT") return;
    throw error;
  }
  for (const entry of entries) {
    if (!entry.isDirectory()) continue;
    const cssPath = join(cache, entry.name, "style.css");
    let css;
    try {
      css = await readFile(cssPath, "utf8");
    } catch (error) {
      if (error.code === "ENOENT") continue;
      throw error;
    }
    const replacements = new Map();
    for (const match of css.matchAll(/url\(["']?([^\s)"']+)["']?\)/g)) {
      const marker = `/.vinext/fonts/${entry.name}/`;
      const index = match[1].lastIndexOf(marker);
      if (index < 0) continue;
      const filename = match[1].slice(index + marker.length);
      if (!filename || /[/\\]/.test(filename) || filename === "..") continue;
      const asset = join(cache, entry.name, filename);
      await access(asset);
      replacements.set(match[0], `url(${JSON.stringify(asset)})`);
    }
    let updated = css;
    for (const [before, after] of replacements) updated = updated.split(before).join(after);
    if (updated !== css) await writeFile(cssPath, updated);
  }
}
