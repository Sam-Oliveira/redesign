#!/usr/bin/env python3
"""
Scans an Obsidian "Blog Posts" folder and syncs any note there into this
site's _posts/ as a Jekyll blog post, converting Obsidian-specific syntax
(wikilinks, image embeds, callouts) to plain Markdown/Liquid, then commits
and pushes.

Run manually whenever you want to publish: `python3 bin/publish_from_obsidian.py`.
Safe to run repeatedly: it tracks what it has already published in
.synced.json inside the watched folder, and only re-syncs a note if its
content actually changed.
"""
import hashlib
import json
import re
import subprocess
import sys
from datetime import date
from pathlib import Path

VAULT_DIR = Path.home() / "Documents/PhD/Research/Obsidian/Work"
WATCH_DIR = VAULT_DIR / "Blog Posts"
REPO_DIR = Path.home() / "Documents/PhD/Careers/sam-site"
POSTS_DIR = REPO_DIR / "_posts"
IMG_DIR = REPO_DIR / "assets/img/blog"
MANIFEST = WATCH_DIR / ".synced.json"
LOG_FILE = WATCH_DIR / ".sync.log"

WIKILINK_RE = re.compile(r"\[\[([^\]|]+)(?:\|([^\]]+))?\]\]")
EMBED_RE = re.compile(r"!\[\[([^\]]+)\]\]")
CALLOUT_RE = re.compile(r"^(>\s*)\[!(\w+)\]([+-]?)\s*(.*)$")
MATH_RE = re.compile(r"\$\$.*?\$\$|\$[^$\n]*?\$", re.DOTALL)


def log(msg):
    line = f"{date.today()} {msg}"
    print(line)
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")


def slugify(text):
    text = re.sub(r"[^a-zA-Z0-9\s-]", "", text).strip().lower()
    return re.sub(r"[\s_]+", "-", text)


def find_in_vault(filename):
    """Find an attachment by filename anywhere in the vault."""
    matches = list(VAULT_DIR.rglob(filename))
    return matches[0] if matches else None


def convert_wikilinks(text):
    def repl(m):
        target, display = m.group(1), m.group(2)
        return display if display else target
    return WIKILINK_RE.sub(repl, text)


def convert_embeds(text, slug):
    def repl(m):
        fname = m.group(1).strip()
        src = find_in_vault(fname)
        if src is None:
            log(f"  WARNING: embedded file not found in vault: {fname}")
            return f"*(missing image: {fname})*"
        dest_dir = IMG_DIR / slug
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / src.name
        dest.write_bytes(src.read_bytes())
        return f"![]({{{{ site.baseurl }}}}/assets/img/blog/{slug}/{src.name})"
    return EMBED_RE.sub(repl, text)


def convert_callouts(text):
    out_lines = []
    for line in text.split("\n"):
        m = CALLOUT_RE.match(line)
        if m:
            prefix, kind, _fold, title = m.groups()
            label = title.strip() if title.strip() else kind.capitalize()
            out_lines.append(f"{prefix}**{label}**")
        else:
            out_lines.append(line)
    return "\n".join(out_lines)


def convert_math_escapes(text):
    """Escape `_` and `|` inside $...$/$$...$$ math so kramdown's GFM parser
    doesn't mistake LaTeX subscripts for italics or table syntax (both
    characters are otherwise indistinguishable from Markdown emphasis and
    pipe-table delimiters to kramdown, which runs before MathJax)."""
    def repl(m):
        s = m.group(0)
        s = re.sub(r"(?<!\\)\|", r"\\|", s)
        s = re.sub(r"(?<!\\)_", r"\\_", s)
        return s
    return MATH_RE.sub(repl, text)


def load_manifest():
    if MANIFEST.exists():
        return json.loads(MANIFEST.read_text())
    return {}


def save_manifest(m):
    MANIFEST.write_text(json.dumps(m, indent=2))


def content_hash(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def build_post(note_path, manifest):
    raw = note_path.read_text(encoding="utf-8")
    h = content_hash(raw)
    key = note_path.name

    entry = manifest.get(key)
    if entry and entry["hash"] == h:
        return None  # unchanged, nothing to do

    title = note_path.stem
    if entry:
        post_date = entry["date"]
        slug = entry["slug"]
    else:
        post_date = date.today().isoformat()
        slug = slugify(title)

    body = raw
    # Strip a leading H1 matching the title (Obsidian convention); keep the rest.
    body = re.sub(rf"^#\s*{re.escape(title)}\s*\n+", "", body, count=1)
    body = convert_embeds(body, slug)
    body = convert_wikilinks(body)
    body = convert_callouts(body)
    body = convert_math_escapes(body)

    frontmatter = (
        "---\n"
        "layout: post\n"
        f"title: {json.dumps(title)}\n"
        f"date: {post_date} 09:00:00-0600\n"
        "description:\n"
        "tags:\n"
        "related_posts: false\n"
        "---\n\n"
    )

    post_filename = f"{post_date}-{slug}.md"
    (POSTS_DIR / post_filename).write_text(frontmatter + body.strip() + "\n")

    manifest[key] = {"hash": h, "date": post_date, "slug": slug, "post_filename": post_filename}
    return post_filename


def git(*args):
    return subprocess.run(["git", "-C", str(REPO_DIR), *args], capture_output=True, text=True)


def main():
    WATCH_DIR.mkdir(parents=True, exist_ok=True)
    POSTS_DIR.mkdir(parents=True, exist_ok=True)
    manifest = load_manifest()

    changed = []
    for note in WATCH_DIR.glob("*.md"):
        result = build_post(note, manifest)
        if result:
            changed.append(result)
            log(f"synced: {note.name} -> _posts/{result}")

    if not changed:
        return

    save_manifest(manifest)

    git("add", "-A")
    status = git("status", "--porcelain")
    if not status.stdout.strip():
        log("no git changes to commit (unexpected)")
        return

    msg = "Blog: " + ", ".join(changed)
    commit = git("commit", "-m", msg)
    if commit.returncode != 0:
        log(f"git commit failed: {commit.stderr}")
        return

    push = git("push")
    if push.returncode != 0:
        log(f"git push failed: {push.stderr}")
    else:
        log(f"pushed: {msg}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        log(f"ERROR: {e}")
        sys.exit(1)
