# Migrating sam-oliveira.github.io to al-folio

This folder is a full al-folio site with your real content already migrated in
from the old Hugo Blox site: bio, education, experience, all 7 projects (with
their original PDFs/slides), a CV page (from `assets/json/resume.json`), a
photography page (your real film photos), news items, and a first blog post.

Your **current live site is untouched** — nothing here affects it until you
do the final swap step below.

## 1. Create the staging repo and push

1. On GitHub, create a new **empty** repository named `redesign` under your
   account (Settings icon → New repository → no README/gitignore/license).
2. Unpack this folder wherever you keep your projects, then from inside it
   run (the delivered archive has no `.git` yet, to keep the download small):

   ```bash
   git init
   git add -A
   git commit -m "Initial al-folio site migrated from Hugo Blox Academic CV theme"
   git remote add origin https://github.com/sam-oliveira/redesign.git
   git branch -M main
   git push -u origin main
   ```

3. In the new repo's **Settings → Pages**, set "Build and deployment" →
   Source: **GitHub Actions** (al-folio ships its own deploy workflow at
   `.github/workflows/deploy.yml`, which will run automatically on push).
4. After the Action finishes (check the **Actions** tab), your staged site
   is live at:

   ```
   https://sam-oliveira.github.io/redesign/
   ```

   This is a completely separate URL from your real site, so you can poke at
   it, share the link with labmates, etc. without touching production.

## 2. Review and iterate

Things you'll likely want to do before going live:
- Read through every page for tone/accuracy — I wrote the bio/project copy
  based on your old site's content, but you know it best.
- Add a real Google Scholar ID to `_data/socials.yml` once you have one.
- Add real publications to `_bibliography/papers.bib` as BibTeX entries (see
  the commented example already in that file).
- Add more blog posts under `_posts/` as you write them.
- Swap `assets/img/prof_pic.jpg` for a higher-res photo if you'd like.
- The photography page pulls in 18 of your film photos from the old site's
  demo gallery folder — trim/reorder in `_pages/photography.md` as you like.

Push more commits to `main` any time — the Action redeploys automatically.

## 3. Go live at sam-oliveira.github.io (the swap)

GitHub Pages ties your **root** `https://sam-oliveira.github.io/` URL to
whichever repo is named exactly `sam-oliveira.github.io`. The cleanest way
to promote the redesign without downtime is to **rename repos** — renaming
a repo to `<username>.github.io` makes GitHub Pages serve it at the root
automatically, DNS/custom-domain settings included.

1. **Two config changes first**, in this `redesign` repo, then commit & push:
   - In `_config.yml`, change:
     ```yaml
     url: https://sam-oliveira.github.io
     baseurl: ""
     ```
     (currently `baseurl: /redesign` for staging — see the comment above
     those two lines in `_config.yml`).
   - Wait for the Action to redeploy and re-check `/redesign/` still works
     with the new baseurl locally before the final step (or just proceed —
     worst case you fix and repush before renaming).

2. **Archive the old repo** (don't delete — keep the history):
   - Go to `https://github.com/sam-oliveira/Sam-Oliveira.github.io` →
     Settings → rename it to something like `Sam-Oliveira.github.io-legacy`.
   - This immediately takes your OLD site offline at the root URL — do this
     right before step 3, not long before.

3. **Promote the new repo**:
   - Go to `https://github.com/sam-oliveira/redesign` → Settings → rename it
     to `Sam-Oliveira.github.io`.
   - GitHub Pages will automatically start serving it at
     `https://sam-oliveira.github.io/` within a few minutes.
   - If Pages doesn't pick it up immediately, check Settings → Pages in the
     renamed repo and re-save the GitHub Actions source.

4. Confirm the live site works, check a few internal links, and you're done.
   The old repo is preserved at its `-legacy` name if you ever want to look
   back at it or revert.

## Rollback

If something looks wrong after the swap, you can rename the two repos back
to their original names at any time — renames are instant and don't lose
any git history.
