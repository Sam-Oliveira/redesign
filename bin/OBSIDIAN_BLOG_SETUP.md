# Obsidian → blog auto-publish

## One-time setup (run these in your Mac's own Terminal, not through Claude)

```bash
mkdir -p ~/Library/LaunchAgents
cp "/Users/sdolivei/Documents/PhD/Careers/sam-site/bin/com.samoliveira.blogsync.plist" ~/Library/LaunchAgents/
launchctl load -w ~/Library/LaunchAgents/com.samoliveira.blogsync.plist
```

That's it. From now on, every ~2 minutes, your Mac checks the
`Blog Posts` folder inside your Obsidian vault for new or edited notes and,
if it finds any, converts and pushes them automatically.

## How to publish a note

1. Write your note anywhere in Obsidian as usual.
2. When ready to publish, move (not copy) it into the vault's Blog Posts folder.
3. Wait ~2 minutes. It'll show up in `_posts/` in this repo, committed and
   pushed, and live at sam-oliveira.github.io/redesign/blog/ shortly after
   (via the existing GitHub Actions deploy).
4. Editing the note again while it's still in `Blog Posts` will trigger a
   re-sync/re-push of the same post (same URL/date), so you can fix typos
   after the fact.

## What gets converted automatically

- `[[Wikilink]]` and `[[Wikilink|Display text]]` → plain text (not linked,
  since there's nothing on the public site for it to link to).
- `![[image.png]]` → the image file is copied into
  `assets/img/blog/<post-slug>/` and the reference is rewritten to a normal
  Markdown image.
- `> [!note] Title` callouts → a bolded blockquote. Not fancy, but safe and
  readable; can be prettied up later with a proper custom-blockquote include
  if you want.
- A leading `# Note Title` heading matching the filename is stripped (Jekyll
  adds its own title from frontmatter).

## Debugging

- `Blog Posts/.sync.log` — the script's own log (what it converted/skipped/failed).
- `Blog Posts/.launchd.log` — stdout/stderr from launchd itself, if the
  script fails to run at all (e.g. wrong python3 path).
- `Blog Posts/.synced.json` — the manifest tracking what's already published
  and its assigned date/slug. Don't edit by hand unless you know what you're
  doing — deleting a note's entry here will make it treat that note as brand
  new (new date) on the next sync.

## To turn it off

```bash
launchctl unload ~/Library/LaunchAgents/com.samoliveira.blogsync.plist
```
