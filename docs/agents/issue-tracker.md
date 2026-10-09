# Issue tracker: Local Markdown

Issues and specs for this repo live as markdown files in `.scratch/`.

## Conventions

- One feature per directory: `.scratch/<feature-slug>/`
- The spec is `.scratch/<feature-slug>/spec.md`
- Implementation issues are one file per ticket: `.scratch/<feature-slug>/issues/<NN>.md`
- `<NN>` is a zero-padded sequential number starting at `01` (`01`, `02`, `03`, …)
- The descriptive title lives inside the file as the first heading: `# <NN>: <title>`
- Never put the title in the filename
- The next `<NN>` in a feature directory is one greater than the highest existing number there. Never reuse a number in that directory, including numbers of tickets that already exist
- Do not rename or move existing `.scratch/` files. Whether a publish step creates a new file or edits an existing one is defined under publishing below
- Triage state for an ordinary implementation issue is a `Status:` line near the top of the issue file. The value is one of the five roles in `triage-labels.md`
- Comments and conversation history append to the bottom of the file under a `## Comments` heading

## Historical tickets

The naming, heading, field formatting, and status conventions specified here apply to newly created tickets. Historical tickets may have different formats, such as `# 1: ...`, `Status: complete`, `Status: accepted`, or bold Markdown fields. Preserve them unchanged unless the user explicitly requests migration. Do not interpret historical status values as Wayfinder lifecycle states. Existing implementation tickets must not enter the Wayfinder frontier merely because they share an effort directory.

## When a skill says "publish to the issue tracker"

Write only to the destination that matches the artifact:

- Specification: `.scratch/<feature-slug>/spec.md`
- Implementation ticket: `.scratch/<feature-slug>/issues/<NN>.md` (one ticket per file)
- Wayfinder map: `.scratch/<effort>/map.md`
- Wayfinder child ticket: `.scratch/<effort>/issues/<NN>.md`

Create the feature or effort directory when it does not already exist. Do not create a generic markdown file directly under `.scratch/` as a substitute for one of these paths.

When publishing a NEW specification, map, or ticket, never replace an existing file. If the target path already exists, stop and request clarification. However, existing files MAY be modified as explicitly required by the invoked skill or the user's authorized task: claiming a Wayfinder ticket, changing its lifecycle status, recording an answer, appending comments, updating the Wayfinder map's Decisions-so-far, or revising an existing specification. Preserve all unrelated content. Never replace an entire existing document just to update one field or section.

## When a skill says "fetch the relevant ticket"

Read the file at the referenced path. The user will normally pass the path or the issue number directly.

## Wayfinding operations

Used by `/wayfinder`. The map is one file. Each child ticket is its own file. Wayfinder lifecycle states are separate from the five triage roles used on ordinary implementation issues.

- **Map**: `.scratch/<effort>/map.md` (the Notes / Decisions-so-far / Fog body).
- **Child ticket**: `.scratch/<effort>/issues/<NN>.md`, numbered from `01` with the same zero-padded sequential rule as implementation tickets. The question is the body. The title is the first heading inside the file. A `Type:` line records the ticket type (`research` / `prototype` / `grilling` / `task`). A `Status:` line records the Wayfinder lifecycle.
- **Wayfinder status values.** These three are the only valid lifecycle values:
  - `Status: open` — newly created and available for claiming. Every new Wayfinder ticket starts here.
  - `Status: claimed` — currently being investigated.
  - `Status: resolved` — investigation completed.
- **Blocking**: a `Blocked by: NN, NN` line near the top. Use `Blocked by: None` when there are no dependencies. A dependency is satisfied only when that ticket's `Status` is `resolved`.
- **Frontier**: among Wayfinder child tickets in `.scratch/<effort>/issues/` whose `Status` is `open` and whose `Blocked by` dependencies are all `resolved`, the lowest number wins. Existing implementation tickets do not enter the frontier merely because they share the effort directory. Historical status values are not Wayfinder lifecycle states.
- **Claim**: allowed only when `Status` is `open` and every `Blocked by` dependency is `resolved`. Set `Status: claimed` and save before any work.
- **Resolve**: append the answer under an `## Answer` heading, set `Status: resolved`, then append a context pointer (gist + link) to the map's Decisions-so-far in `map.md`.
