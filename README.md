# Matchwinners Animator – Frontend Suite

This repository now ships a fresh, purpose-built frontend for configuring Matchwinners animations. The legacy wizard and misc. UI assets have been removed; all authoring happens inside three focused tabs:

1. **Global Settings** (`/global-settings.html`) – canvas defaults, fonts, phases.
2. **Phase Configurator** (`/phase-configurator.html`) – per-phase segments, modules, transitions.
3. **Export** (`/export.html`) – preview the stitched project, export MP4, import/export JSON archives.

`SPEC.md` remains the canonical data-model reference that each UI consumes.

---

## Project Layout

```
animator/
├─ global-settings.html            # Vite entry (tab 1)
├─ phase-configurator.html         # Vite entry (tab 2)
├─ export.html                     # Vite entry (tab 3)
├─ phase-preview.html              # Lightweight iframe preview renderer
├─ src/
│  ├─ global-settings/             # Tab 1 UI logic + styles
│  ├─ phase-configurator/          # Tab 2 UI logic + styles
│  ├─ export/                      # Tab 3 UI logic + styles
│  ├─ shared/                      # Shared storage helpers (global settings, URLs, etc.)
│  ├─ core/, modules/, renderer/   # Rendering engine + module renderers
│  └─ exporter/, assets/, …        # Supporting libraries
├─ SPEC.md                         # Comprehensive spec (data model, renderer rules)
├─ vite.config.ts                  # Vite build config (only new tabs are bundled)
└─ package.json / tsconfig.*       # Tooling configuration
```

---

## Requirements

- Node.js 20+ (recommended via `nvm`)
- npm 9+

---

## Setup

```bash
npm install
```

Installs Vite, TypeScript, ESLint, and other dev dependencies.

---

## Development Workflow

Start the dev server:

```bash
npm run dev
```

Vite automatically opens `http://localhost:5173/global-settings.html`. Navigate to the other tabs manually:

- `http://localhost:5173/phase-configurator.html?phase=phase-1`
- `http://localhost:5173/export.html`

### Live Preview (Phase Configurator)

The configurator writes per-phase configs to `localStorage` (`mw-phase-config:<phaseId>`). Saving your phase enables the iframe preview inside the same tab and feeds the Export tab.

### Global Settings ↔ Phase Config

Both tabs share `localStorage` (`mw-global-settings`). Updating project name, fonts, or phases in Global Settings updates the tab strip everywhere. Use “Configure Phase” buttons to jump into a phase-specific configurator instance.

---

## Build

```bash
npm run build
```

- Runs `tsc`.
- Builds each Vite entry: `/global-settings.html`, `/phase-configurator.html`, `/export.html`, `/phase-preview.html`.

Output is written to `dist/`:

```
dist/
├─ global-settings.html
├─ phase-configurator.html
├─ export.html
├─ phase-preview.html
└─ assets/...
```

Deploy the entire `dist/` directory to your static host (S3, Vercel, etc.).

---

## Linting

```bash
npm run lint
```

ESLint rules cover TypeScript files inside `src/`.

---

## Data Model Quick Reference

The UI and renderer operate on a single `Project` schema (`SPEC.md`):

- **Global Settings**: canvas width/height/fps/background, font defaults.
- **Phases**: layout slots, segment groups, segments, tracks.
- **Segments**: module bindings per slot, transitions, background overrides.
- **Tracks**: multi-segment modules (snail ticker, persistent logos).
- **Export Settings**: width, height, fps, codec, bit depth, filename.

Both previews and the Export tab run the same renderer to ensure what you see locally matches the exported MP4.

---

## Import / Export

- **Save JSON (Export tab)** – downloads the entire project (global settings + per-phase configs + cached assets when possible).
- **Import JSON** – restores `localStorage` state and per-phase data. Use this to hand off projects between machines.
- **Export MP4** – POSTs to `/api/export` with the JSON payload (implement the backend to render headless).

---

## Legacy Scripts

The Python scripts (`brand_*.py`, `render_brand_sequence.py`) remain for reference/testing but are no longer part of the core frontend. They can ingest the same project JSON if you need CLI exports; see the respective README files (`README_brand_config.md`) for usage.

---

For detailed animation rules, transitions, and module parameters, read `SPEC.md`. For questions or contributions, open an issue or PR. Happy animating!
