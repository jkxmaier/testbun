# Atlas — Scientific Workbench

A standalone, dark scientific visualization workspace built with **Vanilla HTML, CSS and JavaScript**, using **Three.js** as a generic 3D/OpenGL-style rendering foundation.

## Features

- 60 px header and 20 px status footer
- Two-pane workspace with a draggable divider and pane maximize/restore
- Three.js scene, camera, lights, grid, axes and OrbitControls
- Demo 3D scene to verify the renderer is working
- GLB, GLTF and OBJ import (file picker or drag-and-drop)
- Contextual toolbars with settings drawers
- Image gallery, local image import and preview
- Searchable/sortable example data table and CSV export
- Canvas-based example plots
- No application build step and no backend

## Run locally

Three.js is loaded as ES modules from jsDelivr, so serve the folder over HTTP (opening `index.html` directly with `file://` is not sufficient).

### Option A: Python

```bash
cd science-workbench
python -m http.server 8000
```

Open <http://localhost:8000>.

### Option B: Node

```bash
npx serve .
```

Open the local URL printed by the server.

## Project structure

```text
science-workbench/
├── index.html
├── package.json
├── README.md
└── src/
    ├── app.js
    └── styles.css
```

## Three.js integration boundary

The Three.js setup is intentionally generic. It does not impose a molecular data model or a specific scientific representation.

In `src/app.js`, the viewer core is isolated under:

```js
// Three.js viewer core.
```

The main integration points are:

- `scene`: your scene graph
- `camera`: perspective camera
- `renderer`: WebGL renderer
- `controls`: OrbitControls (replace if your interaction model differs)
- `modelRoot`: container for imported or application-owned objects
- `buildDemoScene()`: demo geometry; replace this with your own scene construction
- `fitCamera(object)`: generic bounding-box camera framing
- `animate()`: render loop

A straightforward route for integrating your own visualizer is to expose a small adapter with methods such as:

```js
const viewerAdapter = {
  mount(container) {},
  loadData(data) {},
  setBackground(color) {},
  resetCamera() {},
  resize(width, height) {},
  dispose() {}
};
```

Then call the adapter from the existing toolbar and file/data workflows. Keep the workspace UI independent of renderer-specific object types.

## Import support

- `.glb`, `.gltf`: Three.js `GLTFLoader`
- `.obj`: Three.js `OBJLoader`

The loaders are imported from Three.js addons via the import map in `index.html`. For GLTF files that reference external texture/buffer files, prefer `.glb` or package the companion assets with resolvable relative paths.

## Notes

- The initial table, metrics, plots and demo scene are illustrative placeholder data.
- Image examples use remote Unsplash URLs; internet access is required for those example thumbnails. Local images can be added through the Images tab.
- Three.js itself is loaded from a CDN; internet access is required unless you vendor the Three.js distribution locally.
- This is a frontend starter, not a production data platform. Add your scientific parsers, persistence, backend/API and domain-specific analysis as needed.
