import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { GLTFLoader } from "three/addons/loaders/GLTFLoader.js";
import { OBJLoader } from "three/addons/loaders/OBJLoader.js";

const $ = (s, root=document) => root.querySelector(s);
const $$ = (s, root=document) => [...root.querySelectorAll(s)];
const status = (message) => { $("#statusMessage").textContent = message; };
let toastTimer;
function toast(message) {
  const el = $("#toast"); el.textContent = message; el.classList.add("show");
  clearTimeout(toastTimer); toastTimer = setTimeout(() => el.classList.remove("show"), 2300);
}

// -----------------------------------------------------------------------------
// Three.js viewer core. Replace/extend this module boundary to integrate a
// domain-specific renderer while keeping the workspace UI unchanged.
// -----------------------------------------------------------------------------
const mount = $("#threeViewport");
const scene = new THREE.Scene();
scene.background = new THREE.Color("#0c1521");
scene.fog = new THREE.Fog("#0c1521", 18, 42);
const camera = new THREE.PerspectiveCamera(42, 1, 0.1, 1000);
camera.position.set(7.5, 5.5, 9.5);
const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false, powerPreference: "high-performance" });
renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
renderer.outputColorSpace = THREE.SRGBColorSpace;
renderer.toneMapping = THREE.ACESFilmicToneMapping;
renderer.toneMappingExposure = 1.15;
mount.appendChild(renderer.domElement);
const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true; controls.dampingFactor = 0.06; controls.target.set(0,0,0);
scene.add(new THREE.HemisphereLight(0xbad8ff, 0x263247, 2.1));
const keyLight = new THREE.DirectionalLight(0xffffff, 2.7); keyLight.position.set(7,10,8); scene.add(keyLight);
const rimLight = new THREE.DirectionalLight(0x5bd7df, 1.4); rimLight.position.set(-8,2,-7); scene.add(rimLight);
const grid = new THREE.GridHelper(18, 18, 0x36506a, 0x203246); grid.position.y = -2.15; scene.add(grid);
const axes = new THREE.AxesHelper(2.4); axes.position.set(-4.5,-2.1,-4.5); scene.add(axes);
const modelRoot = new THREE.Group(); modelRoot.name = "Model root"; scene.add(modelRoot);
let loadedModel = null;
let frameCount = 0, lastFpsTime = performance.now();
const clock = new THREE.Clock();

function resizeRenderer() {
  const w = Math.max(1, mount.clientWidth), h = Math.max(1, mount.clientHeight);
  camera.aspect = w/h; camera.updateProjectionMatrix(); renderer.setSize(w,h,false);
}
const resizeObserver = new ResizeObserver(resizeRenderer); resizeObserver.observe(mount);
resizeRenderer();

// A generic procedural demo object: a connected set of nodes and links.
// This is deliberately not a molecular renderer; it demonstrates the generic
// Three.js scene, camera, lights, object hierarchy and selection foundation.
function buildDemoScene() {
  modelRoot.clear();
  const group = new THREE.Group(); group.name = "Demo network";
  const points = [];
  const n = 38;
  for (let i=0;i<n;i++) {
    const t = i/(n-1)*Math.PI*5.2;
    const radius = 2.15 + 0.52*Math.sin(i*0.57);
    points.push(new THREE.Vector3(Math.cos(t)*radius, (i/(n-1)-.5)*5.3, Math.sin(t)*radius));
  }
  const nodeGeo = new THREE.SphereGeometry(0.105, 18, 14);
  const nodeMatA = new THREE.MeshStandardMaterial({color:0x67d4dc, roughness:.3, metalness:.12, emissive:0x12383b, emissiveIntensity:.35});
  const nodeMatB = new THREE.MeshStandardMaterial({color:0xf0b879, roughness:.36, metalness:.08, emissive:0x382817, emissiveIntensity:.28});
  points.forEach((p,i)=>{
    const node = new THREE.Mesh(nodeGeo, i%6===0 ? nodeMatB : nodeMatA);
    node.position.copy(p); node.scale.setScalar(i%6===0?1.28:1); node.userData.selectable=true; group.add(node);
  });
  const linkMat = new THREE.MeshStandardMaterial({color:0x7894aa, roughness:.5, metalness:.15, transparent:true, opacity:.72});
  for(let i=0;i<points.length-1;i++){
    const a=points[i], b=points[i+1], mid=a.clone().add(b).multiplyScalar(.5);
    const dir=b.clone().sub(a); const len=dir.length();
    const link=new THREE.Mesh(new THREE.CylinderGeometry(.025,.025,len,8),linkMat);
    link.position.copy(mid); link.quaternion.setFromUnitVectors(new THREE.Vector3(0,1,0),dir.normalize()); group.add(link);
    if(i%3===0 && i+4<points.length){
      const c=points[i+4], d=c.clone().sub(a), l=d.length();
      const cross=new THREE.Mesh(new THREE.CylinderGeometry(.012,.012,l,6),new THREE.MeshStandardMaterial({color:0x426276,transparent:true,opacity:.42}));
      cross.position.copy(a.clone().add(c).multiplyScalar(.5)); cross.quaternion.setFromUnitVectors(new THREE.Vector3(0,1,0),d.normalize()); group.add(cross);
    }
  }
  const ring = new THREE.Mesh(new THREE.TorusGeometry(3.5,.012,6,120),new THREE.MeshBasicMaterial({color:0x3b6474,transparent:true,opacity:.38}));
  ring.rotation.x=Math.PI/2; ring.position.y=-1.2; group.add(ring);
  modelRoot.add(group); loadedModel=group; fitCamera(group);
  $("#objectCount").textContent = String(group.children.length);
}
function fitCamera(object) {
  const box = new THREE.Box3().setFromObject(object);
  if(box.isEmpty()) return;
  const sphere = box.getBoundingSphere(new THREE.Sphere());
  const distance = Math.max(5, sphere.radius*2.6);
  controls.target.copy(sphere.center);
  camera.position.copy(sphere.center).add(new THREE.Vector3(distance*.78,distance*.55,distance));
  camera.near=Math.max(.01,distance/100); camera.far=distance*100; camera.updateProjectionMatrix();
  controls.update();
}
buildDemoScene();

function animate() {
  requestAnimationFrame(animate);
  controls.update(); renderer.render(scene,camera);
  frameCount++;
  const now=performance.now();
  if(now-lastFpsTime>1000){ $("#fpsValue").textContent=String(Math.round(frameCount*1000/(now-lastFpsTime))); frameCount=0; lastFpsTime=now; }
}
animate();

const raycaster = new THREE.Raycaster(), pointer = new THREE.Vector2();
renderer.domElement.addEventListener("click", (event)=>{
  const rect=renderer.domElement.getBoundingClientRect();
  pointer.x=((event.clientX-rect.left)/rect.width)*2-1;
  pointer.y=-((event.clientY-rect.top)/rect.height)*2+1;
  raycaster.setFromCamera(pointer,camera);
  const hits=raycaster.intersectObject(modelRoot,true);
  if(hits.length){ const obj=hits[0].object; if(obj.userData.selectable){ toast(`Selected scene node · ${obj.id}`); status(`Selected object ${obj.id}`); } }
});

function loadModelFile(file) {
  const ext=file.name.split(".").pop().toLowerCase();
  const url=URL.createObjectURL(file);
  const finish=(object)=>{
    if(loadedModel) modelRoot.remove(loadedModel);
    object.name=file.name; loadedModel=object; modelRoot.add(object);
    // Normalize imported object to a useful view scale, without modifying source geometry.
    const box=new THREE.Box3().setFromObject(object), size=box.getSize(new THREE.Vector3());
    const maxDim=Math.max(size.x,size.y,size.z)||1;
    object.scale.multiplyScalar(6/maxDim);
    fitCamera(object);
    $("#sceneStatus").textContent="LOCAL MODEL";
    $("#modelLabel").textContent=file.name.toUpperCase().slice(0,28);
    $("#objectCount").textContent=String(object.children.length||1);
    status(`Loaded ${file.name}`); toast(`Loaded ${file.name}`);
    URL.revokeObjectURL(url);
  };
  const fail=(err)=>{console.error(err);toast(`Could not load ${file.name}`);status("Model load failed");URL.revokeObjectURL(url);};
  if(ext==="obj") new OBJLoader().load(url,finish,undefined,fail);
  else if(ext==="gltf"||ext==="glb") new GLTFLoader().load(url,gltf=>finish(gltf.scene),undefined,fail);
  else {toast("Supported 3D formats: GLB, GLTF, OBJ");URL.revokeObjectURL(url);}
}
$("#modelInput").addEventListener("change",e=>[...e.target.files].forEach(loadModelFile));
$("#openProjectBtn").addEventListener("click",()=>$("#modelInput").click());
$("#resetCamera").addEventListener("click",()=>{if(loadedModel)fitCamera(loadedModel);toast("Camera reset")});
$("#toggleGrid").addEventListener("click",e=>{grid.visible=!grid.visible;e.currentTarget.classList.toggle("active",grid.visible)});
$("#toggleAxes").addEventListener("click",e=>{axes.visible=!axes.visible;e.currentTarget.classList.toggle("active",axes.visible)});

// Drag/drop model support.
const viewport=$("#viewportWrap"), dropHint=$("#dropHint");
["dragenter","dragover"].forEach(name=>viewport.addEventListener(name,e=>{e.preventDefault();dropHint.classList.add("visible")}));
["dragleave","drop"].forEach(name=>viewport.addEventListener(name,e=>{e.preventDefault();dropHint.classList.remove("visible")}));
viewport.addEventListener("drop",e=>[...(e.dataTransfer.files||[])].forEach(loadModelFile));

// Pane resizing and maximize controls.
const splitter=$("#splitter"), workspace=$("#workspace"), leftPane=$("#visualPane"), rightPane=$("#dataPane");
let dragging=false;
splitter.addEventListener("pointerdown",e=>{dragging=true;splitter.classList.add("dragging");splitter.setPointerCapture(e.pointerId)});
splitter.addEventListener("pointermove",e=>{
  if(!dragging)return;
  const rect=workspace.getBoundingClientRect(); let pct=(e.clientX-rect.left)/rect.width*100;
  pct=Math.max(28,Math.min(72,pct)); leftPane.style.flex=`0 0 ${pct}%`; leftPane.style.width=`${pct}%`; resizeRenderer();
});
function stopDrag(){dragging=false;splitter.classList.remove("dragging")}
splitter.addEventListener("pointerup",stopDrag); splitter.addEventListener("pointercancel",stopDrag);
splitter.addEventListener("keydown",e=>{if(e.key==="ArrowLeft"||e.key==="ArrowRight"){e.preventDefault();let w=parseFloat(leftPane.style.width)||55;w+=e.key==="ArrowRight"?2:-2;w=Math.max(28,Math.min(72,w));leftPane.style.flex=`0 0 ${w}%`;leftPane.style.width=`${w}%`;resizeRenderer()}});
function maximize(pane){const already=pane.classList.contains("maximized");leftPane.classList.remove("maximized");rightPane.classList.remove("maximized");workspace.classList.remove("has-max");if(!already){pane.classList.add("maximized");workspace.classList.add("has-max")}setTimeout(resizeRenderer,240)}
$("#maximize3d").addEventListener("click",()=>maximize(leftPane));$("#maximizeData").addEventListener("click",()=>maximize(rightPane));

// Data tabs, tables, plots.
$$(".data-tab").forEach(btn=>btn.addEventListener("click",()=>switchTab(btn.dataset.tab)));
$$("[data-switch-tab]").forEach(btn=>btn.addEventListener("click",()=>switchTab(btn.dataset.switchTab)));
function switchTab(name){$$(".data-tab").forEach(b=>b.classList.toggle("active",b.dataset.tab===name));$$(".tab-panel").forEach(p=>p.classList.toggle("active",p.id===`tab-${name}`));if(name==="plot")drawPlots()}
const rows=[
 ["GLY 12","A",1.42,.94],["TYR 18","A",2.86,.89],["SER 24","A",3.14,.91],["TRP 31","A",1.08,.97],
 ["ASP 45","A",4.21,.82],["ARG 52","B",2.03,.93],["PHE 61","B",1.77,.96],["ASN 68","B",3.66,.87],
 ["LEU 74","B",2.51,.90],["GLU 89","B",4.02,.84],["TYR 97","A",1.31,.98],["LYS 104","A",3.28,.88]
].map((r,i)=>({residue:r[0],chain:r[1],depth:r[2],score:r[3],index:i}));
let sortKey="residue",sortAsc=true;
function renderTable(filter=""){
  const filtered=rows.filter(r=>`${r.residue} ${r.chain}`.toLowerCase().includes(filter.toLowerCase()));
  filtered.sort((a,b)=>{let av=a[sortKey],bv=b[sortKey];return (typeof av==="number"?av-bv:String(av).localeCompare(String(bv)))*(sortAsc?1:-1)});
  $("#metricsBody").innerHTML=filtered.map(r=>`<tr><td>${r.residue}</td><td>${r.chain}</td><td>${r.depth.toFixed(2)}</td><td><span class="score-pill">${r.score.toFixed(2)}</span></td><td><span class="status-pill">● PASS</span></td></tr>`).join("");
}
renderTable();
$("#tableSearch").addEventListener("input",e=>renderTable(e.target.value));
$$("th[data-sort]").forEach(th=>th.addEventListener("click",()=>{const key=th.dataset.sort;sortAsc=sortKey===key?!sortAsc:true;sortKey=key;renderTable($("#tableSearch").value)}));
$("#exportCsv").addEventListener("click",()=>{
  const csv=["residue,chain,depth_angstrom,score,status",...rows.map(r=>`${r.residue},${r.chain},${r.depth},${r.score},PASS`)].join("\n");
  downloadBlob(new Blob([csv],{type:"text/csv"}),"residue_metrics.csv");toast("CSV exported");
});
function downloadBlob(blob,name){const a=document.createElement("a");const u=URL.createObjectURL(blob);a.href=u;a.download=name;a.click();URL.revokeObjectURL(u)}
function drawPlots(){
  const canvas=$("#plotCanvas"),ctx=canvas.getContext("2d"),rect=canvas.getBoundingClientRect(),dpr=window.devicePixelRatio||1;
  canvas.width=Math.max(1,rect.width*dpr);canvas.height=185*dpr;ctx.scale(dpr,dpr);
  const w=rect.width,h=185;ctx.clearRect(0,0,w,h);
  const pad={l:28,r:8,t:12,b:23},pw=w-pad.l-pad.r,ph=h-pad.t-pad.b;
  ctx.strokeStyle="#29394b";ctx.lineWidth=1;ctx.fillStyle="#71869c";ctx.font="9px DM Mono, monospace";
  for(let i=0;i<=4;i++){let y=pad.t+ph*i/4;ctx.beginPath();ctx.moveTo(pad.l,y);ctx.lineTo(w-pad.r,y);ctx.stroke();ctx.fillText((4-i).toFixed(0),4,y+3)}
  ctx.beginPath();rows.forEach((r,i)=>{const x=pad.l+i/(rows.length-1)*pw,y=pad.t+ph-(r.depth/5)*ph;i?ctx.lineTo(x,y):ctx.moveTo(x,y)});ctx.strokeStyle="#67d4dc";ctx.lineWidth=2;ctx.stroke();
  rows.forEach((r,i)=>{const x=pad.l+i/(rows.length-1)*pw,y=pad.t+ph-(r.depth/5)*ph;ctx.beginPath();ctx.arc(x,y,2.5,0,Math.PI*2);ctx.fillStyle="#a1f0ed";ctx.fill()});
  ctx.fillStyle="#71869c";ctx.fillText("1",pad.l,h-5);ctx.fillText(String(rows.length),w-15,h-5);
  const bins=[0,0,0,0,0];rows.forEach(r=>bins[Math.min(4,Math.floor(r.score*5))]++);
  $("#barChart").innerHTML=bins.map((v,i)=>`<span style="height:${Math.max(10,v/Math.max(...bins)*76)}px" data-label="${(i*.2).toFixed(1)}"></span>`).join("");
}
window.addEventListener("resize",()=>{resizeRenderer();if($("#tab-plot").classList.contains("active"))drawPlots()});

// Image gallery and preview.
const imageRecords=[
 {name:"Surface rendering",src:"https://images.unsplash.com/photo-1530210124550-912dc1381cb8?w=1000&q=90"},
 {name:"Density map",src:"https://images.unsplash.com/photo-1532094349884-543bc11b234d?w=1000&q=90"},
 {name:"Microscopy",src:"https://images.unsplash.com/photo-1579154204601-01588f351e67?w=1000&q=90"}
];
function addImageRecord(name,src){
  imageRecords.push({name,src});
  const card=document.createElement("button");card.className="gallery-card";card.dataset.preview=name;
  const img=document.createElement("img");img.src=src;img.alt=name;
  const span=document.createElement("span");span.textContent=name;
  card.append(img,span);$("#imageGallery").append(card);
  const overview=document.createElement("button");overview.className="image-card";overview.dataset.preview=name;
  const oi=document.createElement("img");oi.src=src;oi.alt=name;
  const cap=document.createElement("span");cap.className="image-card-caption";cap.innerHTML=`<b></b><small>LOCAL IMAGE</small>`;cap.querySelector("b").textContent=name;
  overview.append(oi,cap);$("#overviewImages").append(overview);
  $("#imageCount").textContent=String(imageRecords.length).padStart(2,"0");
}
function showPreview(name){
  const rec=imageRecords.find(r=>r.name===name)||imageRecords[0];if(!rec)return;
  $("#previewTitle").textContent=rec.name;$("#previewImage").src=rec.src;$("#previewModal").classList.add("open");
}
document.addEventListener("click",e=>{const card=e.target.closest("[data-preview]");if(card)showPreview(card.dataset.preview)});
$("#closePreview").addEventListener("click",()=>$("#previewModal").classList.remove("open"));
$("#previewModal").addEventListener("click",e=>{if(e.target.id==="previewModal")$("#previewModal").classList.remove("open")});
$("#uploadImagesBtn").addEventListener("click",()=>$("#imageInput").click());
$("#imageDrop").addEventListener("click",()=>$("#imageInput").click());
$("#imageInput").addEventListener("change",e=>[...e.target.files].forEach(file=>{if(!file.type.startsWith("image/"))return;addImageRecord(file.name,URL.createObjectURL(file));status(`Added image ${file.name}`)}));
$("#imageDrop").addEventListener("dragover",e=>{e.preventDefault();$("#imageDrop").style.borderColor="var(--cyan)"});
$("#imageDrop").addEventListener("dragleave",()=>$("#imageDrop").style.borderColor="");
$("#imageDrop").addEventListener("drop",e=>{e.preventDefault();$("#imageDrop").style.borderColor="";[...e.dataTransfer.files].forEach(file=>{if(file.type.startsWith("image/"))addImageRecord(file.name,URL.createObjectURL(file))})});

// Contextual settings drawers.
const drawer=$("#settingsDrawer"),backdrop=$("#drawerBackdrop"),drawerContent=$("#drawerContent");
const toolContent={
 scene:{eyebrow:"VIEWPORT SETTINGS",title:"Scene settings",html:`<div class="setting-group"><div class="setting-title">SCENE</div><label class="setting-row">Background <input type="color" id="bgColor" value="#0c1521"></label><label class="setting-row">Show grid <input class="toggle" type="checkbox" id="gridToggle" checked></label><label class="setting-row">Show axes <input class="toggle" type="checkbox" id="axesToggle" checked></label></div><div class="setting-group"><div class="setting-title">CAMERA</div><label class="setting-row">Field of view <input type="range" min="25" max="80" value="42" id="fovRange"></label><label class="setting-row">Damping <input class="toggle" type="checkbox" id="dampingToggle" checked></label></div><button class="drawer-action" id="drawerReset">RESET VIEW</button>`},
 layers:{eyebrow:"SCENE HIERARCHY",title:"Layers",html:`<div class="setting-group"><div class="setting-title">OBJECT TREE</div><label class="setting-row">Model root <input class="toggle" type="checkbox" checked id="modelVisible"></label><label class="setting-row">Reference grid <input class="toggle" type="checkbox" checked id="gridVisible"></label><label class="setting-row">World axes <input class="toggle" type="checkbox" checked id="axesVisible"></label></div><p class="muted-note">Imported objects appear under Model root. Extend this panel to expose your renderer's own scene graph.</p>`},
 appearance:{eyebrow:"RENDERING",title:"Appearance",html:`<div class="setting-group"><div class="setting-title">RENDER MODE</div><label class="setting-row">Tone mapping <select id="toneSelect"><option value="aces">ACES Filmic</option><option value="linear">Linear</option><option value="none">No tone mapping</option></select></label><label class="setting-row">Exposure <input type="range" min=".5" max="2" step=".05" value="1.15" id="exposureRange"></label><label class="setting-row">Antialiasing <span class="positive">ENABLED</span></label></div><div class="setting-group"><div class="setting-title">LIGHTING</div><label class="setting-row">Key light <input class="toggle" type="checkbox" checked id="keyLightToggle"></label><label class="setting-row">Rim light <input class="toggle" type="checkbox" checked id="rimLightToggle"></label></div>`},
 measure:{eyebrow:"TOOLS",title:"Measurement",html:`<div class="setting-group"><div class="setting-title">MEASUREMENT TOOLS</div><p class="muted-note">The generic viewer currently supports object picking. Add domain-specific measurement logic through the viewer adapter in <code>app.js</code>.</p><button class="drawer-action" id="pickMode">ENABLE PICKING</button></div>`},
 import:{eyebrow:"ASSET IMPORT",title:"Import model",html:`<div class="setting-group"><div class="setting-title">SUPPORTED FORMATS</div><p class="muted-note">GLB, GLTF and OBJ files are supported in this starter implementation.</p><button class="drawer-action" id="drawerImport">CHOOSE 3D FILE</button></div><div class="setting-group"><div class="setting-title">DRAG AND DROP</div><p class="muted-note">Drop a model directly into the 3D viewport. For additional formats, add a loader or connect your own visualization engine.</p></div>`},
 settings:{eyebrow:"ENGINE",title:"Viewer settings",html:`<div class="setting-group"><div class="setting-title">RENDERER INFORMATION</div><p class="muted-note">Three.js WebGLRenderer · ACES Filmic tone mapping · OrbitControls · device pixel ratio capped at 2.</p></div><div class="setting-group"><div class="setting-title">INTEGRATION POINT</div><p class="muted-note">The scene, camera, renderer and animation loop are isolated near the top of <code>src/app.js</code>. Replace the demo scene builder with your own scene adapter.</p></div>`},
 inspector:{eyebrow:"DATA WORKSPACE",title:"Inspector",html:`<p class="muted-note">Use the tabs to inspect images, tabular metrics and plots. This is a UI shell with illustrative sample data.</p>`},
 filters:{eyebrow:"DATA WORKSPACE",title:"Filters",html:`<div class="setting-group"><div class="setting-title">TABLE FILTERS</div><label class="setting-row">Minimum score <input type="range" min="0" max="1" step=".01" value="0" id="minScore"></label><p class="muted-note">The table's text filter and sorting are active. Wire additional domain filters to your dataset here.</p></div>`},
 visuals:{eyebrow:"DATA WORKSPACE",title:"Visualization options",html:`<div class="setting-group"><div class="setting-title">PLOT OPTIONS</div><p class="muted-note">The example plot is drawn with Canvas 2D. Replace it with Plotly, ECharts or your preferred plotting library when connecting real analysis output.</p></div>`},
 export:{eyebrow:"DATA WORKSPACE",title:"Export data",html:`<div class="setting-group"><div class="setting-title">EXPORT</div><p class="muted-note">Export the current example residue metrics as CSV.</p><button class="drawer-action" id="drawerExport">EXPORT CSV</button></div>`},
 "data-settings":{eyebrow:"DATA WORKSPACE",title:"Data settings",html:`<div class="setting-group"><div class="setting-title">DATA SOURCE</div><p class="muted-note">This starter runs entirely in the browser. Connect your own API, local file parser or analysis service in the data adapter layer.</p></div>`}
};
function openDrawer(key){
  const cfg=toolContent[key]||toolContent.scene;
  $("#drawerEyebrow").textContent=cfg.eyebrow;$("#drawerTitle").textContent=cfg.title;drawerContent.innerHTML=cfg.html;
  drawer.classList.add("open");backdrop.classList.add("open");drawer.setAttribute("aria-hidden","false");
  bindDrawerControls();
}
function closeDrawer(){drawer.classList.remove("open");backdrop.classList.remove("open");drawer.setAttribute("aria-hidden","true")}
function bindDrawerControls(){
  const bind=(id,event,fn)=>{const el=$("#"+id);if(el)el.addEventListener(event,fn)};
  bind("bgColor","input",e=>scene.background.set(e.target.value));
  bind("gridToggle","change",e=>grid.visible=e.target.checked);
  bind("axesToggle","change",e=>axes.visible=e.target.checked);
  bind("gridVisible","change",e=>grid.visible=e.target.checked);
  bind("axesVisible","change",e=>axes.visible=e.target.checked);
  bind("modelVisible","change",e=>modelRoot.visible=e.target.checked);
  bind("fovRange","input",e=>{camera.fov=+e.target.value;camera.updateProjectionMatrix()});
  bind("dampingToggle","change",e=>controls.enableDamping=e.target.checked);
  bind("exposureRange","input",e=>renderer.toneMappingExposure=+e.target.value);
  bind("toneSelect","change",e=>{renderer.toneMapping=e.target.value==="aces"?THREE.ACESFilmicToneMapping:e.target.value==="linear"?THREE.LinearToneMapping:THREE.NoToneMapping});
  bind("keyLightToggle","change",e=>keyLight.visible=e.target.checked);
  bind("rimLightToggle","change",e=>rimLight.visible=e.target.checked);
  bind("drawerReset","click",()=>{if(loadedModel)fitCamera(loadedModel);camera.fov=42;camera.updateProjectionMatrix();toast("View reset")});
  bind("drawerImport","click",()=>$("#modelInput").click());
  bind("drawerExport","click",()=>$("#exportCsv").click());
  bind("pickMode","click",()=>toast("Click a selectable object in the viewport"));
}
$$("[data-tool]").forEach(btn=>btn.addEventListener("click",()=>{ $$(".tool-rail [data-tool]").forEach(b=>b.classList.toggle("selected",b===btn));openDrawer(btn.dataset.tool)}));
$$("[data-data-tool]").forEach(btn=>btn.addEventListener("click",()=>{$$(".data-rail [data-data-tool]").forEach(b=>b.classList.toggle("selected",b===btn));openDrawer(btn.dataset.dataTool)}));
$("#closeDrawer").addEventListener("click",closeDrawer);backdrop.addEventListener("click",closeDrawer);
$("#helpBtn").addEventListener("click",()=>toast("Drag to orbit · Scroll to zoom · Right-drag to pan · Drag divider to resize"));
document.addEventListener("keydown",e=>{if(e.key==="Escape"){closeDrawer();$("#previewModal").classList.remove("open")}});
function updateClock(){ $("#clock").textContent=new Date().toLocaleTimeString([], {hour12:false}); }
updateClock();setInterval(updateClock,1000);
$("#rendererLabel").textContent=renderer.capabilities.isWebGL2?"WEBGL 2":"WEBGL";
status("Workspace initialized");
