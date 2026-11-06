// WebXR viewer for an MQ3DRecon web location (specs/WEB_LOCATION_EXPORT.md).
// In VR: point a controller and pull the trigger to aim, release to teleport (upward-facing surfaces only);
// push a thumbstick left/right to snap-turn 30°; squeeze a grip to switch splat <-> mesh.
// On desktop: drag to orbit, scroll to zoom, M to switch splat <-> mesh.
import * as THREE from 'three';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { VRButton } from 'three/addons/webxr/VRButton.js';
import { XRControllerModelFactory } from 'three/addons/webxr/XRControllerModelFactory.js';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

const EYE_HEIGHT = 1.6;
const SNAP_TURN = THREE.MathUtils.degToRad(30);
const statusEl = document.getElementById('status');
const setStatus = (text) => { statusEl.textContent = text; };

const id = new URLSearchParams(location.search).get('loc');
if (!id) throw new Error('missing ?loc=');
const base = `/locations/${encodeURIComponent(id)}/`;
const loc = await (await fetch(base + 'location.json', { cache: 'no-store' })).json();
document.title = loc.name;
document.getElementById('title').textContent = loc.name;

// ---------------------------------------------------------------- scene
const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.xr.enabled = true;
renderer.xr.setReferenceSpaceType('local-floor');
document.body.appendChild(renderer.domElement);
document.body.appendChild(VRButton.createButton(renderer));

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x101014);
const camera = new THREE.PerspectiveCamera(70, window.innerWidth / window.innerHeight, 0.05, 500);

const spawn = new THREE.Vector3(...loc.spawn.position);
const spawnYaw = THREE.MathUtils.degToRad(loc.spawn.yaw_deg);
const forward = new THREE.Vector3(0, 0, -1).applyAxisAngle(new THREE.Vector3(0, 1, 0), spawnYaw);
camera.position.copy(spawn).add(new THREE.Vector3(0, EYE_HEIGHT, 0));
const controls = new OrbitControls(camera, renderer.domElement);
controls.target.copy(camera.position).addScaledVector(forward, 1.0);
controls.update();

const loader = new GLTFLoader();
const colliders = [];
setStatus('Loading model…');
const [visual, collider] = await Promise.all([
  loader.loadAsync(base + loc.assets.visual),
  loader.loadAsync(base + loc.assets.collider),
]);
visual.scene.traverse((node) => {
  if (!node.isMesh) return;
  const map = node.material && node.material.map;  // textured (baked) locations keep their texture, unlit
  node.material = map
    ? new THREE.MeshBasicMaterial({ map, side: THREE.DoubleSide })
    : new THREE.MeshBasicMaterial({ vertexColors: true, side: THREE.DoubleSide });
});
scene.add(visual.scene);

// Gaussian splat (photoreal), when the location has one: shown by default, mesh kept as a toggle.
let splat = null;
if (loc.assets.splat) {
  setStatus('Loading splats…');
  const { SparkRenderer, SplatMesh } = await import('@sparkjsdev/spark');
  scene.add(new SparkRenderer({ renderer }));
  splat = new SplatMesh({ url: base + loc.assets.splat });  // already Y-up (no COLMAP flip)
  scene.add(splat);
  visual.scene.visible = false;
}
function toggleRepresentation() {
  if (!splat) return;
  splat.visible = !splat.visible;
  visual.scene.visible = !splat.visible;
}
window.addEventListener('keydown', (event) => { if (event.key === 'm' || event.key === 'M') toggleRepresentation(); });

collider.scene.traverse((node) => { if (node.isMesh) { node.visible = false; colliders.push(node); } });
scene.add(collider.scene);

// Invisible floor at floor_y so teleport always has ground, even where the scan missed the floor.
const [minX, , minZ] = loc.bounds.min;
const [maxX, , maxZ] = loc.bounds.max;
const floor = new THREE.Mesh(
  new THREE.PlaneGeometry(maxX - minX + 4, maxZ - minZ + 4).rotateX(-Math.PI / 2),
  new THREE.MeshBasicMaterial({ visible: false }),
);
floor.position.set((minX + maxX) / 2, loc.floor_y, (minZ + maxZ) / 2);
scene.add(floor);
colliders.push(floor);
setStatus(`${splat ? `${(loc.stats.splats / 1000).toFixed(0)}k splats (M: mesh) · ` : ''}${(loc.stats.visual_triangles / 1000).toFixed(0)}k triangles · ${loc.stats.textured ? `${loc.stats.texture_size}px texture` : 'vertex colors'} · floor: ${loc.floor_source ?? 'mesh'}`);

// ---------------------------------------------------------------- VR rig: world_from_base transform
const rig = new THREE.Matrix4();
let baseReferenceSpace = null;

function applyRig() {
  if (!baseReferenceSpace) return;
  const inverse = rig.clone().invert();
  const position = new THREE.Vector3();
  const quaternion = new THREE.Quaternion();
  inverse.decompose(position, quaternion, new THREE.Vector3());
  const offset = new XRRigidTransform(
    { x: position.x, y: position.y, z: position.z, w: 1 },
    { x: quaternion.x, y: quaternion.y, z: quaternion.z, w: quaternion.w },
  );
  renderer.xr.setReferenceSpace(baseReferenceSpace.getOffsetReferenceSpace(offset));
}

renderer.xr.addEventListener('sessionstart', () => {
  baseReferenceSpace = renderer.xr.getReferenceSpace();
  rig.compose(new THREE.Vector3(spawn.x, loc.floor_y, spawn.z),
    new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(0, 1, 0), spawnYaw), new THREE.Vector3(1, 1, 1));
  applyRig();
});

function headWorldPosition() {
  const xrCamera = renderer.xr.getCamera();
  return new THREE.Vector3().setFromMatrixPosition(xrCamera.matrixWorld);
}

function teleportTo(point) {
  // Move the user's head (horizontally) onto the target and their floor (vertically) onto its height.
  // The rig is yaw + translation, so its translation's y is the current world height of the user's floor.
  const head = headWorldPosition();
  const floorY = new THREE.Vector3().setFromMatrixPosition(rig).y;
  rig.premultiply(new THREE.Matrix4().makeTranslation(point.x - head.x, point.y - floorY, point.z - head.z));
  applyRig();
}

function snapTurn(angle) {
  const head = headWorldPosition();
  const turn = new THREE.Matrix4().makeTranslation(head.x, 0, head.z)
    .multiply(new THREE.Matrix4().makeRotationY(angle))
    .multiply(new THREE.Matrix4().makeTranslation(-head.x, 0, -head.z));
  rig.premultiply(turn);
  applyRig();
}

// ---------------------------------------------------------------- controllers
const raycaster = new THREE.Raycaster();
const marker = new THREE.Mesh(
  new THREE.RingGeometry(0.15, 0.2, 32).rotateX(-Math.PI / 2),
  new THREE.MeshBasicMaterial({ color: 0x66ff99, transparent: true, opacity: 0.85 }),
);
marker.visible = false;
scene.add(marker);
const modelFactory = new XRControllerModelFactory();
const controllers = [0, 1].map((index) => {
  const controller = renderer.xr.getController(index);
  controller.userData = { aiming: false, target: null, turnArmed: true };
  const pointer = new THREE.Line(
    new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(0, 0, 0), new THREE.Vector3(0, 0, -1)]),
    new THREE.LineBasicMaterial({ color: 0xffffff }),
  );
  pointer.scale.z = 5;
  controller.add(pointer);
  controller.addEventListener('selectstart', () => { controller.userData.aiming = true; });
  controller.addEventListener('selectend', () => {
    controller.userData.aiming = false;
    if (controller.userData.target) teleportTo(controller.userData.target);
    controller.userData.target = null;
    marker.visible = false;
  });
  controller.addEventListener('connected', (event) => { controller.userData.inputSource = event.data; });
  controller.addEventListener('squeezestart', toggleRepresentation);
  scene.add(controller);
  const grip = renderer.xr.getControllerGrip(index);
  grip.add(modelFactory.createControllerModel(grip));
  scene.add(grip);
  return controller;
});

const normalMatrix = new THREE.Matrix3();
function updateAim(controller) {
  const origin = new THREE.Vector3().setFromMatrixPosition(controller.matrixWorld);
  const direction = new THREE.Vector3(0, 0, -1).transformDirection(controller.matrixWorld);
  raycaster.set(origin, direction);
  const hit = raycaster.intersectObjects(colliders, false)[0];
  controller.userData.target = null;
  if (!hit || !hit.face) return;
  normalMatrix.getNormalMatrix(hit.object.matrixWorld);
  const normal = hit.face.normal.clone().applyMatrix3(normalMatrix).normalize();
  if (Math.abs(normal.y) < 0.6) return;  // walls are not walkable
  controller.userData.target = hit.point.clone();
  marker.position.copy(hit.point).add(new THREE.Vector3(0, 0.01, 0));
  marker.visible = true;
}

function updateSnapTurn(controller) {
  const gamepad = controller.userData.inputSource?.gamepad;
  if (!gamepad || gamepad.axes.length < 4) return;
  const x = gamepad.axes[2];
  if (controller.userData.turnArmed && Math.abs(x) > 0.7) {
    snapTurn(x > 0 ? -SNAP_TURN : SNAP_TURN);
    controller.userData.turnArmed = false;
  } else if (Math.abs(x) < 0.3) {
    controller.userData.turnArmed = true;
  }
}

// ---------------------------------------------------------------- loop
window.addEventListener('resize', () => {
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
});

renderer.setAnimationLoop(() => {
  if (renderer.xr.isPresenting) {
    let aiming = false;
    for (const controller of controllers) {
      if (controller.userData.aiming) { updateAim(controller); aiming = true; }
      updateSnapTurn(controller);
    }
    if (!aiming) marker.visible = false;
  } else {
    controls.update();
  }
  renderer.render(scene, camera);
});

// Debug handle (inspect from the browser console).
window.mq3drecon = { scene, camera, renderer, controls, visual, splat, loc };
