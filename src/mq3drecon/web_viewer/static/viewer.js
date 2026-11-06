// WebXR viewer for an MQ3DRecon web location (specs/WEB_LOCATION_EXPORT.md).
// In VR (movement in the style of Google Earth VR):
//   left thumbstick   glide where the left controller points (push forward/back, tilt sideways to strafe).
//                     Walls stop you (you slide along them), you walk up/down ledges up to 40 cm,
//                     and the floor follows you.
//   right thumbstick  left/right snap-turns 30°; up/down flies up/down; click lands you on the ground.
//                     The higher you fly, the faster you glide.
//   trigger           hold to aim, release to teleport (upward-facing surfaces only).
//   grip              switch splat <-> mesh, only with &mesh=1 in the URL. By default visitors see only the
//                     splat: the scanned mesh is the (invisible) collider and looks rough next to it.
//   A comfort vignette narrows the view while you glide or fly (add &vignette=0 to the URL to turn it off).
// On desktop: drag to orbit, scroll to zoom, WASD / arrow keys to move (Shift = faster, Q/E = down/up),
// M to switch splat <-> mesh.
import * as THREE from 'three';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { VRButton } from 'three/addons/webxr/VRButton.js';
import { XRControllerModelFactory } from 'three/addons/webxr/XRControllerModelFactory.js';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { MeshBVH, acceleratedRaycast } from 'three-mesh-bvh';

const EYE_HEIGHT = 1.6;
const SNAP_TURN = THREE.MathUtils.degToRad(30);
const MOVE_SPEED = 1.5;       // m/s at full stick, on the ground
const FLY_SPEED = 1.2;        // m/s up/down at full stick
const MAX_ALTITUDE = 30;      // m above the ground
const STEP_HEIGHT = 0.4;      // tallest ledge you can walk onto
const BODY_RADIUS = 0.25;     // how close you get to a wall
const DEAD_ZONE = 0.15;
const WALKABLE_NORMAL_Y = 0.6;
const UP = new THREE.Vector3(0, 1, 0);
const DOWN = new THREE.Vector3(0, -1, 0);

const params = new URLSearchParams(location.search);
const comfortVignette = params.get('vignette') !== '0';
const meshToggleInVr = params.get('mesh') === '1';  // grip switches splat <-> mesh only when asked for
const statusEl = document.getElementById('status');
const setStatus = (text) => { statusEl.textContent = text; };

const id = params.get('loc');
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
scene.add(camera);  // so head-locked children (the comfort vignette) render; in VR three.js moves it with the head

const spawn = new THREE.Vector3(...loc.spawn.position);
const spawnYaw = THREE.MathUtils.degToRad(loc.spawn.yaw_deg);
const spawnForward = new THREE.Vector3(0, 0, -1).applyAxisAngle(UP, spawnYaw);
camera.position.copy(spawn).add(new THREE.Vector3(0, EYE_HEIGHT, 0));
const controls = new OrbitControls(camera, renderer.domElement);
controls.target.copy(camera.position).addScaledVector(spawnForward, 1.0);
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

// The collider is invisible and double-sided: a scan is a single surface, so it blocks from both sides.
// A BVH makes each ray cast ~0.05 ms instead of ~20 ms (100k triangles); walking casts several per frame.
collider.scene.traverse((node) => {
  if (!node.isMesh) return;
  node.visible = false;
  node.material.side = THREE.DoubleSide;
  node.geometry.boundsTree = new MeshBVH(node.geometry);
  node.raycast = acceleratedRaycast;
  colliders.push(node);
});
scene.add(collider.scene);

// Invisible floor at floor_y so teleport and walking always have ground, even where the scan missed the floor.
const [minX, , minZ] = loc.bounds.min;
const [maxX, , maxZ] = loc.bounds.max;
const floor = new THREE.Mesh(
  new THREE.PlaneGeometry(maxX - minX + 4, maxZ - minZ + 4).rotateX(-Math.PI / 2),
  new THREE.MeshBasicMaterial({ visible: false, side: THREE.DoubleSide }),
);
floor.position.set((minX + maxX) / 2, loc.floor_y, (minZ + maxZ) / 2);
scene.add(floor);
colliders.push(floor);
setStatus(`${splat ? `${(loc.stats.splats / 1000).toFixed(0)}k splats (M: mesh) · ` : ''}${(loc.stats.visual_triangles / 1000).toFixed(0)}k triangles · ${loc.stats.textured ? `${loc.stats.texture_size}px texture` : 'vertex colors'} · floor: ${loc.floor_source ?? 'mesh'}`);

// Comfort vignette: a head-locked black ring that closes in while you glide (Google Earth VR's "tunnel vision").
const vignette = new THREE.Mesh(
  new THREE.PlaneGeometry(4, 4),
  new THREE.ShaderMaterial({
    transparent: true,
    depthTest: false,
    depthWrite: false,
    uniforms: { strength: { value: 0 } },
    vertexShader: 'varying vec2 vPos; void main() { vPos = position.xy; gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0); }',
    // Angle off the view axis (the plane sits 0.3 m ahead): clear inside ~26°, fully dark beyond ~49°.
    fragmentShader: 'uniform float strength; varying vec2 vPos; void main() { float angle = atan(length(vPos), 0.3); gl_FragColor = vec4(0.0, 0.0, 0.0, strength * smoothstep(0.45, 0.85, angle)); }',
  }),
);
vignette.position.z = -0.3;
vignette.renderOrder = 10000;
vignette.frustumCulled = false;
vignette.visible = false;
camera.add(vignette);

// ---------------------------------------------------------------- collision queries (world coordinates)
const probeRay = new THREE.Raycaster();
probeRay.firstHitOnly = true;  // three-mesh-bvh: nearest hit per mesh is all any query here needs
const normalMatrix = new THREE.Matrix3();

function castColliders(origin, direction, far = Infinity) {
  probeRay.set(origin, direction);
  probeRay.far = far;
  return probeRay.intersectObjects(colliders, false).filter((hit) => hit.face);
}

// The hit surface's normal, turned to face back along the ray (scan winding does not matter).
function facingNormal(hit, direction) {
  normalMatrix.getNormalMatrix(hit.object.matrixWorld);
  const normal = hit.face.normal.clone().applyMatrix3(normalMatrix).normalize();
  return normal.dot(direction) > 0 ? normal.negate() : normal;
}

// Height of the first surface below (x, fromY, z), or null when there is none.
function groundBelow(x, z, fromY) {
  const hit = castColliders(new THREE.Vector3(x, fromY, z), DOWN)[0];
  return hit ? hit.point.y : null;
}

// Clip a horizontal move `step` for a body at `position` with its floor at floorY and eyes at eyeY:
// rays at knee and chest height; on a hit, keep only the part of the move along the wall.
function collideStep(position, floorY, eyeY, step) {
  const probes = [floorY + STEP_HEIGHT + 0.05];
  if (eyeY - 0.2 > probes[0] + 0.2) probes.push(eyeY - 0.2);
  for (let attempt = 0; attempt < 3; attempt++) {
    const length = step.length();
    if (length < 1e-6) return step.set(0, 0, 0);
    const direction = step.clone().divideScalar(length);
    let nearest = null;
    for (const y of probes) {
      const hit = castColliders(new THREE.Vector3(position.x, y, position.z), direction, length + BODY_RADIUS)[0];
      if (hit && (!nearest || hit.distance < nearest.distance)) nearest = hit;
    }
    if (!nearest) return step;
    const normal = facingNormal(nearest, direction).setY(0);
    if (normal.lengthSq() < 1e-4) return step.set(0, 0, 0);
    normal.normalize();
    step.addScaledVector(normal, -step.dot(normal));  // slide
    step.y = 0;
  }
  return step.set(0, 0, 0);  // wedged into a corner
}

// ---------------------------------------------------------------- VR rig: world_from_base transform
// The rig is yaw + translation; its translation's y is the world height of the user's floor (feetY).
const rig = new THREE.Matrix4();
let baseReferenceSpace = null;
let feetY = loc.floor_y;
let flying = false;      // true after flying up with the right stick, until you land
let settle = 0;          // seconds of ground-following left (after gliding stops, or after landing)

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
  feetY = loc.floor_y;
  flying = false;
  rig.compose(new THREE.Vector3(spawn.x, feetY, spawn.z),
    new THREE.Quaternion().setFromAxisAngle(UP, spawnYaw), new THREE.Vector3(1, 1, 1));
  applyRig();
});

function headWorldPosition() {
  const xrCamera = renderer.xr.getCamera();
  return new THREE.Vector3().setFromMatrixPosition(xrCamera.matrixWorld);
}

function rigFloorY() {
  return new THREE.Vector3().setFromMatrixPosition(rig).y;
}

function moveRig(dx, dy, dz) {
  rig.premultiply(new THREE.Matrix4().makeTranslation(dx, dy, dz));
}

function teleportTo(point) {
  // Move the user's head (horizontally) onto the target and their floor (vertically) onto its height.
  const head = headWorldPosition();
  moveRig(point.x - head.x, point.y - rigFloorY(), point.z - head.z);
  feetY = point.y;
  flying = false;
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
const marker = new THREE.Mesh(
  new THREE.RingGeometry(0.15, 0.2, 32).rotateX(-Math.PI / 2),
  new THREE.MeshBasicMaterial({ color: 0x66ff99, transparent: true, opacity: 0.85 }),
);
marker.visible = false;
scene.add(marker);
const modelFactory = new XRControllerModelFactory();
const controllers = [0, 1].map((index) => {
  const controller = renderer.xr.getController(index);
  controller.userData = { index, aiming: false, target: null, turnArmed: true, landArmed: true, inputSource: null };
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
  controller.addEventListener('disconnected', () => { controller.userData.inputSource = null; });
  if (meshToggleInVr) controller.addEventListener('squeezestart', toggleRepresentation);
  scene.add(controller);
  const grip = renderer.xr.getControllerGrip(index);
  grip.add(modelFactory.createControllerModel(grip));
  scene.add(grip);
  return controller;
});

function isMoveHand(controller) {
  const handedness = controller.userData.inputSource?.handedness;
  return handedness === 'left' || (handedness !== 'right' && controller.userData.index === 0);
}

function readStick(controller) {
  const gamepad = controller.userData.inputSource?.gamepad;
  if (!gamepad || gamepad.axes.length < 4) return null;
  const shape = (v) => (Math.abs(v) < DEAD_ZONE ? 0 : Math.sign(v) * (Math.abs(v) - DEAD_ZONE) / (1 - DEAD_ZONE));
  return {
    rawX: gamepad.axes[2],
    x: shape(gamepad.axes[2]),
    y: shape(gamepad.axes[3]),  // forward is negative
    click: Boolean(gamepad.buttons[3]?.pressed),
  };
}

function updateAim(controller) {
  const origin = new THREE.Vector3().setFromMatrixPosition(controller.matrixWorld);
  const direction = new THREE.Vector3(0, 0, -1).transformDirection(controller.matrixWorld);
  const hit = castColliders(origin, direction)[0];
  controller.userData.target = null;
  if (!hit) return;
  if (facingNormal(hit, direction).y < WALKABLE_NORMAL_Y) return;  // walls and ceilings are not walkable
  controller.userData.target = hit.point.clone();
  marker.position.copy(hit.point).add(new THREE.Vector3(0, 0.01, 0));
  marker.visible = true;
}

// Right stick: snap turn on X, fly on Y, click to land. Returns how hard the user is flying (0..1).
function updateTurnAndFly(controller, stick, dt) {
  if (controller.userData.turnArmed && Math.abs(stick.rawX) > 0.7) {
    snapTurn(stick.rawX > 0 ? -SNAP_TURN : SNAP_TURN);
    controller.userData.turnArmed = false;
  } else if (Math.abs(stick.rawX) < 0.3) {
    controller.userData.turnArmed = true;
  }
  if (stick.click && controller.userData.landArmed) {
    flying = false;
    settle = 3;
  }
  controller.userData.landArmed = !stick.click;
  if (stick.y === 0 || Math.abs(stick.rawX) > Math.abs(stick.y) + 0.2) return 0;  // turning, not flying
  const head = headWorldPosition();
  let rise = -stick.y * FLY_SPEED * dt;
  if (rise > 0) {
    const ceiling = castColliders(head, UP, 0.15 + rise)[0];
    if (ceiling) rise = Math.max(0, ceiling.distance - 0.15);
  }
  const ground = groundBelow(head.x, head.z, feetY + STEP_HEIGHT) ?? loc.floor_y;
  const next = Math.min(Math.max(feetY + rise, ground), ground + MAX_ALTITUDE);
  flying = next > ground + 0.02;
  feetY = next;
  return Math.abs(stick.y);
}

// Left stick: glide where the controller points (flattened onto the floor). Returns how hard (0..1).
function updateGlide(controller, stick, dt) {
  const amount = Math.min(1, Math.hypot(stick.x, stick.y));
  if (amount === 0) return 0;
  const forward = new THREE.Vector3(0, 0, -1).transformDirection(controller.matrixWorld).setY(0);
  if (forward.lengthSq() < 0.04) forward.set(0, 0, -1).transformDirection(renderer.xr.getCamera().matrixWorld).setY(0);
  forward.normalize();
  const right = new THREE.Vector3(-forward.z, 0, forward.x);
  const head = headWorldPosition();
  const ground = flying ? (groundBelow(head.x, head.z, feetY + STEP_HEIGHT) ?? loc.floor_y) : feetY;
  const speed = MOVE_SPEED * Math.min(1 + (feetY - ground) * 0.5, 10);  // faster when flying high
  const step = forward.multiplyScalar(-stick.y).addScaledVector(right, stick.x).normalize()
    .multiplyScalar(amount * speed * dt);
  collideStep(head, feetY, head.y, step);
  moveRig(step.x, 0, step.z);
  settle = 0.4;
  return amount;
}

// On the ground, ease the floor onto whatever is underfoot (ledges up to STEP_HEIGHT, any drop).
function followGround(dt) {
  const head = headWorldPosition();
  const target = groundBelow(head.x, head.z, feetY + STEP_HEIGHT);
  if (target === null) return;
  feetY += (target - feetY) * Math.min(1, dt * 12);
}

function updateLocomotion(dt) {
  let effort = 0;
  for (const controller of controllers) {
    const stick = readStick(controller);
    if (!stick) continue;
    effort = Math.max(effort, isMoveHand(controller) ? updateGlide(controller, stick, dt) : updateTurnAndFly(controller, stick, dt));
  }
  if (!flying && settle > 0) {
    followGround(dt);
    settle -= dt;
  }
  const before = rigFloorY();
  if (effort > 0 || Math.abs(feetY - before) > 1e-5) {
    moveRig(0, feetY - before, 0);
    applyRig();
  }
  return effort;
}

function updateVignette(effort, dt) {
  const material = vignette.material;
  const target = comfortVignette ? 0.9 * effort : 0;
  material.uniforms.strength.value += (target - material.uniforms.strength.value) * Math.min(1, dt * 6);
  vignette.visible = material.uniforms.strength.value > 0.01;
}

// ---------------------------------------------------------------- desktop keys
const keys = new Set();
window.addEventListener('keydown', (event) => {
  if (event.key === 'm' || event.key === 'M') toggleRepresentation();
  keys.add(event.key.toLowerCase());
});
window.addEventListener('keyup', (event) => { keys.delete(event.key.toLowerCase()); });
window.addEventListener('blur', () => { keys.clear(); });

function desktopMove(dt) {
  const held = (...names) => names.some((name) => keys.has(name));
  const ahead = (held('w', 'arrowup') ? 1 : 0) - (held('s', 'arrowdown') ? 1 : 0);
  const side = (held('d', 'arrowright') ? 1 : 0) - (held('a', 'arrowleft') ? 1 : 0);
  const rise = (held('e') ? 1 : 0) - (held('q') ? 1 : 0);
  if (!ahead && !side && !rise) return;
  const boost = held('shift') ? 3 : 1;
  const forward = camera.getWorldDirection(new THREE.Vector3()).setY(0);
  if (forward.lengthSq() < 1e-6) forward.set(0, 0, -1);
  forward.normalize();
  const right = new THREE.Vector3(-forward.z, 0, forward.x);
  const step = forward.multiplyScalar(ahead).addScaledVector(right, side).normalize().multiplyScalar(MOVE_SPEED * boost * dt);
  step.y = rise * FLY_SPEED * boost * dt;
  camera.position.add(step);
  controls.target.add(step);
}

// ---------------------------------------------------------------- loop
window.addEventListener('resize', () => {
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
});

const clock = new THREE.Clock();
renderer.setAnimationLoop(() => {
  const dt = Math.min(clock.getDelta(), 0.1);
  if (renderer.xr.isPresenting) {
    let aiming = false;
    for (const controller of controllers) {
      if (controller.userData.aiming) { updateAim(controller); aiming = true; }
    }
    if (!aiming) marker.visible = false;
    updateVignette(updateLocomotion(dt), dt);
  } else {
    desktopMove(dt);
    controls.update();
    updateVignette(0, dt);
  }
  renderer.render(scene, camera);
});

// Debug handle (inspect from the browser console).
window.mq3drecon = { scene, camera, renderer, controls, visual, splat, loc, colliders, groundBelow, collideStep, castColliders };
