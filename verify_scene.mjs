/**
 * Sub-1-Second Node Pre-Flight Gate (/home/antigravity/archimedes-eureka-3d/verify_scene.mjs)
 * Evaluates 30 timestamps (0..30s) for:
 *  - Zero NaN matrices or positions across scene graph
 *  - Strict WebGL budgets (<= 55 draws, <= 55,000 tris, <= 25 materials, <= 3 lights)
 *  - Archimedes foot grounding when standing on floor (t < 9.8 || t > 15.2, |minY - 0.0| <= 0.025)
 *  - King Hiero foot grounding across all beats (|minY - 0.0| <= 0.025)
 *  - Camera NDC safe viewport bounds (|ndcX| <= 0.90, ndcY in [-0.58 .. +0.88])
 */
import fs from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import * as THREE from 'three';
import { MaterialRegistry } from './src/engine/baseRig.js';
import { createBaseDirector } from './src/engine/baseDirector.js';
import { buildArchimedesScene } from './src/scene/archimedesStage.js';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const timeline = JSON.parse(await fs.readFile(path.join(__dirname, 'config', 'timeline.json'), 'utf-8'));
const manifest = await fs
  .readFile(path.join(__dirname, 'audio', 'output', 'voice_manifest.json'), 'utf-8')
  .then((txt) => JSON.parse(txt))
  .catch(() => null);

const BUDGETS = {
  maxDraws: 55,
  maxTris: 55000,
  maxMaterials: 25,
  maxLights: 3,
};

function countSceneMetrics(root) {
  let draws = 0;
  let totalTris = 0;
  let lights = 0;
  const uniqueMaterials = new Set();

  function walk(obj) {
    if (!obj.visible) return;
    if (obj.isLight) lights++;

    if (obj.isInstancedMesh) {
      draws++;
      if (obj.material) {
        if (Array.isArray(obj.material)) obj.material.forEach((m) => uniqueMaterials.add(m));
        else uniqueMaterials.add(obj.material);
      }
      const g = obj.geometry;
      if (g) {
        const trisPerInst = g.index
          ? g.index.count / 3
          : g.attributes.position
            ? g.attributes.position.count / 3
            : 0;
        totalTris += Math.round(trisPerInst * obj.count);
      }
    } else if (obj.isMesh || obj.isSprite || obj.isLine || obj.isLineSegments || obj.isPoints) {
      draws++;
      if (obj.material) {
        if (Array.isArray(obj.material)) obj.material.forEach((m) => uniqueMaterials.add(m));
        else uniqueMaterials.add(obj.material);
      }
      if (obj.geometry) {
        const g = obj.geometry;
        const tris = g.index
          ? g.index.count / 3
          : g.attributes.position
            ? g.attributes.position.count / 3
            : 0;
        totalTris += Math.round(tris);
      }
    }

    for (let i = 0; i < obj.children.length; i++) {
      walk(obj.children[i]);
    }
  }

  walk(root);
  return { draws, totalTris, uniqueMaterials: uniqueMaterials.size, lights };
}

function findNaNNode(root) {
  let bad = null;
  root.traverse((obj) => {
    if (bad) return;
    const p = obj.position;
    if (!Number.isFinite(p.x) || !Number.isFinite(p.y) || !Number.isFinite(p.z)) {
      bad = `${obj.name || obj.type} (position)`;
      return;
    }
    const elems = obj.matrixWorld.elements;
    for (let i = 0; i < 16; i++) {
      if (!Number.isFinite(elems[i])) {
        bad = `${obj.name || obj.type} (matrixWorld[${i}])`;
        return;
      }
    }
  });
  return bad;
}

function resolveAnchorVec(raw, outVec) {
  if (!raw) return null;
  if (raw.isVector3 || (typeof raw.x === 'number' && typeof raw.y === 'number' && typeof raw.z === 'number')) {
    outVec.set(raw.x, raw.y, raw.z);
    return { vec: outVec, visible: raw.visible !== false };
  }
  const src = raw.worldPos || raw.pos || raw.position;
  if (src && typeof src.x === 'number') {
    outVec.set(src.x, src.y, src.z);
    return { vec: outVec, visible: raw.visible !== false };
  }
  if (Array.isArray(raw) && raw.length >= 3) {
    outVec.set(raw[0], raw[1], raw[2]);
    return { vec: outVec, visible: true };
  }
  return null;
}

const width = timeline.resolution?.width || 1080;
const height = timeline.resolution?.height || 1920;
const duration = Number(timeline.duration || 30.0);

const scene = new THREE.Scene();
scene.background = new THREE.Color('#2B1E16');
scene.fog = new THREE.FogExp2('#2B1E16', 0.052);

const camera = new THREE.PerspectiveCamera(46.0, width / height, 0.1, 80);

scene.add(new THREE.HemisphereLight('#FFF2D8', '#3B281A', 1.20));
const keyLight = new THREE.DirectionalLight('#FFEAC8', 2.45);
keyLight.position.set(2.4, 4.6, 3.6);
keyLight.target.position.set(0.0, 0.85, 0.0);
keyLight.castShadow = true;
scene.add(keyLight);
scene.add(keyLight.target);
const rimLight = new THREE.DirectionalLight('#38BDF8', 1.18);
rimLight.position.set(-3.2, 3.0, -2.8);
scene.add(rimLight);

const registry = new MaterialRegistry();
const stage = buildArchimedesScene(scene, registry, timeline);
const director = createBaseDirector(camera, timeline);

const timestamps = Array.from({ length: 30 }, (_, i) => +((i + 0.5) * (duration / 30)).toFixed(2));
const errors = [];
const warnings = [];
const tmpVec = new THREE.Vector3();

let peakMetrics = { draws: 0, totalTris: 0, uniqueMaterials: 0, registryMaterials: 0, lights: 0 };

for (const t of timestamps) {
  stage.update(t, manifest);
  director.update(t, stage.getState ? stage.getState() : {});

  scene.updateMatrixWorld(true);
  camera.updateMatrixWorld(true);
  camera.updateProjectionMatrix();

  const nanNode = findNaNNode(scene);
  if (nanNode) errors.push(`t=${t.toFixed(2)}s NaN detected at ${nanNode}`);

  const m = countSceneMetrics(scene);
  peakMetrics.draws = Math.max(peakMetrics.draws, m.draws);
  peakMetrics.totalTris = Math.max(peakMetrics.totalTris, m.totalTris);
  peakMetrics.uniqueMaterials = Math.max(peakMetrics.uniqueMaterials, m.uniqueMaterials);
  peakMetrics.registryMaterials = Math.max(peakMetrics.registryMaterials, registry.count());
  peakMetrics.lights = Math.max(peakMetrics.lights, m.lights);

  const allAnchors = stage.getAnchors ? stage.getAnchors() : {};

  // Archimedes standing grounding check (outside bathtub beat 9.8..15.2s)
  if (t < 9.8 || t > 15.2) {
    let minFootY = null;
    if (stage.archimedes?.legL?.shoe && stage.archimedes?.legR?.shoe) {
      const bL = new THREE.Box3().setFromObject(stage.archimedes.legL.shoe);
      const bR = new THREE.Box3().setFromObject(stage.archimedes.legR.shoe);
      minFootY = Math.min(bL.min.y, bR.min.y);
    } else if (allAnchors.archimedesFeet) {
      const res = resolveAnchorVec(allAnchors.archimedesFeet, tmpVec);
      if (res) minFootY = res.vec.y;
    }
    if (minFootY === null || Math.abs(minFootY - 0.0) > 0.025) {
      errors.push(`t=${t.toFixed(2)}s Archimedes foot grounding error minY=${minFootY}`);
    }
  }

  // King Hiero standing grounding check across all beats
  {
    let minKingFootY = null;
    if (stage.king?.legL?.shoe && stage.king?.legR?.shoe) {
      const bL = new THREE.Box3().setFromObject(stage.king.legL.shoe);
      const bR = new THREE.Box3().setFromObject(stage.king.legR.shoe);
      minKingFootY = Math.min(bL.min.y, bR.min.y);
    } else if (allAnchors.kingFeet) {
      const res = resolveAnchorVec(allAnchors.kingFeet, tmpVec);
      if (res) minKingFootY = res.vec.y;
    }
    if (minKingFootY === null || Math.abs(minKingFootY - 0.0) > 0.025) {
      errors.push(`t=${t.toFixed(2)}s King Hiero foot grounding error minY=${minKingFootY}`);
    }
  }

  const requiredKeys = ['archimedesHead', 'archimedesFeet', 'kingHead', 'kingFeet', 'slateBoard', 'centerStage'];
  for (const key of requiredKeys) {
    const resolved = resolveAnchorVec(allAnchors[key], tmpVec);
    if (!resolved) {
      errors.push(`t=${t.toFixed(2)}s missing required anchor "${key}"`);
      continue;
    }
    const ndc = resolved.vec.clone().project(camera);
    if (ndc.z < -1 || ndc.z > 1) {
      errors.push(`t=${t.toFixed(2)}s "${key}" outside clip plane (ndcZ=${ndc.z.toFixed(3)})`);
      continue;
    }
    if (Math.abs(ndc.x) > 0.90) {
      errors.push(`t=${t.toFixed(2)}s "${key}" clipped horizontal safe bound (ndcX=${ndc.x.toFixed(3)})`);
    }
    if (ndc.y < -0.58 || ndc.y > 0.88) {
      errors.push(`t=${t.toFixed(2)}s "${key}" outside vertical safe bound (ndcY=${ndc.y.toFixed(3)}, allowed=[-0.58..0.88])`);
    }
  }

  // Horizontal overlap gate between Archimedes and King Hiero
  const archHead = resolveAnchorVec(allAnchors.archimedesHead, new THREE.Vector3());
  const kingHead = resolveAnchorVec(allAnchors.kingHead, new THREE.Vector3());
  if (archHead && kingHead) {
    const ndcA = archHead.vec.project(camera);
    const ndcK = kingHead.vec.project(camera);
    if (Math.abs(ndcA.x - ndcK.x) < 0.16) {
      errors.push(`t=${t.toFixed(2)}s Archimedes & King horizontal overlap |ndcX_a - ndcX_b|=${Math.abs(ndcA.x - ndcK.x).toFixed(3)} < 0.16`);
    }
  }
}

if (peakMetrics.draws > BUDGETS.maxDraws) errors.push(`Draw budget exceeded: ${peakMetrics.draws} > ${BUDGETS.maxDraws}`);
if (peakMetrics.totalTris > BUDGETS.maxTris) errors.push(`Triangle budget exceeded: ${peakMetrics.totalTris} > ${BUDGETS.maxTris}`);
if (peakMetrics.uniqueMaterials > BUDGETS.maxMaterials) errors.push(`Material budget exceeded: ${peakMetrics.uniqueMaterials} > ${BUDGETS.maxMaterials}`);
if (peakMetrics.lights > BUDGETS.maxLights) errors.push(`Light budget exceeded: ${peakMetrics.lights} > ${BUDGETS.maxLights}`);

console.log('[VerifyScene] Peak Metrics:', JSON.stringify(peakMetrics));
if (errors.length > 0) {
  console.error(`[VerifyScene] FAILED (${errors.length} errors):`);
  for (const e of errors) console.error('  ERR:', e);
  process.exit(1);
}
console.log('[VerifyScene] ALL 30 TIMESTAMPS PASSED BUDGET, GROUNDING, & SAFE-FRAME GATES!');
