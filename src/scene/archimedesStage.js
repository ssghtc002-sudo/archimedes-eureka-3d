/**
 * Syracuse Royal Court, Hydro-Lab Centerpieces, Slate Equations & Cast Choreography
 * (/home/antigravity/archimedes-eureka-3d/src/scene/archimedesStage.js)
 *
 * Exports:
 *  - buildArchimedesScene(scene, registry, timeline)
 *    -> { archimedes, king, update(t, manifest), getAnchors(), getState() }
 *  - computePhonemeEnergy(manifest, t, speaker, timeline)
 *  - applyLipSync(rig, energy, t, smile)
 *
 * Strictly adheres to WebGL performance budgets:
 *  - <= 55 visible draw calls
 *  - <= 55,000 triangles
 *  - <= 25 materials in MaterialRegistry
 *  - 100% Node-safe (graceful fallback when document is undefined)
 */
import * as THREE from 'three';
import { mergeGeometries } from 'three/addons/utils/BufferGeometryUtils.js';
import {
  MaterialRegistry,
  buildPixarHumanoid as createHumanoidRig,
  safeMerge,
  createGlowSprite,
  createContactShadowDisc,
} from '../engine/baseRig.js';

function makeCanvas(w, h) {
  if (typeof document !== 'undefined') {
    const c = document.createElement('canvas');
    c.width = w;
    c.height = h;
    return c;
  }
  return null;
}

function clamp(v, lo, hi) {
  return v < lo ? lo : v > hi ? hi : v;
}

function clamp01(v) {
  return v < 0 ? 0 : v > 1 ? 1 : v;
}

function lerp(a, b, t) {
  return a + (b - a) * clamp01(t);
}

function smoothstep(e0, e1, x) {
  const u = clamp01((x - e0) / Math.max(1e-5, e1 - e0));
  return u * u * (3 - 2 * u);
}

/**
 * Merges geometries with per-entry sRGB color baked into vertex color attribute
 * so multi-colored rigid assemblies render in 1 draw call with 1 shared material.
 */
function mergeWithVertexColors(entries) {
  const prep = entries.map(({ geo, color }) => {
    const g = geo.index ? geo.toNonIndexed() : geo.clone();
    if (!g.attributes.normal) g.computeVertexNormals();
    const count = g.attributes.position.count;
    if (!g.attributes.uv) {
      g.setAttribute('uv', new THREE.BufferAttribute(new Float32Array(count * 2), 2));
    }
    const c = new THREE.Color(color);
    const colArr = new Float32Array(count * 3);
    for (let i = 0; i < count; i++) {
      colArr[i * 3] = c.r;
      colArr[i * 3 + 1] = c.g;
      colArr[i * 3 + 2] = c.b;
    }
    g.setAttribute('color', new THREE.BufferAttribute(colArr, 3));
    for (const key of Object.keys(g.attributes)) {
      if (key !== 'position' && key !== 'normal' && key !== 'uv' && key !== 'color') {
        g.deleteAttribute(key);
      }
    }
    return g;
  });
  const merged = mergeGeometries(prep, false);
  for (const g of prep) g.dispose();
  return merged;
}

function cloneTransformed(mesh) {
  mesh.updateMatrix();
  const g = mesh.geometry.clone();
  g.applyMatrix4(mesh.matrix);
  return g;
}

/**
 * Computes phoneme energy [0..1] for a speaker at timestamp t (seconds)
 * from either manifest.lipSync[speaker], manifest.lines, or timeline.dialogue.
 */
export function computePhonemeEnergy(manifest, t, speaker, timeline = null) {
  const fps = manifest?.fps || timeline?.fps || 30;
  const frameIdx = Math.round(t * fps);

  const lipArr = manifest?.lipSync?.[speaker];
  if (Array.isArray(lipArr) && lipArr.length > 0) {
    const raw = lipArr[clamp(frameIdx, 0, lipArr.length - 1)];
    if (typeof raw === 'number') return clamp01(raw);
    if (raw && typeof raw === 'object') return clamp01(Number(raw.open ?? raw.open01 ?? raw.energy ?? 0));
  }

  const lines = Array.isArray(manifest?.lines) && manifest.lines.length > 0
    ? manifest.lines
    : Array.isArray(timeline?.dialogue)
      ? timeline.dialogue
      : [];

  for (let i = 0; i < lines.length; i++) {
    const line = lines[i];
    if (line.speaker !== speaker) continue;
    if (t < line.start - 0.02 || t > line.end + 0.04) continue;

    if (Array.isArray(line.words) && line.words.length > 0) {
      for (let j = 0; j < line.words.length; j++) {
        const w = line.words[j];
        if (t >= w.start - 0.015 && t <= w.end + 0.02) {
          const wDur = Math.max(0.06, w.end - w.start);
          const u = clamp01((t - w.start) / wDur);
          const env = Math.sin(u * Math.PI);
          const syllables = Math.max(1, Math.min(4, Math.ceil((w.word || '').length / 3)));
          const osc = 0.55 + 0.45 * Math.pow(Math.sin(u * Math.PI * syllables), 2);
          return clamp01(env * osc * 0.95);
        }
      }
      return 0.08;
    }

    const dur = Math.max(0.1, line.end - line.start);
    const u = clamp01((t - line.start) / dur);
    const env = Math.sin(u * Math.PI);
    const osc = 0.35 + 0.65 * Math.abs(Math.sin(t * 19.5)) * (0.6 + 0.4 * Math.cos(t * 9.2));
    return clamp01(Math.min(1.0, env * 1.35) * osc);
  }

  return 0;
}

/**
 * Applies lip-sync jaw/mouth opening and subtle speech cadence to a character rig.
 */
export function applyLipSync(rig, energy, t = 0, smile = 0.25) {
  if (!rig || typeof rig.setLipSync !== 'function') return;
  const e = clamp01(energy);
  const flutter = e > 0.05 ? e * (0.86 + 0.14 * Math.sin(t * 34.0)) : 0;
  rig.setLipSync(clamp01(flutter), smile);
}

/**
 * Multi-ring radial Syracuse travertine/marble floor geometry with vertex-color vignette.
 */
function buildRadialMarbleFloorGeometry(rings = [0, 2.2, 4.5, 8.8], segs = 40) {
  const positions = [];
  const normals = [];
  const uvs = [];
  const colors = [];
  const indices = [];

  const ringColors = [
    new THREE.Color(1.0, 0.99, 0.96),
    new THREE.Color(0.95, 0.92, 0.86),
    new THREE.Color(0.58, 0.52, 0.46),
    new THREE.Color(0.16, 0.13, 0.11),
  ];

  for (let rIdx = 0; rIdx < rings.length; rIdx++) {
    const r = rings[rIdx];
    const col = ringColors[rIdx];
    for (let s = 0; s <= segs; s++) {
      const theta = (s / segs) * Math.PI * 2;
      const x = Math.cos(theta) * r;
      const z = Math.sin(theta) * r;
      positions.push(x, 0, z);
      normals.push(0, 1, 0);
      uvs.push(x * 0.38 + 0.5, z * 0.38 + 0.5);
      colors.push(col.r, col.g, col.b);
    }
  }

  const stride = segs + 1;
  for (let rIdx = 0; rIdx < rings.length - 1; rIdx++) {
    for (let s = 0; s < segs; s++) {
      const a = rIdx * stride + s;
      const b = a + 1;
      const c = (rIdx + 1) * stride + s;
      const d = c + 1;
      indices.push(a, c, b);
      indices.push(b, c, d);
    }
  }

  const geo = new THREE.BufferGeometry();
  geo.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
  geo.setAttribute('normal', new THREE.Float32BufferAttribute(normals, 3));
  geo.setAttribute('uv', new THREE.Float32BufferAttribute(uvs, 2));
  geo.setAttribute('color', new THREE.Float32BufferAttribute(colors, 3));
  geo.setIndex(indices);
  return geo;
}

function createSyracuseMarbleFloorTexture() {
  const c = makeCanvas(512, 512);
  if (!c) return null;
  const ctx = c.getContext('2d');

  ctx.fillStyle = '#EAE0CC';
  ctx.fillRect(0, 0, 512, 512);

  const tiles = 4;
  const step = 512 / tiles;
  const tones = ['#EFE6D5', '#E4D8C1', '#F3EBDD', '#DFD2B9'];

  for (let r = 0; r < tiles; r++) {
    for (let k = 0; k < tiles; k++) {
      const x = k * step;
      const y = r * step;
      ctx.fillStyle = tones[(r * 3 + k) % tones.length];
      ctx.fillRect(x, y, step, step);

      // Subtle marble veins
      ctx.strokeStyle = 'rgba(138, 118, 92, 0.22)';
      ctx.lineWidth = 1.6;
      ctx.beginPath();
      ctx.moveTo(x + 12, y + 24 + ((r + k) % 3) * 18);
      ctx.bezierCurveTo(x + step * 0.4, y + 60, x + step * 0.65, y + 18, x + step - 10, y + step - 24);
      ctx.stroke();

      // Gold-ochre Greek mosaic border inset
      ctx.strokeStyle = 'rgba(180, 140, 72, 0.35)';
      ctx.lineWidth = 2;
      ctx.strokeRect(x + 8, y + 8, step - 16, step - 16);

      // Tile grout seam
      ctx.strokeStyle = 'rgba(92, 74, 52, 0.45)';
      ctx.lineWidth = 2.5;
      ctx.strokeRect(x, y, step, step);
    }
  }

  const tex = new THREE.CanvasTexture(c);
  tex.colorSpace = THREE.SRGBColorSpace;
  tex.wrapS = THREE.RepeatWrapping;
  tex.wrapT = THREE.RepeatWrapping;
  tex.repeat.set(3.0, 3.0);
  tex.needsUpdate = true;
  return tex;
}

function createSyracuseWallTexture() {
  const c = makeCanvas(512, 256);
  if (!c) return null;
  const ctx = c.getContext('2d');
  const g = ctx.createRadialGradient(256, 128, 30, 256, 128, 320);
  g.addColorStop(0, '#4A3728');
  g.addColorStop(0.55, '#332419');
  g.addColorStop(1, '#1E150F');
  ctx.fillStyle = g;
  ctx.fillRect(0, 0, 512, 256);

  // Subtle Greek stone course lines
  ctx.strokeStyle = 'rgba(255, 236, 200, 0.06)';
  ctx.lineWidth = 1.5;
  for (let y = 32; y < 256; y += 32) {
    ctx.beginPath();
    ctx.moveTo(0, y);
    ctx.lineTo(512, y);
    ctx.stroke();
  }

  const tex = new THREE.CanvasTexture(c);
  tex.colorSpace = THREE.SRGBColorSpace;
  tex.needsUpdate = true;
  return tex;
}

/**
 * Pre-renders the 6 Syracuse Equation Slate textures (1 per beat).
 * Keep main equations high and centered (y = 140..280 of 512px) so Archimedes & King never occlude them.
 */
function createSlateTextures() {
  const specs = [
    {
      main: 'CROWN = 100% GOLD?',
      sub: 'NO MELTING ALLOWED! • HIERO II',
      mainColor: '#FFE066',
      glowColor: '#F59E0B',
      subColor: '#BAE6FD',
      badge: 'ROYAL TEST',
    },
    {
      main: 'V_crown = L × W × H ???',
      sub: 'CALIPERS SLIP! • TOO MANY SPIKES!',
      mainColor: '#FDA4AF',
      glowColor: '#F43F5E',
      subColor: '#FDE68A',
      badge: 'GEOMETRY FAILS',
    },
    {
      main: 'EUREKA!  V_water = V_body',
      sub: 'DISPLACED WATER = EXACT VOLUME!',
      mainColor: '#67E8F9',
      glowColor: '#06B6D4',
      subColor: '#FEF08A',
      badge: 'BATHTUB DISCOVERY',
    },
    {
      main: 'ΔV_crown > ΔV_gold  (SILVER!)',
      sub: 'LIGHTER SILVER = BULKIER CROWN = MORE SPILL!',
      mainColor: '#38BDF8',
      glowColor: '#0284C7',
      subColor: '#FCA5A5',
      badge: 'DUAL TANK TRAP',
    },
    {
      main: 'F_b = ρ · g · V   (BUSTED!)',
      sub: 'BUOYANT PUSH = WEIGHT OF DISPLACED WATER',
      mainColor: '#FDE047',
      glowColor: '#EAB308',
      subColor: '#FF8A8A',
      badge: 'GOLDSMITH BUSTED',
    },
    {
      main: "ARCHIMEDES' PRINCIPLE ✨",
      sub: 'F_b = ρ · g · V  •  WHY GIANT SHIPS FLOAT!',
      mainColor: '#FEF08A',
      glowColor: '#F59E0B',
      subColor: '#67E8F9',
      badge: 'SYRACUSE TRIUMPH',
    },
  ];

  return specs.map((spec, idx) => {
    const c = makeCanvas(1024, 512);
    if (!c) return null;
    const ctx = c.getContext('2d');

    // Deep Syracuse dark bronze-slate background
    const bg = ctx.createRadialGradient(512, 230, 40, 512, 256, 620);
    bg.addColorStop(0, '#1E2D3B');
    bg.addColorStop(0.65, '#15202B');
    bg.addColorStop(1, '#0D151D');
    ctx.fillStyle = bg;
    ctx.fillRect(0, 0, 1024, 512);

    // Inner Greek gold border line
    ctx.strokeStyle = 'rgba(245, 196, 81, 0.42)';
    ctx.lineWidth = 4;
    ctx.strokeRect(22, 22, 980, 468);

    ctx.strokeStyle = 'rgba(103, 232, 249, 0.18)';
    ctx.lineWidth = 2;
    ctx.strokeRect(32, 32, 960, 448);

    // Top-center small Greek header pill
    ctx.fillStyle = 'rgba(245, 158, 11, 0.18)';
    ctx.beginPath();
    ctx.roundRect(352, 46, 320, 42, 21);
    ctx.fill();
    ctx.strokeStyle = 'rgba(253, 224, 71, 0.55)';
    ctx.lineWidth = 2;
    ctx.stroke();

    ctx.font = '800 22px "DejaVu Sans", sans-serif';
    ctx.fillStyle = '#FDE68A';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText(spec.badge, 512, 68);

    // Beat-specific chalk doodles on margins (kept clear of main title text)
    ctx.save();
    ctx.strokeStyle = 'rgba(226, 244, 255, 0.26)';
    ctx.lineWidth = 3;
    if (idx === 0) {
      // Crown outline neatly tucked in lower-left margin below divider
      ctx.beginPath();
      ctx.moveTo(110, 420);
      ctx.lineTo(118, 378);
      ctx.lineTo(138, 400);
      ctx.lineTo(158, 370);
      ctx.lineTo(178, 400);
      ctx.lineTo(198, 378);
      ctx.lineTo(206, 420);
      ctx.closePath();
      ctx.stroke();
    } else if (idx === 1) {
      // Crossed-out chalk lines
      ctx.strokeStyle = 'rgba(251, 113, 133, 0.45)';
      ctx.lineWidth = 5;
      ctx.beginPath();
      ctx.moveTo(190, 140);
      ctx.lineTo(830, 235);
      ctx.stroke();
    } else if (idx === 2 || idx === 3) {
      // Rising water wave lines
      ctx.strokeStyle = 'rgba(56, 189, 248, 0.38)';
      ctx.beginPath();
      for (let x = 140; x <= 884; x += 32) {
        const y = 375 + Math.sin(x * 0.05) * 8;
        if (x === 140) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      }
      ctx.stroke();
    }
    ctx.restore();

    // Main Equation Text (high & centered at y = 195 so characters never block it)
    ctx.save();
    ctx.font = idx === 3 ? '900 58px "DejaVu Sans", sans-serif' : '900 64px "DejaVu Sans", sans-serif';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.shadowColor = spec.glowColor;
    ctx.shadowBlur = 22;
    ctx.strokeStyle = 'rgba(8, 15, 26, 0.9)';
    ctx.lineWidth = 8;
    ctx.strokeText(spec.main, 512, 192);
    ctx.fillStyle = spec.mainColor;
    ctx.fillText(spec.main, 512, 192);
    ctx.restore();

    // Glowing divider bar
    ctx.save();
    ctx.strokeStyle = spec.subColor;
    ctx.lineWidth = 4;
    ctx.beginPath();
    ctx.moveTo(220, 258);
    ctx.lineTo(804, 258);
    ctx.stroke();
    ctx.restore();

    // Subtitle / physics takeaway line at y = 310
    ctx.save();
    ctx.font = '800 31px "DejaVu Sans", sans-serif';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillStyle = spec.subColor;
    ctx.fillText(spec.sub, 512, 310);
    ctx.restore();

    const tex = new THREE.CanvasTexture(c);
    tex.colorSpace = THREE.SRGBColorSpace;
    tex.needsUpdate = true;
    return tex;
  });
}

/**
 * Builds curly Greek scholar hair + olive laurel fillet for Archimedes.
 */
function buildArchimedesHairGeometry() {
  const entries = [];
  const addCurl = (x, y, z, r, sx = 1, sy = 1, sz = 1, color = '#E2E8F0') => {
    const g = new THREE.SphereGeometry(r, 9, 7);
    g.scale(sx, sy, sz);
    g.translate(x, y, z);
    entries.push({ geo: g, color });
  };

  // Curly silver-white Greek philosopher locks around temples, sides, and back
  addCurl(0.0, 0.045, -0.02, 0.145, 1.18, 0.68, 1.12);
  addCurl(0.0, -0.04, -0.13, 0.125, 1.32, 1.0, 0.78);
  addCurl(-0.12, -0.07, -0.11, 0.088, 1.05, 1.1, 0.85);
  addCurl(0.12, -0.07, -0.11, 0.088, 1.05, 1.1, 0.85);

  for (const s of [-1, 1]) {
    addCurl(s * 0.168, 0.015, 0.01, 0.068, 0.95, 0.95, 1.05);
    addCurl(s * 0.185, -0.035, -0.025, 0.072, 1.05, 0.92, 0.95);
    addCurl(s * 0.172, -0.095, -0.055, 0.065, 1.0, 0.9, 0.9);
    addCurl(s * 0.115, 0.068, 0.045, 0.058, 1.1, 0.82, 0.92);
  }
  addCurl(-0.045, 0.078, 0.065, 0.052, 1.15, 0.75, 0.88);
  addCurl(0.048, 0.078, 0.062, 0.052, 1.15, 0.75, 0.88);

  // Cobalt-and-gold Greek scholar headband fillet
  const band = new THREE.TorusGeometry(0.178, 0.014, 7, 20);
  band.rotateX(Math.PI * 0.5 - 0.12);
  band.scale(0.97, 1.0, 0.97);
  band.translate(0, 0.012, 0.006);
  entries.push({ geo: band, color: '#1D4ED8' });

  return mergeWithVertexColors(entries);
}

/**
 * Tucked Greek beard & mustache strictly inside head/jaw envelope (jawTrim: true).
 */
function buildTuckedGreekBeardEntries(beardHex) {
  const entries = [];
  // Mustache tucked neatly under nose (y = 0.156, above mouth cavity)
  const mid = new THREE.SphereGeometry(0.017, 8, 6);
  mid.scale(1.3, 0.65, 0.78);
  mid.translate(0, 0.156, 0.170);
  entries.push({ geo: mid, color: beardHex });

  for (const s of [-1, 1]) {
    const wing = new THREE.CapsuleGeometry(0.013, 0.028, 4, 7);
    wing.rotateZ(s * 1.22);
    wing.rotateY(s * 0.26);
    wing.translate(s * 0.025, 0.153, 0.164);
    entries.push({ geo: wing, color: beardHex });

    // Tucked jawline strap (strictly inside skull radius <= 0.155)
    const jaw = new THREE.CapsuleGeometry(0.016, 0.048, 4, 7);
    jaw.rotateZ(s * -0.58);
    jaw.rotateX(0.32);
    jaw.translate(s * 0.068, 0.095, 0.118);
    entries.push({ geo: jaw, color: beardHex });
  }

  // Tucked chin beard below lowerLip (y = 0.064, z = 0.130)
  const chin = new THREE.SphereGeometry(0.056, 10, 8);
  chin.scale(0.92, 0.96, 0.68);
  chin.translate(0, 0.064, 0.130);
  entries.push({ geo: chin, color: beardHex });

  return entries;
}

/**
 * Builds King Hiero II's royal dark hair + gleaming golden royal diadem crown.
 */
function buildKingCrownAndHairGeometry() {
  const entries = [];

  // Dark royal hair cap under the diadem
  const hairCap = new THREE.SphereGeometry(0.178, 14, 10);
  hairCap.scale(0.96, 0.62, 0.96);
  hairCap.translate(0, -0.01, -0.015);
  entries.push({ geo: hairCap, color: '#272220' });

  // Gleaming Gold Royal Diadem Band
  const band = new THREE.CylinderGeometry(0.184, 0.178, 0.068, 20);
  band.translate(0, 0.042, 0);
  entries.push({ geo: band, color: '#F59E0B' });

  const rimBot = new THREE.TorusGeometry(0.182, 0.012, 7, 20);
  rimBot.rotateX(Math.PI * 0.5);
  rimBot.translate(0, 0.010, 0);
  entries.push({ geo: rimBot, color: '#FDE047' });

  // 7 Royal Gold Crown Spikes around the diadem
  const numSpikes = 7;
  for (let i = 0; i < numSpikes; i++) {
    const ang = -Math.PI * 0.65 + (i / (numSpikes - 1)) * Math.PI * 1.3;
    const isCenter = i === 3;
    const h = isCenter ? 0.095 : 0.072;
    const r = isCenter ? 0.028 : 0.022;
    const spike = new THREE.ConeGeometry(r, h, 6);
    const sx = Math.sin(ang) * 0.176;
    const sz = Math.cos(ang) * 0.176;
    spike.translate(sx, 0.076 + h * 0.45, sz);
    entries.push({ geo: spike, color: '#FDE047' });
  }

  // Front ruby jewel cabochon
  const ruby = new THREE.SphereGeometry(0.022, 8, 6);
  ruby.scale(1.0, 1.2, 0.6);
  ruby.translate(0, 0.044, 0.184);
  entries.push({ geo: ruby, color: '#E11D48' });

  return mergeWithVertexColors(entries);
}

/**
 * Consolidates a humanoid rig's static sub-parts into vertex-colored meshes
 * while preserving `legL.shoe`, `legR.shoe`, `groundFeet`, and `setLipSync`.
 */
function consolidateHumanoid(char, colors, beardHex, customHeadgearGeo, multiMat) {
  const { skin, suit, accent, trouser, shoe } = colors;

  // 1. Merge chest coat + accent sash/placket
  if (char.chestGroup.children.length >= 2) {
    const coatMesh = char.chestGroup.children[0];
    const accentMesh = char.chestGroup.children[1];
    const mergedChest = mergeWithVertexColors([
      { geo: cloneTransformed(coatMesh), color: suit },
      { geo: cloneTransformed(accentMesh), color: accent },
    ]);
    char.chestGroup.remove(coatMesh);
    char.chestGroup.remove(accentMesh);
    coatMesh.geometry.dispose();
    accentMesh.geometry.dispose();
    const chestMesh = new THREE.Mesh(mergedChest, multiMat);
    chestMesh.castShadow = true;
    char.chestGroup.add(chestMesh);
  }

  // 2. Merge head skin + tucked Greek beard
  {
    const headSkinMesh = char.headGroup.children[0];
    const entries = [{ geo: cloneTransformed(headSkinMesh), color: skin }];
    if (beardHex) {
      entries.push(...buildTuckedGreekBeardEntries(beardHex));
    }
    const mergedHead = mergeWithVertexColors(entries);
    char.headGroup.remove(headSkinMesh);
    headSkinMesh.geometry.dispose();
    const newHead = new THREE.Mesh(mergedHead, multiMat);
    newHead.castShadow = true;
    char.headGroup.add(newHead);
  }

  // 3. Merge mouth cavity + teeth, keep lowerLip articulated
  let mouthCavityTeeth = null;
  let mouthLowerLip = null;
  {
    const mouthGroup = char.headGroup.children.find(
      (c) => c.isGroup && Math.abs(c.position.y - 0.125) < 0.01
    );
    if (mouthGroup && mouthGroup.children.length >= 3) {
      const [cavMesh, teethMesh, lipMesh] = mouthGroup.children;
      const mergedMouth = mergeWithVertexColors([
        { geo: cloneTransformed(cavMesh), color: '#2A0B0F' },
        { geo: cloneTransformed(teethMesh), color: '#F6F2E8' },
      ]);
      mouthGroup.remove(cavMesh);
      mouthGroup.remove(teethMesh);
      cavMesh.geometry.dispose();
      teethMesh.geometry.dispose();
      mouthCavityTeeth = new THREE.Mesh(mergedMouth, multiMat);
      mouthGroup.add(mouthCavityTeeth);
      mouthLowerLip = lipMesh;
    }
  }

  // 4. Merge forearm + hand on each arm
  for (const arm of [char.armL, char.armR]) {
    const forearmMesh = arm.elbow.children[0];
    const handMesh = arm.hand;
    if (forearmMesh && handMesh) {
      const mergedFH = mergeWithVertexColors([
        { geo: cloneTransformed(forearmMesh), color: skin }, // bare lower arms in Greek tunic/toga!
        { geo: cloneTransformed(handMesh), color: skin },
      ]);
      arm.elbow.remove(forearmMesh);
      arm.elbow.remove(handMesh);
      forearmMesh.geometry.dispose();
      handMesh.geometry.dispose();
      const fhMesh = new THREE.Mesh(mergedFH, multiMat);
      arm.elbow.add(fhMesh);
      arm.hand = fhMesh;
    }
  }

  // 5. Merge shin + sandal shoe on each leg (preserving leg.shoe for verify_scene.mjs!)
  const legShoeMeshes = [];
  for (const leg of [char.legL, char.legR]) {
    const knee = leg.hip.children.find((c) => c.isGroup);
    if (knee && knee.children.length >= 2) {
      const shinMesh = knee.children[0];
      const ankle = knee.children[1];
      const shoeMesh = leg.shoe;
      ankle.updateMatrix();
      shoeMesh.updateMatrix();
      const shoeMat4 = new THREE.Matrix4().multiplyMatrices(ankle.matrix, shoeMesh.matrix);
      const shoeGeoInKnee = shoeMesh.geometry.clone();
      shoeGeoInKnee.applyMatrix4(shoeMat4);

      const mergedSS = mergeWithVertexColors([
        { geo: cloneTransformed(shinMesh), color: trouser },
        { geo: shoeGeoInKnee, color: shoe },
      ]);
      shoeGeoInKnee.dispose();
      knee.remove(shinMesh);
      knee.remove(ankle);
      shinMesh.geometry.dispose();
      shoeMesh.geometry.dispose();

      const ssMesh = new THREE.Mesh(mergedSS, multiMat);
      ssMesh.castShadow = true;
      knee.add(ssMesh);
      leg.shin = ssMesh;
      leg.shoe = ssMesh;
      legShoeMeshes.push(ssMesh);
    }
  }

  // 6. Attach custom headgear (hair / royal crown)
  if (customHeadgearGeo) {
    const hgMesh = new THREE.Mesh(customHeadgearGeo, multiMat);
    hgMesh.castShadow = true;
    char.headgearGroup.add(hgMesh);
    char.headgearGroup.visible = true;
  }

  char.setHatPop = (pop01 = 0) => {
    char.headgearGroup.visible = true;
    const p = clamp01(pop01);
    char.headgearGroup.position.y = 0.345 + p * 0.22;
    char.headgearGroup.rotation.z = p * 0.18;
    char.headgearGroup.rotation.y = p * 0.85;
  };

  char.setLipSync = (open01 = 0, smile = 0.25) => {
    const o = clamp01(open01);
    if (mouthCavityTeeth) {
      mouthCavityTeeth.visible = o > 0.04 || smile > 0.48;
      mouthCavityTeeth.scale.set(1.0 + smile * 0.14, 0.32 + o * 1.15, 1.0);
    }
    if (mouthLowerLip) {
      mouthLowerLip.position.y = -0.012 - o * 0.030;
      mouthLowerLip.scale.x = 1.0 + Math.max(0, smile) * 0.12;
    }
  };

  let groundedRigY = 0;
  char.groundFeet = (surfaceY = 0.0) => {
    char.rig.position.y = 0;
    char.root.updateMatrixWorld(true);
    let minWorldY = Infinity;
    for (const m of legShoeMeshes) {
      const box = new THREE.Box3().setFromObject(m);
      if (box.min.y < minWorldY) minWorldY = box.min.y;
    }
    if (!Number.isFinite(minWorldY)) return 0;
    const scaleY = char.root.scale.y || 1;
    const delta = (surfaceY - minWorldY) / scaleY;
    char.rig.position.y = delta;
    groundedRigY = delta;
    char.root.updateMatrixWorld(true);
    if (char.contactShadow) char.contactShadow.position.y = 0.004;
    return delta;
  };

  const origReset = char.resetPose;
  char.resetPose = () => {
    origReset();
    char.rig.position.set(0, groundedRigY, 0);
    char.headgearGroup.visible = true;
    char.setHatPop(0);
  };

  return legShoeMeshes;
}

/**
 * Builds the Spiky Royal Gold Crown geometry (used on the pedestal, in tanks, and on the balance).
 */
function buildSpikyGoldCrownGeometry(scale = 1.0) {
  const parts = [];
  const baseBand = new THREE.CylinderGeometry(0.115 * scale, 0.102 * scale, 0.055 * scale, 16);
  baseBand.translate(0, 0.028 * scale, 0);
  parts.push(baseBand);

  const rim = new THREE.TorusGeometry(0.108 * scale, 0.010 * scale, 6, 16);
  rim.rotateX(Math.PI * 0.5);
  rim.translate(0, 0.012 * scale, 0);
  parts.push(rim);

  const spikes = 8;
  for (let i = 0; i < spikes; i++) {
    const a = (i / spikes) * Math.PI * 2;
    const h = (i % 2 === 0 ? 0.095 : 0.072) * scale;
    const cone = new THREE.ConeGeometry(0.022 * scale, h, 5);
    cone.translate(Math.cos(a) * 0.105 * scale, (0.052 + h * 0.45) * scale, Math.sin(a) * 0.105 * scale);
    parts.push(cone);
  }
  return safeMerge(parts);
}

/**
 * Builds the Greek Syracuse Court environment, Equation Slate, Bathtub, Hydro-Pedestal,
 * and both characters (`archimedes` and `king`).
 */
export function buildArchimedesScene(scene, registry = new MaterialRegistry(), timeline = {}) {
  const matReg = registry || new MaterialRegistry();
  const wc = timeline?.worldContract || {};
  const floorY = wc.floorY ?? 0.0;
  const archimedesBasePos = new THREE.Vector3(...(wc.archimedesPos || [-0.60, 0.0, 0.02]));
  const kingBasePos = new THREE.Vector3(...(wc.kingPos || [0.60, 0.0, 0.02]));
  const centerStagePos = new THREE.Vector3(...(wc.tablePos || [0.0, 0.0, 0.22]));
  const bathtubBasePos = new THREE.Vector3(...(wc.bathtubPos || [-0.14, 0.0, 0.18]));
  const slatePos = new THREE.Vector3(...(wc.slatePos || [0.0, 1.98, -1.12]));

  const rootGroup = new THREE.Group();
  rootGroup.name = 'archimedes_syracuse_stage';
  if (scene) scene.add(rootGroup);

  // =========================================================================
  // 1. SHARED MATERIAL PALETTE (Compact count <= 18 materials total)
  // =========================================================================
  const multiMat = matReg.getStandard('shared_vertex_matte', {
    color: '#FFFFFF',
    vertexColors: true,
    roughness: 0.52,
    metalness: 0.05,
  });

  const floorTex = createSyracuseMarbleFloorTexture();
  const floorMat = matReg.getStandard('syracuse_marble_floor', {
    map: floorTex || null,
    color: floorTex ? '#FFFFFF' : '#E6DBC6',
    roughness: 0.34,
    metalness: 0.04,
    vertexColors: true,
  });

  const wallTex = createSyracuseWallTexture();
  const wallMat = matReg.getStandard('syracuse_court_wall', {
    map: wallTex || null,
    color: wallTex ? '#FFFFFF' : '#332419',
    roughness: 0.88,
    metalness: 0.02,
  });

  const marbleMat = matReg.getStandard('syracuse_carved_marble', {
    color: '#F3ECE0',
    roughness: 0.38,
    metalness: 0.04,
  });

  const bronzeMat = matReg.getStandard('syracuse_aged_bronze', {
    color: '#C88A36',
    roughness: 0.28,
    metalness: 0.72,
  });

  const goldMat = matReg.getStandard('royal_pure_gold', {
    color: '#FFD43B',
    emissive: '#B45309',
    emissiveIntensity: 0.18,
    roughness: 0.20,
    metalness: 0.84,
  });

  const silverMat = matReg.getStandard('sneaky_silver_core', {
    color: '#E2E8F0',
    emissive: '#38BDF8',
    emissiveIntensity: 0.22,
    roughness: 0.22,
    metalness: 0.82,
  });

  const waterMat = matReg.getStandard('syracuse_aqua_water', {
    color: '#22D3EE',
    emissive: '#0891B2',
    emissiveIntensity: 0.35,
    roughness: 0.14,
    metalness: 0.08,
    transparent: true,
    opacity: 0.84,
  });

  const glassMat = matReg.getStandard('hydro_tank_glass', {
    color: '#E0F2FE',
    roughness: 0.12,
    metalness: 0.10,
    transparent: true,
    opacity: 0.36,
    depthWrite: false,
  });

  const cyanGlowMat = matReg.getStandard('eureka_cyan_glow', {
    color: '#67E8F9',
    emissive: '#06B6D4',
    emissiveIntensity: 1.15,
    roughness: 0.20,
    metalness: 0.10,
  });

  const slateTextures = createSlateTextures();
  const slateFaceMat = matReg.getBasic('syracuse_slate_face', {
    map: slateTextures[0] || null,
    color: slateTextures[0] ? '#FFFFFF' : '#1E2D3B',
  });

  // =========================================================================
  // 2. STATIC SYRACUSE COURT ENVIRONMENT (Merged per material -> 4 draws!)
  // =========================================================================
  const floorMesh = new THREE.Mesh(buildRadialMarbleFloorGeometry([0, 2.2, 4.5, 8.8], 40), floorMat);
  floorMesh.position.y = floorY;
  floorMesh.receiveShadow = true;
  floorMesh.name = 'syracuse_floor';
  rootGroup.add(floorMesh);

  const wallMesh = new THREE.Mesh(new THREE.PlaneGeometry(16.0, 7.2), wallMat);
  wallMesh.position.set(0, floorY + 3.6, -1.62);
  wallMesh.receiveShadow = true;
  wallMesh.name = 'syracuse_backdrop_wall';
  rootGroup.add(wallMesh);

  // Merged Greek Doric Columns + Capitals + Architrave + Low Plinth Wall (positioned cleanly behind slate z=-1.36)
  const marbleEnvParts = [];
  const colPositions = [-1.42, -0.88, 0.88, 1.42];
  const colZ = -1.36;
  for (const cx of colPositions) {
    const basePlinth = new THREE.BoxGeometry(0.30, 0.14, 0.30);
    basePlinth.translate(cx, 0.07, colZ);
    marbleEnvParts.push(basePlinth);

    const torusBase = new THREE.CylinderGeometry(0.125, 0.138, 0.08, 14);
    torusBase.translate(cx, 0.18, colZ);
    marbleEnvParts.push(torusBase);

    const shaft = new THREE.CylinderGeometry(0.102, 0.118, 2.12, 14);
    shaft.translate(cx, 1.28, colZ);
    marbleEnvParts.push(shaft);

    const echinus = new THREE.CylinderGeometry(0.148, 0.104, 0.09, 14);
    echinus.translate(cx, 2.38, colZ);
    marbleEnvParts.push(echinus);

    const abacus = new THREE.BoxGeometry(0.32, 0.08, 0.32);
    abacus.translate(cx, 2.46, colZ);
    marbleEnvParts.push(abacus);
  }

  // Overhead Greek Architrave & Cornice Beam
  const architrave = new THREE.BoxGeometry(3.60, 0.18, 0.36);
  architrave.translate(0, 2.59, -1.34);
  marbleEnvParts.push(architrave);

  const cornice = new THREE.BoxGeometry(3.76, 0.08, 0.42);
  cornice.translate(0, 2.72, -1.32);
  marbleEnvParts.push(cornice);

  // Back dado stone Wainscot
  const dadoWall = new THREE.BoxGeometry(12.0, 0.52, 0.08);
  dadoWall.translate(0, 0.26, -1.56);
  marbleEnvParts.push(dadoWall);

  const dadoCap = new THREE.BoxGeometry(12.0, 0.06, 0.12);
  dadoCap.translate(0, 0.55, -1.54);
  marbleEnvParts.push(dadoCap);

  const colonnadeMesh = new THREE.Mesh(safeMerge(marbleEnvParts), marbleMat);
  colonnadeMesh.receiveShadow = true;
  colonnadeMesh.name = 'syracuse_colonnade';
  rootGroup.add(colonnadeMesh);

  // =========================================================================
  // 3. SYRACUSE EQUATION SLATE AT [0.0, 1.98, -1.12] (2 draws: bronze frame + face)
  // =========================================================================
  const slateGroup = new THREE.Group();
  slateGroup.name = 'syracuse_slate_group';
  slateGroup.position.copy(slatePos);
  rootGroup.add(slateGroup);

  const slateW = 2.08;
  const slateH = 1.04;
  const frameT = 0.065;
  const frameD = 0.065;

  const bronzeEnvParts = [];
  const topB = new THREE.BoxGeometry(slateW + frameT * 2, frameT, frameD);
  topB.translate(0, slateH * 0.5 + frameT * 0.5, 0);
  bronzeEnvParts.push(topB);

  const botB = new THREE.BoxGeometry(slateW + frameT * 2, frameT, frameD);
  botB.translate(0, -slateH * 0.5 - frameT * 0.5, 0);
  bronzeEnvParts.push(botB);

  const leftB = new THREE.BoxGeometry(frameT, slateH, frameD);
  leftB.translate(-slateW * 0.5 - frameT * 0.5, 0, 0);
  bronzeEnvParts.push(leftB);

  const rightB = new THREE.BoxGeometry(frameT, slateH, frameD);
  rightB.translate(slateW * 0.5 + frameT * 0.5, 0, 0);
  bronzeEnvParts.push(rightB);

  const ledge = new THREE.BoxGeometry(slateW + 0.10, 0.03, 0.10);
  ledge.translate(0, -slateH * 0.5 - frameT + 0.015, 0.045);
  bronzeEnvParts.push(ledge);

  // Rear mounting struts attaching slate cleanly onto inner columns (z = -1.12 -> -1.30)
  for (const bx of [-0.88, 0.88]) {
    const strut = new THREE.BoxGeometry(0.08, 0.08, 0.16);
    strut.translate(bx, 0, -0.09);
    bronzeEnvParts.push(strut);
  }

  const slateFrameMesh = new THREE.Mesh(safeMerge(bronzeEnvParts), bronzeMat);
  slateFrameMesh.name = 'slate_bronze_frame';
  slateGroup.add(slateFrameMesh);

  const slateFaceMesh = new THREE.Mesh(new THREE.PlaneGeometry(slateW, slateH), slateFaceMat);
  slateFaceMesh.position.set(0, 0, 0.022);
  slateFaceMesh.name = 'slate_equation_face';
  slateGroup.add(slateFaceMesh);

  // =========================================================================
  // 4. CHARACTERS: ARCHIMEDES & KING HIERO II
  // =========================================================================
  const archimedes = createHumanoidRig(
    {
      id: 'archimedes',
      skinColor: '#E2AC86',
      suitColor: '#F5EFE0',
      accentColor: '#1D4ED8',
      trouserColor: '#E2AC86', // bare lower legs above Greek sandals
      shoeColor: '#5C341E',
      hairColor: '#E2E8F0',
      irisColor: '#2563EB',
      hasMustache: false,
      hasGoatee: false,
      headgear: 'none',
      hasMonocle: false,
      castShadow: true,
    },
    matReg
  );
  const archShoeMeshes = consolidateHumanoid(
    archimedes,
    {
      skin: '#E2AC86',
      suit: '#F5EFE0',
      accent: '#1D4ED8',
      trouser: '#E2AC86',
      shoe: '#5C341E',
    },
    '#E2E8F0', // jawTrim: true tucked Greek philosopher beard
    buildArchimedesHairGeometry(),
    multiMat
  );
  archimedes.root.scale.setScalar(0.95);
  archimedes.root.position.copy(archimedesBasePos);
  archimedes.groundFeet(floorY);
  rootGroup.add(archimedes.root);

  // Archimedes right-hand Brass Caliper prop (visible in Beat 2)
  const handCaliperParts = [];
  const calSpine = new THREE.CylinderGeometry(0.008, 0.008, 0.14, 8);
  calSpine.translate(0, 0.04, 0.04);
  handCaliperParts.push(calSpine);
  for (const s of [-1, 1]) {
    const jaw = new THREE.TorusGeometry(0.065, 0.009, 6, 12, Math.PI * 0.75);
    jaw.rotateY(Math.PI * 0.5);
    jaw.rotateX(s * 0.45);
    jaw.translate(s * 0.022, 0.02, 0.09);
    handCaliperParts.push(jaw);
  }
  const handCaliperMesh = new THREE.Mesh(safeMerge(handCaliperParts), bronzeMat);
  handCaliperMesh.visible = false;
  archimedes.armR.propMount.add(handCaliperMesh);

  const king = createHumanoidRig(
    {
      id: 'king_hiero',
      skinColor: '#DCA37B',
      suitColor: '#581C87', // Royal Tyrian purple toga
      accentColor: '#F59E0B', // Gleaming gold royal trim
      trouserColor: '#DCA37B',
      shoeColor: '#4A2511',
      hairColor: '#272220',
      irisColor: '#4B2E1E',
      hasMustache: false,
      hasGoatee: false,
      headgear: 'none',
      hasMonocle: false,
      castShadow: true,
    },
    matReg
  );
  const kingShoeMeshes = consolidateHumanoid(
    king,
    {
      skin: '#DCA37B',
      suit: '#581C87',
      accent: '#F59E0B',
      trouser: '#DCA37B',
      shoe: '#4A2511',
    },
    '#272220', // jawTrim: true royal dark beard
    buildKingCrownAndHairGeometry(),
    multiMat
  );
  king.root.scale.setScalar(0.98);
  king.root.position.copy(kingBasePos);
  king.groundFeet(floorY);
  rootGroup.add(king.root);

  // =========================================================================
  // 5. INTERACTIVE CENTERPIECE A: BEAT 3 BRONZE-RIMMED MARBLE BATHTUB
  // =========================================================================
  const bathtubGroup = new THREE.Group();
  bathtubGroup.name = 'eureka_bathtub_assembly';
  bathtubGroup.position.copy(bathtubBasePos);
  rootGroup.add(bathtubGroup);

  // 5a. Solid thick-walled marble tub basin via LatheGeometry + bronze rim, claws & spout (2 draws)
  const tubProfile = [
    new THREE.Vector2(0.001, 0.03),
    new THREE.Vector2(0.285, 0.03),
    new THREE.Vector2(0.355, 0.50),
    new THREE.Vector2(0.362, 0.52),
    new THREE.Vector2(0.335, 0.525),
    new THREE.Vector2(0.320, 0.50),
    new THREE.Vector2(0.255, 0.075),
    new THREE.Vector2(0.001, 0.075),
  ];
  const tubLathe = new THREE.LatheGeometry(tubProfile, 24);
  tubLathe.scale(1.22, 1.0, 1.00);
  const tubBasinMesh = new THREE.Mesh(tubLathe, marbleMat);
  tubBasinMesh.castShadow = true;
  tubBasinMesh.receiveShadow = true;
  bathtubGroup.add(tubBasinMesh);

  const tubBronzeParts = [];
  const tubRim = new THREE.TorusGeometry(0.348, 0.022, 8, 24);
  tubRim.rotateX(Math.PI * 0.5);
  tubRim.scale(1.22, 1.00, 1.0);
  tubRim.translate(0, 0.52, 0);
  tubBronzeParts.push(tubRim);

  const tubMidBand = new THREE.TorusGeometry(0.318, 0.012, 6, 22);
  tubMidBand.rotateX(Math.PI * 0.5);
  tubMidBand.scale(1.22, 1.00, 1.0);
  tubMidBand.translate(0, 0.28, 0);
  tubBronzeParts.push(tubMidBand);

  for (const [fx, fz] of [
    [-0.31, 0.21],
    [0.31, 0.21],
    [-0.31, -0.21],
    [0.31, -0.21],
  ]) {
    const claw = new THREE.SphereGeometry(0.038, 8, 6);
    claw.scale(1.1, 0.8, 1.3);
    claw.translate(fx, 0.03, fz);
    tubBronzeParts.push(claw);
  }

  // Overflow spout on right side (+X) pointing cleanly toward the glass beaker (x = 0.52)
  const spout = new THREE.CylinderGeometry(0.020, 0.024, 0.11, 10);
  spout.rotateZ(Math.PI * 0.44);
  spout.translate(0.45, 0.45, 0);
  tubBronzeParts.push(spout);

  const tubBronzeMesh = new THREE.Mesh(safeMerge(tubBronzeParts), bronzeMat);
  bathtubGroup.add(tubBronzeMesh);

  // 5b. Rising turquoise bath water disc strictly inside the tub inner wall
  const tubWaterGeo = new THREE.CylinderGeometry(0.312, 0.292, 0.055, 22);
  tubWaterGeo.scale(1.20, 1.0, 0.98);
  const tubWaterMesh = new THREE.Mesh(tubWaterGeo, waterMat);
  tubWaterMesh.position.set(0, 0.36, 0);
  bathtubGroup.add(tubWaterMesh);

  // 5c. Side Glass Beaker + Overflowing Water Stream + Rising Beaker Water
  const beakerWallGeo = new THREE.CylinderGeometry(0.076, 0.072, 0.28, 16, 1, true);
  beakerWallGeo.translate(0.52, 0.145, 0);
  const beakerBaseGeo = new THREE.CylinderGeometry(0.072, 0.072, 0.012, 16);
  beakerBaseGeo.translate(0.52, 0.006, 0);
  const beakerGlassMesh = new THREE.Mesh(safeMerge([beakerWallGeo, beakerBaseGeo]), glassMat);
  bathtubGroup.add(beakerGlassMesh);

  const beakerWaterMesh = new THREE.Mesh(new THREE.CylinderGeometry(0.068, 0.066, 0.22, 14), waterMat);
  beakerWaterMesh.position.set(0.52, 0.02, 0);
  bathtubGroup.add(beakerWaterMesh);

  const overflowStreamMesh = new THREE.Mesh(new THREE.CylinderGeometry(0.013, 0.017, 0.31, 8), waterMat);
  overflowStreamMesh.position.set(0.505, 0.29, 0);
  overflowStreamMesh.rotation.z = -0.10;
  bathtubGroup.add(overflowStreamMesh);

  // 5d. Bobbing Golden Rubber Ducky (1 vertex-colored mesh)
  const duckBody = new THREE.SphereGeometry(0.042, 10, 8);
  duckBody.scale(1.25, 0.88, 0.95);
  const duckHead = new THREE.SphereGeometry(0.028, 10, 8);
  duckHead.translate(0.030, 0.034, 0);
  const duckBill = new THREE.ConeGeometry(0.013, 0.030, 6);
  duckBill.rotateZ(-Math.PI * 0.5);
  duckBill.translate(0.060, 0.032, 0);
  const duckyMesh = new THREE.Mesh(
    mergeWithVertexColors([
      { geo: duckBody, color: '#FACC15' },
      { geo: duckHead, color: '#FDE047' },
      { geo: duckBill, color: '#F97316' },
    ]),
    multiMat
  );
  duckyMesh.position.set(0.21, 0.44, 0.16);
  bathtubGroup.add(duckyMesh);

  // 5e. Upward Cyan Buoyancy Arrows flanking Archimedes inside water (1 merged mesh)
  const arrowParts = [];
  for (const ax of [-0.26, 0.26]) {
    const stem = new THREE.CylinderGeometry(0.011, 0.011, 0.16, 8);
    stem.translate(ax, 0.08, 0.04);
    arrowParts.push(stem);
    const head = new THREE.ConeGeometry(0.028, 0.065, 8);
    head.translate(ax, 0.19, 0.04);
    arrowParts.push(head);
  }
  const tubBuoyancyArrows = new THREE.Mesh(safeMerge(arrowParts), cyanGlowMat);
  tubBuoyancyArrows.position.set(0, 0.42, 0);
  bathtubGroup.add(tubBuoyancyArrows);

  // 5f. InstancedMesh of 24 Splash Water Droplets (frustumCulled = false)
  const DROPLET_COUNT = 24;
  const dropletGeo = new THREE.SphereGeometry(0.020, 7, 6);
  const splashInstanced = new THREE.InstancedMesh(dropletGeo, waterMat, DROPLET_COUNT);
  splashInstanced.frustumCulled = false;
  bathtubGroup.add(splashInstanced);

  // =========================================================================
  // 6. INTERACTIVE CENTERPIECE B: MARBLE HYDRO-PEDESTAL (Beats 1, 2, 4, 5, 6)
  // =========================================================================
  const pedestalGroup = new THREE.Group();
  pedestalGroup.name = 'hydro_pedestal_assembly';
  pedestalGroup.position.copy(centerStagePos);
  rootGroup.add(pedestalGroup);

  const pedParts = [];
  const pedBase = new THREE.BoxGeometry(0.56, 0.08, 0.40);
  pedBase.translate(0, 0.04, 0);
  pedParts.push(pedBase);

  const pedCol = new THREE.CylinderGeometry(0.16, 0.19, 0.56, 16);
  pedCol.scale(1.25, 1.0, 0.88);
  pedCol.translate(0, 0.36, 0);
  pedParts.push(pedCol);

  const pedTop = new THREE.BoxGeometry(0.70, 0.08, 0.44);
  pedTop.translate(0, 0.68, 0);
  pedParts.push(pedTop);

  const pedestalMesh = new THREE.Mesh(safeMerge(pedParts), marbleMat);
  pedestalMesh.castShadow = true;
  pedestalMesh.receiveShadow = true;
  pedestalGroup.add(pedestalMesh);

  const tableTopY = 0.72;

  // 6a. Spiky Royal Gold Crown + Pure Gold Ingot Bar + Silver Core (3 draws)
  const crownMesh = new THREE.Mesh(buildSpikyGoldCrownGeometry(0.84), goldMat);
  crownMesh.castShadow = true;
  crownMesh.position.set(-0.14, tableTopY, 0.02);
  pedestalGroup.add(crownMesh);

  // Inner sneaky silver core tucked neatly inside the crown band (exposed in Beat 5!)
  const silverCoreMesh = new THREE.Mesh(new THREE.CylinderGeometry(0.056, 0.052, 0.048, 12), silverMat);
  silverCoreMesh.position.set(0, 0.042, 0);
  silverCoreMesh.visible = false;
  crownMesh.add(silverCoreMesh);

  const goldBarGeo = new THREE.BoxGeometry(0.15, 0.062, 0.088);
  const goldBarMesh = new THREE.Mesh(goldBarGeo, goldMat);
  goldBarMesh.castShadow = true;
  goldBarMesh.position.set(0.14, tableTopY + 0.031, 0.02);
  pedestalGroup.add(goldBarMesh);

  // 6b. Beat 4: Twin Glass Water Displacement Tanks + Rising Water Levels (3 draws)
  const twinTanksGroup = new THREE.Group();
  twinTanksGroup.position.set(0, tableTopY, 0);
  twinTanksGroup.visible = false;
  pedestalGroup.add(twinTanksGroup);

  const tankGlassParts = [];
  for (const tx of [-0.175, 0.175]) {
    const cyl = new THREE.CylinderGeometry(0.142, 0.142, 0.38, 18, 1, true);
    cyl.translate(tx, 0.19, 0.02);
    tankGlassParts.push(cyl);
    const disc = new THREE.CylinderGeometry(0.142, 0.142, 0.012, 18);
    disc.translate(tx, 0.006, 0.02);
    tankGlassParts.push(disc);
  }
  const twinTankGlassMesh = new THREE.Mesh(safeMerge(tankGlassParts), glassMat);
  twinTanksGroup.add(twinTankGlassMesh);

  const tankWaterLeft = new THREE.Mesh(new THREE.CylinderGeometry(0.133, 0.133, 0.20, 16), waterMat);
  tankWaterLeft.position.set(-0.175, 0.112, 0.02);
  twinTanksGroup.add(tankWaterLeft);

  const tankWaterRight = new THREE.Mesh(new THREE.CylinderGeometry(0.133, 0.133, 0.20, 16), waterMat);
  tankWaterRight.position.set(0.175, 0.112, 0.02);
  twinTanksGroup.add(tankWaterRight);

  // 6c. Beat 5: Bronze Hydrostatic Balance Beam + Articulated Side-Stirrup Scale Pans
  const balanceGroup = new THREE.Group();
  balanceGroup.position.set(0, tableTopY, -0.07);
  balanceGroup.visible = false;
  pedestalGroup.add(balanceGroup);

  const standParts = [];
  const pillar = new THREE.CylinderGeometry(0.022, 0.032, 0.54, 10);
  pillar.translate(0, 0.27, 0);
  standParts.push(pillar);
  const standBase = new THREE.CylinderGeometry(0.085, 0.105, 0.032, 12);
  standBase.translate(0, 0.016, 0);
  standParts.push(standBase);
  const balanceStandMesh = new THREE.Mesh(safeMerge(standParts), bronzeMat);
  balanceGroup.add(balanceStandMesh);

  const crossbarGeo = new THREE.CylinderGeometry(0.013, 0.013, 0.44, 10);
  crossbarGeo.rotateZ(Math.PI * 0.5);
  const balanceBeamMesh = new THREE.Mesh(crossbarGeo, bronzeMat);
  balanceBeamMesh.position.set(0, 0.52, 0);
  balanceGroup.add(balanceBeamMesh);

  function buildScalePanMesh() {
    const parts = [];
    // Flat round bronze dish at y = -0.25 below hook
    const dish = new THREE.CylinderGeometry(0.114, 0.102, 0.014, 16);
    dish.translate(0, -0.25, 0);
    parts.push(dish);
    // Outer side-stirrup wires (clear of the crown & gold bar in the center!)
    for (const sx of [-1, 1]) {
      const wire = new THREE.CylinderGeometry(0.004, 0.004, 0.272, 6);
      wire.rotateZ(sx * 0.415);
      wire.translate(sx * 0.056, -0.125, 0);
      parts.push(wire);
    }
    const hook = new THREE.SphereGeometry(0.012, 6, 6);
    parts.push(hook);
    return new THREE.Mesh(safeMerge(parts), bronzeMat);
  }
  const leftPanMesh = buildScalePanMesh();
  const rightPanMesh = buildScalePanMesh();
  balanceGroup.add(leftPanMesh);
  balanceGroup.add(rightPanMesh);

  // 6d. Beat 5-6: Contained Glass Wave Basin + Miniature Floating Greek Trireme Ship
  const shipTankGroup = new THREE.Group();
  shipTankGroup.position.set(0, tableTopY, 0.13);
  shipTankGroup.visible = false;
  pedestalGroup.add(shipTankGroup);

  const shipGlassParts = [];
  const tankW = 0.36;
  const tankH = 0.14;
  const tankD = 0.22;
  const wallT = 0.008;
  const tankFloor = new THREE.BoxGeometry(tankW, wallT, tankD);
  tankFloor.translate(0, wallT * 0.5, 0);
  shipGlassParts.push(tankFloor);
  const frontWall = new THREE.BoxGeometry(tankW, tankH, wallT);
  frontWall.translate(0, tankH * 0.5, tankD * 0.5 - wallT * 0.5);
  shipGlassParts.push(frontWall);
  const backWall = new THREE.BoxGeometry(tankW, tankH, wallT);
  backWall.translate(0, tankH * 0.5, -tankD * 0.5 + wallT * 0.5);
  shipGlassParts.push(backWall);
  const leftWall = new THREE.BoxGeometry(wallT, tankH, tankD - wallT * 2);
  leftWall.translate(-tankW * 0.5 + wallT * 0.5, tankH * 0.5, 0);
  shipGlassParts.push(leftWall);
  const rightWall = new THREE.BoxGeometry(wallT, tankH, tankD - wallT * 2);
  rightWall.translate(tankW * 0.5 - wallT * 0.5, tankH * 0.5, 0);
  shipGlassParts.push(rightWall);
  const shipTankGlassMesh = new THREE.Mesh(safeMerge(shipGlassParts), glassMat);
  shipTankGroup.add(shipTankGlassMesh);

  const shipTankWaterMesh = new THREE.Mesh(new THREE.BoxGeometry(tankW - 0.02, 0.095, tankD - 0.02), waterMat);
  shipTankWaterMesh.position.set(0, 0.056, 0);
  shipTankGroup.add(shipTankWaterMesh);

  const shipHull = new THREE.CylinderGeometry(0.058, 0.028, 0.25, 10);
  shipHull.rotateZ(Math.PI * 0.5);
  shipHull.scale(1.0, 0.65, 0.72);
  const shipMast = new THREE.CylinderGeometry(0.006, 0.008, 0.16, 6);
  shipMast.translate(0, 0.085, 0);
  const shipSail = new THREE.BoxGeometry(0.12, 0.095, 0.010);
  shipSail.translate(0, 0.095, 0.014);
  const shipMesh = new THREE.Mesh(
    mergeWithVertexColors([
      { geo: shipHull, color: '#92400E' },
      { geo: shipMast, color: '#D97706' },
      { geo: shipSail, color: '#FEF3C7' },
    ]),
    multiMat
  );
  shipMesh.position.set(0, 0.112, 0);
  shipTankGroup.add(shipMesh);

  // Beat 6 royal bronze display cushion for the pure crown
  const crownCushionMesh = new THREE.Mesh(new THREE.CylinderGeometry(0.105, 0.118, 0.036, 16), bronzeMat);
  crownCushionMesh.position.set(-0.21, tableTopY + 0.018, 0.03);
  crownCushionMesh.visible = false;
  pedestalGroup.add(crownCushionMesh);

  // 6e. Beat 6: Celebratory Golden Sparks / Confetti InstancedMesh (frustumCulled = false)
  const CONFETTI_COUNT = 24;
  const confettiGeo = new THREE.BoxGeometry(0.028, 0.028, 0.008);
  const confettiInstanced = new THREE.InstancedMesh(confettiGeo, goldMat, CONFETTI_COUNT);
  confettiInstanced.frustumCulled = false;
  confettiInstanced.visible = false;
  rootGroup.add(confettiInstanced);

  // =========================================================================
  // 7. FRAME UPDATE & BEAT CHOREOGRAPHY (0..30s)
  // =========================================================================
  const dummy = new THREE.Object3D();
  const state = {
    t: 0,
    beatIndex: 0,
    beatId: 'beat_1_royal_crown_hook',
    archimedesInTub: false,
  };

  function update(t = 0, manifest = null) {
    const time = clamp(Number(t) || 0, 0, 30.0);

    // Determine active beat index (0..5)
    let beatIdx = 0;
    if (time < 4.8) beatIdx = 0;
    else if (time < 9.8) beatIdx = 1;
    else if (time <= 15.2) beatIdx = 2;
    else if (time < 20.4) beatIdx = 3;
    else if (time < 25.4) beatIdx = 4;
    else beatIdx = 5;

    const beatIds = [
      'beat_1_royal_crown_hook',
      'beat_2_ruler_struggle',
      'beat_3_bathtub_eureka',
      'beat_4_dual_tank_trap',
      'beat_5_busted_buoyancy',
      'beat_6_syracuse_finale',
    ];
    state.t = time;
    state.beatIndex = beatIdx;
    state.beatId = beatIds[beatIdx];
    state.archimedesInTub = beatIdx === 2;

    // 1. Update Syracuse Equation Slate texture per beat
    if (slateTextures[beatIdx] && slateFaceMat.map !== slateTextures[beatIdx]) {
      slateFaceMat.map = slateTextures[beatIdx];
      slateFaceMat.needsUpdate = true;
    }

    // 2. Reset base character poses
    archimedes.resetPose();
    king.resetPose();

    archimedes.root.position.copy(archimedesBasePos);
    archimedes.root.rotation.set(0, 0.22, 0);
    archimedes.contactShadow.visible = beatIdx !== 2;

    king.root.position.set(beatIdx === 2 ? 0.64 : kingBasePos.x, 0.0, beatIdx === 2 ? 0.06 : kingBasePos.z);
    king.root.rotation.set(0, beatIdx === 2 ? -0.26 : -0.22, 0);
    king.contactShadow.visible = true;

    handCaliperMesh.visible = false;
    silverCoreMesh.visible = false;
    twinTanksGroup.visible = false;
    balanceGroup.visible = false;
    shipTankGroup.visible = false;
    crownCushionMesh.visible = false;
    confettiInstanced.visible = false;
    crownMesh.scale.setScalar(1.0);

    const breathe = Math.sin(time * 2.6) * 0.018;
    let archSmile = 0.25;
    let kingSmile = 0.15;

    if (beatIdx === 0) {
      // =====================================================================
      // BEAT 1 (0.0 .. 4.8s): King Hiero II presents suspect Crown & Gold Bar
      // =====================================================================
      bathtubGroup.visible = false;
      bathtubGroup.position.set(bathtubBasePos.x, 0, -0.65);
      pedestalGroup.visible = true;
      pedestalGroup.position.copy(centerStagePos);

      crownMesh.visible = true;
      goldBarMesh.visible = true;
      const gleamBump = time >= 1.80 && time <= 2.35 ? Math.sin(((time - 1.80) / 0.55) * Math.PI) * 0.05 : 0;
      crownMesh.position.set(-0.14, tableTopY + gleamBump, 0.02);
      crownMesh.rotation.set(0, time * 0.85, 0);
      goldBarMesh.position.set(0.14, tableTopY + 0.031, 0.02);
      goldBarMesh.rotation.set(0, -0.22, 0);

      // King Hiero gestures warmly above/right of the gold bar; outer arm hangs outside robe
      king.chestGroup.rotation.x = 0.04 + breathe;
      king.headGroup.rotation.y = -0.18;
      king.armL.shoulder.rotation.set(-1.18 + Math.sin(time * 4.5) * 0.08, 0.28, -0.02);
      king.armL.elbow.rotation.set(-0.92, 0, 0);
      king.armR.shoulder.rotation.set(-0.12, 0, 0.26);
      king.armR.elbow.rotation.set(-0.46, 0, 0);
      kingSmile = -0.15;

      // Archimedes inspects crown cleanly from left without overlapping spikes; recoils at 3.40s
      const boomShock = smoothstep(3.35, 3.55, time) * (1.0 - smoothstep(4.2, 4.75, time) * 0.5);
      archimedes.chestGroup.rotation.x = 0.08 - 0.18 * boomShock + breathe;
      archimedes.headGroup.rotation.y = 0.20;
      archimedes.headGroup.rotation.x = 0.06 - 0.22 * boomShock;
      archimedes.armR.shoulder.rotation.set(-1.35 - 0.35 * boomShock, -0.24, -0.04 + 0.14 * boomShock);
      archimedes.armR.elbow.rotation.set(-0.88, 0, 0);
      archimedes.armL.shoulder.rotation.set(-0.14 - 0.55 * boomShock, 0, -0.26 - 0.14 * boomShock);
      archimedes.armL.elbow.rotation.set(-0.48, 0, 0);
      archimedes.setHatPop(boomShock * 0.45);
      archSmile = boomShock > 0.2 ? -0.20 : 0.22;
    } else if (beatIdx === 1) {
      // =====================================================================
      // BEAT 2 (4.8 .. 9.8s): Caliper Struggle & Geometry Gives Up
      // =====================================================================
      bathtubGroup.visible = false;
      bathtubGroup.position.set(bathtubBasePos.x, 0, -0.65);
      pedestalGroup.visible = true;
      pedestalGroup.position.copy(centerStagePos);

      handCaliperMesh.visible = true;
      crownMesh.visible = true;
      goldBarMesh.visible = true;

      // Caliper boing at 5.60s makes the crown wobble!
      const boingDt = time - 5.60;
      const wobble = boingDt >= 0 && boingDt < 1.2 ? Math.sin(boingDt * 24.0) * Math.exp(-boingDt * 3.5) * 0.24 : 0;
      crownMesh.position.set(-0.12, tableTopY + Math.abs(wobble) * 0.07, 0.03);
      crownMesh.rotation.set(wobble * 0.45, time * 0.5, wobble);
      goldBarMesh.position.set(0.15, tableTopY + 0.031, 0.02);

      // Archimedes holds brass calipers safely above & left of crown while scratching temple
      const struggleOsc = Math.sin(time * 9.5);
      archimedes.root.rotation.y = 0.24;
      archimedes.chestGroup.rotation.x = 0.08 + breathe;
      archimedes.headGroup.rotation.y = 0.16 + struggleOsc * 0.06;
      archimedes.armR.shoulder.rotation.set(-1.42 + struggleOsc * 0.07, -0.28, 0.02);
      archimedes.armR.elbow.rotation.set(-0.82, 0, 0);
      archimedes.armL.shoulder.rotation.set(-1.32, 0.12, -0.28);
      archimedes.armL.elbow.rotation.set(-1.18, 0, 0);
      archSmile = -0.18;

      // King Hiero crosses arms cleanly in front of chest (zero chest clipping)
      king.chestGroup.rotation.x = -0.04 + breathe;
      king.headGroup.rotation.y = -0.22;
      king.armL.shoulder.rotation.set(-0.76, 0.72, -0.22);
      king.armL.elbow.rotation.set(-1.12, 0, 0);
      king.armR.shoulder.rotation.set(-0.86, -0.72, 0.22);
      king.armR.elbow.rotation.set(-1.12, 0, 0);
      kingSmile = -0.12;
    } else if (beatIdx === 2) {
      // =====================================================================
      // BEAT 3 (9.8 .. 15.2s): EUREKA BATHTUB SPLASH!
      // =====================================================================
      pedestalGroup.visible = false;
      pedestalGroup.position.set(centerStagePos.x, 0, -0.65);
      bathtubGroup.visible = true;
      bathtubGroup.position.copy(bathtubBasePos);

      // Archimedes springs high first before arcing inward over the left rim into the basin
      const hopU = smoothstep(9.8, 10.32, time);
      const hopX = smoothstep(9.92, 10.32, time);
      const hopArc = time < 10.35 ? Math.sin(Math.pow(hopU, 0.72) * Math.PI) * 0.64 : 0;
      const splashBob = time >= 10.35 ? Math.sin((time - 10.35) * 6.5) * 0.012 : 0;

      archimedes.root.position.set(
        lerp(archimedesBasePos.x, bathtubBasePos.x, hopX),
        lerp(0.0, -0.12, hopU) + hopArc + splashBob,
        lerp(archimedesBasePos.z, 0.07, hopX)
      );
      archimedes.root.rotation.set(0, lerp(0.22, 0.10, hopU), 0);
      archimedes.rig.position.set(0, 0, 0);

      // Seated pose: thighs forward & knees bent DOWN inside tub basin (zero wall piercing!)
      const legTuck = smoothstep(9.8, 10.22, time);
      const legBounce = time >= 10.35 ? Math.sin(time * 8.0) * 0.05 : 0;
      archimedes.pelvis.rotation.x = -0.10 * legTuck;
      archimedes.chestGroup.rotation.x = 0.10 * legTuck;
      archimedes.legL.hip.rotation.set(-1.52 * legTuck + legBounce, 0, -0.06 * legTuck);
      archimedes.legL.knee.rotation.x = 1.62 * legTuck;
      archimedes.legR.hip.rotation.set(-1.52 * legTuck - legBounce, 0, 0.06 * legTuck);
      archimedes.legR.knee.rotation.x = 1.62 * legTuck;

      // Joyful Eureka arms raised above tub rim and clear of King Hiero!
      const eurekaRaise = smoothstep(11.4, 12.4, time);
      const wave = Math.sin(time * 8.5) * 0.10;
      archimedes.armL.shoulder.rotation.set(lerp(-1.34, -2.26, eurekaRaise) + wave, 0.10, -0.34);
      archimedes.armL.elbow.rotation.set(-0.38, 0, 0);
      archimedes.armR.shoulder.rotation.set(lerp(-1.34, -2.26, eurekaRaise) - wave, -0.14, 0.28);
      archimedes.armR.elbow.rotation.set(-0.42, 0, 0);
      archimedes.eyeL.brow.position.y += 0.026;
      archimedes.setHatPop(eurekaRaise * 0.35);
      archSmile = 0.88;

      // Rising turquoise tub water + overflowing stream into side beaker
      const waterRise = smoothstep(10.15, 10.85, time);
      tubWaterMesh.position.y = lerp(0.34, 0.45, waterRise) + Math.sin(time * 7.0) * 0.007;
      overflowStreamMesh.visible = time >= 10.30;
      const beakerFill = smoothstep(10.35, 12.4, time);
      beakerWaterMesh.scale.set(1, Math.max(0.1, beakerFill), 1);
      beakerWaterMesh.position.set(0.52, 0.015 + beakerFill * 0.11, 0);

      // Bobbing golden rubber ducky (pops at sfx_07 11.40s!)
      const duckDt = time - 11.40;
      const duckPop = duckDt >= 0 && duckDt < 0.7 ? Math.sin((duckDt / 0.7) * Math.PI) * 0.12 : 0;
      duckyMesh.position.set(0.21, tubWaterMesh.position.y + 0.038 + duckPop, 0.16);
      duckyMesh.rotation.set(Math.sin(time * 5.2) * 0.12, time * 1.8, Math.cos(time * 4.5) * 0.10);

      // Glowing upward cyan buoyancy arrows flanking Archimedes inside water
      tubBuoyancyArrows.visible = time >= 11.8;
      tubBuoyancyArrows.position.y = 0.40 + Math.sin(time * 5.5) * 0.035;

      // Update 24 splash water droplets inside tub radius
      const splashAge = Math.max(0, time - 10.30);
      for (let i = 0; i < DROPLET_COUNT; i++) {
        const cycle = (splashAge * 1.6 + i * 0.13) % 1.1;
        const angle = (i / DROPLET_COUNT) * Math.PI * 2 + i * 0.7;
        const rad = 0.11 + cycle * 0.22;
        const dy = 0.44 + cycle * 0.42 - 0.55 * cycle * cycle;
        dummy.position.set(Math.cos(angle) * rad, Math.max(0.20, dy), Math.sin(angle) * rad * 0.85);
        const s = time >= 10.25 ? Math.max(0.2, 1.05 - cycle * 0.65) : 0.001;
        dummy.scale.setScalar(s);
        dummy.updateMatrix();
        splashInstanced.setMatrixAt(i, dummy.matrix);
      }
      splashInstanced.instanceMatrix.needsUpdate = true;

      // King Hiero stands clear on the right (x = 0.64), reacting in amazement
      king.chestGroup.rotation.x = -0.06 + breathe;
      king.headGroup.rotation.y = -0.26;
      king.eyeL.brow.position.y += 0.024;
      king.armL.shoulder.rotation.set(-1.18, 0.30, -0.10);
      king.armL.elbow.rotation.set(-0.88, 0, 0);
      king.armR.shoulder.rotation.set(-0.42, -0.08, 0.26);
      king.armR.elbow.rotation.set(-0.56, 0, 0);
      kingSmile = 0.35;
    } else if (beatIdx === 3) {
      // =====================================================================
      // BEAT 4 (15.2 .. 20.4s): TWIN GLASS WATER DISPLACEMENT TANKS
      // =====================================================================
      bathtubGroup.visible = false;
      bathtubGroup.position.set(bathtubBasePos.x, 0, -0.65);
      pedestalGroup.visible = true;
      pedestalGroup.position.copy(centerStagePos);

      twinTanksGroup.visible = true;
      crownMesh.visible = true;
      goldBarMesh.visible = true;

      // Dunking gold bar (left tank x = -0.175) and bulkier crown (right tank x = +0.175)
      const dunkU = smoothstep(15.6, 16.4, time);
      goldBarMesh.position.set(-0.175, lerp(tableTopY + 0.44, tableTopY + 0.08, dunkU), 0.02);
      goldBarMesh.rotation.set(0, time * 0.6, 0);

      crownMesh.position.set(0.175, lerp(tableTopY + 0.44, tableTopY + 0.06, dunkU), 0.02);
      crownMesh.rotation.set(0, -time * 0.6, 0);

      // Left tank rises modestly; Right (Crown) tank rises MUCH higher (ΔV_crown > ΔV_gold)!
      const leftH = lerp(0.85, 1.08, dunkU);
      const rightH = lerp(0.85, 1.72, dunkU);
      tankWaterLeft.scale.set(1, leftH, 1);
      tankWaterLeft.position.set(-0.175, 0.012 + 0.10 * leftH, 0.02);

      tankWaterRight.scale.set(1, rightH, 1);
      tankWaterRight.position.set(0.175, 0.012 + 0.10 * rightH, 0.02);

      // Archimedes gestures excitedly ABOVE and LEFT of the tanks (zero tank/water overlap!)
      archimedes.root.rotation.y = 0.24;
      archimedes.chestGroup.rotation.x = 0.06 + breathe;
      archimedes.headGroup.rotation.y = 0.18;
      archimedes.armR.shoulder.rotation.set(-1.84 + Math.sin(time * 6.0) * 0.06, -0.26, -0.08);
      archimedes.armR.elbow.rotation.set(-0.62, 0, 0);
      archimedes.armL.shoulder.rotation.set(-0.22, 0, -0.26);
      archimedes.armL.elbow.rotation.set(-0.52, 0, 0);
      archSmile = 0.72;

      // King Hiero gasps with hand tucked cleanly over his own upper chest (zero tank overlap!)
      king.chestGroup.rotation.x = 0.06 + breathe;
      king.headGroup.rotation.y = -0.24;
      king.eyeL.brow.position.y += 0.022;
      king.armL.shoulder.rotation.set(-1.16, 0.44, 0.04);
      king.armL.elbow.rotation.set(-1.18, 0, 0);
      king.armR.shoulder.rotation.set(-0.22, 0, 0.26);
      king.armR.elbow.rotation.set(-0.56, 0, 0);
      kingSmile = -0.20;
    } else if (beatIdx === 4) {
      // =====================================================================
      // BEAT 5 (20.4 .. 25.4s): BUSTED BY BUOYANCY (Hydro-Balance + Silver Core)
      // =====================================================================
      bathtubGroup.visible = false;
      bathtubGroup.position.set(bathtubBasePos.x, 0, -0.65);
      pedestalGroup.visible = true;
      pedestalGroup.position.copy(centerStagePos);

      balanceGroup.visible = true;
      crownMesh.visible = true;
      silverCoreMesh.visible = true;
      goldBarMesh.visible = true;

      // Balance beam tilts hard at sfx_11 (20.85s); articulated pans stay upright!
      const tiltU = smoothstep(20.6, 21.15, time);
      const tiltAngle = tiltU * 0.34 + Math.sin(time * 7.0) * 0.025 * tiltU;
      balanceBeamMesh.rotation.z = tiltAngle;

      const armSpan = 0.20;
      const leftHookX = -armSpan * Math.cos(tiltAngle);
      const leftHookY = 0.52 - armSpan * Math.sin(tiltAngle);
      const rightHookX = armSpan * Math.cos(tiltAngle);
      const rightHookY = 0.52 + armSpan * Math.sin(tiltAngle);

      leftPanMesh.position.set(leftHookX, leftHookY, 0);
      rightPanMesh.position.set(rightHookX, rightHookY, 0);

      // Gold bar rests cleanly on left pan dish; Crown rests cleanly on right pan dish
      goldBarMesh.position.set(leftHookX, tableTopY + leftHookY - 0.212, -0.07);
      goldBarMesh.rotation.set(0, 0.15, 0);
      crownMesh.position.set(rightHookX, tableTopY + rightHookY - 0.242, -0.07);
      crownMesh.rotation.set(0, time * 1.4, 0);

      // Show contained floating Greek ship in front of balance stand starting at 23.2s
      if (time >= 23.2) {
        shipTankGroup.visible = true;
        shipTankGroup.scale.setScalar(0.72);
        shipTankGroup.position.set(0, tableTopY, 0.13);
        shipMesh.position.set(0, 0.112 + Math.sin(time * 4.5) * 0.010, 0);
        shipMesh.rotation.set(0, 0.12, Math.sin(time * 3.8) * 0.06);
      }

      // King Hiero shakes royal fist safely outside & right of right pan/crown
      const fistShake = Math.sin(time * 12.0) * 0.10;
      king.chestGroup.rotation.x = 0.04 + breathe;
      king.headGroup.rotation.y = -0.18;
      king.armL.shoulder.rotation.set(-1.76 + fistShake, 0.48, 0.18);
      king.armL.elbow.rotation.set(-0.88, 0, 0);
      king.armR.shoulder.rotation.set(-0.28, 0, 0.26);
      king.armR.elbow.rotation.set(-0.62, 0, 0);
      kingSmile = -0.25;

      // Archimedes presents the buoyancy proof clearly above & left of the balance pan
      archimedes.chestGroup.rotation.x = -0.04 + breathe;
      archimedes.headGroup.rotation.y = 0.16;
      archimedes.armR.shoulder.rotation.set(-1.78, -0.28, -0.04);
      archimedes.armR.elbow.rotation.set(-0.68, 0, 0);
      archimedes.armL.shoulder.rotation.set(-0.26, 0, -0.26);
      archimedes.armL.elbow.rotation.set(-0.56, 0, 0);
      archSmile = 0.78;
    } else {
      // =====================================================================
      // BEAT 6 (25.4 .. 30.0s): SYRACUSE FINALE (Floating Ship + Pure Crown + Confetti)
      // =====================================================================
      bathtubGroup.visible = false;
      bathtubGroup.position.set(bathtubBasePos.x, 0, -0.65);
      pedestalGroup.visible = true;
      pedestalGroup.position.copy(centerStagePos);

      // Ship floats inside contained clear glass wave tank on right side of pedestal
      shipTankGroup.visible = true;
      shipTankGroup.scale.setScalar(1.0);
      shipTankGroup.position.set(0.11, tableTopY, 0.03);
      shipMesh.position.set(0, 0.112 + Math.sin(time * 4.2) * 0.012, 0);
      shipMesh.rotation.set(0, Math.sin(time * 1.5) * 0.16, Math.sin(time * 3.6) * 0.06);

      // Pure gleaming crown displayed cleanly on bronze cushion on left side of pedestal
      crownCushionMesh.visible = true;
      crownMesh.visible = true;
      crownMesh.position.set(-0.21, tableTopY + 0.038 + Math.sin(time * 5.0) * 0.010, 0.03);
      crownMesh.rotation.set(0, time * 1.6, 0);
      goldBarMesh.visible = false;

      // Golden celebratory confetti raining around outer flanks & top beam (keeping faces & slate text clear!)
      confettiInstanced.visible = true;
      const confElapsed = Math.max(0, time - 25.4);
      for (let i = 0; i < CONFETTI_COUNT; i++) {
        const fall = ((confElapsed * 0.55 + i * 0.17) % 1.8);
        const side = i % 2 === 0 ? -1 : 1;
        const lane = i % 4;
        const cx = lane === 0 ? (((i * 19) % 44) / 100 - 0.22) : side * (0.78 + ((i * 13) % 24) / 100);
        const cy = lane === 0 ? 2.68 - fall * 0.12 : 2.38 - fall * 1.05;
        const cz = lane === 0 ? -0.45 : (((i * 23) % 60) / 100 - 0.20);
        dummy.position.set(cx, cy, cz);
        dummy.rotation.set(time * 3.2 + i, time * 2.4 + i * 0.5, i);
        dummy.scale.setScalar(1.0);
        dummy.updateMatrix();
        confettiInstanced.setMatrixAt(i, dummy.matrix);
      }
      confettiInstanced.instanceMatrix.needsUpdate = true;

      // Archimedes & King Hiero celebrate with high wide arms (clear of crown & tank!)
      const cheer = Math.sin(time * 7.5) * 0.10;
      archimedes.root.rotation.y = 0.22;
      archimedes.chestGroup.rotation.x = -0.06 + breathe;
      archimedes.armR.shoulder.rotation.set(-2.12 + cheer, -0.10, 0.34);
      archimedes.armR.elbow.rotation.set(-0.32, 0, 0);
      archimedes.armL.shoulder.rotation.set(-1.55 - cheer, 0.12, -0.36);
      archimedes.armL.elbow.rotation.set(-0.45, 0, 0);
      archSmile = 0.94;

      king.root.rotation.y = -0.22;
      king.chestGroup.rotation.x = -0.05 + breathe;
      king.armL.shoulder.rotation.set(-2.05 - cheer, 0.10, -0.34);
      king.armL.elbow.rotation.set(-0.35, 0, 0);
      king.armR.shoulder.rotation.set(-1.45 + cheer, -0.12, 0.34);
      king.armR.elbow.rotation.set(-0.48, 0, 0);
      kingSmile = 0.88;
    }

    // 3. Drive Lip-Sync on both Archimedes and King Hiero II
    const archEnergy = computePhonemeEnergy(manifest, time, 'archimedes', timeline);
    const kingEnergy = computePhonemeEnergy(manifest, time, 'king', timeline);
    applyLipSync(archimedes, archEnergy, time, archSmile);
    applyLipSync(king, kingEnergy, time, kingSmile);

    // 4. MANDATORY FOOT GROUNDING ENFORCEMENT
    // Archimedes grounded at y = 0.0 in Beats 1, 2, 4, 5, 6 (t < 9.8 || t > 15.2)
    if (time < 9.8 || time > 15.2) {
      archimedes.groundFeet(floorY);
    } else {
      archimedes.root.updateMatrixWorld(true);
    }
    // King Hiero grounded at y = 0.0 across ALL timestamps 0..30s!
    king.groundFeet(floorY);
  }

  // Prune intermediate baseRig materials that were consolidated into vertex-colored multiMat
  {
    const usedMats = new Set([matReg.contactShadowMat]);
    rootGroup.traverse((obj) => {
      if (obj.material) {
        if (Array.isArray(obj.material)) obj.material.forEach((m) => usedMats.add(m));
        else usedMats.add(obj.material);
      }
    });
    for (const [k, m] of matReg.map.entries()) {
      if (!usedMats.has(m)) {
        m.dispose();
        matReg.map.delete(k);
      }
    }
  }

  // Initialize scene at t = 0
  update(0, null);

  function getAnchors() {
    archimedes.root.updateMatrixWorld(true);
    king.root.updateMatrixWorld(true);

    const archimedesHead = new THREE.Vector3(0, 0.38, 0).applyMatrix4(archimedes.headGroup.matrixWorld);
    const kingHead = new THREE.Vector3(0, 0.38, 0).applyMatrix4(king.headGroup.matrixWorld);

    let archMinY = Infinity;
    for (const m of archShoeMeshes) {
      const b = new THREE.Box3().setFromObject(m);
      if (b.min.y < archMinY) archMinY = b.min.y;
    }
    const archimedesFeet = new THREE.Vector3(
      archimedes.root.position.x,
      Number.isFinite(archMinY) ? archMinY : archimedes.root.position.y,
      archimedes.root.position.z
    );

    let kingMinY = Infinity;
    for (const m of kingShoeMeshes) {
      const b = new THREE.Box3().setFromObject(m);
      if (b.min.y < kingMinY) kingMinY = b.min.y;
    }
    const kingFeet = new THREE.Vector3(
      king.root.position.x,
      Number.isFinite(kingMinY) ? kingMinY : king.root.position.y,
      king.root.position.z
    );

    const slateBoard = slatePos.clone();
    const centerStage = new THREE.Vector3(centerStagePos.x, 0.55, centerStagePos.z);

    return {
      archimedesHead,
      archimedesFeet,
      kingHead,
      kingFeet,
      slateBoard,
      centerStage,
    };
  }

  function getState() {
    return { ...state };
  }

  return {
    archimedes,
    king,
    update,
    getAnchors,
    getState,
  };
}
