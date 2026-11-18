/**
 * Archimedes Eureka Main Application & Deterministic Frame Engine
 * (/home/antigravity/archimedes-eureka-3d/src/main.js)
 * Wires strict 3-light studio, dynamic per-beat FogExp2 atmospheric mood interpolation,
 * Syracuse Stage + Cast (`src/scene/archimedesStage.js`), declarative `baseDirector.js`,
 * clean boxless YouTube HUD (`hudCompositor.js`), and deterministic WebCodecs frame hooks.
 */
import * as THREE from 'three';
import { MaterialRegistry } from './engine/baseRig.js';
import { createBaseDirector } from './engine/baseDirector.js';
import { createHudCompositor, createFrameCompositor } from './engine/hudCompositor.js';
import { buildArchimedesScene } from './scene/archimedesStage.js';

function clamp01(v) {
  return v < 0 ? 0 : v > 1 ? 1 : v;
}

function lerp(a, b, t) {
  return a + (b - a) * t;
}

function smoothstep(e0, e1, x) {
  const u = clamp01((x - e0) / Math.max(1e-5, e1 - e0));
  return u * u * (3 - 2 * u);
}

const MOODS = {
  royalCourt: {
    bg: new THREE.Color('#2B1E16'),
    fogDensity: 0.052,
    hemiSky: new THREE.Color('#FFF2D8'),
    hemiGround: new THREE.Color('#3B281A'),
    hemiInt: 1.20,
    keyColor: new THREE.Color('#FFEAC8'),
    keyInt: 2.45,
    rimColor: new THREE.Color('#38BDF8'),
    rimInt: 1.18,
  },
  aquaBath: {
    bg: new THREE.Color('#0E2B3D'),
    fogDensity: 0.056,
    hemiSky: new THREE.Color('#E0F7FF'),
    hemiGround: new THREE.Color('#14354A'),
    hemiInt: 1.22,
    keyColor: new THREE.Color('#F0FAFF'),
    keyInt: 2.35,
    rimColor: new THREE.Color('#22D3EE'),
    rimInt: 1.38,
  },
  hydroLab: {
    bg: new THREE.Color('#1B2438'),
    fogDensity: 0.054,
    hemiSky: new THREE.Color('#EAF4FF'),
    hemiGround: new THREE.Color('#222D42'),
    hemiInt: 1.18,
    keyColor: new THREE.Color('#FFF0D4'),
    keyInt: 2.40,
    rimColor: new THREE.Color('#60A5FA'),
    rimInt: 1.28,
  },
  syracuseGold: {
    bg: new THREE.Color('#382412'),
    fogDensity: 0.048,
    hemiSky: new THREE.Color('#FFF6DC'),
    hemiGround: new THREE.Color('#472D16'),
    hemiInt: 1.28,
    keyColor: new THREE.Color('#FFE5A3'),
    keyInt: 2.60,
    rimColor: new THREE.Color('#38BDF8'),
    rimInt: 1.25,
  },
};

function blendMoods(a, b, alpha, outBg, outHemiSky, outHemiGround, outKey, outRim) {
  outBg.lerpColors(a.bg, b.bg, alpha);
  outHemiSky.lerpColors(a.hemiSky, b.hemiSky, alpha);
  outHemiGround.lerpColors(a.hemiGround, b.hemiGround, alpha);
  outKey.lerpColors(a.keyColor, b.keyColor, alpha);
  outRim.lerpColors(a.rimColor, b.rimColor, alpha);
  return {
    fogDensity: lerp(a.fogDensity, b.fogDensity, alpha),
    hemiInt: lerp(a.hemiInt, b.hemiInt, alpha),
    keyInt: lerp(a.keyInt, b.keyInt, alpha),
    rimInt: lerp(a.rimInt, b.rimInt, alpha),
  };
}

async function init() {
  const [timeline, manifest] = await Promise.all([
    fetch('/config/timeline.json').then((r) => r.json()),
    fetch('/audio/output/voice_manifest.json').then((r) => (r.ok ? r.json() : null)).catch(() => null),
  ]);

  const width = timeline.resolution?.width || 1080;
  const height = timeline.resolution?.height || 1920;
  const fps = timeline.fps || 30;
  const duration = timeline.duration || 30.0;
  const totalFrames = timeline.totalFrames || 900;

  const webglCanvas = document.getElementById('webgl-canvas');
  const hudCanvas = document.getElementById('hud-canvas');

  const renderer = new THREE.WebGLRenderer({
    canvas: webglCanvas,
    antialias: true,
    preserveDrawingBuffer: true,
    powerPreference: 'high-performance',
  });
  renderer.setPixelRatio(1);
  renderer.setSize(width, height, false);
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  renderer.toneMappingExposure = 1.08;
  renderer.outputColorSpace = THREE.SRGBColorSpace;
  renderer.shadowMap.enabled = true;
  renderer.shadowMap.type = THREE.PCFShadowMap;
  renderer.shadowMap.autoUpdate = false;

  const scene = new THREE.Scene();
  scene.background = MOODS.royalCourt.bg.clone();
  scene.fog = new THREE.FogExp2(MOODS.royalCourt.bg.getHex(), MOODS.royalCourt.fogDensity);

  const camera = new THREE.PerspectiveCamera(46.0, width / height, 0.1, 80);

  // Strict 3-Light Compiled Cap
  const hemiLight = new THREE.HemisphereLight('#FFF2D8', '#3B281A', 1.20);
  scene.add(hemiLight);

  const keyLight = new THREE.DirectionalLight('#FFEAC8', 2.45);
  keyLight.position.set(2.4, 4.6, 3.6);
  keyLight.target.position.set(0.0, 0.85, 0.0);
  keyLight.castShadow = true;
  keyLight.shadow.mapSize.set(1024, 1024);
  keyLight.shadow.camera.left = -2.4;
  keyLight.shadow.camera.right = 2.4;
  keyLight.shadow.camera.top = 2.4;
  keyLight.shadow.camera.bottom = -2.4;
  keyLight.shadow.camera.near = 0.5;
  keyLight.shadow.camera.far = 16.0;
  keyLight.shadow.bias = -0.0004;
  keyLight.shadow.normalBias = 0.025;
  scene.add(keyLight);
  scene.add(keyLight.target);

  const rimLight = new THREE.DirectionalLight('#38BDF8', 1.18);
  rimLight.position.set(-3.2, 3.0, -2.8);
  rimLight.castShadow = false;
  scene.add(rimLight);

  const registry = new MaterialRegistry();
  const stage = buildArchimedesScene(scene, registry, timeline);
  const director = createBaseDirector(camera, timeline);

  const sfxEvents = Array.isArray(timeline?.sfxEvents) ? timeline.sfxEvents : [];
  function drawOverlay(ctx, t) {
    for (let i = 0; i < sfxEvents.length; i++) {
      const ev = sfxEvents[i];
      const dt = t - ev.time;
      if (dt < 0 || dt > 0.28) continue;
      if (ev.type === 'eureka_splash' || ev.type === 'eureka_chime' || ev.type === 'anvil_slam') {
        const a = 0.28 * Math.exp(-dt * 12.0);
        if (a > 0.01) {
          const isAqua = ev.type === 'eureka_splash';
          const g = ctx.createRadialGradient(width * 0.5, height * 0.42, 20, width * 0.5, height * 0.42, height * 0.55);
          g.addColorStop(0.0, isAqua ? `rgba(186, 230, 253, ${a.toFixed(3)})` : `rgba(254, 243, 199, ${a.toFixed(3)})`);
          g.addColorStop(1.0, 'rgba(56, 189, 248, 0)');
          ctx.fillStyle = g;
          ctx.fillRect(0, 0, width, height);
        }
      }
    }
  }

  const hud = createHudCompositor(timeline, manifest, hudCanvas, {
    subtitleY: 1650,
    accentColor: '#38BDF8',
    accentGlow: 'rgba(56, 189, 248, 0.50)',
    drawOverlay,
  });
  const compositor = createFrameCompositor(webglCanvas, hudCanvas, width, height);

  const tmpBg = new THREE.Color();
  const tmpHemiSky = new THREE.Color();
  const tmpHemiGround = new THREE.Color();
  const tmpKey = new THREE.Color();
  const tmpRim = new THREE.Color();

  function updateAtmosphere(t) {
    let scalars;
    if (t < 15.0) {
      const u = smoothstep(9.5, 10.2, t);
      scalars = blendMoods(MOODS.royalCourt, MOODS.aquaBath, u, tmpBg, tmpHemiSky, tmpHemiGround, tmpKey, tmpRim);
    } else if (t < 24.8) {
      const u = smoothstep(15.0, 15.6, t);
      scalars = blendMoods(MOODS.aquaBath, MOODS.hydroLab, u, tmpBg, tmpHemiSky, tmpHemiGround, tmpKey, tmpRim);
    } else {
      const u = smoothstep(24.8, 25.6, t);
      scalars = blendMoods(MOODS.hydroLab, MOODS.syracuseGold, u, tmpBg, tmpHemiSky, tmpHemiGround, tmpKey, tmpRim);
    }

    scene.background.copy(tmpBg);
    scene.fog.color.copy(tmpBg);
    scene.fog.density = scalars.fogDensity;

    hemiLight.color.copy(tmpHemiSky);
    hemiLight.groundColor.copy(tmpHemiGround);
    hemiLight.intensity = scalars.hemiInt;
    keyLight.color.copy(tmpKey);
    keyLight.intensity = scalars.keyInt;
    rimLight.color.copy(tmpRim);
    rimLight.intensity = scalars.rimInt;
  }

  function renderFrame(frameIndex = 0) {
    const f = Math.max(0, Math.min(totalFrames - 1, Math.floor(frameIndex)));
    const t = f / fps;

    updateAtmosphere(t);
    stage.update(t, manifest);
    director.update(t, stage.getState ? stage.getState() : {});

    // Rebuild shadow map on frame 0 and every 2nd frame for 40% faster SwiftShader rasterization
    if (f === 0 || f % 2 === 0) {
      renderer.shadowMap.needsUpdate = true;
    }
    renderer.render(scene, camera);
    hud.render(t);

    return {
      frame: f,
      time: +t.toFixed(3),
      calls: renderer.info.render.calls,
      tris: renderer.info.render.triangles,
      materials: registry.count(),
    };
  }

  window.__RENDER_FRAME__ = renderFrame;
  window.__GET_COMPOSITE_CANVAS__ = () => compositor.composite();
  window.__SCENE_READY__ = true;

  const isRenderMode = new URLSearchParams(window.location.search).get('render') === '1';
  if (!isRenderMode) {
    let startTs = performance.now();
    function loop(now) {
      const elapsed = ((now - startTs) / 1000) % duration;
      renderFrame(Math.floor(elapsed * fps));
      requestAnimationFrame(loop);
    }
    requestAnimationFrame(loop);
  } else {
    renderFrame(0);
  }
}

init().catch((err) => {
  console.error('[ArchimedesMain Error]', err);
});
