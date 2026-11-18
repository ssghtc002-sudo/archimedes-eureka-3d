"""
Archimedes Eureka! Comedy Physics Audio, Mediterranean BGM & Foley Generator
(/home/antigravity/archimedes-eureka-3d/audio/generate_audio.py)

Synthesizes:
  1. All 12 multi-character dialogue lines via Kokoro ONNX (`am_puck` narrator,
     `bm_george` Archimedes pan -0.15, `am_michael` King Hiero pan +0.16)
     using shared `/home/antigravity/shorts-engine/audio/dspCore.py`.
  2. Lively 30.0s Mediterranean Greek comedic instrumental BGM bed (116 BPM, stereo 44.1 kHz):
     - Plucked bouzouki & ancient Greek lyre arpeggios, warm acoustic bass pizzicato,
       and light tambourine + doumbek hand-percussion groove.
     - Swells on beat transitions and ducks -5 dB under dialogue.
  3. All 14 procedural Foley SFX cues in `config/timeline.json`:
     `royal_horn`, `crown_gleam`, `vine_boom`, `caliper_boing`, `chalk_screech`,
     `eureka_splash`, `duck_squeak`, `eureka_chime`, `tank_dunk`, `water_overflow`,
     `anvil_slam`, `ship_horn_bell`, `fanfare_swell`, `crown_ding`.
  4. Mastered 44.1 kHz stereo WAV (`30.000s`, `-14.0 LUFS`, true peak <= -1.0 dBFS),
     `audio/output/voice_manifest.json` (with 900-frame lipSync curves for narrator,
     archimedes, king), and `output/subtitles.srt`.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
from scipy import signal

SHORTS_ENGINE_ROOT = Path("/home/antigravity/shorts-engine")
if str(SHORTS_ENGINE_ROOT) not in sys.path:
    sys.path.insert(0, str(SHORTS_ENGINE_ROOT))

from audio.dspCore import (
    CHANNELS,
    DURATION,
    FPS,
    PEAK_CEILING,
    SR,
    TOTAL_FRAMES,
    TOTAL_SAMPLES,
    add_stereo_event,
    adsr_env,
    butter_filter,
    compute_lip_sync,
    master_and_export,
    midi_to_hz,
    pan_stereo,
    peaking_eq,
    synthesize_dialogue,
    write_srt,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TIMELINE_PATH = PROJECT_ROOT / "config" / "timeline.json"

BPM = 116.0
BEAT = 60.0 / BPM         # 0.51724138 s
EIGHTH = BEAT * 0.5       # 0.25862069 s
SIXTEENTH = BEAT * 0.25   # 0.12931034 s

# Optional phonetic clarity hints for Kokoro TTS (display subtitles keep original timeline.json text)
TTS_OVERRIDES: dict[str, str] = {
    "line_01": "Archimedes! Is my crown pure gold, or fake?",
    "line_02": "One catch, melt the King's crown, and you're history!",
    "line_03": "A gold bar is easy, but how do you measure spiky curves?",
    "line_04": "Rulers fail, calipers slip, and geometry gives up!",
    "line_05": "Then Archimedes hops into a brim-full bath, splash!",
    "line_06": "Eureka! My body pushes out its exact volume of water!",
    "line_07": "Cheat with lighter silver, and the crown must be bulkier!",
}


def load_timeline(path: Path = TIMELINE_PATH) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Mediterranean Greek Comedic Instruments (116 BPM)
# ---------------------------------------------------------------------------
def synth_bouzouki_pluck(
    freq: float,
    dur: float = 0.36,
    gain: float = 0.19,
    tremolo: bool = False,
    bright: float = 1.0,
) -> np.ndarray:
    """
    Bright Mediterranean double-course plucked bouzouki / cittern with crisp plectrum transient
    and warm wooden soundboard resonance.
    """
    n = int(round(dur * SR))
    if n <= 0:
        return np.zeros(0, dtype=np.float32)
    t = np.linspace(0.0, dur, n, endpoint=False, dtype=np.float32)

    # Double-course slight unison detune (+1.4 cents / -1.4 cents)
    f1 = freq * 0.9992
    f2 = freq * 1.0008
    decay_rate = 3.2 if tremolo else 7.4

    str1 = (
        0.58 * np.sin(2.0 * math.pi * f1 * t) * np.exp(-t * decay_rate)
        + 0.28 * bright * np.sin(2.0 * math.pi * (f1 * 2.0) * t) * np.exp(-t * (decay_rate * 1.5))
        + 0.14 * bright * np.sin(2.0 * math.pi * (f1 * 3.0) * t) * np.exp(-t * (decay_rate * 2.1))
        + 0.07 * bright * np.sin(2.0 * math.pi * (f1 * 4.01) * t) * np.exp(-t * (decay_rate * 2.9))
    )
    str2 = (
        0.52 * np.sin(2.0 * math.pi * f2 * t + 0.35) * np.exp(-t * decay_rate)
        + 0.25 * bright * np.sin(2.0 * math.pi * (f2 * 2.0) * t + 0.7) * np.exp(-t * (decay_rate * 1.5))
        + 0.12 * bright * np.sin(2.0 * math.pi * (f2 * 3.0) * t + 1.1) * np.exp(-t * (decay_rate * 2.1))
    )

    # Crisp plectrum pick click
    pick = np.sin(2.0 * math.pi * min(freq * 5.5, 3900.0) * t) * np.exp(-t * 130.0) * 0.18 * bright
    sig = 0.55 * (str1 + str2) + pick

    if tremolo:
        # Expressive rapid Mediterranean plectrum tremolo (13.5 Hz)
        trem_mod = 0.72 + 0.28 * np.sin(2.0 * math.pi * 13.5 * t)
        env = adsr_env(n, attack=0.008, decay=0.06, sustain=0.82, release=0.06)
        sig = sig * trem_mod * env
    else:
        attack = np.clip(t / 0.0022, 0.0, 1.0)
        sig = sig * attack
        fade = min(n, int(0.025 * SR))
        if fade > 1:
            sig[-fade:] *= np.linspace(1.0, 0.0, fade, dtype=np.float32)

    return (sig * gain).astype(np.float32)


def synth_ancient_lyre(freq: float, dur: float = 0.52, gain: float = 0.16) -> np.ndarray:
    """Warm resonant ancient Greek kithara / tortoise-shell lyre finger-pluck."""
    n = int(round(dur * SR))
    if n <= 0:
        return np.zeros(0, dtype=np.float32)
    t = np.linspace(0.0, dur, n, endpoint=False, dtype=np.float32)

    # Gentle warm harmonic series with soft finger-pad attack
    harm = (
        0.66 * np.sin(2.0 * math.pi * freq * t) * np.exp(-t * 4.8)
        + 0.24 * np.sin(2.0 * math.pi * (freq * 2.0) * t) * np.exp(-t * 8.5)
        + 0.09 * np.sin(2.0 * math.pi * (freq * 3.002) * t) * np.exp(-t * 14.0)
        + 0.04 * np.sin(2.0 * math.pi * (freq * 4.005) * t) * np.exp(-t * 22.0)
    )
    attack = np.clip(t / 0.0045, 0.0, 1.0)
    sig = harm * attack * gain
    fade = min(n, int(0.04 * SR))
    if fade > 1:
        sig[-fade:] *= np.linspace(1.0, 0.0, fade, dtype=np.float32)
    return sig.astype(np.float32)


def synth_bass_pizz(freq: float, dur: float = 0.42, gain: float = 0.25, slide: float = 0.0) -> np.ndarray:
    """Warm buoyant acoustic upright bass / laouto bass pizzicato."""
    n = int(round(dur * SR))
    if n <= 0:
        return np.zeros(0, dtype=np.float32)
    t = np.linspace(0.0, dur, n, endpoint=False, dtype=np.float32)

    if slide != 0.0:
        f_curve = freq * (1.0 + slide * np.clip(t / max(0.08, dur * 0.8), 0.0, 1.0))
    else:
        f_curve = freq * (1.0 + 0.032 * np.exp(-t * 90.0))
    phase = 2.0 * math.pi * np.cumsum(f_curve) / SR

    body = (
        0.74 * np.sin(phase) * np.exp(-t * 4.9)
        + 0.23 * np.sin(phase * 2.0) * np.exp(-t * 9.6)
        + 0.09 * np.sin(phase * 3.0) * np.exp(-t * 15.5)
    )
    pluck = np.sin(2.0 * math.pi * (freq * 4.0) * t) * np.exp(-t * 72.0) * 0.11
    attack = np.clip(t / 0.0038, 0.0, 1.0)
    sig = butter_filter((body + pluck) * attack, "low", 740.0, order=2)
    fade = min(n, int(0.032 * SR))
    if fade > 1:
        sig[-fade:] *= np.linspace(1.0, 0.0, fade, dtype=np.float32)
    return (sig * gain).astype(np.float32)


def synth_doumbek(stroke: str = "dum", gain: float = 0.20) -> np.ndarray:
    """Mediterranean goblet hand drum (toubeleki / doumbek): warm 'dum' center or crisp 'tek' rim."""
    if stroke == "dum":
        dur = 0.17
        n = int(round(dur * SR))
        t = np.linspace(0.0, dur, n, endpoint=False, dtype=np.float32)
        f = 132.0 * (0.48 ** np.clip(t / 0.06, 0.0, 1.0)) + 52.0
        head = np.sin(2.0 * math.pi * np.cumsum(f) / SR) * np.exp(-t * 21.0)
        slap = np.sin(2.0 * math.pi * 290.0 * t) * np.exp(-t * 85.0) * 0.22
        attack = np.clip(t / 0.0025, 0.0, 1.0)
        return ((head + slap) * attack * gain).astype(np.float32)
    else:
        dur = 0.095
        n = int(round(dur * SR))
        t = np.linspace(0.0, dur, n, endpoint=False, dtype=np.float32)
        rim = (
            0.65 * np.sin(2.0 * math.pi * 410.0 * t) * np.exp(-t * 48.0)
            + 0.35 * np.sin(2.0 * math.pi * 685.0 * t) * np.exp(-t * 68.0)
        )
        finger = np.sin(2.0 * math.pi * 1480.0 * t) * np.exp(-t * 150.0) * 0.24
        attack = np.clip(t / 0.0015, 0.0, 1.0)
        return ((rim + finger) * attack * gain).astype(np.float32)


def synth_tambourine(rng: np.random.Generator, accent: bool = False, gain: float = 0.075) -> np.ndarray:
    """Crisp Mediterranean brass-jingle tambourine shake/tap."""
    dur = 0.095 if accent else 0.065
    n = int(round(dur * SR))
    t = np.linspace(0.0, dur, n, endpoint=False, dtype=np.float32)
    noise = butter_filter(rng.standard_normal(n).astype(np.float32), "band", (4800.0, 10800.0), order=2)
    jingle = (
        0.45 * np.sin(2.0 * math.pi * 6150.0 * t)
        + 0.35 * np.sin(2.0 * math.pi * 7820.0 * t)
        + 0.25 * np.sin(2.0 * math.pi * 9340.0 * t)
    ) * np.exp(-t * (42.0 if accent else 58.0))
    env = np.sin(math.pi * np.clip(t / dur, 0.0, 1.0)) ** (1.15 if accent else 1.55)
    sig = (noise * 0.72 + jingle * 0.28) * env * gain * (1.28 if accent else 0.85)
    return sig.astype(np.float32)


def synth_mediterranean_pad(
    midis: list[int],
    dur: float,
    gain: float = 0.16,
    attack: float = 0.08,
    release: float = 0.22,
) -> np.ndarray:
    """Warm Mediterranean accordion/brass-lyre harmonic pad that swells on transitions."""
    n = int(round(dur * SR))
    if n <= 0:
        return np.zeros((0, CHANNELS), dtype=np.float32)
    t = np.linspace(0.0, dur, n, endpoint=False, dtype=np.float32)

    left = np.zeros(n, dtype=np.float32)
    right = np.zeros(n, dtype=np.float32)
    vib = 1.0 + 0.0025 * np.sin(2.0 * math.pi * 4.8 * t)

    for idx, m in enumerate(midis):
        f = midi_to_hz(m)
        fl = f * (0.9986 - 0.0004 * idx)
        fr = f * (1.0014 + 0.0004 * idx)
        pl = 2.0 * math.pi * np.cumsum(fl * vib) / SR
        pr = 2.0 * math.pi * np.cumsum(fr * vib) / SR

        wl = 0.56 * np.sin(pl) + 0.26 * np.sin(2.0 * pl) + 0.12 * np.sin(3.0 * pl) + 0.05 * np.sin(4.0 * pl)
        wr = 0.56 * np.sin(pr) + 0.26 * np.sin(2.0 * pr) + 0.12 * np.sin(3.0 * pr) + 0.05 * np.sin(4.0 * pr)
        w = 1.0 / (1.0 + 0.12 * idx)
        left += wl * w
        right += wr * w

    norm = max(1.0, len(midis) * 0.62)
    stereo = np.stack([left / norm, right / norm], axis=-1).astype(np.float32)
    stereo = butter_filter(stereo, "low", 2300.0, order=2)
    # Carve pocket around 2 kHz for vocal clarity
    stereo[:, 0] = peaking_eq(stereo[:, 0], 2100.0, 0.9, -2.6)
    stereo[:, 1] = peaking_eq(stereo[:, 1], 2100.0, 0.9, -2.6)

    env = adsr_env(n, attack=attack, decay=0.14, sustain=0.80, release=release)
    return (stereo * env[:, None] * gain).astype(np.float32)


def apply_stereo_echo(bus: np.ndarray, delay_sec: float = 0.26, feedback: float = 0.26) -> np.ndarray:
    d = int(round(delay_sec * SR))
    out = np.copy(bus)
    if d <= 0 or d >= len(bus):
        return out
    tap1 = np.zeros_like(bus)
    tap1[d:, 0] = bus[:-d, 1] * feedback
    tap1[d:, 1] = bus[:-d, 0] * feedback
    d2 = d * 2
    tap2 = np.zeros_like(bus)
    if d2 < len(bus):
        tap2[d2:, 0] = bus[:-d2, 0] * (feedback * 0.38)
        tap2[d2:, 1] = bus[:-d2, 1] * (feedback * 0.38)
    wet = butter_filter(tap1 + tap2, "low", 2800.0, order=2)
    return (out + wet).astype(np.float32)


def generate_bgm(timeline: dict[str, Any]) -> np.ndarray:
    """
    Compose 30.0s lively Mediterranean Greek comedic instrumental BGM bed (116 BPM)
    with plucked bouzouki/lyre arpeggios, warm bass pizzicato, light tambourine/doumbek
    groove, and transition swells at beat boundaries.
    """
    rng = np.random.default_rng(20260287)
    bgm = np.zeros((TOTAL_SAMPLES, CHANNELS), dtype=np.float32)
    plucked_layer = np.zeros((TOTAL_SAMPLES, CHANNELS), dtype=np.float32)

    # ------------------------------------------------------------------
    # BEAT 1 (0.00..4.80s): Royal Syracuse Hook (D Dorian / Hijaz flavor)
    # ------------------------------------------------------------------
    # Warm royal lyre chord bed at opening
    add_stereo_event(bgm, synth_mediterranean_pad([50, 57, 62, 66, 69], 2.2, gain=0.15), 0.05)
    add_stereo_event(bgm, synth_mediterranean_pad([55, 62, 67, 70, 74], 2.1, gain=0.14), 2.45)

    b1_melody = [
        # (time_s, midi, dur, gain, pan, inst)
        (0.06, 62, 0.30, 0.20, -0.24, "bouzouki"),
        (0.32, 66, 0.28, 0.18, 0.22, "lyre"),
        (0.58, 69, 0.32, 0.20, -0.20, "bouzouki"),
        (0.84, 74, 0.34, 0.21, 0.24, "bouzouki"),
        (1.10, 72, 0.28, 0.18, -0.22, "lyre"),
        (1.36, 69, 0.28, 0.18, 0.20, "bouzouki"),
        (1.62, 66, 0.30, 0.19, -0.18, "lyre"),
        (1.88, 69, 0.36, 0.20, 0.22, "bouzouki"),
        (2.40, 74, 0.30, 0.21, -0.24, "bouzouki"),
        (2.66, 70, 0.28, 0.18, 0.22, "lyre"),
        (2.92, 69, 0.28, 0.19, -0.20, "bouzouki"),
        (3.18, 67, 0.28, 0.18, 0.20, "lyre"),
        (3.70, 66, 0.28, 0.19, -0.22, "bouzouki"),
        (3.96, 69, 0.28, 0.20, 0.24, "lyre"),
        (4.22, 74, 0.34, 0.22, -0.18, "bouzouki"),
        (4.48, 76, 0.34, 0.23, 0.18, "bouzouki"),
    ]
    for st_s, m, dur_s, g, pan, inst in b1_melody:
        wav = synth_bouzouki_pluck(midi_to_hz(m), dur_s, g) if inst == "bouzouki" else synth_ancient_lyre(midi_to_hz(m), dur_s, g)
        add_stereo_event(plucked_layer, pan_stereo(wav, pan), st_s)

    b1_bass = [
        (0.06, 38, 0.42, 0.26), (0.58, 45, 0.38, 0.23), (1.10, 38, 0.42, 0.25), (1.62, 45, 0.38, 0.23),
        (2.14, 43, 0.42, 0.25), (2.66, 50, 0.38, 0.23), (3.18, 45, 0.38, 0.24), (3.70, 38, 0.42, 0.26),
        (4.22, 45, 0.38, 0.25),
    ]
    for st_s, m, dur_s, g in b1_bass:
        add_stereo_event(bgm, pan_stereo(synth_bass_pizz(midi_to_hz(m), dur_s, g), -0.04), st_s)

    # ------------------------------------------------------------------
    # BEAT 2 (4.80..9.80s): Quirky Geometric Struggle (playful head-scratching plucks)
    # ------------------------------------------------------------------
    b2_notes = [69, 70, 69, 67, 66, 67, 64, 62, 69, 72, 70, 69, 67, 64, 66, 62, 64, 67, 69]
    b2_bass_notes = [38, 45, 43, 42, 40, 45, 38, 43, 45, 38]
    t_cur = 4.82
    idx = 0
    while t_cur < 9.55:
        mn = b2_notes[idx % len(b2_notes)]
        pan = -0.26 if idx % 2 == 0 else 0.26
        if idx % 3 == 1:
            wav = synth_ancient_lyre(midi_to_hz(mn + 5), 0.36, 0.16)
        else:
            wav = synth_bouzouki_pluck(midi_to_hz(mn), 0.30, 0.18, bright=0.92)
        add_stereo_event(plucked_layer, pan_stereo(wav, pan), t_cur)

        if idx % 2 == 0:
            bn = b2_bass_notes[(idx // 2) % len(b2_bass_notes)]
            slide = -0.035 if idx % 4 == 2 else 0.0
            add_stereo_event(bgm, pan_stereo(synth_bass_pizz(midi_to_hz(bn), 0.38, 0.24, slide=slide), 0.0), t_cur)
        t_cur += EIGHTH
        idx += 1

    # ------------------------------------------------------------------
    # BEAT 3 (9.80..15.20s): Bathtub Eureka! (Upward Lyre Glissando -> Joyous Island Groove)
    # ------------------------------------------------------------------
    # Shimmering upward Greek lyre glissando at beat 2->3 transition (9.65..10.10s)
    gliss_midis = [57, 59, 62, 64, 66, 69, 71, 74, 76, 78, 81]
    for i, m in enumerate(gliss_midis):
        add_stereo_event(
            plucked_layer,
            pan_stereo(synth_ancient_lyre(midi_to_hz(m), 0.45, 0.15 + 0.005 * i), -0.35 + 0.07 * i),
            9.64 + i * 0.042,
        )

    add_stereo_event(bgm, synth_mediterranean_pad([50, 57, 62, 66, 74], 2.5, gain=0.16), 10.10)
    add_stereo_event(bgm, synth_mediterranean_pad([55, 62, 67, 71, 74], 2.4, gain=0.17), 12.60)

    b3_melody = [74, 78, 81, 78, 76, 74, 71, 69, 74, 76, 78, 81, 83, 81, 78, 74]
    b3_bass = [38, 45, 43, 50, 45, 52, 38, 45]
    t_cur = 10.15
    step = 0
    while t_cur < 15.05:
        mn = b3_melody[step % len(b3_melody)]
        pan = -0.28 if step % 2 == 0 else 0.28
        is_trem = (step % 4 == 0)
        wav = synth_bouzouki_pluck(midi_to_hz(mn), 0.32, 0.20, tremolo=is_trem, bright=1.10)
        add_stereo_event(plucked_layer, pan_stereo(wav, pan), t_cur)
        if step % 2 == 1:
            lyre_w = synth_ancient_lyre(midi_to_hz(mn - 5), 0.38, 0.14)
            add_stereo_event(plucked_layer, pan_stereo(lyre_w, -pan), t_cur)
        if step % 2 == 0:
            bn = b3_bass[(step // 2) % len(b3_bass)]
            add_stereo_event(bgm, pan_stereo(synth_bass_pizz(midi_to_hz(bn), 0.36, 0.25), 0.0), t_cur)
        t_cur += EIGHTH
        step += 1

    # ------------------------------------------------------------------
    # BEAT 4 (15.20..20.40s): Dual Tank Density Trap (Inquisitive Detective Bouzouki & Lyre)
    # ------------------------------------------------------------------
    add_stereo_event(bgm, synth_mediterranean_pad([52, 59, 64, 67, 71], 2.5, gain=0.15), 15.20)
    add_stereo_event(bgm, synth_mediterranean_pad([45, 52, 61, 64, 69], 2.5, gain=0.16), 17.75)

    b4_melody = [64, 67, 71, 72, 71, 67, 66, 64, 69, 72, 76, 75, 76, 72, 69, 66]
    b4_bass = [40, 47, 45, 42, 40, 47, 45, 45]
    t_cur = 15.22
    step = 0
    while t_cur < 20.20:
        mn = b4_melody[step % len(b4_melody)]
        pan = -0.25 if step % 2 == 0 else 0.25
        wav = synth_bouzouki_pluck(midi_to_hz(mn), 0.28, 0.18, bright=1.02)
        add_stereo_event(plucked_layer, pan_stereo(wav, pan), t_cur)
        if step % 2 == 0:
            bn = b4_bass[(step // 2) % len(b4_bass)]
            add_stereo_event(bgm, pan_stereo(synth_bass_pizz(midi_to_hz(bn), 0.36, 0.25), -0.03), t_cur)
        t_cur += EIGHTH
        step += 1

    # ------------------------------------------------------------------
    # BEAT 5 (20.40..25.40s): Busted by Buoyancy! (Heroic Accusation -> Triumphant Swell)
    # ------------------------------------------------------------------
    # Transition pickup flourish at 20.20..20.45s
    for i, m in enumerate([64, 66, 69, 71, 74, 78]):
        add_stereo_event(
            plucked_layer,
            pan_stereo(synth_bouzouki_pluck(midi_to_hz(m), 0.24, 0.16 + 0.01 * i, bright=1.12), -0.24 + 0.09 * i),
            20.12 + i * 0.05,
        )

    add_stereo_event(bgm, synth_mediterranean_pad([50, 57, 62, 66, 74], 2.4, gain=0.18), 20.45)
    add_stereo_event(bgm, synth_mediterranean_pad([55, 62, 67, 71, 79], 2.4, gain=0.19), 22.90)

    b5_melody = [74, 78, 81, 86, 83, 81, 78, 74, 76, 79, 83, 86, 85, 81, 78, 74]
    b5_bass = [38, 45, 43, 50, 45, 49, 38, 45]
    t_cur = 20.45
    step = 0
    while t_cur < 25.25:
        mn = b5_melody[step % len(b5_melody)]
        pan = -0.30 if step % 2 == 0 else 0.30
        wav = synth_bouzouki_pluck(midi_to_hz(mn), 0.30, 0.20, tremolo=(step % 3 == 0), bright=1.14)
        add_stereo_event(plucked_layer, pan_stereo(wav, pan), t_cur)
        if step % 2 == 0:
            bn = b5_bass[(step // 2) % len(b5_bass)]
            add_stereo_event(bgm, pan_stereo(synth_bass_pizz(midi_to_hz(bn), 0.36, 0.26), 0.0), t_cur)
        t_cur += EIGHTH
        step += 1

    # ------------------------------------------------------------------
    # BEAT 6 (25.40..30.00s): Syracuse Finale! (Celebratory Full Mediterranean Ensemble + Button Sting)
    # ------------------------------------------------------------------
    add_stereo_event(bgm, synth_mediterranean_pad([50, 57, 62, 66, 74, 81], 2.1, gain=0.20), 25.40)
    add_stereo_event(bgm, synth_mediterranean_pad([55, 62, 67, 71, 79, 83], 2.1, gain=0.21), 27.50)

    b6_melody = [81, 78, 74, 78, 81, 86, 83, 81, 79, 76, 74, 78, 81, 85, 86]
    b6_bass = [38, 45, 43, 50, 45, 49, 50, 45]
    t_cur = 25.42
    step = 0
    while t_cur < 29.40:
        mn = b6_melody[step % len(b6_melody)]
        pan = -0.32 if step % 2 == 0 else 0.32
        wav = synth_bouzouki_pluck(midi_to_hz(mn), 0.32, 0.21, tremolo=True, bright=1.18)
        add_stereo_event(plucked_layer, pan_stereo(wav, pan), t_cur)
        lyre_w = synth_ancient_lyre(midi_to_hz(mn - 12), 0.40, 0.15)
        add_stereo_event(plucked_layer, pan_stereo(lyre_w, -pan * 0.8), t_cur)

        if step % 2 == 0:
            bn = b6_bass[(step // 2) % len(b6_bass)]
            add_stereo_event(bgm, pan_stereo(synth_bass_pizz(midi_to_hz(bn), 0.36, 0.27), 0.0), t_cur)
        t_cur += EIGHTH
        step += 1

    # Cheeky Final Syracuse Button Sting at 29.62s!
    add_stereo_event(
        bgm,
        synth_mediterranean_pad([50, 57, 62, 66, 74, 78, 86], 0.34, gain=0.26, attack=0.012, release=0.15),
        29.62,
    )
    add_stereo_event(bgm, pan_stereo(synth_bass_pizz(midi_to_hz(38), 0.32, 0.30), 0.0), 29.62)
    add_stereo_event(
        plucked_layer,
        pan_stereo(synth_bouzouki_pluck(midi_to_hz(86), 0.28, 0.24, tremolo=True, bright=1.25), -0.15),
        29.62,
    )
    add_stereo_event(
        plucked_layer,
        pan_stereo(synth_ancient_lyre(midi_to_hz(90), 0.22, 0.20), 0.18),
        29.74,
    )

    # Add subtle Mediterranean courtyard echo to plucked strings
    bgm += apply_stereo_echo(plucked_layer, delay_sec=EIGHTH, feedback=0.24)

    # ------------------------------------------------------------------
    # Light Tambourine & Doumbek Hand-Percussion Groove Across All 30s
    # ------------------------------------------------------------------
    perc_t = 0.06
    p_step = 0
    while perc_t < 29.55:
        beat_pos = p_step % 4
        if beat_pos == 0:
            add_stereo_event(bgm, pan_stereo(synth_doumbek("dum", gain=0.20), 0.0), perc_t)
        elif beat_pos == 2:
            add_stereo_event(bgm, pan_stereo(synth_doumbek("dum", gain=0.15), -0.06), perc_t)
        elif beat_pos in (1, 3):
            add_stereo_event(
                bgm,
                pan_stereo(synth_doumbek("tek", gain=0.14), 0.20 if beat_pos == 1 else -0.20),
                perc_t,
            )

        tamb = synth_tambourine(rng, accent=(beat_pos in (1, 3)), gain=0.075)
        add_stereo_event(bgm, pan_stereo(tamb, -0.26 if p_step % 2 == 0 else 0.26), perc_t)
        perc_t += EIGHTH
        p_step += 1

    # ------------------------------------------------------------------
    # Beat-Transition Swells (boost BGM +35% across beat transition windows)
    # ------------------------------------------------------------------
    t_axis = np.linspace(0.0, DURATION, TOTAL_SAMPLES, endpoint=False, dtype=np.float32)
    swell_env = np.ones(TOTAL_SAMPLES, dtype=np.float32)
    transition_centers = [0.15, 2.52, 4.80, 7.42, 9.80, 12.52, 15.20, 17.92, 20.40, 22.88, 25.40, 27.72, 29.65]
    for tc in transition_centers:
        swell_env += 0.36 * np.exp(-0.5 * ((t_axis - tc) / 0.22) ** 2).astype(np.float32)
    bgm *= swell_env[:, None]

    # Clean master intro fade & tail fade strictly inside 30.000s
    fade_in = int(0.08 * SR)
    fade_out = int(0.09 * SR)
    bgm[:fade_in] *= np.linspace(0.0, 1.0, fade_in, dtype=np.float32)[:, None]
    bgm[-fade_out:] *= np.linspace(1.0, 0.0, fade_out, dtype=np.float32)[:, None]
    return bgm


# ---------------------------------------------------------------------------
# All 14 Synchronized Archimedes Comedy Foley SFX Cues
# ---------------------------------------------------------------------------
def synth_sfx(sfx_type: str) -> np.ndarray:
    rng = np.random.default_rng(sum(ord(c) * (i + 1) for i, c in enumerate(sfx_type)) + 287)

    if sfx_type == "royal_horn":
        # Majestic ancient Syracuse bronze salpinx / royal fanfare horn call (D4 -> A4 -> D5)
        dur = 0.86
        n = int(round(dur * SR))
        t = np.linspace(0.0, dur, n, endpoint=False, dtype=np.float32)
        out = np.zeros(n, dtype=np.float32)
        notes = [(0.00, 0.22, 293.66, 0.72), (0.18, 0.24, 440.00, 0.82), (0.36, 0.48, 587.33, 0.95)]
        for st_s, d_s, freq, amp in notes:
            idx0 = int(round(st_s * SR))
            ln = min(int(round(d_s * SR)), n - idx0)
            if ln <= 4:
                continue
            tt = t[:ln]
            scoop = 1.0 - 0.025 * np.exp(-tt * 65.0)
            vib = 1.0 + 0.004 * np.sin(2.0 * math.pi * 5.5 * tt)
            ph = 2.0 * math.pi * np.cumsum(freq * scoop * vib) / SR
            brass = (
                0.50 * np.sin(ph)
                + 0.30 * np.sin(2.0 * ph)
                + 0.16 * np.sin(3.0 * ph)
                + 0.09 * np.sin(4.0 * ph)
                + 0.05 * np.sin(5.0 * ph)
            )
            env = adsr_env(ln, attack=0.015, decay=0.06, sustain=0.82, release=0.08)
            out[idx0 : idx0 + ln] += brass * env * amp
        out = butter_filter(out, "low", 3400.0, order=2)
        return np.tanh(out * 1.15).astype(np.float32) * 0.85

    if sfx_type == "crown_gleam":
        # Sparkling golden metallic crown gleam + crystalline high harmonic cascade
        dur = 0.74
        n = int(round(dur * SR))
        t = np.linspace(0.0, dur, n, endpoint=False, dtype=np.float32)
        out = np.zeros(n, dtype=np.float32)
        gleam_freqs = [1174.66, 1479.98, 1760.00, 2349.32, 2959.96, 3520.00]
        for idx, fr in enumerate(gleam_freqs):
            d = int(round(idx * 0.038 * SR))
            tt = t[: n - d]
            shim = 0.82 + 0.18 * np.sin(2.0 * math.pi * 16.0 * tt)
            ping = (
                0.68 * np.sin(2.0 * math.pi * fr * tt) * np.exp(-tt * 6.5)
                + 0.32 * np.sin(2.0 * math.pi * (fr * 2.003) * tt) * np.exp(-tt * 11.0)
            )
            out[d:] += ping * shim * np.clip(tt / 0.003, 0.0, 1.0) * 0.28
        fade = int(0.06 * SR)
        out[-fade:] *= np.linspace(1.0, 0.0, fade, dtype=np.float32)
        return out.astype(np.float32)

    if sfx_type == "vine_boom":
        # Deep punchy comedic sub-bass drop + warm impact thump
        dur = 0.78
        n = int(round(dur * SR))
        t = np.linspace(0.0, dur, n, endpoint=False, dtype=np.float32)
        f = 135.0 * (0.24 ** np.clip(t / 0.22, 0.0, 1.0)) + 30.0
        sub = np.sin(2.0 * math.pi * np.cumsum(f) / SR) * np.exp(-t * 4.4)
        crack = butter_filter(rng.standard_normal(n).astype(np.float32), "low", 480.0, order=2) * np.exp(-t * 28.0) * 0.35
        thump = np.sin(2.0 * math.pi * 160.0 * t) * np.exp(-t * 42.0) * 0.40
        sig = np.tanh((sub * 0.98 + crack + thump) * 1.45) * 0.88
        fade = int(0.06 * SR)
        sig[-fade:] *= np.linspace(1.0, 0.0, fade, dtype=np.float32)
        return sig.astype(np.float32)

    if sfx_type == "caliper_boing":
        # Metallic brass caliper slip click + elastic cartoon spring BOING!
        dur = 0.66
        n = int(round(dur * SR))
        t = np.linspace(0.0, dur, n, endpoint=False, dtype=np.float32)

        # Crisp metallic slip-snap at t=0..0.07s
        snap = butter_filter(rng.standard_normal(n).astype(np.float32), "band", (1600.0, 6200.0), order=2) * np.exp(-t * 110.0) * 0.55
        metal_clink = np.sin(2.0 * math.pi * 2180.0 * t) * np.exp(-t * 85.0) * 0.42

        # Elastic cartoon jawharpy BOING! starting at 0.045s
        t_b = np.maximum(0.0, t - 0.045)
        b_mask = (t >= 0.045).astype(np.float32)
        wobble = np.sin(2.0 * math.pi * 16.5 * t_b) * np.exp(-t_b * 4.0)
        f_boing = 220.0 + 210.0 * (1.0 - np.exp(-t_b * 18.0)) + 175.0 * wobble
        ph = 2.0 * math.pi * np.cumsum(f_boing) / SR
        boing = (
            0.62 * np.sin(ph)
            + 0.28 * np.sin(ph * 2.0 + 0.4 * np.exp(-t_b * 5.0))
            + 0.14 * np.sin(ph * 3.0)
        ) * np.exp(-t_b * 4.6) * b_mask * 0.72
        out = (snap + metal_clink + boing).astype(np.float32)
        fade = int(0.04 * SR)
        out[-fade:] *= np.linspace(1.0, 0.0, fade, dtype=np.float32)
        return out

    if sfx_type == "chalk_screech":
        # Classic cringe-funny slate chalk squeak + gritty swipe
        dur = 0.36
        n = int(round(dur * SR))
        t = np.linspace(0.0, dur, n, endpoint=False, dtype=np.float32)
        f_squeal = 3120.0 - 640.0 * (t / dur) + 280.0 * np.sin(2.0 * math.pi * 28.0 * t)
        ph = 2.0 * math.pi * np.cumsum(f_squeal) / SR
        squeal = np.sin(ph) + 0.45 * np.sin(ph * 1.48) + 0.22 * np.sin(ph * 2.12)
        grit = butter_filter(rng.standard_normal(n).astype(np.float32), "band", (1800.0, 5600.0), order=2)
        env = (np.sin(math.pi * t / dur) ** 0.75) * (0.75 + 0.25 * np.sin(2.0 * math.pi * 22.0 * t))
        sig = np.tanh((squeal * 0.48 + grit * 0.55) * env * 1.3) * 0.64
        return sig.astype(np.float32)

    if sfx_type == "eureka_splash":
        # Big lively bathtub water plunge: deep hydro cavity thump + broadband splash + bubble droplets
        dur = 0.95
        n = int(round(dur * SR))
        t = np.linspace(0.0, dur, n, endpoint=False, dtype=np.float32)

        # 1. Deep hydro plunge cavity ("ker-PLOP")
        f_plunge = 185.0 * (0.38 ** np.clip(t / 0.09, 0.0, 1.0)) + 68.0 + 140.0 * np.clip((t - 0.07) / 0.18, 0.0, 1.0)
        plunge = np.sin(2.0 * math.pi * np.cumsum(f_plunge) / SR) * np.exp(-t * 9.5) * 0.65

        # 2. Turbulent water splash spray
        raw_noise = rng.standard_normal(n).astype(np.float32)
        splash_mid = butter_filter(raw_noise, "band", (380.0, 2400.0), order=2) * np.exp(-t * 5.2) * np.clip(t / 0.012, 0.0, 1.0)
        splash_hi = butter_filter(raw_noise, "band", (2200.0, 7400.0), order=2) * np.exp(-t * 3.8) * np.clip(t / 0.025, 0.0, 1.0)

        # 3. Individual resonant water bubbles & slosh droplets
        bubbles = np.zeros(n, dtype=np.float32)
        bubble_events = [
            (0.08, 420.0, 780.0), (0.15, 510.0, 940.0), (0.23, 380.0, 690.0),
            (0.32, 620.0, 1120.0), (0.41, 480.0, 880.0), (0.52, 710.0, 1280.0),
            (0.64, 560.0, 990.0),
        ]
        for st_s, f0, f1 in bubble_events:
            d = int(round(st_s * SR))
            blen = min(int(round(0.065 * SR)), n - d)
            if blen <= 4:
                continue
            tt = t[:blen]
            fb = f0 + (f1 - f0) * ((tt / 0.065) ** 1.4)
            b_wave = np.sin(2.0 * math.pi * np.cumsum(fb) / SR) * np.sin(math.pi * tt / 0.065) * 0.34
            bubbles[d : d + blen] += b_wave

        sig = np.tanh((plunge + splash_mid * 0.72 + splash_hi * 0.48 + bubbles) * 1.25) * 0.88
        fade = int(0.07 * SR)
        sig[-fade:] *= np.linspace(1.0, 0.0, fade, dtype=np.float32)
        return sig.astype(np.float32)

    if sfx_type == "duck_squeak":
        # Cute comical double rubber-ducky squeak ("squeak-SQUEAK!")
        dur = 0.42
        n = int(round(dur * SR))
        t = np.linspace(0.0, dur, n, endpoint=False, dtype=np.float32)
        out = np.zeros(n, dtype=np.float32)
        for st_s, s_dur, f_base, amp in [(0.00, 0.14, 1180.0, 0.78), (0.16, 0.21, 1340.0, 0.88)]:
            d = int(round(st_s * SR))
            slen = min(int(round(s_dur * SR)), n - d)
            tt = t[:slen]
            arch = np.sin(math.pi * tt / s_dur)
            f_curve = f_base + 540.0 * (arch ** 1.3)
            ph = 2.0 * math.pi * np.cumsum(f_curve) / SR
            reed = (0.68 * np.sin(ph) + 0.26 * np.sin(2.0 * ph) + 0.10 * np.sin(3.0 * ph)) * (arch ** 0.8)
            air = butter_filter(rng.standard_normal(slen).astype(np.float32), "band", (2200.0, 5800.0), order=2) * arch * 0.14
            out[d : d + slen] += (reed + air) * amp
        return np.tanh(out * 1.15).astype(np.float32) * 0.82

    if sfx_type == "eureka_chime":
        # Radiant lightbulb-moment golden harp/bell chord chime + upward sparkle
        dur = 0.96
        n = int(round(dur * SR))
        t = np.linspace(0.0, dur, n, endpoint=False, dtype=np.float32)
        out = np.zeros(n, dtype=np.float32)
        # Bright D major 9th celestial bell arpeggio
        chime_notes = [
            (0.00, 587.33, 0.36), (0.04, 739.99, 0.32), (0.08, 880.00, 0.32),
            (0.12, 1174.66, 0.36), (0.16, 1318.51, 0.30), (0.20, 1760.00, 0.34),
            (0.25, 2349.32, 0.28),
        ]
        for st_s, fr, amp in chime_notes:
            d = int(round(st_s * SR))
            tt = t[: n - d]
            bell = (
                0.64 * np.sin(2.0 * math.pi * fr * tt) * np.exp(-tt * 3.8)
                + 0.24 * np.sin(2.0 * math.pi * (fr * 2.0) * tt) * np.exp(-tt * 6.5)
                + 0.12 * np.sin(2.0 * math.pi * (fr * 2.998) * tt) * np.exp(-tt * 10.5)
            )
            out[d:] += bell * np.clip(tt / 0.003, 0.0, 1.0) * amp
        fade = int(0.08 * SR)
        out[-fade:] *= np.linspace(1.0, 0.0, fade, dtype=np.float32)
        return np.tanh(out * 1.1).astype(np.float32) * 0.85

    if sfx_type == "tank_dunk":
        # Twin heavy gold/crown submersions into glass water tanks ("glug-plop!" + glass tank ring)
        dur = 0.72
        n = int(round(dur * SR))
        t = np.linspace(0.0, dur, n, endpoint=False, dtype=np.float32)
        out = np.zeros(n, dtype=np.float32)

        for st_s, f_lo, f_hi, g_amp in [(0.00, 135.0, 520.0, 0.78), (0.22, 120.0, 610.0, 0.86)]:
            d = int(round(st_s * SR))
            ln = min(int(round(0.42 * SR)), n - d)
            tt = t[:ln]
            # Upward water-cavity pitch sweep ("bloop-glug!")
            f_glug = f_lo + (f_hi - f_lo) * (np.clip(tt / 0.18, 0.0, 1.0) ** 1.6)
            glug = np.sin(2.0 * math.pi * np.cumsum(f_glug) / SR) * np.exp(-tt * 11.5) * 0.68
            slosh = butter_filter(rng.standard_normal(ln).astype(np.float32), "band", (450.0, 2600.0), order=2) * np.exp(-tt * 9.0) * 0.42
            glass = np.sin(2.0 * math.pi * 1680.0 * tt) * np.exp(-tt * 18.0) * 0.20
            out[d : d + ln] += (glug + slosh + glass) * g_amp
        fade = int(0.05 * SR)
        out[-fade:] *= np.linspace(1.0, 0.0, fade, dtype=np.float32)
        return np.tanh(out * 1.2).astype(np.float32) * 0.85

    if sfx_type == "water_overflow":
        # Vigorous pouring & bubbling water stream spilling into a glass measurement beaker
        dur = 0.88
        n = int(round(dur * SR))
        t = np.linspace(0.0, dur, n, endpoint=False, dtype=np.float32)

        raw = rng.standard_normal(n).astype(np.float32)
        stream = butter_filter(raw, "band", (550.0, 3400.0), order=2)
        flutter = 0.68 + 0.32 * np.sin(2.0 * math.pi * 19.0 * t)
        env = np.sin(math.pi * np.clip(t / dur, 0.0, 1.0)) ** 0.75
        out = stream * flutter * env * 0.48

        # Rising beaker fill pitch bubbles (pitch climbs as beaker fills!)
        for idx in range(11):
            st_s = 0.03 + idx * 0.072
            d = int(round(st_s * SR))
            blen = min(int(round(0.058 * SR)), n - d)
            if blen <= 4:
                continue
            tt = t[:blen]
            f0 = 420.0 + idx * 55.0
            f1 = f0 * 1.75
            fb = f0 + (f1 - f0) * ((tt / 0.058) ** 1.3)
            bub = np.sin(2.0 * math.pi * np.cumsum(fb) / SR) * np.sin(math.pi * tt / 0.058) * 0.36
            out[d : d + blen] += bub

        fade = int(0.06 * SR)
        out[-fade:] *= np.linspace(1.0, 0.0, fade, dtype=np.float32)
        return np.tanh(out * 1.2).astype(np.float32) * 0.84

    if sfx_type == "anvil_slam":
        # Heavy bronze balance-beam / anvil metallic slam + sub impact + bronze bell ring
        dur = 0.85
        n = int(round(dur * SR))
        t = np.linspace(0.0, dur, n, endpoint=False, dtype=np.float32)

        f_sub = 155.0 * (0.22 ** np.clip(t / 0.14, 0.0, 1.0)) + 36.0
        sub = np.sin(2.0 * math.pi * np.cumsum(f_sub) / SR) * np.exp(-t * 5.8) * 0.82
        clank = butter_filter(rng.standard_normal(n).astype(np.float32), "band", (600.0, 4800.0), order=2) * np.exp(-t * 38.0) * 0.65
        # Inharmonic metallic anvil/bronze modes
        metal = (
            0.45 * np.sin(2.0 * math.pi * 685.0 * t) * np.exp(-t * 6.5)
            + 0.35 * np.sin(2.0 * math.pi * 1192.0 * t) * np.exp(-t * 8.2)
            + 0.25 * np.sin(2.0 * math.pi * 1845.0 * t) * np.exp(-t * 11.5)
            + 0.16 * np.sin(2.0 * math.pi * 2790.0 * t) * np.exp(-t * 16.0)
        ) * np.clip(t / 0.002, 0.0, 1.0) * 0.58
        sig = np.tanh((sub + clank + metal) * 1.28) * 0.88
        fade = int(0.06 * SR)
        sig[-fade:] *= np.linspace(1.0, 0.0, fade, dtype=np.float32)
        return sig.astype(np.float32)

    if sfx_type == "ship_horn_bell":
        # Warm maritime ship horn + crisp brass ship's bell double-ding ("ding-ding!")
        dur = 0.86
        n = int(round(dur * SR))
        t = np.linspace(0.0, dur, n, endpoint=False, dtype=np.float32)

        # Warm low-mid ship horn (D3 + A3)
        f_h = 146.83
        horn = (
            0.55 * np.sin(2.0 * math.pi * f_h * t)
            + 0.35 * np.sin(2.0 * math.pi * (f_h * 1.5) * t)
            + 0.25 * np.sin(2.0 * math.pi * (f_h * 2.0) * t)
            + 0.14 * np.sin(2.0 * math.pi * (f_h * 3.0) * t)
        )
        horn_env = adsr_env(n, attack=0.04, decay=0.12, sustain=0.72, release=0.22)
        out = butter_filter(horn * horn_env, "low", 1200.0, order=2) * 0.52

        # Twin crisp brass ship's bell dings at 0.08s and 0.28s
        for st_s, amp in [(0.08, 0.52), (0.28, 0.62)]:
            d = int(round(st_s * SR))
            tt = t[: n - d]
            bell = (
                0.62 * np.sin(2.0 * math.pi * 1760.0 * tt) * np.exp(-tt * 5.8)
                + 0.28 * np.sin(2.0 * math.pi * 3520.0 * tt) * np.exp(-tt * 9.5)
                + 0.16 * np.sin(2.0 * math.pi * 4780.0 * tt) * np.exp(-tt * 16.0)
            ) * np.clip(tt / 0.002, 0.0, 1.0)
            out[d:] += bell * amp

        fade = int(0.06 * SR)
        out[-fade:] *= np.linspace(1.0, 0.0, fade, dtype=np.float32)
        return np.tanh(out * 1.15).astype(np.float32) * 0.84

    if sfx_type == "fanfare_swell":
        # Triumphant royal Syracuse brass & lyre fanfare chord swell (D major add9)
        dur = 0.92
        n = int(round(dur * SR))
        t = np.linspace(0.0, dur, n, endpoint=False, dtype=np.float32)
        freqs = [293.66, 440.00, 587.33, 739.99, 880.00, 1174.66]
        scoop = 1.0 - 0.018 * np.exp(-t * 55.0)
        chord = np.zeros(n, dtype=np.float32)
        for idx, fr in enumerate(freqs):
            ph = 2.0 * math.pi * np.cumsum(fr * scoop) / SR
            voice = 0.56 * np.sin(ph) + 0.28 * np.sin(2.0 * ph) + 0.14 * np.sin(3.0 * ph)
            chord += voice / (1.0 + 0.15 * idx)
        chord /= len(freqs) * 0.55
        env = adsr_env(n, attack=0.06, decay=0.14, sustain=0.82, release=0.22)
        sig = butter_filter(chord * env, "low", 3200.0, order=2)
        return np.tanh(sig * 1.2).astype(np.float32) * 0.84

    if sfx_type == "crown_ding":
        # Pure high-karat gold crown ping + sparkling harmonic tail
        dur = 0.72
        n = int(round(dur * SR))
        t = np.linspace(0.0, dur, n, endpoint=False, dtype=np.float32)
        ding = (
            0.62 * np.sin(2.0 * math.pi * 2349.32 * t) * np.exp(-t * 5.2)
            + 0.28 * np.sin(2.0 * math.pi * 3520.00 * t) * np.exp(-t * 7.8)
            + 0.16 * np.sin(2.0 * math.pi * 4698.64 * t) * np.exp(-t * 12.0)
        )
        attack = np.clip(t / 0.002, 0.0, 1.0)
        out = ding * attack * 0.82
        fade = int(0.06 * SR)
        out[-fade:] *= np.linspace(1.0, 0.0, fade, dtype=np.float32)
        return out.astype(np.float32)

    return np.zeros(int(0.1 * SR), dtype=np.float32)


def _bound_sfx(wav: np.ndarray, ceiling: float = 0.85) -> np.ndarray:
    pk = float(np.max(np.abs(wav)))
    if pk > ceiling:
        wav = (wav * (ceiling / pk)).astype(np.float32)
    return wav


def measure_integrated_lufs_bs1770(stereo: np.ndarray, sr: int = SR) -> float:
    """Exact ITU-R BS.1770-4 K-weighted integrated loudness (LUFS) in pure SciPy/NumPy."""
    # Stage 1: high-shelf head acoustics filter (+4 dB above ~1.5 kHz)
    f0 = 1681.974450955533
    g_db = 3.999843853973347
    q0 = 0.7071752369554196
    k = math.tan(math.pi * f0 / sr)
    vh = 10.0 ** (g_db / 20.0)
    vb = vh ** 0.4996667741545416
    a0 = 1.0 + k / q0 + k * k
    b_shelf = np.array(
        [(vh + vb * k / q0 + k * k) / a0, 2.0 * (k * k - vh) / a0, (vh - vb * k / q0 + k * k) / a0],
        dtype=np.float64,
    )
    a_shelf = np.array([1.0, 2.0 * (k * k - 1.0) / a0, (1.0 - k / q0 + k * k) / a0], dtype=np.float64)

    # Stage 2: RLB high-pass filter (~38 Hz)
    f1 = 38.13547087602444
    q1 = 0.5003270373238773
    k1 = math.tan(math.pi * f1 / sr)
    a1_0 = 1.0 + k1 / q1 + k1 * k1
    b_hp = np.array([1.0, -2.0, 1.0], dtype=np.float64) / a1_0
    a_hp = np.array([1.0, 2.0 * (k1 * k1 - 1.0) / a1_0, (1.0 - k1 / q1 + k1 * k1) / a1_0], dtype=np.float64)

    y = signal.lfilter(b_shelf, a_shelf, stereo.astype(np.float64), axis=0)
    y = signal.lfilter(b_hp, a_hp, y, axis=0)

    # 400 ms gating blocks with 75% overlap (100 ms step)
    block_len = int(round(0.400 * sr))
    hop_len = int(round(0.100 * sr))
    powers = []
    for start in range(0, len(y) - block_len + 1, hop_len):
        blk = y[start : start + block_len]
        powers.append(float(np.mean(np.sum(blk * blk, axis=1))))
    powers_arr = np.array(powers, dtype=np.float64)

    # Absolute gate at -70 LUFS
    lufs_blocks = -0.691 + 10.0 * np.log10(np.maximum(powers_arr, 1e-18))
    abs_mask = lufs_blocks > -70.0
    if not np.any(abs_mask):
        return -70.0
    rel_thresh = -0.691 + 10.0 * math.log10(float(np.mean(powers_arr[abs_mask]))) - 10.0
    rel_mask = abs_mask & (lufs_blocks > rel_thresh)
    if not np.any(rel_mask):
        return -70.0
    return -0.691 + 10.0 * math.log10(float(np.mean(powers_arr[rel_mask])))


def calibrate_master_lufs_and_peak(
    project_root: Path,
    timeline: dict[str, Any],
    target_lufs: float = -14.0,
    ceiling_dbfs: float = -1.05,
) -> tuple[float, float]:
    """
    Fine-tune exported `master_audio.wav` so integrated loudness hits `-14.0 LUFS`
    while strictly respecting true-peak ceiling `<= -1.0 dBFS`.
    """
    master_path = project_root / timeline["audio"]["masterPath"]
    manifest_path = project_root / timeline["audio"]["manifestPath"]

    mix, sr = sf.read(str(master_path), dtype="float32")
    ceiling_lin = 10.0 ** (ceiling_dbfs / 20.0)  # ~0.886

    for _ in range(4):
        cur_lufs = measure_integrated_lufs_bs1770(mix, sr)
        diff_db = target_lufs - cur_lufs
        if abs(diff_db) < 0.05:
            break
        gain = 10.0 ** (diff_db / 20.0)
        mix = mix * gain
        # Soft limiter above 0.76 up to ceiling_lin so peak never exceeds -1.05 dBFS
        abs_m = np.abs(mix)
        knee = 0.76
        over = abs_m > knee
        if np.any(over):
            sign = np.sign(mix[over])
            excess = (abs_m[over] - knee) / max(1e-6, ceiling_lin - knee)
            mix[over] = sign * (knee + (ceiling_lin - knee) * np.tanh(excess))

    # 4x oversampled true-peak protection
    up4 = signal.resample_poly(mix, 4, 1, axis=0)
    true_peak = float(np.max(np.abs(up4)))
    if true_peak > ceiling_lin:
        mix *= (ceiling_lin / true_peak)

    sf.write(str(master_path), mix, sr, subtype="PCM_16")

    # Re-read quantized PCM_16 to report exact final file metrics
    final_wav, _ = sf.read(str(master_path), dtype="float32")
    final_lufs = measure_integrated_lufs_bs1770(final_wav, sr)
    final_up4 = signal.resample_poly(final_wav, 4, 1, axis=0)
    final_tp_db = 20.0 * math.log10(max(1e-9, float(np.max(np.abs(final_up4)))))
    final_sample_pk_db = 20.0 * math.log10(max(1e-9, float(np.max(np.abs(final_wav)))))

    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["loudnessLufs"] = round(final_lufs, 2)
        manifest["truePeakDbfs"] = round(final_tp_db, 2)
        manifest["peakDbfs"] = round(final_sample_pk_db, 3)
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    return final_lufs, final_tp_db


def main() -> None:
    timeline = load_timeline(TIMELINE_PATH)

    print("[Audio] Synthesizing 12 Kokoro dialogue lines (am_puck / bm_george / am_michael)...")
    dialogue_bus, speaker_mono_buses, manifest_lines = synthesize_dialogue(
        timeline,
        tts_overrides=TTS_OVERRIDES,
    )

    print("[Audio] Synthesizing 116 BPM Mediterranean Greek comedic BGM bed...")
    bgm_bus = generate_bgm(timeline)

    sfx_events = timeline.get("sfxEvents", [])
    print(f"[Audio] Synthesizing {len(sfx_events)} procedural Archimedes Foley SFX cues...")
    sfx_bus = np.zeros((TOTAL_SAMPLES, CHANNELS), dtype=np.float32)
    for ev in sfx_events:
        wav = _bound_sfx(synth_sfx(ev["type"])) * float(ev.get("gain", 0.68))
        add_stereo_event(sfx_bus, pan_stereo(wav, float(ev.get("pan", 0.0))), float(ev["time"]))

    master_and_export(
        PROJECT_ROOT,
        timeline,
        dialogue_bus,
        bgm_bus,
        sfx_bus,
        speaker_mono_buses,
        manifest_lines,
    )

    final_lufs, final_tp_db = calibrate_master_lufs_and_peak(
        PROJECT_ROOT,
        timeline,
        target_lufs=float(timeline["audio"].get("loudnessLufs", -14.0)),
        ceiling_dbfs=-1.08,
    )
    print(f"[Audio] Calibrated Master Loudness: {final_lufs:.2f} LUFS | True Peak: {final_tp_db:.2f} dBFS")


if __name__ == "__main__":
    main()
