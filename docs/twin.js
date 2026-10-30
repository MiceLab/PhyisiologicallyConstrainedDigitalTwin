// Browser port of twin.py and app.py's simulation loop, for the static GitHub Pages explorer.
// The decoder is evaluated by hand from the weights export_web.py writes; selftest.mjs checks
// the whole dose -> pi/ra -> decoder -> overlap path against the TF original.

export const STEPS = 288;          // 24 h of 5-minute steps
export const STEP_MIN = 5;
export const EVAL_SEED = 42;

// ---------------------------------------------------------------- physiological models
const TMAX_I = 55, KE = 0.138, VI = 0.12;
const TMAX_RESCUE_CHO = 40, BIO_RESCUE_CHO = 0.8;

export function plasmaInsulin(insulin, dt = 5) {
  // twin.calculate_plasma_insulin with the equilibrium start at 0.
  const kDia = 1 / TMAX_I, n = insulin.length;
  const pi = new Float64Array(n);
  let c1 = 0, c2 = 0, p = 0, u = insulin[0];
  for (let k = 0; k < n; k++) {
    if (k > 0) u = insulin[k - 1];
    [c1, c2, p] = [c1 + dt * (u / dt - kDia * c1), c2 + dt * kDia * (c1 - c2),
                   (1 - KE * dt) * p + dt * kDia * c2 / VI];
    pi[k] = p;
  }
  return pi;
}

export function raHovorka(cho, ts = 5) {
  // twin.model_simulation_ra; expm of the 2-compartment chain written out in closed form.
  const T = TMAX_RESCUE_CHO, a = Math.exp(-ts / T), r = ts / T;
  const g1 = BIO_RESCUE_CHO * T * (1 - a), g2 = BIO_RESCUE_CHO * T * (1 - a * (1 + r));
  const ra = new Float64Array(cho.length);
  let x1 = 0, x2 = 0;
  for (let k = 0; k < cho.length; k++) {
    ra[k] = x2 / T;
    [x1, x2] = [a * x1 + g1 * cho[k], r * a * x1 + a * x2 + g2 * cho[k]];
  }
  return ra;
}

// ---------------------------------------------------------------- decoder
const f32 = s => new Float32Array(Uint8Array.from(atob(s), c => c.charCodeAt(0)).buffer);

export function loadRun(json) {
  const w = {};
  for (const [name, l] of Object.entries(json.weights)) w[name] = { k: f32(l.k), b: f32(l.b) };
  const phi = json.latent_phi, dim = w.dense_5.k.length / w.dense_5.b.length;
  return { ...json, w, latentDim: dim,
           // a per-dimension phi only fits a latent of that width (app._latent_dynamics)
           phi: phi.length === dim ? phi : Array(dim).fill(phi.reduce((s, v) => s + v, 0) / phi.length) };
}

function dense(x, { k, b }) {
  const n = b.length, y = Float64Array.from(b);
  for (let i = 0; i < x.length; i++) for (let j = 0; j < n; j++) y[j] += x[i] * k[i * n + j];
  return y;
}

function perRow(x, width, layer) {        // TimeDistributed(Dense) over [rows, width]
  const out = [];
  for (let r = 0; r < x.length; r += width) out.push(...dense(x.subarray(r, r + width), layer));
  return Float64Array.from(out);
}

const leaky = y => y.map(v => (v > 0 ? v : 0.2 * v));
const softplus = v => Math.max(v, 0) + Math.log1p(Math.exp(-Math.abs(v)));
const sigmoid = v => 1 / (1 + Math.exp(-v));

export function decode(run, z, pi, ra) {
  // The Keras graph in models/*/decoder_epoch_*.keras: a latent-driven baseline plus
  // softplus-positive per-step gains on pi (lowering) and ra (raising), through a sigmoid.
  const { w, min_gain: c } = run;
  const g = perRow(leaky(dense(dense(leaky(dense([...pi, ...ra], w.dense_8)), w.dense_9), w.dense_10)),
                   16, w.time_distributed_1);
  const base = perRow(leaky(dense(leaky(dense(z, w.dense_5)), w.dense_6)), 64, w.time_distributed);
  const pre = base.map((v, t) => v + run.control_scale *
    (-pi[t] * (softplus(g[2 * t]) + c) + ra[t] * (softplus(g[2 * t + 1]) + c)));
  return dense(pre, w.dense_12).map(sigmoid);
}

// ---------------------------------------------------------------- day simulation
export function emit(preds, t, overlap) {
  // app.emit: average the `overlap` most recent predictions of instant t.
  const lo = Math.max(0, t - overlap + 1);
  let sum = 0;
  for (let s = lo; s <= t; s++) sum += preds[s][t - s];
  return sum / (t + 1 - lo);
}

export function normalRng(seed) {
  // mulberry32 + Box-Muller: seeded so a slider move replays the same day. Not numpy's
  // stream, so OU/resample days differ from the Python app's; frozen ones match.
  let a = seed >>> 0;
  const u = () => {
    a = (a + 0x6D2B79F5) >>> 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
  return () => Math.sqrt(-2 * Math.log(1 - u())) * Math.cos(2 * Math.PI * u());
}

export function stepOU(z, phi, normal, scale, mu) {
  // twin.step_latent_ou: exact OU step, marginal stays N(mu, scale^2).
  if (phi.every(p => p >= 1)) return z;
  return z.map((v, i) => mu[i] + phi[i] * (v - mu[i]) + Math.sqrt(Math.max(1 - phi[i] ** 2, 0)) * scale * normal());
}

export function newDay(run, overlap = run.after, mode = 'ou') {
  const n = run.before + STEPS + run.after;
  const s = { run, overlap, mode, cho: new Float64Array(n), bolus: new Float64Array(n), bg: [], t: 0 };
  refreshInputs(s);
  return s;
}

export function refreshInputs(s) {
  s.pi = plasmaInsulin(s.bolus).map(v => v / s.run.pi_max);
  s.ra = raHovorka(s.cho).map(v => v / s.run.ra_max);
  s.dirty = false;
}

export function takeDose(s, cho, bolus) {
  const i = s.run.before + Math.min(s.t, STEPS - 1);
  s.cho[i] += cho;
  s.bolus[i] += bolus;
  s.dirty = true;
}

export function* generate(s, z) {
  // app.run_all and app.stream in one: yields the latent after each 5-minute step. A dose
  // taken mid-run is picked up on the next step, as in Play.
  const normal = normalRng(EVAL_SEED), z0 = z.slice(), preds = [], width = s.run.before + s.run.after;
  s.bg = [];
  for (let t = 0; t < STEPS; t++) {
    if (s.dirty) refreshInputs(s);
    preds.push(decode(s.run, z, s.pi.subarray(t, t + width), s.ra.subarray(t, t + width)));
    s.bg.push(emit(preds, t, s.overlap));
    if (s.mode === 'resample') z = z.map(() => normal() * s.run.latent_scale);
    else if (s.mode === 'ou') z = stepOU(z, s.run.phi, normal, s.run.latent_scale, z0);
    yield z;
  }
}

export function runAll(s, z) {
  if (s.dirty) refreshInputs(s);
  for (const _ of generate(s, z));    // eslint-disable-line no-unused-vars
}
