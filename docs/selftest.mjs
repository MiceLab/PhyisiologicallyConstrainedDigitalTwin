// node docs/selftest.mjs -- the browser port must reproduce the TF reference day export_web.py
// stored with each model, and keep app.selftest's dose/emit properties.
import { readFileSync, readdirSync } from 'node:fs';
import assert from 'node:assert/strict';
import { loadRun, newDay, runAll, takeDose, emit, STEPS } from './twin.js';

const dir = new URL('./models/', import.meta.url);
const maxDiff = (a, b) => Math.max(...Array.from(a, (v, i) => Math.abs(v - b[i])));

for (const file of readdirSync(dir)) {
  const run = loadRun(JSON.parse(readFileSync(new URL(file, dir))));
  const ref = run.reference;

  const s = newDay(run, run.after, 'frozen');
  for (const [t, g] of Object.entries(ref.cho)) s.cho[run.before + +t] = g;
  for (const [t, u] of Object.entries(ref.bolus)) s.bolus[run.before + +t] = u;
  s.dirty = true;
  runAll(s, ref.z);
  const d = [maxDiff(s.pi, ref.pi), maxDiff(s.ra, ref.ra), maxDiff(s.bg, ref.bg)];
  console.log(run.dataset, 'max |js - tf|  pi', d[0].toExponential(1), 'ra', d[1].toExponential(1),
              'bg', d[2].toExponential(1));
  assert(d.every(x => x < 1e-4), `${run.dataset}: JS port drifted from the TF reference`);

  // carbs raise, insulin lowers; pi/ra are causal (BG is not: the decoder sees 18 steps ahead)
  const z = Array(run.latentDim).fill(0), mean = a => a.reduce((x, y) => x + y, 0) / a.length;
  const day = (cho, bolus) => {
    const s = newDay(run, run.after, 'frozen');
    s.t = 12;
    takeDose(s, cho, bolus);
    runAll(s, z);
    return s;
  };
  const flat = day(0, 0), meal = day(60, 0), dose = day(0, 8);
  const b = run.before + 12;
  assert.deepEqual(meal.ra.slice(0, b + 1), flat.ra.slice(0, b + 1), 'Ra changed before the meal');
  assert.deepEqual(dose.pi.slice(0, b + 1), flat.pi.slice(0, b + 1), 'PI changed before the bolus');
  assert(mean(meal.bg.slice(12)) > mean(flat.bg.slice(12)), 'carbs did not raise generated glucose');
  assert(mean(dose.bg.slice(12)) < mean(flat.bg.slice(12)), 'insulin did not lower generated glucose');

  // the three latent modes must give three different days
  const days = ['frozen', 'ou', 'resample'].map(m => { const s = newDay(run, run.after, m); runAll(s, z); return s.bg; });
  assert(days.every(b => b.length === STEPS));
  assert(maxDiff(days[0], days[1]) > 1e-6 && maxDiff(days[1], days[2]) > 1e-6, 'a latent mode is ignored');
}

const preds = [0, 1, 2, 3].map(s => [s, s, s, s]);     // window s emits s
assert.equal(emit(preds, 3, 1), 3);
assert.equal(emit(preds, 3, 4), 1.5);
assert.equal(emit(preds, 0, 18), 0);
console.log('selftest ok');
