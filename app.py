"""
Interactive Gradio explorer for the trained VAE decoders.

Pick a dataset / model / epoch (the shipped best-of, from evaluation_numerical.figures'
RUNS, is shown in bold and preselected), set the latent by hand with one slider per
dimension, then either press Play to stream the generation forward in 5-minute steps or
hand the model a meal + bolus and see what it does with it.

Play does not freeze the latent: z walks the same Ornstein-Uhlenbeck path
glycemia_evaluation.simulation uses (step_latent_ou, phi from RUNS), so the generated day
drifts between physiological regimes instead of being one frozen "patient". It mean-reverts
to wherever the sliders were left rather than to 0, so the latent you picked stays the
centre of the behaviour; the sliders track the drift live and are locked while it plays.

A dose taken at clock time T is written into the raw cho/bolus timelines at that step and
pushed through the SAME physiological models the training data came from
(data_extraction.metabolical_models) and the SAME pi/ra scalers, so what reaches the
decoder is what training saw. Because the shipped models take before=0/after=18, the
decoder's pi/ra window at step t is [t, t+18) -- so a meal shows up in the generated
glucose from the step it is taken, not 18 steps later.

Both panels share the x axis (wall-clock time) and the y axis: pi/ra are drawn in their
scaled units multiplied by 400, the same [0, 400] span the glucose panel uses.

Run:  APP_AUTH=user:pass python app.py   (then open the printed URL)
      python app.py --selftest          (assert the dose path works)
"""
import glob
import json
import os
import re
import sys
import time
from datetime import datetime, timedelta

os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')

import numpy as np
import gradio as gr
from matplotlib.figure import Figure
import matplotlib.dates as mdates
from tensorflow.keras.models import load_model

# twin.RUNS names the shipped model/epoch; importing twin also applies the plot style.
from twin import (
    MODELS_PATH, RUNS, GEN, PI, RA, INSULIN, CHO, EUGLY, EVAL_SEED, FASTING_BG_SCALED,
    calculate_plasma_insulin, model_simulation_ra, load_pi_ra_scalers, resolve_before_after,
    bg_descale_params, generate_fake_samples, step_latent_ou)

STEP_MIN = 5
STEPS = 288          # 24 h of 5-minute steps
MAX_LATENT = 8       # sliders built up front; only the model's own dims stay visible
BG_MAX = 400.0       # shared y span of both panels (see module docstring)

# ponytail: one global session, no per-browser state. This is a local single-user
# explorer; give it gr.State + a session dict if it ever gets served to more than one.
S = {}

# Unicode sans-serif bold, so the best model/epoch can be bold inside a plain <select>.
# Dropdown choices are (label, value) pairs, so the value stays the real ASCII name.
_BOLD = str.maketrans(
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789",
    "".join(chr(0x1D5D4 + i) for i in range(26))
    + "".join(chr(0x1D5EE + i) for i in range(26))
    + "".join(chr(0x1D7EC + i) for i in range(10)))


def bold(text):
    return str(text).translate(_BOLD)


# ---------------------------------------------------------------- model discovery
def _cfg(model):
    path = os.path.join(MODELS_PATH, model, 'training_config.json')
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}


def list_models(dataset):
    """Model directories trained on `dataset` that still have per-epoch decoders on disk.
    training_config.json is authoritative; the directory name is the fallback for runs
    from before it was written."""
    names = []
    for name in sorted(os.listdir(MODELS_PATH)):
        cfg = _cfg(name)
        belongs = cfg.get('dataset_folder') == dataset if cfg else f'_{dataset}_' in name
        if belongs and glob.glob(os.path.join(MODELS_PATH, name, 'decoder_epoch_*.keras')):
            names.append(name)
    return names


def model_choices(dataset):
    best = RUNS[dataset]['model']
    names = list_models(dataset)
    if best in names:                      # shipped model first, and bold
        names.insert(0, names.pop(names.index(best)))
    return [(bold(n) if n == best else n, n) for n in names], (names[0] if names else None)


def epoch_choices(dataset, model):
    epochs = sorted(int(re.search(r'_(\d+)\.keras$', p).group(1))
                    for p in glob.glob(os.path.join(MODELS_PATH, model, 'decoder_epoch_*.keras')))
    if not epochs:
        return [], None
    best = RUNS[dataset]['epoch'] if model == RUNS[dataset]['model'] else epochs[-1]
    if best not in epochs:
        best = epochs[-1]
    return [(bold(e) if e == best else str(e), e) for e in epochs], best


# ---------------------------------------------------------------- session
def _latent_dynamics(dataset, latent_dim):
    """(phi, scale) for the OU latent path, from figures.RUNS -- the same values
    glycemia_evaluation.simulation is driven with. They were calibrated on each dataset's
    SHIPPED model+epoch (estimate_latent_phi.py, latent_audit.py's 'escala' column), so for
    any other checkpoint in the dropdown they are the best available estimate, not a fitted
    one. safe's phi is per-dimension (three dims decorrelate in ~45 min, a fourth takes
    2.5 h), which only applies to a latent of that width -- hence the collapse to its mean
    when the selected model's latent_dim differs."""
    cfg = RUNS[dataset]
    phi = cfg.get('latent_phi', 1.0)
    if np.ndim(phi) > 0 and len(np.atleast_1d(phi)) != latent_dim:
        phi = float(np.mean(phi))
    return phi, float(cfg.get('latent_scale', 1.0))


def build(dataset, model, epoch):
    before, after, uses_hist = resolve_before_after(model)
    pi_scaler, ra_scaler = load_pi_ra_scalers(dataset)
    offset, span = bg_descale_params(dataset)
    decoder = load_model(os.path.join(MODELS_PATH, model, f'decoder_epoch_{int(epoch):03d}.keras'),
                         compile=False)
    S.update(dataset=dataset, model=model, epoch=int(epoch), decoder=decoder,
             latent_dim=int(decoder.inputs[0].shape[-1]), before=before, after=after,
             uses_hist=uses_hist, pi_scaler=pi_scaler, ra_scaler=ra_scaler,
             offset=offset, span=span, playing=False)
    S['phi'], S['ou_scale'] = _latent_dynamics(dataset, S['latent_dim'])
    # Carried across a model change rather than reset, so comparing two checkpoints does not
    # silently compare them under different emission rules. Capped at `after`: there are only
    # `after` predictions of any instant, so a larger number would be the same as full overlap
    # while implying it was something else.
    S['overlap'] = max(1, min(int(S.get('overlap', S['after'])), S['after']))
    S.setdefault('latent_mode', 'ou')
    reset()


def reset():
    """Clear the dose timelines and the generated trajectory. The arrays are
    before+STEPS+after long so step t's decoder window is exactly arr[t : t+before+after]."""
    n = S['before'] + STEPS + S['after']
    S['cho'] = np.zeros(n)
    S['bolus'] = np.zeros(n)
    S['bg'] = []
    S['t'] = 0
    refresh_inputs()


def refresh_inputs():
    """Re-derive pi/ra from the whole raw dose timeline. Both models are causal, so past
    values are unaffected by a dose just added and only the future response changes."""
    pi_raw = calculate_plasma_insulin(np.zeros_like(S['bolus']), S['bolus'])
    ra_raw = model_simulation_ra(S['cho'])
    S['pi'] = S['pi_scaler'].transform(pi_raw.reshape(-1, 1)).ravel().astype('float32')
    S['ra'] = S['ra_scaler'].transform(ra_raw.reshape(-1, 1)).ravel().astype('float32')


def latent(values):
    return np.asarray(values[:S['latent_dim']], dtype='float32').reshape(1, S['latent_dim'])


def step_window(t, z):
    """One decoder pass at step t; returns the whole emitted (scaled) window.

    The window predicts instants t..t+after-1, so instant t+k sits at index k. Returning
    all of it is what lets `emit` average the overlapping predictions the way
    glycemia_evaluation does -- taking index 0 and discarding the rest is `overlap=1`.
    """
    width = S['before'] + S['after']
    pi = S['pi'][t:t + width].reshape(1, 1, width)
    ra = S['ra'][t:t + width].reshape(1, 1, width)
    hist = None
    if S['uses_hist'] and S['before']:
        # Autoregressive: condition on our own output, seeded at fasting -- never on data
        # the model wouldn't have live (same rule as glycemia_evaluation.simulate_next_step).
        tail = S['bg'][-S['before']:]
        tail = [FASTING_BG_SCALED] * (S['before'] - len(tail)) + list(tail)
        hist = np.asarray(tail, dtype='float32').reshape(1, 1, S['before'])
    bg, _ = generate_fake_samples(S['decoder'], S['latent_dim'], pi, ra, bg_hist=hist, z=z)
    return np.asarray(bg).flatten()


def emit(preds, t, overlap):
    """Average the `overlap` most recent predictions of instant t.

    `preds[s]` is the window decoded at step s, which predicts instant t at index t-s. The
    windows covering t all START at or before t, so this is causal and needs no lookahead:
    by the time step t is emitted every contributor has been decoded. Early instants simply
    average fewer, exactly as the replay does at the start of a group.
    """
    lo = max(0, t - overlap + 1)
    return float(np.mean([preds[s][t - s] for s in range(lo, t + 1)]))


def next_latent(z, z0, rng):
    """One step of the selected latent dynamics. Shared by the static recompute and Play so
    the two cannot drift apart.

    The three modes are the three arms the paper compares: the measured \\ac{OU} process,
    a frozen latent (phi = 1) and an independent redraw each step (phi = 0). Frozen has the
    best per-meal fidelity and the worst amplitude; resampling is the opposite.
    """
    mode = S.get('latent_mode', 'ou')
    if mode == 'frozen':
        return z
    if mode == 'resample':
        return (rng.standard_normal((1, S['latent_dim'])) * S['ou_scale']).astype('float32')
    return step_latent_ou(z, S['phi'], rng, scale=S['ou_scale'], mu=z0)


def run_all(z):
    """Recompute the whole 24 h from the current dose timelines. Deliberately does NOT
    move S['t'] -- that is the live cursor a dose is taken at, and only Play advances it.

    Seeded off EVAL_SEED so a slider move replays the same day rather than redrawing it,
    which matters once the latent mode is anything but frozen."""
    rng = np.random.default_rng(EVAL_SEED)
    z0, preds, S['bg'] = z.copy(), [], []
    for t in range(STEPS):
        preds.append(step_window(t, z))
        S['bg'].append(emit(preds, t, S['overlap']))
        z = next_latent(z, z0, rng)


# ---------------------------------------------------------------- plotting
def _start_time(text):
    try:
        return datetime.strptime(str(text).strip(), '%H:%M')
    except ValueError:
        return datetime.strptime('08:00', '%H:%M')


def figure(start_text):
    times = [_start_time(start_text) + timedelta(minutes=STEP_MIN * i) for i in range(STEPS)]
    fig = Figure(figsize=(11, 7))
    ax_bg, ax_in = fig.subplots(2, 1, sharex=True, sharey=True)

    ax_bg.axhspan(70, 180, color=EUGLY, alpha=0.25, lw=0, label='70-180 mg/dL')
    if S['bg']:
        bg = np.asarray(S['bg']) * S['span'] + S['offset']
        ax_bg.plot(times[:len(bg)], bg, color=GEN, label='generated BG')
    ax_bg.set(ylabel='mg/dL', ylim=(0, BG_MAX),
              title=f"{S['dataset']} - {S['model']} - epoch {S['epoch']}")
    ax_bg.legend(loc='upper left')

    b = S['before']
    ax_in.plot(times, S['pi'][b:b + STEPS] * BG_MAX, color=PI, label='plasma insulin (scaled x400)')
    ax_in.plot(times, S['ra'][b:b + STEPS] * BG_MAX, color=RA, label='glucose Ra (scaled x400)')
    ax_in.set(ylabel='scaled x400', xlabel='time of day', ylim=(0, BG_MAX))
    ax_in.legend(loc='upper left')

    for ax in (ax_bg, ax_in):
        for i in np.nonzero(S['cho'][b:b + STEPS])[0]:
            ax.axvline(times[i], color=CHO, ls='--', lw=1)
        for i in np.nonzero(S['bolus'][b:b + STEPS])[0]:
            ax.axvline(times[i], color=INSULIN, ls=':', lw=1)
    ax_in.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    ax_in.xaxis.set_major_locator(mdates.HourLocator(interval=2))
    return fig


# ---------------------------------------------------------------- callbacks
def on_dataset(dataset):
    choices, value = model_choices(dataset)
    return gr.update(choices=choices, value=value)


def on_model(dataset, model):
    choices, value = epoch_choices(dataset, model)
    return gr.update(choices=choices, value=value)


def on_epoch(dataset, model, epoch, start, *zs):
    build(dataset, model, epoch)
    run_all(latent(zs))
    return [figure(start)] + _slider_visibility()


def on_load(dataset, model, epoch, start, *zs):
    """Same as on_epoch but generates nothing: the app opens on empty axes.

    Loading the decoder is not the same as committing to a day. Generating on load meant the
    first thing shown was 288 steps of an unconditioned latent walk over a day with no doses
    in it, which is a real output of the model but not one anybody asked for, and it cost a
    full pass before the user had touched anything.
    """
    build(dataset, model, epoch)
    return [figure(start)] + _slider_visibility()


def _slider_visibility():
    return [gr.update(visible=i < S['latent_dim']) for i in range(MAX_LATENT)]


def on_latent(start, *zs):
    if S.get('playing'):
        return gr.skip()
    run_all(latent(zs))
    return figure(start)


def on_overlap(value, start, *zs):
    S['overlap'] = max(1, min(int(value), S['after']))
    if S.get('playing'):
        return gr.skip()
    run_all(latent(zs))
    return figure(start)


def on_latent_mode(value, start, *zs):
    S['latent_mode'] = str(value).split()[0].lower()
    if S.get('playing'):
        return gr.skip()
    run_all(latent(zs))
    return figure(start)


def on_reset(start, *zs):
    """Back to the state the app opens in: doses cleared and nothing generated.

    Reset used to clear the doses and immediately generate a fresh day, which was consistent
    when the app generated on load. Now that it opens empty, resetting to a curve would make
    Reset the one control that cannot reach the initial state."""
    reset()
    return figure(start)


def take_dose(cho, bolus, start, *zs):
    """A meal/bolus happens NOW: write it into the raw timelines at the current step and
    let refresh_inputs turn it into the pi/ra the decoder is fed."""
    t = min(S['t'], STEPS - 1)
    S['cho'][S['before'] + t] += float(cho or 0)
    S['bolus'][S['before'] + t] += float(bolus or 0)
    if S.get('playing'):
        S['dirty'] = True        # the streaming loop picks it up on its next step
        return gr.skip()
    refresh_inputs()
    run_all(latent(zs))
    return figure(start)


def stream(start, delay, *zs):
    """Play: generate step by step, advancing z one OU step per 5 simulated minutes.

    Reverting to z0 (the slider values) instead of to 0 is the deliberate difference from
    `simulation`, which reverts to whatever it drew z from -- here that draw is the user's.
    Seeded off EVAL_SEED, so the same sliders replay the same day."""
    z = latent(zs)
    z0 = z.copy()
    rng = np.random.default_rng(EVAL_SEED)
    preds = []
    S['bg'] = []
    S['t'] = 0
    S['playing'] = True
    try:
        for t in range(STEPS):
            if S.pop('dirty', False):
                refresh_inputs()
            preds.append(step_window(t, z))
            S['bg'].append(emit(preds, t, S['overlap']))
            S['t'] = t + 1
            z = next_latent(z, z0, rng)
            time.sleep(delay)
            yield [figure(start)] + _z_updates(z)
    finally:
        S['playing'] = False


def _z_updates(z):
    """Push the drifted latent back onto the sliders so the OU walk is visible. Hidden dims
    get a no-op update, and `interactive` is left alone so the play lock survives."""
    values = np.asarray(z).flatten()
    return [gr.update(value=round(float(values[i]), 3)) if i < S['latent_dim'] else gr.update()
            for i in range(MAX_LATENT)]


def _lock(flag):
    return [gr.update(interactive=flag) for _ in range(MAX_LATENT)]


def stop_stream():
    """Stop Play, leaving the day generated so far on screen.

    `cancels` closes the generator, whose own `finally` clears this flag, but it is cleared
    here as well: the two happen on different threads and a handler landing in between would
    otherwise still believe a generation is running and skip its own work.
    """
    S['playing'] = False
    return _lock(True)


# ---------------------------------------------------------------- ui
def ui():
    datasets = list(RUNS)
    with gr.Blocks(title='Glycemia VAE explorer') as demo:
        gr.Markdown('## Glycemia VAE explorer\nBest model and epoch are **bold** and '
                    'preselected. Play walks the latent along an OU path (phi from RUNS) '
                    'and locks the sliders while they track the drift.')
        with gr.Row():
            dataset = gr.Dropdown(datasets, value=datasets[0], label='Dataset')
            mc, mv = model_choices(datasets[0])
            model = gr.Dropdown(mc, value=mv, label='Model', scale=3)
            ec, ev = epoch_choices(datasets[0], mv)
            epoch = gr.Dropdown(ec, value=ev, label='Epoch')
        with gr.Row():
            start = gr.Textbox('08:00', label='Start time (HH:MM)', scale=1)
            delay = gr.Slider(0.0, 0.3, 0.03, step=0.01, label='Play delay (s/step)', scale=2)
        with gr.Row():
            overlap = gr.Dropdown([18, 9, 6, 3, 2, 1], value=18, label='Overlap (predictions '
                                  'averaged per instant)', scale=2)
            latent_mode = gr.Radio(['ou (measured phi)', 'frozen (phi=1)',
                                    'resample (phi=0)'], value='ou (measured phi)',
                                   label='Latent dynamics', scale=3)
        with gr.Row():
            zs = [gr.Slider(-4, 4, 0.0, step=0.05, label=f'z[{i}]', visible=i < 4)
                  for i in range(MAX_LATENT)]
        with gr.Row():
            cho = gr.Number(40, label='CHO (g)')
            bolus = gr.Number(4, label='Bolus (U)')
            take = gr.Button('Take meal and bolus')
            gen = gr.Button('Generate day')
            play = gr.Button('Play generation', variant='primary')
            stop = gr.Button('Stop', variant='stop')
            clear = gr.Button('Reset')
        plot = gr.Plot()

        dataset.change(on_dataset, dataset, model)
        model.change(on_model, [dataset, model], epoch)
        epoch.change(on_epoch, [dataset, model, epoch, start] + zs, [plot] + zs)
        overlap.change(on_overlap, [overlap, start] + zs, plot)
        latent_mode.change(on_latent_mode, [latent_mode, start] + zs, plot)
        for slider in zs:
            slider.release(on_latent, [start] + zs, plot)
        # queue=False so a dose can be taken while the generator is streaming.
        take.click(take_dose, [cho, bolus, start] + zs, plot, queue=False)
        # With the app opening empty, this is how a day is asked for without moving a slider
        # or taking a dose. on_latent already does exactly this, so it is the same path.
        gen.click(on_latent, [start] + zs, plot)
        clear.click(on_reset, [start] + zs, plot)
        # Held as a named event so Stop can cancel it. Cancelling the chain would also skip
        # the re-lock that follows, which is why stop_stream re-locks the sliders itself.
        playing = play.click(lambda: _lock(False), None, zs) \
                      .then(stream, [start, delay] + zs, [plot] + zs)
        playing.then(lambda: _lock(True), None, zs)
        # queue=False so Stop is served while the generator still holds the queue -- the same
        # reason take_dose uses it. Queued, Stop would only arrive once Play had finished.
        stop.click(stop_stream, None, zs, cancels=[playing], queue=False)
        demo.load(on_load, [dataset, model, epoch, start] + zs, [plot] + zs)
    return demo


def selftest():
    """Smallest check that the whole dose path works: doses are causal, carbs raise the
    generated glucose and insulin lowers it (the monotonicity the decoder is trained for)."""
    ds = 'safe'
    build(ds, RUNS[ds]['model'], RUNS[ds]['epoch'])
    z = latent([0.0] * MAX_LATENT)

    # The dose checks are about the decoder's monotonicity, so the latent is held: with it
    # moving, the difference between two runs mixes the dose response with the latent walk
    # and the comparison stops being causal.
    S['latent_mode'] = 'frozen'
    run_all(z)
    flat = float(np.mean(S['bg']))

    t = 12
    reset()
    S['cho'][S['before'] + t] = 60.0
    refresh_inputs()
    assert np.allclose(S['ra'][:S['before'] + t], S['ra'][0]), 'Ra changed before the meal'
    run_all(z)
    meal = float(np.mean(S['bg'][t:]))

    reset()
    S['bolus'][S['before'] + t] = 8.0
    refresh_inputs()
    assert np.allclose(S['pi'][:S['before'] + t], S['pi'][0]), 'PI changed before the bolus'
    run_all(z)
    dose = float(np.mean(S['bg'][t:]))

    print(f'no dose {flat:.4f} | meal {meal:.4f} | bolus {dose:.4f} (scaled)')
    assert meal > flat, 'carbs did not raise generated glucose'
    assert dose < flat, 'insulin did not lower generated glucose'

    # The OU walk Play uses: marginal must stay N(z0, ou_scale^2). A wrong `mu` would drag
    # the latent to 0 over the day and quietly discard whatever the sliders were set to.
    z0 = np.array([[1.5, -1.0, 0.5, 0.0]][0][:S['latent_dim']], dtype='float32').reshape(1, -1)
    rng = np.random.default_rng(EVAL_SEED)
    walk = [z0]
    for _ in range(20000):
        walk.append(step_latent_ou(walk[-1], S['phi'], rng, scale=S['ou_scale'], mu=z0))
    walk = np.concatenate(walk[1000:], axis=0)
    drift = np.abs(walk.mean(axis=0) - z0.flatten())
    spread = walk.std(axis=0)
    print('OU drift', np.round(drift, 3), 'sd', np.round(spread, 3),
          'target', round(S['ou_scale'], 3))
    assert (drift < 0.1).all(), f'OU latent drifted off the slider values: {drift}'
    assert np.allclose(spread, S['ou_scale'], atol=0.1), f'OU spread off target: {spread}'

    # `emit` against a hand-built case, because an off-by-one here would silently average the
    # wrong instants: window s predicts instant s+k at index k, so instant 3 is predicted by
    # window 0 at index 3, window 1 at index 2, and so on.
    preds = [np.arange(s, s + 4, dtype=float) * 0 + s for s in range(4)]   # window s emits s
    assert emit(preds, 3, 1) == 3.0, 'overlap=1 must be the newest prediction alone'
    assert emit(preds, 3, 4) == 1.5, 'overlap=4 must average windows 0..3'
    assert emit(preds, 0, 18) == 0.0, 'instant 0 has exactly one predictor'

    # Every overlap setting must produce a DIFFERENT trajectory. That is all this can assert.
    #
    # It is tempting to also require that more averaging is smoother, and that is what the
    # cohort replay shows, but it does not hold on a single synthetic day: measured here the
    # ordering flips with the dose configuration, monotone increasing with overlap on a day
    # carrying two meals and decreasing on a day carrying one. Averaging predictions of the
    # same instant made at lead times from 0 to 85 minutes is not a low-pass filter when the
    # inputs move fast, since the long-lead predictions come from a different input window.
    # Asserting a direction here would be asserting the setup, not the behaviour.
    S['latent_mode'] = 'frozen'
    w = S['after']
    rough, traces = {}, {}
    for ov in (1, 2, 3, 6, 9, w):
        S['overlap'] = ov
        run_all(z)
        traces[ov] = np.array(S['bg'])
        rough[ov] = float(np.mean(np.abs(np.diff(traces[ov][w:]))))
    print('mean |step| by overlap', {k: round(v, 6) for k, v in rough.items()})
    for ov in (2, 3, 6, 9, w):
        assert not np.allclose(traces[1], traces[ov]), f'overlap {ov} matched overlap 1'

    # The three latent modes must actually differ. Frozen holds z, resample redraws it, and
    # OU is between the two; if a mode were ignored two of these would be identical.
    S['overlap'] = S['after']
    traces = {}
    for mode in ('frozen', 'ou', 'resample'):
        S['latent_mode'] = mode
        run_all(z)
        traces[mode] = np.array(S['bg'])
    for a, b in (('frozen', 'ou'), ('ou', 'resample'), ('frozen', 'resample')):
        assert not np.allclose(traces[a], traces[b]), f'{a} and {b} produced the same day'
    S['latent_mode'] = 'ou'
    print('selftest ok')


if __name__ == '__main__':
    if '--selftest' in sys.argv:
        selftest()
    else:
        # Resolved here, not at import time, so --selftest and a plain import don't
        # require APP_AUTH to be set. Format: "user1:pass1,user2:pass2".
        auth = [tuple(p.split(':', 1)) for p in os.environ['APP_AUTH'].split(',')]
        ui().launch(auth=auth)
