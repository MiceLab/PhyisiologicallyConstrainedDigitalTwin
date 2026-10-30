"""
Everything app.py needs besides Gradio: the shipped models, the physiological input models,
the scalers and the decoder call.

Extracted from the RaModeling research repo (src/config.py, evaluation_numerical/figures.py,
evaluation_numerical/glycemia_evaluation.py, evaluation_numerical/utils.py,
data_extraction/metabolical_models.py). Only what the explorer uses is kept; the functions
are unchanged apart from paths, so the decoder sees exactly what it saw in training.
"""
import json
import os
import re

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm

ROOT = os.path.dirname(os.path.abspath(__file__))
MODELS_PATH = os.path.join(ROOT, 'models')

# ---------------------------------------------------------------- shipped models
# The best-scoring run of each family. Family statistics, not this run's count, describe
# the method (see the research repo's RUNS.md). latent_phi is the measured per-dimension
# Ornstein-Uhlenbeck persistence; the app's Play walks the latent with it.
RUNS = {
    'safe': dict(model='2026_10_02_safe_kra30_kpi20_b0_a18_kl0.001_fb0.5_ev0-0.25-0.25_lr0.001_sd1_sbev025_1',
                      epoch=32,
                      latent_scale=1.0,
                      latent_phi=[0.9681, 0.0000, 0.0000, 0.0000]),
    't1dexi': dict(model='2026_09_10_t1dexi_kra30_kpi20_b0_a18_kl0.001_fb0.5_ev0-0.25-0_lr0.001_sd2_hypo252',
                   epoch=34,
                   latent_scale=1.0,
                   latent_phi=[0.0000, 0.0000, 0.9730, 0.9525]),
}

FASTING_BG_SCALED = 100.0 / 400.0  # ~100 mg/dL fasting baseline, in the model's [0,1] BG scale
EVAL_SEED = 42

# ---------------------------------------------------------------- plot style
plt.rcParams.update({
    'figure.dpi': 100, 'figure.facecolor': 'white',
    'figure.constrained_layout.use': True,
    'font.family': 'serif', 'mathtext.fontset': 'dejavuserif', 'font.size': 12,
    'axes.titlesize': 13, 'axes.labelsize': 12,
    'xtick.labelsize': 10, 'ytick.labelsize': 10, 'legend.fontsize': 10,
    'axes.edgecolor': '#2B2B2B', 'axes.linewidth': 0.9,
    'axes.grid': True, 'grid.alpha': 0.3, 'grid.linewidth': 0.5, 'grid.color': '#B0B0B0',
    'axes.axisbelow': True,
    'legend.frameon': True, 'legend.framealpha': 0.95, 'legend.edgecolor': '#B0B0B0',
    'legend.fancybox': False,
    'lines.linewidth': 1.6,
})
GEN = '#FFB703'         # generated BG
INSULIN = '#023047'     # insulin bolus (event)
PI = '#219EBC'          # plasma insulin (curve)
CHO = '#FB8500'         # CHO meal (event)
RA = '#F4A261'          # glucose Ra (curve)
EUGLY = '#9E9E9E'       # 70-180 mg/dL band

# ---------------------------------------------------------------- physiological models
TMAX_I = 55       # Time-to-maximum absorption for short-acting insulin (minutes)
KE = 0.138        # Insulin elimination rate constant
vi = 0.12         # Insulin distribution volume (L/kg)
TMAX_RESCUE_CHO = 40    # minutes
BIO_RESCUE_CHO = 0.8    # bioavailability fraction


def calculate_plasma_insulin(basal, bolus, dt=5):
    """Discrete-time plasma insulin from basal/bolus (U per dt step)."""
    k_dia = 1 / TMAX_I
    ke = KE
    total_insulin = basal + bolus
    length = len(total_insulin)
    c1 = np.zeros(length)
    c2 = np.zeros(length)
    pi = np.zeros(length)
    c1_eq, c2_eq, pi_eq = 0, 0, 0
    for k in range(length):
        if k == 0:
            c1[k] = c1_eq + dt*(total_insulin[0]/dt - k_dia*c1_eq)
            c2[k] = c2_eq + dt*(k_dia*(c1_eq - c2_eq))
            pi[k] = (1 - ke*dt)*pi_eq + (dt*k_dia*c2_eq)/vi
        else:
            c1[k] = c1[k-1] + dt*(total_insulin[k-1]/dt - k_dia*c1[k-1])
            c2[k] = c2[k-1] + dt*(k_dia*(c1[k-1] - c2[k-1]))
            pi[k] = (1 - ke*dt)*pi[k-1] + (dt*k_dia*c2[k-1])/vi
    return pi


def model_simulation_ra(cho, ts=5):
    """Hovorka rescue-CHO glucose appearance rate for grams of carbs per ts-minute step,
    solved exactly (the system is linear with piecewise-constant input)."""
    cho = np.asarray(cho, dtype=float)
    a_matrix = np.array([[-1 / TMAX_RESCUE_CHO, 0.0],
                         [1 / TMAX_RESCUE_CHO, -1 / TMAX_RESCUE_CHO]])
    b_vector = np.array([BIO_RESCUE_CHO, 0.0])
    augmented = np.zeros((3, 3))
    augmented[:2, :2] = a_matrix
    augmented[:2, 2] = b_vector
    discretized = expm(augmented * ts)
    phi, gamma = discretized[:2, :2], discretized[:2, 2]
    xb = np.zeros(2)
    history = np.zeros((len(cho), 2))
    for step in range(len(cho)):
        # history[k] is the state at t = k*ts; cho[k] first shows up at history[k+1].
        history[step, :] = xb
        xb = phi @ xb + gamma * cho[step]
    return history[:, 1] / TMAX_RESCUE_CHO


# ---------------------------------------------------------------- scalers
# The training MinMax scalers reduced to their only parameter: min is 0 for both channels,
# so scaled = raw / max. No scaler file is shipped, so no training data travels with the app.
PI_RA_MAX = {
    'safe': (40.73834816, 11.37655517),
    't1dexi': (14.16640201, 8.90626483),
}


class MaxScaler:
    def __init__(self, data_max):
        self.data_max = data_max

    def transform(self, x):
        return np.asarray(x) / self.data_max


def load_pi_ra_scalers(dataset):
    pi_max, ra_max = PI_RA_MAX[dataset]
    return MaxScaler(pi_max), MaxScaler(ra_max)


def bg_descale_params(dataset):
    """(offset, span) such that mg/dL = offset + span * scaled. Both datasets scale glucose
    by a fixed [0, 400] mg/dL."""
    return 0.0, 400.0


# ---------------------------------------------------------------- decoder
_MODEL_NAME_BA_RE = re.compile(r'_b(\d+)_a(\d+)$')


def resolve_before_after(model_name):
    """(before, after, uses_bg_hist) the decoder was trained with: training_config.json,
    else the directory name's _b<before>_a<after> suffix. Raises rather than guess, since a
    wrong window can run without error and silently misalign the inputs."""
    config_path = os.path.join(MODELS_PATH, model_name, 'training_config.json')
    if os.path.exists(config_path):
        with open(config_path) as f:
            cfg = json.load(f)
        before = cfg.get('before_points')
        after = cfg.get('after_points', cfg.get('out_window_size'))
        if before is not None and after is not None:
            return before, after, cfg.get('uses_bg_hist', False)
    match = _MODEL_NAME_BA_RE.search(model_name)
    if match:
        return int(match.group(1)), int(match.group(2)), False
    raise ValueError(f"Could not resolve the before/after window for model '{model_name}'")


def generate_fake_samples(g_model, latent_dim, pi, ra, bg_hist=None, z=None):
    """One decoder pass conditioned on latent z (n, latent_dim). Returns (bg_pred, sigma);
    sigma is None unless the decoder is heteroscedastic."""
    decoder_inputs = [np.asarray(z, dtype=np.float32),
                      np.asarray(pi, dtype=np.float32), np.asarray(ra, dtype=np.float32)]
    if bg_hist is not None:
        decoder_inputs.append(np.asarray(bg_hist, dtype=np.float32))
    # Direct call, NOT model.predict(): predict() re-traces on repeated single-sample calls
    # and leaks memory over a long rollout.
    out = g_model(decoder_inputs, training=False)
    if isinstance(out, (list, tuple)):
        return out[0].numpy(), np.exp(0.5 * out[1].numpy())
    return out.numpy(), None


def step_latent_ou(z, phi, rng, scale=1.0, mu=None, sd=None):
    """One exact discrete Ornstein-Uhlenbeck step, scalar or per-dimension phi:

        z[t] = m + phi * (z[t-1] - m) + sqrt(1 - phi^2) * s * eps

    so the marginal stays N(m, s^2). phi=1 freezes z, phi=0 redraws it every step."""
    phi = np.asarray(phi, dtype=np.float32)
    if np.all(phi >= 1.0):
        return z
    z = np.asarray(z, dtype=np.float32)
    m = np.asarray(mu, dtype=np.float32) if mu is not None else 0.0
    s = np.asarray(sd, dtype=np.float32) if sd is not None else scale
    eps = rng.normal(0.0, 1.0, size=z.shape).astype(np.float32)
    # Floor at exactly 0: at phi=1 even a tiny innovation would accumulate into a random walk.
    innov = np.sqrt(np.maximum(1.0 - phi ** 2, 0.0))
    return (m + phi * (z - m) + innov * s * eps).astype(np.float32)
