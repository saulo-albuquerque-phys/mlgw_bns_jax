"""
Quick test of the TaylorF2 notebook pipeline:
1. Build GW network from existing cleaned data
2. Evaluate TaylorF2 waveform
3. Check likelihood at literature values for GW170817
4. Scan key parameters around the best-fit to verify the peak is near literature values
"""
import os, sys, time
from functools import partial
import numpy as np

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)

print("JAX devices:", jax.devices())

# ── TaylorF2 waveform ────────────────────────────────────────────────
import sharpy.GW_likelihood as _gw_mod
from sharpy.utils import McQ2Masses
from astropy import constants as const

M_sun = const.M_sun.value
G = const.G.value
c = const.c.value
pc = const.pc.value


def TaylorF2_template(params, frequency_array):
    Mc       = params[6]
    q        = params[7]
    phi_c    = params[4]
    logdist  = params[2]
    cos_iota = jnp.cos(params[3])

    distance = jnp.exp(logdist)
    nu = q / ((1 + q) ** 2)

    Mc_kg = Mc * M_sun
    r = distance * pc * 1e6

    M = Mc_kg / (nu ** (3.0 / 5.0))

    pi_M = G * jnp.pi * M
    v = jnp.power(pi_M * frequency_array, 1.0 / 3.0) / c
    gamma_e = jnp.float64(0.5772156649015329)

    amp = (jnp.power(jnp.pi, -2.0 / 3.0) * jnp.sqrt(5.0 / 24.0)
           * jnp.power(G * Mc_kg / c**3, 5.0 / 6.0)
           * jnp.power(frequency_array, -7.0 / 6.0)
           * (c / r))

    v2 = v**2;  v3 = v**3;  v4 = v**4
    v5 = v**5;  v6 = v**6;  v7 = v**7
    log_v = jnp.log(v)

    phi_plus = (3.0 / (128.0 * nu * v5)) * (
        1.0
        + v2 * (20.0 / 9.0) * (743.0 / 336.0 + nu * 11.0 / 4.0)
        - v3 * (16.0 * jnp.pi)
        + v4 * 10.0 * (3058673.0 / 1016064.0 + nu * 5429.0 / 1008.0 + nu**2 * 617.0 / 144.0)
        + v5 * jnp.pi * (38645.0 / 756.0 - nu * 65.0 / 9.0) * (1.0 + 3.0 * log_v)
        + v6 * (11583231236531.0 / 4694215680.0 - jnp.pi**2 * 640.0 / 3.0
                - 6848.0 * gamma_e / 21.0 - 6848.0 / 21.0 * log_v
                + nu * (-15737765635.0 / 3048192.0 + 2255.0 * jnp.pi**2 / 12.0)
                + nu**2 * 76055.0 / 1728.0 - nu**3 * 127825.0 / 1296.0)
        + v7 * jnp.pi * (77096675.0 / 254016.0 + nu * 378515.0 / 1512.0 - nu**2 * 74045.0 / 756.0)
    )
    phi_plus = phi_plus + jnp.pi - jnp.pi / 4.0
    phi_cross = phi_plus + jnp.pi / 2.0

    phase_factor = jnp.exp(-1j * phi_c)
    exp_phi_plus = jnp.exp(1j * phi_plus)
    exp_phi_cross = jnp.exp(1j * phi_cross)

    cos_iota_sq = cos_iota**2
    h_plus  = phase_factor * amp * ((1.0 + cos_iota_sq) / 2.0) * exp_phi_plus
    h_cross = phase_factor * amp * cos_iota * exp_phi_cross

    return h_plus, h_cross


_gw_mod.template = TaylorF2_template
print("SHARPy template patched with TaylorF2.")

# ── Event parameters ─────────────────────────────────────────────────
TRIGGER_TIME = 1187008882.43
SEGMENT_DURATION = 128.0
SAMPLING_RATE = 4096
F_LOWER = 23.0
F_UPPER = 2000.0
N_FREQ_POINTS = 3000
DATA_START_GPS = 1187008114
DATA_DURATION = 1024

FIXED_RA  = 3.44616
FIXED_DEC = -0.408084

DATA_DIR = "gw170817_data"

# ── Build network (use existing cleaned files for testing) ───────────
from sharpy.GW_likelihood import GWNetwork, log_likelihood_det, antenna_pattern_functions
from sharpy.utils import TimeDelayFromEarthCenter

# Try BWCLEANED first, fallback to CLEANED
prefix = "BWCLEANED"
test_file = os.path.join(DATA_DIR, f"H-H1_{prefix}_4KHZ-{DATA_START_GPS}-{DATA_DURATION}.txt")
if not os.path.isfile(test_file):
    prefix = "CLEANED"
    print(f"BWCLEANED files not found, falling back to {prefix} files for testing")

data_files = {
    det: os.path.join(DATA_DIR, f"{det[0]}-{det}_{prefix}_4KHZ-{DATA_START_GPS}-{DATA_DURATION}.txt")
    for det in ["H1", "L1", "V1"]
}
for det, f in data_files.items():
    assert os.path.isfile(f), f"Missing: {f}"
    print(f"{det}: {os.path.basename(f)}")

detector_settings = {}
for det in ["H1", "L1", "V1"]:
    detector_settings[det] = dict(
        data_file=data_files[det], channel="GWOSC",
        trigger_time=TRIGGER_TIME, duration=SEGMENT_DURATION,
        sampling_rate=SAMPLING_RATE,
        f_lower=F_LOWER, f_upper=F_UPPER,
        psd_file=None, psd_method="welch",
        download_data=False, zero_noise=False,
    )

print(f"\nBuilding GW network (segment={SEGMENT_DURATION}s)...")
t0 = time.time()
gw_network = GWNetwork(detector_settings, injection_parameters=None)
print(f"Network built in {time.time() - t0:.2f} s")

# ── Resample to low-res grid ────────────────────────────────────────
from scipy.interpolate import interp1d as _interp1d

batched_det = gw_network.batched_detector
f_orig = np.array(batched_det.Frequency[0])
n_det = len(batched_det.latitude)
f_new = np.linspace(F_LOWER, F_UPPER, N_FREQ_POINTS)
df_new = f_new[1] - f_new[0]

print(f"\nOriginal grid: {len(f_orig)} bins, df = {1/SEGMENT_DURATION:.4f} Hz")
print(f"New grid: {N_FREQ_POINTS} bins, df = {df_new:.4f} Hz  ({len(f_orig)/N_FREQ_POINTS:.0f}x reduction)")

new_FrequencySeries_list = []
new_PSD_list = []

for i in range(n_det):
    f_det = np.array(batched_det.Frequency[i])
    sf_det = np.array(batched_det.FrequencySeries[i])
    psd_det = np.array(batched_det.PowerSpectralDensity[i])

    phase_corr = 2.0 * np.pi * f_det * (SEGMENT_DURATION - 1.0)
    sf_rotated = sf_det * np.exp(1j * phase_corr)

    sf_new_real = _interp1d(f_det, sf_rotated.real, kind='cubic',
                            bounds_error=False, fill_value=0.0)(f_new)
    sf_new_imag = _interp1d(f_det, sf_rotated.imag, kind='cubic',
                            bounds_error=False, fill_value=0.0)(f_new)
    sf_new = sf_new_real + 1j * sf_new_imag

    log_psd = np.log(np.where(psd_det > 0, psd_det, 1e-100))
    psd_new = np.exp(
        _interp1d(f_det, log_psd, kind='cubic',
                  bounds_error=False, fill_value=np.log(1e-100))(f_new)
    )

    new_FrequencySeries_list.append(sf_new)
    new_PSD_list.append(psd_new)

batched_det = batched_det.replace(
    Frequency=jnp.stack([jnp.array(f_new, dtype=jnp.float64)] * n_det),
    FrequencySeries=jnp.stack([jnp.array(s, dtype=jnp.complex128) for s in new_FrequencySeries_list]),
    PowerSpectralDensity=jnp.stack([jnp.array(p, dtype=jnp.float64) for p in new_PSD_list]),
    sigmasq=jnp.stack([jnp.array(p, dtype=jnp.float64) for p in new_PSD_list]),
    TwoDeltaTOverN=jnp.stack([jnp.float64(2.0 * df_new)] * n_det),
)
gw_network.batched_detector = batched_det

# Monkey-patch project_waveform for low-grid
def project_waveform_lowgrid(params, detector_dictionary):
    f = detector_dictionary.Frequency
    h_plus, h_cross = _gw_mod.template(params, f)
    fplus, fcross = antenna_pattern_functions(
        params,
        detector_dictionary.latitude, detector_dictionary.longitude,
        detector_dictionary.gamma, detector_dictionary.zeta,
        detector_dictionary.trigtime,
    )
    ra, dec = params[0], params[1]
    tc = detector_dictionary.trigtime + params[8]
    timedelay = TimeDelayFromEarthCenter(
        detector_dictionary.latitude, detector_dictionary.longitude,
        detector_dictionary.elevation, ra, dec, tc,
    )
    timeshift = timedelay + params[8]
    shift = 2.0 * np.pi * f * timeshift
    h = (fplus * h_plus + fcross * h_cross) * (jnp.cos(shift) - 1j * jnp.sin(shift))
    return h

_gw_mod.project_waveform = project_waveform_lowgrid
print("Monkey-patched project_waveform for low-grid.")
print(f"Resampled to {N_FREQ_POINTS}-point grid.")

# ── Likelihood setup ─────────────────────────────────────────────────
batched_detector = gw_network.batched_detector
log_likelihood_full = partial(log_likelihood_det, detector_list=batched_detector)

def log_likelihood_reduced(params_9):
    params_13 = jnp.concatenate([
        jnp.array([FIXED_RA, FIXED_DEC]),
        params_9,
        jnp.array([0.0, 0.0]),
    ])
    return log_likelihood_full(params_13)

# ── Test at literature values for GW170817 ────────────────────────────
print("\n" + "="*70)
print("TESTING LIKELIHOOD AT GW170817 LITERATURE VALUES")
print("="*70)

# GW170817 literature values (Abbott+ 2019, PRX 9, 011001):
# Mc = 1.186 +0.001/-0.001 M_sun
# q = 0.73-1.0 (90% CI)
# chi_eff = 0.00 +0.02/-0.01
# D_L ~ 40 Mpc
# theta_JN ~ 2.5-2.7 rad (or ~0.4-0.6 rad, degenerate)

# params_9: [logdist, incl, phic, pol, mc, q, tc, chi1, chi2]
literature_params = jnp.array([
    jnp.log(40.0),    # logdist ~ 40 Mpc
    jnp.pi / 6,       # inclination ~ 30 deg (face-on-ish)
    1.0,               # phic (arbitrary)
    0.5,               # pol (arbitrary)
    1.186,             # mc (literature value)
    0.9,               # q ~ 0.9
    0.0,               # tc = 0 (relative to trigger)
    0.0,               # chi1 ~ 0
    0.0,               # chi2 ~ 0
])

logL_lit = log_likelihood_reduced(literature_params)
print(f"\nlogL at literature values = {float(logL_lit):.2f}")

# ── Scan chirp mass around literature value ──────────────────────────
print("\n--- Chirp mass scan ---")
mc_values = np.linspace(1.180, 1.210, 31)
logL_mc = []
for mc in mc_values:
    p = literature_params.at[4].set(mc)
    logL_mc.append(float(log_likelihood_reduced(p)))

best_mc_idx = np.argmax(logL_mc)
print(f"Best Mc = {mc_values[best_mc_idx]:.4f} M_sun (logL = {logL_mc[best_mc_idx]:.2f})")
print(f"Literature Mc = 1.186 M_sun")

# ── Scan mass ratio ──────────────────────────────────────────────────
best_mc = mc_values[best_mc_idx]
p_best = literature_params.at[4].set(best_mc)

print("\n--- Mass ratio scan ---")
q_values = np.linspace(0.5, 1.0, 21)
logL_q = []
for q in q_values:
    p = p_best.at[5].set(q)
    logL_q.append(float(log_likelihood_reduced(p)))

best_q_idx = np.argmax(logL_q)
print(f"Best q = {q_values[best_q_idx]:.3f} (logL = {logL_q[best_q_idx]:.2f})")
print(f"Literature q ~ 0.73-1.0")

# ── Scan distance ───────────────────────────────────────────────────
p_best2 = p_best.at[5].set(q_values[best_q_idx])

print("\n--- Distance scan ---")
d_values = np.linspace(10.0, 75.0, 26)
logL_d = []
for d in d_values:
    p = p_best2.at[0].set(jnp.log(d))
    logL_d.append(float(log_likelihood_reduced(p)))

best_d_idx = np.argmax(logL_d)
print(f"Best D_L = {d_values[best_d_idx]:.1f} Mpc (logL = {logL_d[best_d_idx]:.2f})")
print(f"Literature D_L ~ 40 Mpc")

# ── Scan tc ──────────────────────────────────────────────────────────
p_best3 = p_best2.at[0].set(jnp.log(d_values[best_d_idx]))

print("\n--- tc scan ---")
tc_values = np.linspace(-0.05, 0.05, 21)
logL_tc = []
for tc in tc_values:
    p = p_best3.at[6].set(tc)
    logL_tc.append(float(log_likelihood_reduced(p)))

best_tc_idx = np.argmax(logL_tc)
print(f"Best tc = {tc_values[best_tc_idx]:.4f} s (logL = {logL_tc[best_tc_idx]:.2f})")

# ── Scan inclination ─────────────────────────────────────────────────
p_best4 = p_best3.at[6].set(tc_values[best_tc_idx])

print("\n--- Inclination scan ---")
incl_values = np.linspace(0.1, 3.0, 30)
logL_incl = []
for incl in incl_values:
    p = p_best4.at[1].set(incl)
    logL_incl.append(float(log_likelihood_reduced(p)))

best_incl_idx = np.argmax(logL_incl)
print(f"Best incl = {incl_values[best_incl_idx]:.3f} rad ({np.degrees(incl_values[best_incl_idx]):.1f} deg) "
      f"(logL = {logL_incl[best_incl_idx]:.2f})")
print(f"Literature incl ~ 2.5-2.7 rad (or ~0.4-0.6 rad)")

# ── Final scan: phase + pol grid at best physical params ─────────────
p_best5 = p_best4.at[1].set(incl_values[best_incl_idx])

print("\n--- Phase + polarisation grid scan ---")
phic_values = np.linspace(0, 2*np.pi, 13)
pol_values = np.linspace(0, np.pi, 7)
best_logL = -np.inf
best_phic, best_pol = 0.0, 0.0
for phic in phic_values:
    for pol in pol_values:
        p = p_best5.at[2].set(phic).at[3].set(pol)
        ll = float(log_likelihood_reduced(p))
        if ll > best_logL:
            best_logL = ll
            best_phic, best_pol = phic, pol

p_best6 = p_best5.at[2].set(best_phic).at[3].set(best_pol)
print(f"Best phic = {best_phic:.3f}, pol = {best_pol:.3f} (logL = {best_logL:.2f})")

# ── Summary ──────────────────────────────────────────────────────────
print("\n" + "="*70)
print("SUMMARY — Best-fit from grid scans")
print("="*70)
best_params = p_best6
print(f"  logdist  = {float(best_params[0]):.4f}  ->  D_L = {np.exp(float(best_params[0])):.1f} Mpc")
print(f"  incl     = {float(best_params[1]):.3f} rad ({np.degrees(float(best_params[1])):.1f} deg)")
print(f"  phic     = {float(best_params[2]):.3f}")
print(f"  pol      = {float(best_params[3]):.3f}")
print(f"  Mc       = {float(best_params[4]):.4f} M_sun")
print(f"  q        = {float(best_params[5]):.3f}")
print(f"  tc       = {float(best_params[6]):.5f} s")
print(f"  chi1     = {float(best_params[7]):.3f}")
print(f"  chi2     = {float(best_params[8]):.3f}")
print(f"  logL     = {best_logL:.2f}")
print(f"\nExpected (literature):")
print(f"  Mc ~ 1.186 M_sun, q ~ 0.73-1.0, D_L ~ 40 Mpc, chi_eff ~ 0")
print(f"  incl ~ 2.5-2.7 rad or 0.4-0.6 rad (degenerate)")

# ── Quick SMC run (very few particles, just to test convergence direction) ──
print("\n" + "="*70)
print("QUICK SMC TEST (50 particles — just testing convergence)")
print("="*70)

from sharpy.smc_functions import run_sharpy

prior_bounds = jnp.array([
    [jnp.log(1.0),  jnp.log(75.0)],
    [0.0,           jnp.pi],
    [0.0,           2 * jnp.pi],
    [0.0,           jnp.pi],
    [1.18,          1.21],
    [0.5,           1.0],
    [-0.1,          0.1],
    [-0.5,          0.5],
    [-0.5,          0.5],
])

boundary_conditions = jnp.array([0, 0, 1, 1, 0, 0, 0, 0, 0])

def prior(params):
    return 0.0

OUTDIR = "results_taylorf2_test"
os.makedirs(OUTDIR, exist_ok=True)

print("Running SHARPy with 50 particles...")
t0 = time.time()
result_dict = run_sharpy(
    log_likelihood_reduced, prior,
    prior_bounds, boundary_conditions,
    0.95, 50, 0.3,
    jax.random.PRNGKey(42),
    folder=OUTDIR, label="TF2_test",
)
dt_run = time.time() - t0

samples = np.array(result_dict["posterior_samples"])
logZ, dlogZ = result_dict["logZ"], result_dict["dlogZ"]
print(f"\nDone in {dt_run:.1f} s — log Z = {logZ:.2f} +/- {dlogZ:.2f}")
print(f"Posterior samples: {samples.shape}")

# Print posterior medians
param_names = ["logdist", "incl", "phic", "pol", "mc", "q", "tc", "chi1", "chi2"]
print("\nPosterior medians:")
for j, name in enumerate(param_names):
    med = np.median(samples[:, j])
    lo, hi = np.percentile(samples[:, j], [5, 95])
    extra = ""
    if name == "logdist":
        extra = f"  -> D_L = {np.exp(med):.1f} Mpc [{np.exp(lo):.1f}, {np.exp(hi):.1f}]"
    if name == "incl":
        extra = f"  = {np.degrees(med):.1f} deg"
    print(f"  {name:10s} = {med:8.4f}  [{lo:.4f}, {hi:.4f}]{extra}")

m1_arr = np.zeros(len(samples))
m2_arr = np.zeros(len(samples))
for i in range(len(samples)):
    m1_arr[i], m2_arr[i] = McQ2Masses(samples[i, 4], samples[i, 5])
chi_eff = (m1_arr * samples[:, 7] + m2_arr * samples[:, 8]) / (m1_arr + m2_arr)
print(f"\n  chi_eff   = {np.median(chi_eff):.4f}  [{np.percentile(chi_eff, 5):.4f}, {np.percentile(chi_eff, 95):.4f}]")
print(f"  m1        = {np.median(m1_arr):.3f}  [{np.percentile(m1_arr, 5):.3f}, {np.percentile(m1_arr, 95):.3f}]")
print(f"  m2        = {np.median(m2_arr):.3f}  [{np.percentile(m2_arr, 5):.3f}, {np.percentile(m2_arr, 95):.3f}]")
