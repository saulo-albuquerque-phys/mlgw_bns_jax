"""Refined iterative coordinate-descent grid search for TaylorF2 on GW170817."""
import os, time
import numpy as np
os.environ['JAX_PLATFORMS'] = 'cpu'
import jax
import jax.numpy as jnp
jax.config.update('jax_enable_x64', True)

from astropy import constants as const
M_sun = const.M_sun.value; G = const.G.value; c = const.c.value; pc = const.pc.value


def TaylorF2_template(params, frequency_array):
    Mc = params[6]; q = params[7]; phi_c = params[4]
    logdist = params[2]; cos_iota = jnp.cos(params[3])
    distance = jnp.exp(logdist); nu = q / ((1.0 + q)**2)
    Mc_kg = Mc * M_sun; r = distance * pc * 1e6
    M = Mc_kg / (nu**(3.0 / 5.0))
    pi_M = G * jnp.pi * M
    v = jnp.power(pi_M * frequency_array, 1.0 / 3.0) / c
    gamma_e = jnp.float64(0.5772156649015329)
    amp = (jnp.power(jnp.pi, -2.0/3.0) * jnp.sqrt(5.0/24.0)
           * jnp.power(G * Mc_kg / c**3, 5.0/6.0)
           * jnp.power(frequency_array, -7.0/6.0) * (c / r))
    v2 = v**2; v3 = v**3; v4 = v**4; v5 = v**5; v6 = v**6; v7 = v**7
    log_v = jnp.log(v)
    psi = (3.0 / (128.0 * nu * v5)) * (
        1.0
        + v2 * (20.0/9.0) * (743.0/336.0 + nu*11.0/4.0)
        - v3 * (16.0 * jnp.pi)
        + v4 * 10.0 * (3058673.0/1016064.0 + nu*5429.0/1008.0 + nu**2*617.0/144.0)
        + v5 * jnp.pi * (38645.0/756.0 - nu*65.0/9.0) * (1.0 + 3.0*log_v)
        + v6 * (11583231236531.0/4694215680.0 - jnp.pi**2*640.0/3.0
                - 6848.0*gamma_e/21.0 - 6848.0/21.0*log_v
                + nu*(-15737765635.0/3048192.0 + 2255.0*jnp.pi**2/12.0)
                + nu**2*76055.0/1728.0 - nu**3*127825.0/1296.0)
        + v7 * jnp.pi * (77096675.0/254016.0 + nu*378515.0/1512.0 - nu**2*74045.0/756.0)
    )
    psi -= jnp.pi / 4.0
    cos_iota_sq = cos_iota**2
    h0 = amp * jnp.exp(-1j * psi)
    phase_factor = jnp.exp(-2j * phi_c)
    h_plus  = phase_factor * h0 * ((1.0 + cos_iota_sq) / 2.0)
    h_cross = phase_factor * (-1j) * h0 * cos_iota
    return h_plus, h_cross


import sharpy.GW_likelihood as _gw_mod
_gw_mod.template = TaylorF2_template
from sharpy.GW_likelihood import GWNetwork, log_likelihood_det
from functools import partial

TRIGGER_TIME = 1187008882.43
DATA_DIR = "gw170817_data"
ds = {}
for det in ["H1", "L1", "V1"]:
    prefix = {"H1": "H-H1", "L1": "L-L1", "V1": "V-V1"}[det]
    ds[det] = dict(
        data_file=os.path.join(DATA_DIR, f"{prefix}_BWCLEANED_4KHZ-1187008114-1024.txt"),
        channel="GWOSC", trigger_time=TRIGGER_TIME, duration=128,
        sampling_rate=4096, psd_file=None, psd_method="welch",
        download_data=False, zero_noise=False,
    )

print("Building network...", flush=True)
gw_network = GWNetwork(ds, injection_parameters=None)
print("Done.", flush=True)

bd = gw_network.batched_detector
llf = partial(log_likelihood_det, detector_list=bd)
RA = 3.44616; DEC = -0.408084

def eval_ll(logd, incl, phi_c, psi_pol, mc, q, tc):
    p13 = jnp.array([RA, DEC, logd, incl, phi_c, psi_pol, mc, q, tc, 0.0, 0.0, 0.0, 0.0])
    return llf(p13)

jit_ll = jax.jit(eval_ll)
_ = jit_ll(jnp.log(40.0), 2.5, 0.0, 0.0, 1.186, 0.87, 0.0)
_.block_until_ready()
print("JIT ready.\n", flush=True)

noise_ll = float(jit_ll(jnp.log(500.0), jnp.pi/2, 0.0, 0.0, 1.3, 1.0, 0.1))
print(f"Noise logL: {noise_ll:.2f}\n", flush=True)

# ── Helper: 1D scan ──────────────────────────────────────────────────
def scan_1d(name, vals, make_args, best_args):
    """Scan one parameter, return best value and logL."""
    best_ll = -np.inf
    best_v = vals[0]
    for v in vals:
        args = make_args(v, best_args)
        ll = float(jit_ll(*args))
        if ll > best_ll:
            best_ll = ll; best_v = v
    return best_v, best_ll

# ── Starting point from previous coarse search ───────────────────────
best = dict(logd=jnp.log(60.0), incl=2.0, phi_c=3.09, psi_pol=0.0,
            mc=1.1955, q=0.85, tc=0.0024)

def args_from(b):
    return (b['logd'], b['incl'], b['phi_c'], b['psi_pol'], b['mc'], b['q'], b['tc'])

t0 = time.time()

# ── Iterate: coordinate descent with shrinking grids ──────────────────
for iteration in range(4):
    print(f"{'='*60}\nIteration {iteration+1}\n{'='*60}", flush=True)

    # tc: fine scan (0.05 ms resolution)
    tc_center = best['tc']
    tc_range = 0.02 if iteration == 0 else 0.002
    tc_vals = np.linspace(tc_center - tc_range, tc_center + tc_range, 801)
    for tc in tc_vals:
        ll = float(jit_ll(best['logd'], best['incl'], best['phi_c'], best['psi_pol'],
                          best['mc'], best['q'], tc))
        if ll > float(jit_ll(*args_from(best))):
            best['tc'] = tc
    ll_now = float(jit_ll(*args_from(best)))
    print(f"  tc = {best['tc']:.6f} s  dlogL = {ll_now-noise_ll:.2f}", flush=True)

    # phi_c: full 0-2pi scan then refine
    phi_vals = np.linspace(0, 2*np.pi, 256)
    for phi in phi_vals:
        ll = float(jit_ll(best['logd'], best['incl'], phi, best['psi_pol'],
                          best['mc'], best['q'], best['tc']))
        if ll > float(jit_ll(*args_from(best))):
            best['phi_c'] = phi
    ll_now = float(jit_ll(*args_from(best)))
    print(f"  phi_c = {best['phi_c']:.4f} rad  dlogL = {ll_now-noise_ll:.2f}", flush=True)

    # psi_pol: 0-pi scan
    psi_vals = np.linspace(0, np.pi, 128)
    for psi in psi_vals:
        ll = float(jit_ll(best['logd'], best['incl'], best['phi_c'], psi,
                          best['mc'], best['q'], best['tc']))
        if ll > float(jit_ll(*args_from(best))):
            best['psi_pol'] = psi
    ll_now = float(jit_ll(*args_from(best)))
    print(f"  psi = {best['psi_pol']:.4f} rad  dlogL = {ll_now-noise_ll:.2f}", flush=True)

    # mc: scan with resolution 0.0002
    mc_range = 0.02 if iteration == 0 else 0.005
    mc_vals = np.linspace(max(1.17, best['mc']-mc_range),
                          min(1.21, best['mc']+mc_range), 201)
    for mc in mc_vals:
        ll = float(jit_ll(best['logd'], best['incl'], best['phi_c'], best['psi_pol'],
                          mc, best['q'], best['tc']))
        if ll > float(jit_ll(*args_from(best))):
            best['mc'] = mc
    ll_now = float(jit_ll(*args_from(best)))
    print(f"  mc = {best['mc']:.5f} Msun  dlogL = {ll_now-noise_ll:.2f}", flush=True)

    # Re-tune tc after mc change (very sensitive)
    tc_vals2 = np.linspace(best['tc']-0.002, best['tc']+0.002, 401)
    for tc in tc_vals2:
        ll = float(jit_ll(best['logd'], best['incl'], best['phi_c'], best['psi_pol'],
                          best['mc'], best['q'], tc))
        if ll > float(jit_ll(*args_from(best))):
            best['tc'] = tc
    ll_now = float(jit_ll(*args_from(best)))
    print(f"  tc(ref) = {best['tc']:.6f} s  dlogL = {ll_now-noise_ll:.2f}", flush=True)

    # Re-tune phi_c (fine)
    phi_center = best['phi_c']
    phi_vals2 = np.linspace(phi_center - 0.15, phi_center + 0.15, 201)
    for phi in phi_vals2:
        ll = float(jit_ll(best['logd'], best['incl'], phi, best['psi_pol'],
                          best['mc'], best['q'], best['tc']))
        if ll > float(jit_ll(*args_from(best))):
            best['phi_c'] = phi
    ll_now = float(jit_ll(*args_from(best)))
    print(f"  phi_c(ref) = {best['phi_c']:.4f} rad  dlogL = {ll_now-noise_ll:.2f}", flush=True)

    # q: scan
    q_vals = np.linspace(0.5, 1.0, 101)
    for q in q_vals:
        ll = float(jit_ll(best['logd'], best['incl'], best['phi_c'], best['psi_pol'],
                          best['mc'], q, best['tc']))
        if ll > float(jit_ll(*args_from(best))):
            best['q'] = q
    ll_now = float(jit_ll(*args_from(best)))
    print(f"  q = {best['q']:.3f}  dlogL = {ll_now-noise_ll:.2f}", flush=True)

    # logd: scan (distance 1-75 Mpc)
    logd_vals = np.linspace(np.log(1), np.log(75), 150)
    for logd in logd_vals:
        ll = float(jit_ll(logd, best['incl'], best['phi_c'], best['psi_pol'],
                          best['mc'], best['q'], best['tc']))
        if ll > float(jit_ll(*args_from(best))):
            best['logd'] = logd
    ll_now = float(jit_ll(*args_from(best)))
    print(f"  D_L = {np.exp(best['logd']):.2f} Mpc  dlogL = {ll_now-noise_ll:.2f}", flush=True)

    # incl: scan (0-pi)
    incl_vals = np.linspace(0.01, np.pi-0.01, 200)
    for incl in incl_vals:
        ll = float(jit_ll(best['logd'], incl, best['phi_c'], best['psi_pol'],
                          best['mc'], best['q'], best['tc']))
        if ll > float(jit_ll(*args_from(best))):
            best['incl'] = incl
    ll_now = float(jit_ll(*args_from(best)))
    print(f"  incl = {best['incl']:.4f} rad  dlogL = {ll_now-noise_ll:.2f}", flush=True)

    print(f"\n  >>> Iteration {iteration+1} result: dlogL = {ll_now-noise_ll:.2f}", flush=True)

dt = time.time() - t0
final_ll = float(jit_ll(*args_from(best)))

print(f"\n{'='*60}")
print(f"FINAL RESULT ({dt:.1f}s total)")
print(f"{'='*60}")
print(f"Noise logL:   {noise_ll:.2f}")
print(f"Best  logL:   {final_ll:.2f}")
print(f"Delta logL:   {final_ll - noise_ll:.2f}")
print(f"\nBest-fit parameters:")
print(f"  Mc      = {best['mc']:.5f} M_sun")
print(f"  q       = {best['q']:.3f}")
print(f"  D_L     = {np.exp(best['logd']):.2f} Mpc")
print(f"  iota    = {best['incl']:.4f} rad ({np.degrees(best['incl']):.1f} deg)")
print(f"  tc      = {best['tc']:.6f} s")
print(f"  phi_c   = {best['phi_c']:.4f} rad")
print(f"  psi_pol = {best['psi_pol']:.4f} rad")
print(f"\nLiterature (GW170817):")
print(f"  Mc ~ 1.186 M_sun, q ~ 0.73-1.0, D_L ~ 40 Mpc")
print(f"  iota ~ 147-163 deg (2.57-2.84 rad)")
