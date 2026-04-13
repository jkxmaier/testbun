"""
PyTorch Molecular Mechanics Minimizer  — v2  (robust, GPU-ready)
=================================================================
Changes vs v1
─────────────
1. Auto-build 1-2 / 1-3 exclusion and 1-4 pair lists from topology.
   (v1 left excl_pairs=[] when called from your protein code, so bonded
    atoms were seeing their full LJ/Coulomb interaction — wrong gradients.)

2. Soft-core LJ for clash recovery.
   A linear "cap" replaces the r⁻¹² wall below r_switch, keeping the
   energy and gradient finite even when two heavy atoms sit on top of each
   other.  A steepest-descent pre-phase (with gradient clipping) is run
   first to resolve severe clashes before handing off to L-BFGS.

3. GPU support with CPU → GPU phase transition.
   mol.to_device(device) moves every tensor.  The minimizer runs the
   first `n_cpu_steps` on CPU, then migrates to the target device.

4. Fixed the 1-4 scaling bug introduced in v1 upload:
   was:  sc14 * E_14          (adds a full extra copy)
   now:  (sc14 - 1) * E_14   (corrects from 1.0 to sc14)

Units:  kcal/mol · Å · e  throughout.
"""

import math
import torch
import torch.optim as optim

torch.set_default_dtype(torch.float64)


# =============================================================================
# Topology helpers  (auto-build exclusion / 1-4 lists)
# =============================================================================

def build_excl_and_14(N, bonds, angles, torsions):
    """
    Given connectivity lists, return:
      excl_pairs  — set of (i,j) pairs to zero out entirely (1-2 and 1-3)
      pairs_14    — set of (i,j) pairs to scale (1-4, not already excluded)

    Always call this if you do not have an explicit exclusion list from
    your force-field parser — leaving these empty silently corrupts all
    NB gradients.
    """
    excl = set()

    # 1-2  (bonds)
    for i, j in bonds:
        excl.add((min(i,j), max(i,j)))

    # 1-3  (angles: terminal atoms)
    for i, j, k in angles:
        excl.add((min(i,k), max(i,k)))

    # 1-4  (torsions: terminal atoms, not already in excl)
    pairs14 = set()
    for i, j, k, l in torsions:
        p = (min(i,l), max(i,l))
        if p not in excl:
            pairs14.add(p)

    return list(excl), list(pairs14)


# =============================================================================
# Force field
# =============================================================================

class MolecularFF:
    """
    Parameters
    ----------
    coords, charges, eps, sigma          — per-atom  (N,) / (N,3)
    bonds, angles, torsions, impropers   — index arrays
    bond_k/r0, angle_k/theta0           — bonded FF params
    torsion_V/n/delta, improper_k/psi0  — dihedral FF params
    excl_pairs   — 1-2 + 1-3 pairs (fully excluded).
                   Pass None to auto-build from bonds+angles+torsions.
    pairs_14     — 1-4 pairs (scaled).
                   Pass None to auto-build.
    scale14_coul, scale14_vdw           — 1-4 scale factors  (AMBER: 1/1.2, 0.5)
    r_switch     — Å below which soft-core LJ cap kicks in (default 0.8)
    """

    COUL_CONST = 332.0636   # kcal·Å / (mol·e²)

    def __init__(self,
                 coords, charges, eps, sigma,
                 bonds, angles, torsions, impropers,
                 bond_k, bond_r0,
                 angle_k, angle_theta0,
                 torsion_V, torsion_n, torsion_delta,
                 improper_k, improper_psi0,
                 excl_pairs=None,
                 pairs_14=None,
                 scale14_coul=1/1.2,
                 scale14_vdw=0.5,
                 r_switch=0.8):

        self.device = torch.device('cpu')
        N = len(coords)

        # ── optimizable coordinates ─────────────────────────────────────────
        self.coords = torch.tensor(coords, dtype=torch.float64,
                                   requires_grad=True)

        # ── fixed atom properties ────────────────────────────────────────────
        self.q     = torch.tensor(charges, dtype=torch.float64)
        self.eps   = torch.tensor(eps,     dtype=torch.float64)
        self.sigma = torch.tensor(sigma,   dtype=torch.float64)

        # ── topology ─────────────────────────────────────────────────────────
        def _t(lst, cols):
            return (torch.tensor(list(lst), dtype=torch.long)
                    if len(lst) > 0 else torch.zeros((0, cols), dtype=torch.long))

        self.bonds     = _t(bonds,     2)
        self.angles    = _t(angles,    3)
        self.torsions  = _t(torsions,  4)
        self.impropers = _t(impropers, 4)

        # ── bonded FF parameters ─────────────────────────────────────────────
        self.bond_k        = torch.tensor(bond_k,        dtype=torch.float64)
        self.bond_r0       = torch.tensor(bond_r0,       dtype=torch.float64)
        self.angle_k       = torch.tensor(angle_k,       dtype=torch.float64)
        self.angle_theta0  = torch.tensor(angle_theta0,  dtype=torch.float64)
        self.torsion_V     = torch.tensor(torsion_V,     dtype=torch.float64)
        self.torsion_n     = torch.tensor(torsion_n,     dtype=torch.float64)
        self.torsion_delta = torch.tensor(torsion_delta, dtype=torch.float64)
        self.improper_k    = torch.tensor(improper_k,    dtype=torch.float64)
        self.improper_psi0 = torch.tensor(improper_psi0, dtype=torch.float64)

        # ── auto-build exclusion lists if not provided ────────────────────────
        #   THIS IS THE MOST COMMON BUG: passing empty lists causes wrong NB.
        if excl_pairs is None or len(excl_pairs) == 0:
            excl_pairs, pairs_14 = build_excl_and_14(
                N,
                bonds if len(bonds) else [],
                angles if len(angles) else [],
                torsions if len(torsions) else [],
            )
            print(f"  [FF] Auto-built {len(excl_pairs)} exclusion pairs, "
                  f"{len(pairs_14)} 1-4 pairs from topology.")
        elif pairs_14 is None or len(pairs_14) == 0:
            _, pairs_14 = build_excl_and_14(
                N, bonds, angles, torsions)
            print(f"  [FF] Auto-built {len(pairs_14)} 1-4 pairs from topology.")

        # ── exclusion / scaling masks ─────────────────────────────────────────
        excl_mask = torch.eye(N, dtype=torch.bool)
        for i, j in excl_pairs:
            excl_mask[i, j] = excl_mask[j, i] = True

        mask14 = torch.zeros(N, N, dtype=torch.bool)
        for i, j in pairs_14:
            mask14[i, j] = mask14[j, i] = True

        self.excl_mask = excl_mask
        self.mask14    = mask14
        self.sc14_coul = float(scale14_coul)
        self.sc14_vdw  = float(scale14_vdw)

        # upper-triangle mask (avoid double counting)
        self._triu = torch.ones(N, N, dtype=torch.bool).triu(diagonal=1)

        # LB mixing tables (fixed — recomputed after device move)
        self._r_switch = r_switch
        self._precompute_nb_tables()

    # ── precompute fixed NB mixing tables ────────────────────────────────────
    def _precompute_nb_tables(self):
        self._eps_ij   = torch.sqrt(self.eps[:, None] * self.eps[None, :])
        self._sigma_ij = 0.5 * (self.sigma[:, None] + self.sigma[None, :])
        self._qi_qj    = self.q[:, None] * self.q[None, :]
        N = self.eps.shape[0]
        self._r2_self  = torch.eye(N, dtype=torch.float64,
                                   device=self.eps.device) * 1e10

    # ── move all tensors to a device ─────────────────────────────────────────
    def to_device(self, device):
        """
        Move all tensors (coordinates + all FF tables) to `device`.
        Returns self for chaining.

        Usage:
            mol.to_device(torch.device('cuda'))   # → GPU
            mol.to_device(torch.device('cpu'))    # → back to CPU
        """
        device = torch.device(device)
        if device == self.device:
            return self
        self.device = device

        def _m(t):
            return t.to(device)

        # Move coords carefully (need to preserve grad)
        data = self.coords.detach().to(device)
        self.coords = data.requires_grad_(True)

        self.q     = _m(self.q)
        self.eps   = _m(self.eps)
        self.sigma = _m(self.sigma)

        self.bonds     = _m(self.bonds)
        self.angles    = _m(self.angles)
        self.torsions  = _m(self.torsions)
        self.impropers = _m(self.impropers)

        self.bond_k        = _m(self.bond_k)
        self.bond_r0       = _m(self.bond_r0)
        self.angle_k       = _m(self.angle_k)
        self.angle_theta0  = _m(self.angle_theta0)
        self.torsion_V     = _m(self.torsion_V)
        self.torsion_n     = _m(self.torsion_n)
        self.torsion_delta = _m(self.torsion_delta)
        self.improper_k    = _m(self.improper_k)
        self.improper_psi0 = _m(self.improper_psi0)

        self.excl_mask = _m(self.excl_mask)
        self.mask14    = _m(self.mask14)
        self._triu     = _m(self._triu)

        # Recompute NB tables on new device
        self._precompute_nb_tables()
        return self

    # ── diagnostics ──────────────────────────────────────────────────────────
    def worst_angles(self, n=10):
        """Print the n most-strained angles — useful for diagnosing high E_angle."""
        with torch.no_grad():
            ai, aj, ak = self.angles[:,0], self.angles[:,1], self.angles[:,2]
            u = self.coords[ai] - self.coords[aj]
            v = self.coords[ak] - self.coords[aj]
            cos_t = (u*v).sum(1) / (u.norm(dim=1)*v.norm(dim=1) + 1e-12)
            theta = torch.acos(cos_t.clamp(-1+1e-7, 1-1e-7))
            dtheta = (theta - self.angle_theta0).abs()
            E_each = self.angle_k * (theta - self.angle_theta0)**2
            top = E_each.topk(min(n, len(E_each)))
            print(f"\n  Worst {n} angles:")
            print(f"  {'idx':>6}  {'i':>5} {'j':>5} {'k':>5}  "
                  f"{'θ_now°':>8}  {'θ0°':>8}  {'Δθ°':>7}  {'E':>10}")
            for rank, idx in enumerate(top.indices.tolist()):
                i,j,k = (self.angles[idx,c].item() for c in range(3))
                th  = math.degrees(theta[idx].item())
                th0 = math.degrees(self.angle_theta0[idx].item())
                dth = math.degrees(dtheta[idx].item())
                e   = E_each[idx].item()
                print(f"  {idx:>6}  {i:>5} {j:>5} {k:>5}  "
                      f"{th:>8.2f}  {th0:>8.2f}  {dth:>7.2f}  {e:>10.3f}")

    def check_angle_units(self):
        """
        Sanity check: AMBER delivers angle_theta0 in DEGREES — your loader
        must convert to radians before passing here.
        If min(theta0) > π it is almost certainly still in degrees.
        """
        t0 = self.angle_theta0
        if t0.max().item() > math.pi + 0.1:
            print(f"  ⚠  WARNING: angle_theta0 max = {t0.max().item():.2f} "
                  f"> π rad — values look like they are still in DEGREES!")
            print(f"      Fix:  angle_theta0 = [math.radians(a) for a in angle_theta0]")
        else:
            mn, mx = t0.min().item(), t0.max().item()
            print(f"  ✓  angle_theta0 range: [{math.degrees(mn):.1f}°, "
                  f"{math.degrees(mx):.1f}°]  — looks like radians ✓")

    def nan_report(self):
        """Identify which energy term produced NaN/Inf."""
        results = {}
        for name, fn in [('bond',     self.E_bond),
                         ('angle',    self.E_angle),
                         ('torsion',  self.E_torsion),
                         ('improper', self.E_improper),
                         ('nonbond',  self.E_nonbonded)]:
            try:
                with torch.no_grad():
                    v = fn().item()
                results[name] = v
            except Exception as exc:
                results[name] = f'ERROR: {exc}'
        for k, v in results.items():
            flag = '  ⚠  NaN/Inf!' if isinstance(v, float) and not math.isfinite(v) else ''
            print(f"  {k:10s}  {str(v):>20}{flag}")
        return results

    # =========================================================================
    # Energy terms
    # =========================================================================

    def E_bond(self):
        if self.bonds.shape[0] == 0:
            return self.coords.new_zeros(())
        i, j = self.bonds[:,0], self.bonds[:,1]
        r = (self.coords[i] - self.coords[j]).norm(dim=1)
        return (self.bond_k * (r - self.bond_r0)**2).sum()

    def E_angle(self):
        if self.angles.shape[0] == 0:
            return self.coords.new_zeros(())
        ai, aj, ak = self.angles[:,0], self.angles[:,1], self.angles[:,2]
        u = self.coords[ai] - self.coords[aj]
        v = self.coords[ak] - self.coords[aj]
        cos_t = (u*v).sum(dim=1) / (u.norm(dim=1)*v.norm(dim=1) + 1e-12)
        theta = torch.acos(cos_t.clamp(-1+1e-7, 1-1e-7))
        return (self.angle_k * (theta - self.angle_theta0)**2).sum()

    @staticmethod
    def _dihedral(p0, p1, p2, p3):
        b1 = p1 - p0
        b2 = p2 - p1
        b3 = p3 - p2
        n1  = torch.cross(b1, b2, dim=-1)
        n2  = torch.cross(b2, b3, dim=-1)
        b2n = b2 / (b2.norm(dim=-1, keepdim=True) + 1e-12)
        m1  = torch.cross(n1, b2n, dim=-1)
        x   = (n1*n2).sum(dim=-1)
        y   = (m1*n2).sum(dim=-1)
        return torch.atan2(y, x)

    def E_torsion(self):
        if self.torsions.shape[0] == 0:
            return self.coords.new_zeros(())
        i,j,k,l = (self.torsions[:,c] for c in range(4))
        phi = self._dihedral(self.coords[i], self.coords[j],
                             self.coords[k], self.coords[l])
        return (self.torsion_V *
                (1.0 + torch.cos(self.torsion_n*phi - self.torsion_delta))).sum()

    def E_improper(self):
        if self.impropers.shape[0] == 0:
            return self.coords.new_zeros(())
        i,j,k,l = (self.impropers[:,c] for c in range(4))
        psi = self._dihedral(self.coords[i], self.coords[j],
                             self.coords[k], self.coords[l])
        return (self.improper_k * (psi - self.improper_psi0)**2).sum()

    def E_nonbonded(self, soft_core=False):
        """
        LJ 12-6 + Coulomb.

        soft_core=True  — use a linearised LJ below r_switch so that clashing
                          atoms produce a finite (large but not Inf) energy and
                          a well-defined gradient.  Use this for the pre-minimisation
                          clash-removal phase; switch back to hard-core for
                          production minimisation.
        """
        N = self.coords.shape[0]
        if N < 2:
            return self.coords.new_zeros(())

        diff = self.coords[:, None, :] - self.coords[None, :, :]   # [N,N,3]
        r2   = (diff**2).sum(dim=-1) + self._r2_self               # [N,N]
        r    = r2.sqrt().clamp(min=1e-8)

        # ── Lennard-Jones ─────────────────────────────────────────────────────
        if soft_core:
            # Below r_switch: replace LJ with a linear cap E = a + b*r
            # matched in value and first derivative at r_switch.
            rs  = self._r_switch
            rs2 = rs * rs
            sr2_sw = self._sigma_ij**2 / rs2
            sr6_sw = sr2_sw**3
            # LJ value and slope at r_switch
            E_sw  = 4.0 * self._eps_ij * (sr6_sw**2 - sr6_sw)
            dE_sw = 4.0 * self._eps_ij * (-12.0 * sr6_sw**2 / rs
                                           + 6.0 * sr6_sw / rs)
            E_hard = 4.0 * self._eps_ij * ((self._sigma_ij**2/r2)**3)**2 \
                   - 4.0 * self._eps_ij * (self._sigma_ij**2/r2)**3
            # Linear cap for r < r_switch
            E_soft = E_sw + dE_sw * (r - rs)
            in_soft = (r < rs) & ~self._r2_self.bool()
            E_lj = torch.where(in_soft, E_soft, E_hard)
        else:
            sr2  = self._sigma_ij**2 / r2
            sr6  = sr2**3
            E_lj = 4.0 * self._eps_ij * (sr6**2 - sr6)

        # ── Coulomb ───────────────────────────────────────────────────────────
        E_coul = self.COUL_CONST * self._qi_qj / r

        # ── Exclusions (1-2, 1-3) ─────────────────────────────────────────────
        E_lj   = E_lj.masked_fill(self.excl_mask, 0.0)
        E_coul = E_coul.masked_fill(self.excl_mask, 0.0)

        # ── 1-4 scaling  FIX: (sc14-1)*E_14  not sc14*E_14 ───────────────────
        #   Starting from full matrices (factor 1.0 everywhere that's not excluded),
        #   correct 1-4 pairs from 1.0 to sc14_vdw / sc14_coul:
        E_lj   = E_lj   + (self.sc14_vdw  - 1.0) * E_lj.masked_fill(  ~self.mask14, 0.0)
        E_coul = E_coul + (self.sc14_coul - 1.0) * E_coul.masked_fill(~self.mask14, 0.0)

        # ── Sum upper triangle ────────────────────────────────────────────────
        return (E_lj + E_coul)[self._triu].sum()

    def energy_components(self, soft_core=False):
        Eb  = self.E_bond()
        Ea  = self.E_angle()
        Et  = self.E_torsion()
        Ei  = self.E_improper()
        Enb = self.E_nonbonded(soft_core=soft_core)
        return Eb, Ea, Et, Ei, Enb

    def total_energy(self, soft_core=False):
        return sum(self.energy_components(soft_core=soft_core))


# =============================================================================
# Minimizer — two-phase: clash removal → L-BFGS, with CPU→GPU hand-off
# =============================================================================

def minimize(mol,
             max_steps=500,
             tol_grad=1e-3,
             tol_change=1e-9,
             # Clash removal pre-phase
             clash_steps=50,
             clash_lr=0.01,
             clash_grad_clip=10.0,
             # GPU support
             gpu_device=None,          # e.g. torch.device('cuda:0')
             n_cpu_steps=5,            # run this many L-BFGS steps on CPU first
             verbose=True):
    """
    Two-phase geometry minimisation:

    Phase 0 — Steepest descent with gradient clipping + soft-core LJ.
               Resolves severe clashes that would give NaN/Inf in LJ.
               Runs entirely on CPU (cheap; just removes worst contacts).

    Phase 1 — L-BFGS with hard-core LJ.
               Runs first n_cpu_steps on CPU, then migrates to gpu_device
               if one is provided (and available).

    Parameters
    ----------
    mol            : MolecularFF
    max_steps      : int    max L-BFGS outer iterations
    tol_grad       : float  |F|_max convergence threshold [kcal/mol/Å]
    clash_steps    : int    steepest-descent steps for clash removal
    clash_lr       : float  step size for clash removal [Å per unit gradient]
    clash_grad_clip: float  gradient clipping norm for clash removal
    gpu_device     : torch.device or None
    n_cpu_steps    : int    L-BFGS steps to run on CPU before GPU transfer
    """

    # ── check for NaN in initial structure ───────────────────────────────────
    with torch.no_grad():
        E_init = mol.total_energy(soft_core=True)
    if not torch.isfinite(E_init):
        print("  ⚠  Initial energy is NaN/Inf even with soft-core LJ!")
        print("     Running nan_report to locate the problem term:")
        mol.nan_report()
        return mol.coords.detach().clone(), False

    mol.check_angle_units()

    # =========================================================================
    # Phase 0 — Soft-core steepest descent  (CPU, resolves clashes)
    # =========================================================================
    with torch.no_grad():
        E_hard = mol.total_energy(soft_core=False)
    need_phase0 = not torch.isfinite(E_hard) or E_hard.item() > 1e6

    if need_phase0:
        print(f"\n  ⚡ Phase 0: clash removal  ({clash_steps} SD steps, "
              f"soft-core LJ, grad clip={clash_grad_clip})")
        print(f"  {'Step':>5}  {'E_soft':>14}  {'|F|_max':>10}")
        for step in range(clash_steps):
            if mol.coords.grad is not None:
                mol.coords.grad.zero_()
            E = mol.total_energy(soft_core=True)
            E.backward()
            with torch.no_grad():
                g = mol.coords.grad
                gnorm = g.norm()
                if gnorm > clash_grad_clip:
                    g = g * (clash_grad_clip / gnorm)
                mol.coords.data -= clash_lr * g
                fmax = g.abs().max().item()
                if step % 10 == 0 or step < 3:
                    print(f"  {step:>5}  {E.item():>14.3f}  {fmax:>10.3e}")
            # Re-enable grad for next backward
            mol.coords.requires_grad_(True)

        with torch.no_grad():
            E_now = mol.total_energy(soft_core=False)
        if not torch.isfinite(E_now):
            print("  ⚠  Hard-core energy still NaN after clash removal. "
                  "Try more clash_steps or smaller clash_lr.")
        else:
            print(f"  ✓  Hard-core energy after clash removal: {E_now.item():.3f} kcal/mol")
    else:
        print(f"  ✓  No severe clashes detected (E_hard = {E_hard.item():.3f}). "
              f"Skipping Phase 0.")

    # =========================================================================
    # Phase 1 — L-BFGS with hard-core LJ  (CPU then optional GPU)
    # =========================================================================

    use_gpu = (gpu_device is not None and
               torch.cuda.is_available() and
               n_cpu_steps < max_steps)

    header = (f"{'Step':>6}  {'Device':>5}  {'E_tot':>13}  {'E_bond':>9}  "
              f"{'E_angle':>9}  {'E_tors':>9}  {'E_improp':>9}  "
              f"{'E_nb':>12}  {'|F|_max':>9}")
    if verbose:
        print(f"\n  Phase 1: L-BFGS  (tol={tol_grad:.0e}  max={max_steps})")
        print(header)
        print("─" * len(header))

    converged = False
    fmax = float('nan')
    on_gpu = False

    def _make_optimizer():
        return optim.LBFGS(
            [mol.coords],
            lr=1.0,
            max_iter=20,
            history_size=50,
            tolerance_grad=tol_grad,
            tolerance_change=tol_change,
            line_search_fn='strong_wolfe',
        )

    optimizer = _make_optimizer()

    for step in range(max_steps):

        # ── optional CPU→GPU migration ────────────────────────────────────────
        if use_gpu and not on_gpu and step >= n_cpu_steps:
            print(f"\n  → Migrating to {gpu_device} at step {step} …")
            mol.to_device(gpu_device)
            optimizer = _make_optimizer()   # L-BFGS history is device-specific
            on_gpu = True
            if verbose:
                print(header)
                print("─" * len(header))

        dev_label = 'GPU' if on_gpu else 'CPU'

        # ── closure ───────────────────────────────────────────────────────────
        def closure():
            optimizer.zero_grad()
            E = mol.total_energy(soft_core=False)
            if not torch.isfinite(E):
                # Fall back: return a large finite value so LBFGS backs off
                E = mol.coords.new_tensor(1e12)
            E.backward()
            return E

        try:
            optimizer.step(closure)
        except Exception as exc:
            print(f"  ⚠  L-BFGS step raised: {exc}")
            break

        # ── check for NaN mid-run ─────────────────────────────────────────────
        with torch.no_grad():
            Eb, Ea, Et, Ei, Enb = mol.energy_components()
            E_tot = (Eb + Ea + Et + Ei + Enb)

        if not torch.isfinite(E_tot):
            print(f"  ⚠  NaN/Inf at step {step}. Running nan_report …")
            mol.nan_report()
            break

        grad = mol.coords.grad
        fmax = grad.abs().max().item() if grad is not None else float('nan')

        if verbose and (step % 10 == 0 or step < 5):
            print(f"{step:>6}  {dev_label:>5}  {E_tot.item():>13.5f}  "
                  f"{Eb.item():>9.4f}  {Ea.item():>9.4f}  {Et.item():>9.4f}  "
                  f"{Ei.item():>9.4f}  {Enb.item():>12.5f}  {fmax:>9.2e}")

        if fmax < tol_grad:
            if verbose:
                print(f"\n  ✓ Converged at step {step}   |F|_max = {fmax:.2e} kcal/mol/Å")
            converged = True
            break

    if not converged and verbose:
        print(f"\n  ⚠  Not converged after {max_steps} steps.  |F|_max = {fmax:.2e} kcal/mol/Å")

    # Always return coords on CPU
    return mol.coords.detach().cpu().clone(), converged


# =============================================================================
# XYZ writer
# =============================================================================

def write_xyz(filename, coords, elements, comment=""):
    with open(filename, "w") as fh:
        fh.write(f"{len(elements)}\n{comment}\n")
        for el, (x, y, z) in zip(elements, coords.tolist()):
            fh.write(f"{el:2s}  {x:12.6f}  {y:12.6f}  {z:12.6f}\n")
    print(f"  → Written: {filename}")


# =============================================================================
# Self-contained butane test (no external deps)
# =============================================================================

def build_butane():
    coords = [
        [ 0.000,  0.000,  0.000],
        [ 1.540,  0.000,  0.000],
        [ 2.054,  1.431,  0.000],
        [ 3.594,  1.431,  0.000],
        [-0.380,  1.040,  0.000],
        [-0.380, -0.520,  0.900],
        [-0.380, -0.520, -0.900],
        [ 1.920, -0.520,  0.900],
        [ 1.920, -0.520, -0.900],
        [ 1.680,  1.950,  0.900],
        [ 1.680,  1.950, -0.900],
        [ 3.974,  0.390,  0.000],
        [ 3.974,  1.950,  0.900],
        [ 3.974,  1.950, -0.900],
    ]
    elements = ['C','C','C','C'] + ['H']*10
    charges  = [-0.18,-0.12,-0.12,-0.18] + [0.06]*10
    eps      = [0.066]*4 + [0.030]*10
    sigma    = [3.500]*4 + [2.500]*10

    bonds = [(0,1),(1,2),(2,3),(0,4),(0,5),(0,6),(1,7),(1,8),
             (2,9),(2,10),(3,11),(3,12),(3,13)]
    angles = [(0,1,2),(1,2,3),(4,0,1),(5,0,1),(6,0,1),(4,0,5),(4,0,6),
              (5,0,6),(0,1,7),(0,1,8),(7,1,2),(8,1,2),(7,1,8),(1,2,9),
              (1,2,10),(9,2,3),(10,2,3),(9,2,10),(2,3,11),(2,3,12),
              (2,3,13),(11,3,12),(11,3,13),(12,3,13)]

    torsions = []
    bond_set = {(i,j) for i,j in bonds} | {(j,i) for i,j in bonds}
    for ai,aj in bonds:
        for ak in range(14):
            if ak==ai or (aj,ak) not in bond_set: continue
            for al in range(14):
                if al==aj or al==ai or (ak,al) not in bond_set: continue
                torsions.append((ai,aj,ak,al))

    def atype(i,k):
        C={0,1,2,3}; nc=sum(x in C for x in (i,k))
        return (58.35,math.radians(112.7)) if nc==2 else \
               (37.50,math.radians(110.7)) if nc==1 else \
               (33.00,math.radians(107.8))

    ap = [atype(i,k) for i,j,k in angles]

    return dict(
        coords=coords, charges=charges, eps=eps, sigma=sigma,
        bonds=bonds, angles=angles, torsions=torsions, impropers=[],
        bond_k=[222.0]*3+[309.0]*10, bond_r0=[1.529]*3+[1.090]*10,
        angle_k=[p[0] for p in ap], angle_theta0=[p[1] for p in ap],
        torsion_V=[0.2]*len(torsions), torsion_n=[3.0]*len(torsions),
        torsion_delta=[0.0]*len(torsions),
        improper_k=[], improper_psi0=[],
        # Pass None → auto-build from topology
        excl_pairs=None, pairs_14=None,
    ), elements


if __name__ == '__main__':
    print("=" * 52)
    print("  PyTorch MM Minimizer v2 — Butane test")
    print("=" * 52)

    params, elements = build_butane()
    mol = MolecularFF(**params)

    with torch.no_grad():
        Eb,Ea,Et,Ei,Enb = mol.energy_components()
    print(f"\nInitial energies (kcal/mol):")
    for name, E in [('Bond',Eb),('Angle',Ea),('Torsion',Et),
                    ('Improper',Ei),('Non-bond',Enb)]:
        print(f"  {name:10s} {E.item():>12.5f}")
    print(f"  {'TOTAL':10s} {(Eb+Ea+Et+Ei+Enb).item():>12.5f}")

    mol.worst_angles(n=5)

    write_xyz("./butane_v2_initial.xyz",
              mol.coords.detach(), elements, "Butane v2 — initial")

    gpu = torch.device('cuda') if torch.cuda.is_available() else None
    coords_min, ok = minimize(mol, max_steps=200, tol_grad=1e-4,
                              clash_steps=30, gpu_device=gpu, n_cpu_steps=5)

    with torch.no_grad():
        Eb,Ea,Et,Ei,Enb = mol.energy_components()
    print(f"\nFinal energies (kcal/mol):")
    for name, E in [('Bond',Eb),('Angle',Ea),('Torsion',Et),
                    ('Improper',Ei),('Non-bond',Enb)]:
        print(f"  {name:10s} {E.item():>12.5f}")
    print(f"  {'TOTAL':10s} {(Eb+Ea+Et+Ei+Enb).item():>12.5f}")

    write_xyz("./butane_v2_minimized.xyz",
              coords_min, elements, "Butane v2 — minimized")
