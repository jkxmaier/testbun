"""
PyTorch Molecular Mechanics Minimizer
======================================
All energy terms are differentiable PyTorch tensor operations.
Forces (gradients) are obtained automatically via autograd — no manual
gradient coding required.

Units throughout:  kcal/mol · Å · elementary charge (e)

Energy terms:
  E_bond      — harmonic bond stretching      k(r - r0)²
  E_angle     — harmonic angle bending        k(θ - θ0)²
  E_torsion   — proper dihedral               V[1 + cos(n·φ - δ)]
  E_improper  — harmonic improper torsion     k(ψ - ψ0)²
  E_nonbonded — Lennard-Jones 12-6 + Coulomb  (with 1-2/1-3 exclusions, scaled 1-4)

Optimizer:
  torch.optim.LBFGS with strong-Wolfe line search
  (closure pattern: recompute energy each sub-step)

Test system:
  Butane (C4H10) with OPLS-AA-like parameters
"""

from jmConfig import *
from jmfhlog import *
from jmutils import *
from jmcore import *
from molfuncs import *

from _decomp import decomp_sys

import math
import numpy as np
import torch
import torch.optim as optim
torch.set_default_dtype(torch.float64)


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
# Force field / energy
# =============================================================================

class MolecularFF:
    """
    Molecular mechanics force field as a collection of differentiable
    PyTorch tensor operations.

    Parameters
    ----------
    coords       : (N, 3) array-like   — atom Cartesian coordinates [Å]
    charges      : (N,)                — partial charges [e]
    eps          : (N,)                — LJ ε per atom [kcal/mol]
    sigma        : (N,)                — LJ σ per atom [Å]  (arithmetic mean mixing)
    bonds        : (Z,  2) int         — atom index pairs
    angles       : (ZZ, 3) int         — atom index triples  (i–j–k, j = vertex)
    torsions     : (ZZZ,4) int         — atom index quads    (i–j–k–l)
    impropers    : (M,  4) int         — atom index quads    (i–j–k–l, j = centre)
    bond_k/r0    : (Z,)                — [kcal/mol/Å²], [Å]
    angle_k/θ0   : (ZZ,)               — [kcal/mol/rad²], [rad]
    torsion_V/n/δ: (ZZZ,)              — [kcal/mol], dimensionless, [rad]
    improper_k/ψ0: (M,)                — [kcal/mol/rad²], [rad]
    excl_pairs   : list of (i,j)       — 1-2 and 1-3 pairs (excluded from NB)
    pairs_14     : list of (i,j)       — 1-4 pairs (scaled NB)
    scale14_coul : float               — Coulomb scale for 1-4 (default 0.5)
    scale14_vdw  : float               — LJ scale for 1-4     (default 0.5)
    """

    COUL_CONST = 332.0636   # kcal·Å / (mol·e²)
    # Minimum r used for Coulomb during soft-core phase — prevents 1/r singularity
    # from overwhelming the soft-core LJ linearisation (tunable, units Å).
    SOFT_CORE_COULOMB_MIN_DIST = 0.3

    def __init__(self, nmol, r_switch=0.8):
        """
        """
        t0 = now()
        self.device = torch.device('cpu')

        self._ct = None
        self._atmL = []
        self.coords = []
        self.q      = []
        self.eps    = []
        self.sigma  = []
        self.bonds  = []
        self.angles    = []
        self.torsions  = []
        self.impropers = []

        self.bond_k        = []
        self.bond_r0       = []
        self.angle_k       = []
        self.angle_theta0  = []
        self.torsion_V     = []
        self.torsion_n     = []
        self.torsion_delta = []
        self.improper_k    = []
        self.improper_psi0 = []
        self.excl_mask     = []
        self.mask14        = []
        self.sc14_coul     = []
        self.sc14_vdw      = []

        self._triu     = []
        self._eps_ij   = []
        self._sigma_ij = []
        self._qi_qj    = []
        self._r2_self  = []

        if not mValid(nmol):
            print("⚠️  no valid structure input!")
            del self
            return

        # !!!! TODO: deal with None entries !!!

        _tsys = decomp_sys(nmol)
        if allfalse(_tsys) or len(_tsys) != 10:
            print("⚠️  system can not be minimized!")
            print("    Please ensure proper structure input!")
            del self
            return

        atmL = _tsys[0]
        if allfalse(atmL):
            print("⚠️  no valid structure input!")
            del self
            return

        N = len(atmL)
        coords = aPos(atmL)
        assert(len(_tsys[1]) == len(atmL))
        charges, sigma, eps = tr(_tsys[1])

        bonds = _tsys[2]
        assert(l_len(bonds) == 2 and len(bonds))
        assert(len(_tsys[3]) == len(bonds) and l_len(_tsys[3]) == 2)
        bond_k, bond_r0 = tr(_tsys[3])
        ubnds = ['%d%d' % tuple(sorted(get(atmL, a))) for a in bonds]
        msk = m_uniq(ubnds)
        bonds, bond_k, bond_r0 = mgetE((bonds, bond_k, bond_r0), msk)


        angles = _tsys[4]
        assert(l_len(angles) == 3 and len(angles))
        assert(len(_tsys[5]) == len(angles) and l_len(_tsys[5]) == 2)
        angle_k, angle_theta0  = tr(_tsys[5])
        uangs = []
        for ang in angles:
            a, b, c = get(atmL, ang)
            v = (a,b,c) if sum((a,b,c)) < sum((c,b,a)) else (c,b,a)
            uangs.append('%d%d%d' % v)
        msk = m_uniq(uangs)
        angles, angle_k, angle_theta0 = mgetE((angles, angle_k, angle_theta0), msk)

        torsions = _tsys[6]
        assert(l_len(torsions) == 4 and len(torsions))
        assert(len(_tsys[7]) == len(torsions) and l_len(_tsys[7]) == 3)
        torsion_V, torsion_delta, torsion_n = tr(_tsys[7])  #
        torsion_n = [abs(a) for a in torsion_n]
        utors = []
        for tor in torsions:
            a, b, c, d = get(atmL, tor)
            v = (a,b,c,d) if sum((a,b,c,d)) < sum((d, c,b,a)) else (d,c,b,a)
            utors.append('%d%d%d%d' % v)
        msk = m_uniq(utors)
        torsions, torsion_n, torsion_delta, torsion_V = mgetE(
                        (torsions, torsion_n, torsion_delta, torsion_V ), msk)

        print("@@post uniquies: ", lenE(bonds, angles, torsions))

        # do impropers later
        impropers = []
        improper_k, improper_psi0 = ([], [])

        scale14_coul =  _tsys[-2]
        scale14_vdw = _tsys[-1]

        self._ct = nmol
        self._atmL = atmL

        # ── optimizable coordinates ─────────────────────────────────────────
        self.coords = torch.tensor(coords, dtype=torch.float64,
                                   requires_grad=True)

        # ── fixed atom properties ────────────────────────────────────────────
        self.q     = torch.tensor(charges, dtype=torch.float64)
        self.eps   = torch.tensor(eps,     dtype=torch.float64)
        self.sigma = torch.tensor(sigma,   dtype=torch.float64)

        # ── topology ─────────────────────────────────────────────────────────
        def _t(lst, cols):
            return (torch.tensor(lst, dtype=torch.long)
                    if len(lst) > 0 else torch.zeros((0, cols), dtype=torch.long))

        self.bonds     = _t(bonds,     2)
        self.angles    = _t(angles,    3)
        self.torsions  = _t(torsions,  4)
        self.impropers = _t(impropers, 4)

        # ── bonded FF parameters ─────────────────────────────────────────────
        self.bond_k       = torch.tensor(bond_k,        dtype=torch.float64)
        self.bond_r0      = torch.tensor(bond_r0,       dtype=torch.float64)
        self.angle_k      = torch.tensor(angle_k,       dtype=torch.float64)
        self.angle_theta0 = torch.tensor(angle_theta0,  dtype=torch.float64)
        self.torsion_V     = torch.tensor(torsion_V,    dtype=torch.float64)
        self.torsion_n     = torch.tensor(torsion_n,    dtype=torch.float64)
        self.torsion_delta = torch.tensor(torsion_delta,dtype=torch.float64)
        self.improper_k    = torch.tensor(improper_k,   dtype=torch.float64)
        self.improper_psi0 = torch.tensor(improper_psi0,dtype=torch.float64)

        # ── non-bonded exclusion / scaling masks ─────────────────────────────

        excl_pairs, pairs_14 = build_excl_and_14(
            N,
            bonds if len(bonds) else [],
            angles if len(angles) else [],
            torsions if len(torsions) else [],
        )
        print(f"  [FF] Auto-built {len(excl_pairs)} exclusion pairs, "
              f"{len(pairs_14)} 1-4 pairs from topology.")
        # Removed redundant second build_excl_and_14 call that overwrote pairs_14
        # and printed a confusing duplicate log line.

        # ── exclusion / scaling masks ─────────────────────────────────────────
        excl_mask = torch.eye(N, dtype=torch.bool)
        for i, j in excl_pairs:
            excl_mask[i, j] = excl_mask[j, i] = True

        mask14 = torch.zeros(N, N, dtype=torch.bool)
        for i, j in pairs_14:
            mask14[i, j] = mask14[j, i] = True

        self.excl_mask    = excl_mask
        self.mask14       = mask14
        self.sc14_coul    = float(scale14_coul)
        self.sc14_vdw     = float(scale14_vdw)

        # Pre-build upper-triangle index (avoids double-counting in NB sum)
        self._triu = torch.ones(N, N, dtype=torch.bool).triu(diagonal=1)

        # Lorentz–Berthelot mixing (precomputed, fixed)
        self._eps_ij   = torch.sqrt(self.eps[:, None] * self.eps[None, :])
        self._sigma_ij = 0.5 * (self.sigma[:, None] + self.sigma[None, :])
        self._qi_qj    = self.q[:, None] * self.q[None, :]
        self._r2_self  = torch.eye(N, dtype=torch.float64) * 1e10  # avoids /0

        self._r_switch = float(r_switch)  # must be float for soft-core LJ cutoff arithmetic
        self._precompute_nb_tables()
        print("✓ done MolFF initialization - ", timestr(t0))

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

    # ──────────────────────────────────────────────────────────────────────────
    # Bond stretching   E = k (r − r0)²
    # ──────────────────────────────────────────────────────────────────────────
    def E_bond(self):
        if self.bonds.shape[0] == 0:
            return self.coords.new_zeros(())
        i, j = self.bonds[:, 0], self.bonds[:, 1]
        r = (self.coords[i] - self.coords[j]).norm(dim=1)
        #print("@@@ distances: ", len(r), len(i), len(j))
        return (self.bond_k * (r - self.bond_r0) ** 2).sum()

    # ──────────────────────────────────────────────────────────────────────────
    # Angle bending   E = k (θ − θ0)²
    # ──────────────────────────────────────────────────────────────────────────
    def E_angle(self):
        if self.angles.shape[0] == 0:
            return self.coords.new_zeros(())
        ai, aj, ak = self.angles[:, 0], self.angles[:, 1], self.angles[:, 2]
        u = self.coords[ai] - self.coords[aj]
        v = self.coords[ak] - self.coords[aj]
        cos_t = (u * v).sum(dim=1) / (u.norm(dim=1) * v.norm(dim=1) + 1e-12)
        theta = torch.acos(cos_t.clamp(-1 + 1e-7, 1 - 1e-7))
        return (self.angle_k * (theta - self.angle_theta0) ** 2).sum()

    # ──────────────────────────────────────────────────────────────────────────
    # Dihedral helper  →  φ via atan2 (stable at 0° and 180°)
    # Uses the Praxedes / IUPAC convention returning φ ∈ (−π, π]
    # ──────────────────────────────────────────────────────────────────────────
    @staticmethod
    def _dihedral(p0, p1, p2, p3):
        b1 = p1 - p0          # bond vectors along the chain
        b2 = p2 - p1
        b3 = p3 - p2
        # Normal vectors to the two planes
        n1 = torch.cross(b1, b2, dim=-1)
        n2 = torch.cross(b2, b3, dim=-1)
        # Normalise n1, n2 — prevents NaN gradient when atoms are nearly collinear
        n1  = n1 / n1.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        n2  = n2 / n2.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        b2n = b2 / (b2.norm(dim=-1, keepdim=True) + 1e-12)
        m1  = torch.cross(n1, b2n, dim=-1)
        x   = (n1 * n2).sum(dim=-1)
        y   = (m1 * n2).sum(dim=-1)
        return torch.atan2(y, x)

    # ──────────────────────────────────────────────────────────────────────────
    # Proper torsion   E = V [1 + cos(n·φ − δ)]
    # ──────────────────────────────────────────────────────────────────────────
    def E_torsion(self):
        if self.torsions.shape[0] == 0:
            return self.coords.new_zeros(())
        i, j, k, l = (self.torsions[:, c] for c in range(4))
        phi = self._dihedral(self.coords[i], self.coords[j],
                             self.coords[k], self.coords[l])
        return (self.torsion_V *
                (1.0 + torch.cos(self.torsion_n * phi - self.torsion_delta))).sum()

    # ──────────────────────────────────────────────────────────────────────────
    # Improper torsion   E = k (ψ − ψ0)²   (harmonic restraint)
    # ──────────────────────────────────────────────────────────────────────────
    def E_improper(self):
        if self.impropers.shape[0] == 0:
            return self.coords.new_zeros(())
        i, j, k, l = (self.impropers[:, c] for c in range(4))
        psi = self._dihedral(self.coords[i], self.coords[j],
                             self.coords[k], self.coords[l])
        return (self.improper_k * (psi - self.improper_psi0) ** 2).sum()

    # ──────────────────────────────────────────────────────────────────────────
    # Non-bonded:  LJ 12-6  +  Coulomb
    #   Full N×N matrix, then zero excluded pairs, scale 1-4 pairs.
    # ──────────────────────────────────────────────────────────────────────────
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
        # During soft-core (clash-removal) phase, clamp r to ≥ 0.3 Å so that
        # extremely close atom pairs do not generate astronomical Coulomb
        # gradients that overwhelm the soft-core LJ linearisation.
        r_coul = r.clamp(min=self.SOFT_CORE_COULOMB_MIN_DIST) if soft_core else r
        E_coul = self.COUL_CONST * self._qi_qj / r_coul

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


    def E_nonbonded2(self, soft_core=False):
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
        #r    = r2.sqrt()
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
    # ──────────────────────────────────────────────────────────────────────────
    # Total energy  (returns individual components + sum)
    # ──────────────────────────────────────────────────────────────────────────

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
# Minimizer
# =============================================================================

def minimize(mol,
             adam_steps=300, adam_lr=1e-3,
             max_steps=500,
             tol_grad=1e-3,
             tol_change=1e-9,
             # Clash removal pre-phase
             clash_steps=50,
             clash_lr=0.01,
             clash_grad_clip=10.0,
             clash_min_r=0.7,           # trigger Phase 0 when any non-excl pair < this [Å]
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
    clash_min_r    : float  minimum non-excluded interatomic distance [Å] below
                            which Phase 0 is triggered (default 0.7 Å)
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

    # Also trigger Phase 0 when the minimum non-excluded interatomic distance
    # is below 0.7 Å — a finite total energy can still hide catastrophic close
    # contacts whose 1/r² Coulomb gradient will immediately produce NaN.
    if not need_phase0:
        with torch.no_grad():
            diff = mol.coords[:, None, :] - mol.coords[None, :, :]  # [N,N,3]
            r_all = (diff.square().sum(-1) + mol._r2_self).sqrt()
            # Exclude self-interactions and 1-2/1-3 pairs (same as NB potential)
            exclude = mol.excl_mask | ~mol._triu
            r_nb = r_all.masked_fill(exclude, float('inf'))
            min_r = r_nb.min().item()
        if min_r < clash_min_r:
            need_phase0 = True
            print(f"  ⚠  Close contact detected (min r = {min_r:.3f} Å < {clash_min_r:.2f} Å). "
                  f"Running Phase 0 clash removal.")
        else:
            print(f"  ✓  No severe clashes detected (E_hard = {E_hard.item():.3f}, "
                  f"min r = {min_r:.3f} Å). Skipping Phase 0.")

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
            E_now = mol.total_energy(soft_core=True)
        if not torch.isfinite(E_now):
            print("  ⚠  Soft-core energy still NaN after clash removal. "
                  "Try more clash_steps or smaller clash_lr.")
        else:
            print(f"  ✓  Soft-core energy after clash removal: {E_now.item():.3f} kcal/mol")

    # =========================================================================
    # Phase 1 — L-BFGS with hard-core LJ  (CPU then optional GPU)
    # =========================================================================


    # ── Phase 1: Adam  (robust for ill-conditioned / badly placed H atoms) ──
    print(f"\n  Phase 1: Adam  ({adam_steps} steps, lr={adam_lr})")
    print(f"  {'Step':>6}  {'E_tot':>13}  {'E_bond':>9}  {'E_angle':>9}  {'|F|_max':>9}")

    adam = optim.Adam([mol.coords], lr=adam_lr)
    for step in range(adam_steps):
        adam.zero_grad()
        E = mol.total_energy()
        if not torch.isfinite(E):
            print(f"  ⚠  NaN at Adam step {step}"); break
        E.backward()

        # gradient clipping: prevents Adam exploding on the worst clashes
        torch.nn.utils.clip_grad_norm_([mol.coords], max_norm=50.0)
        adam.step()

        if verbose and (step % 50 == 0 or step < 3):
            with torch.no_grad():
                Eb, Ea, Et, Ei, Enb = mol.energy_components()
            fmax = mol.coords.grad.abs().max().item()
            print(f"  {step:>6}  {(Eb+Ea+Et+Ei+Enb).item():>13.4f}  "
                  f"{Eb.item():>9.4f}  {Ea.item():>9.4f}  {fmax:>9.2e}")

        if mol.coords.grad.abs().max().item() < tol_grad * 10:
            print(f"  ✓ Adam pre-converged at step {step}")
            break


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
    #mol.coords = mol.coords.detach().requires_grad_(True)
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
                # Fall back to a large value that still depends on mol.coords so
                # it has a grad_fn; mol.coords.new_tensor(1e12) is a constant
                # (no grad_fn) and calling backward() on it would raise the
                # "does not require grad" error seen in practice.
                E = mol.coords.sum() * 0.0 + 1e12
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
# XYZ writer (for visualisation in e.g. Avogadro / VMD / MOLDEN)
# =============================================================================

def write_xyz(filename, coords, elements, comment=""):
    with open(filename, "w") as fh:
        fh.write(f"{len(elements)}\n{comment}\n")
        for el, (x, y, z) in zip(elements, coords.tolist()):
            fh.write(f"{el:2s}  {x:12.6f}  {y:12.6f}  {z:12.6f}\n")
    print(f"  → Coordinates written to {filename}")



# =============================================================================
# Testing functions
# =============================================================================
def test_butane_mm():
    banner = "  PyTorch MM Minimizer — Butane (OPLS-AA-like)  "
    print("=" * len(banner))
    print(banner)
    print("=" * len(banner))

    params, elements = build_butane()
    mol = MolecularFF(**params)

    # ── initial energy breakdown ──────────────────────────────────────────────
    with torch.no_grad():
        Eb, Ea, Et, Ei, Enb = mol.energy_components()
    print(f"\nInitial energies (kcal/mol):")
    print(f"  Bond      {Eb.item():>12.5f}")
    print(f"  Angle     {Ea.item():>12.5f}")
    print(f"  Torsion   {Et.item():>12.5f}")
    print(f"  Improper  {Ei.item():>12.5f}")
    print(f"  Non-bond  {Enb.item():>12.5f}")
    print(f"  ──────────────────────────")
    print(f"  TOTAL     {(Eb+Ea+Et+Ei+Enb).item():>12.5f}\n")

    write_xyz("./butane_initial.xyz",
              mol.coords.detach(), elements, comment="Butane — initial geometry")

    # ── minimise ──────────────────────────────────────────────────────────────
    print("L-BFGS minimisation  (tolerance |F|_max < 1e-5 kcal/mol/Å)\n")
    coords_min, converged = minimize(mol, max_steps=400, tol_grad=1e-5)

    # ── final energy breakdown ────────────────────────────────────────────────
    with torch.no_grad():
        Eb, Ea, Et, Ei, Enb = mol.energy_components()
    print(f"\nFinal energies (kcal/mol):")
    print(f"  Bond      {Eb.item():>12.5f}")
    print(f"  Angle     {Ea.item():>12.5f}")
    print(f"  Torsion   {Et.item():>12.5f}")
    print(f"  Improper  {Ei.item():>12.5f}")
    print(f"  Non-bond  {Enb.item():>12.5f}")
    print(f"  ──────────────────────────")
    print(f"  TOTAL     {(Eb+Ea+Et+Ei+Enb).item():>12.5f}\n")

    # ── final coordinates ─────────────────────────────────────────────────────
    atom_names = ['C1','C2','C3','C4',
                  'H4','H5','H6','H7','H8','H9','H10','H11','H12','H13']
    print("Final Cartesian coordinates (Å):")
    print(f"  {'Atom':4s}  {'x':>10}  {'y':>10}  {'z':>10}")
    print("  " + "─"*38)
    for name, xyz in zip(atom_names, coords_min.tolist()):
        print(f"  {name:4s}  {xyz[0]:>10.5f}  {xyz[1]:>10.5f}  {xyz[2]:>10.5f}")

    write_xyz("./butane_minimized.xyz",
              coords_min, elements, comment="Butane — minimized geometry")
    print()


def _torsion_energy(phi_deg, terms):
    phi_rad = np.deg2rad(phi_deg)
    E = 0.0
    for pk, phase, pn in terms:
        n = abs(pn)
        arg = n * phi_rad - np.deg2rad(phase)
        E += pk * (1 + np.cos(arg))          # IDIVF = 1 in your file
    return E


def test_dihedral():
    """
    """
    # Your exact CT-CT-C-N terms
    terms = [
        (-0.01184, 0.00000, -6.0),
        (-0.02974, 0.00000, -5.0),
        (-0.11535, 0.00000, -4.0),
        (0.10642, 0.81093, -3.0),
        (-0.70075, -10.51754, -2.0),
        (0.19389, 12.35042, 1.0)
    ]

    print(_torsion_energy(0, terms))      # → -1.10746
    print(_torsion_energy(180, terms))    # → -1.63961


def test_simple_mm():
    """
    """
    from molgeom import mPrepStructure

    fn = '/mnt/DATA/jmoldev/current/FF/data/1crn.jmz'
    #fn = '/mnt/DATA/jmoldev/current/FF/data/6a5j.pdb'
    #fn = '/mnt/DATA/jmoldev/current/FF/todel.pdb'
    ClearSYS()
    LoadStructure(fn)
    amol, mp = mPrepStructure(CurrSYS(), pH=7.1, verb=True)
    print("@@molprobity scores: ", mp)
    mmobj = MolecularFF(amol)
    assert(isinstance(mmobj, MolecularFF))

    # ── initial energy breakdown ──────────────────────────────────────────────
    with torch.no_grad():
        Eb, Ea, Et, Ei, Enb = mmobj.energy_components()
    print(f"\nInitial energies (kcal/mol):")
    print(f"  Bond      {Eb.item():>12.5f}")
    print(f"  Angle     {Ea.item():>12.5f}")
    print(f"  Torsion   {Et.item():>12.5f}")
    print(f"  Improper  {Ei.item():>12.5f}")
    print(f"  Non-bond  {Enb.item():>12.5f}")
    print(f"  ──────────────────────────")
    print(f"  TOTAL     {(Eb+Ea+Et+Ei+Enb).item():>12.5f}\n")

    # ── minimise ──────────────────────────────────────────────────────────────
    print("L-BFGS minimisation  (tolerance |F|_max < 1e-5 kcal/mol/Å)\n")
    coords_min, converged = minimize(mmobj, max_steps=200,
            tol_grad=1e-3, n_cpu_steps=0, gpu_device=None)  #torch.device('cuda:0'))

    # ── final energy breakdown ────────────────────────────────────────────────
    with torch.no_grad():
        Eb, Ea, Et, Ei, Enb = mmobj.energy_components()
    print(f"\nFinal energies (kcal/mol):")
    print(f"  Bond      {Eb.item():>12.5f}")
    print(f"  Angle     {Ea.item():>12.5f}")
    print(f"  Torsion   {Et.item():>12.5f}")
    print(f"  Improper  {Ei.item():>12.5f}")
    print(f"  Non-bond  {Enb.item():>12.5f}")
    print(f"  ──────────────────────────")
    print(f"  TOTAL     {(Eb+Ea+Et+Ei+Enb).item():>12.5f}\n")

    print("CUDA: ", torch.cuda.is_available())

def test():
    """
    """
    test_simple_mm()


# =============================================================================
# Main
# =============================================================================

if __name__ == '__main__':
    test()
