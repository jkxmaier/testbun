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

import math
import torch
import torch.optim as optim

torch.set_default_dtype(torch.float64)


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

    def __init__(self,
                 coords, charges, eps, sigma,
                 bonds, angles, torsions, impropers,
                 bond_k, bond_r0,
                 angle_k, angle_theta0,
                 torsion_V, torsion_n, torsion_delta,
                 improper_k, improper_psi0,
                 excl_pairs=None,
                 pairs_14=None,
                 scale14_coul=0.5, scale14_vdw=0.5):

        # ── optimizable coordinates ─────────────────────────────────────────
        self.coords = torch.tensor(coords, dtype=torch.float64,
                                   requires_grad=True)
        N = self.coords.shape[0]

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
        self.bond_k       = torch.tensor(bond_k,       dtype=torch.float64)
        self.bond_r0      = torch.tensor(bond_r0,      dtype=torch.float64)
        self.angle_k      = torch.tensor(angle_k,      dtype=torch.float64)
        self.angle_theta0 = torch.tensor(angle_theta0, dtype=torch.float64)
        self.torsion_V     = torch.tensor(torsion_V,    dtype=torch.float64)
        self.torsion_n     = torch.tensor(torsion_n,    dtype=torch.float64)
        self.torsion_delta = torch.tensor(torsion_delta,dtype=torch.float64)
        self.improper_k    = torch.tensor(improper_k,   dtype=torch.float64)
        self.improper_psi0 = torch.tensor(improper_psi0,dtype=torch.float64)

        # ── non-bonded exclusion / scaling masks ─────────────────────────────
        # excl_mask[i,j] = True  →  zero this pair entirely
        excl_mask = torch.eye(N, dtype=torch.bool)   # always exclude self
        if excl_pairs:
            for i, j in excl_pairs:
                excl_mask[i, j] = excl_mask[j, i] = True

        # mask14[i,j] = True  →  apply scale factor instead of 1.0
        mask14 = torch.zeros(N, N, dtype=torch.bool)
        if pairs_14:
            for i, j in pairs_14:
                mask14[i, j] = mask14[j, i] = True

        self.excl_mask    = excl_mask
        self.mask14       = mask14
        self.sc14_coul    = scale14_coul
        self.sc14_vdw     = scale14_vdw

        # Pre-build upper-triangle index (avoids double-counting in NB sum)
        self._triu = torch.ones(N, N, dtype=torch.bool).triu(diagonal=1)

        # Lorentz–Berthelot mixing (precomputed, fixed)
        self._eps_ij   = torch.sqrt(self.eps[:, None] * self.eps[None, :])
        self._sigma_ij = 0.5 * (self.sigma[:, None] + self.sigma[None, :])
        self._qi_qj    = self.q[:, None] * self.q[None, :]
        self._r2_self  = torch.eye(N, dtype=torch.float64) * 1e10  # avoids /0

    # ──────────────────────────────────────────────────────────────────────────
    # Bond stretching   E = k (r − r0)²
    # ──────────────────────────────────────────────────────────────────────────
    def E_bond(self):
        if self.bonds.shape[0] == 0:
            return self.coords.new_zeros(())
        i, j = self.bonds[:, 0], self.bonds[:, 1]
        r = (self.coords[i] - self.coords[j]).norm(dim=1)
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
    def E_nonbonded(self):
        N = self.coords.shape[0]
        if N < 2:
            return self.coords.new_zeros(())

        diff = self.coords[:, None, :] - self.coords[None, :, :]   # [N,N,3]
        r2   = (diff ** 2).sum(dim=-1) + self._r2_self             # [N,N]

        # ── Lennard-Jones ─────────────────────────────────────────────────────
        sr2  = self._sigma_ij ** 2 / r2
        sr6  = sr2 ** 3
        E_lj = 4.0 * self._eps_ij * (sr6 ** 2 - sr6)              # [N,N]

        # ── Coulomb ───────────────────────────────────────────────────────────
        r      = r2.sqrt()
        E_coul = self.COUL_CONST * self._qi_qj / r                 # [N,N]

        # ── Apply exclusions (1-2, 1-3) ───────────────────────────────────────
        E_lj   = E_lj.masked_fill(self.excl_mask, 0.0)
        E_coul = E_coul.masked_fill(self.excl_mask, 0.0)

        # ── Apply 1-4 scaling ─────────────────────────────────────────────────
        #   Full matrix already has factor 1.0 for all non-excluded pairs.
        #   For 1-4 pairs we want factor sc14, so add (sc14 − 1) correction.
        E_lj   = E_lj   + (self.sc14_vdw  - 1.0) * E_lj.masked_fill(  ~self.mask14, 0.0)
        E_coul = E_coul + (self.sc14_coul - 1.0) * E_coul.masked_fill(~self.mask14, 0.0)

        # ── Sum upper triangle to avoid double-counting ───────────────────────
        E_nb = (E_lj + E_coul)[self._triu].sum()
        return E_nb

    # ──────────────────────────────────────────────────────────────────────────
    # Total energy  (returns individual components + sum)
    # ──────────────────────────────────────────────────────────────────────────
    def energy_components(self):
        Eb  = self.E_bond()
        Ea  = self.E_angle()
        Et  = self.E_torsion()
        Ei  = self.E_improper()
        Enb = self.E_nonbonded()
        return Eb, Ea, Et, Ei, Enb

    def total_energy(self):
        return sum(self.energy_components())


# =============================================================================
# Minimizer
# =============================================================================

def minimize(mol, max_steps=500, tol_grad=1e-5, tol_change=1e-9, verbose=True):
    """
    Run L-BFGS geometry optimization with strong-Wolfe line search.

    Returns
    -------
    coords_min : (N, 3) tensor  — minimized coordinates
    converged  : bool
    """
    optimizer = optim.LBFGS(
        [mol.coords],
        lr=1.0,
        max_iter=20,            # inner CG steps per outer step
        history_size=100,
        tolerance_grad=tol_grad,
        tolerance_change=tol_change,
        line_search_fn='strong_wolfe',
    )

    header = (f"{'Step':>6}  {'E_tot':>13}  {'E_bond':>9}  {'E_angle':>9}  "
              f"{'E_tors':>9}  {'E_improp':>9}  {'E_nb':>12}  {'|F|_max':>9}")
    if verbose:
        print(header)
        print("─" * len(header))

    converged = False
    fmax = float('nan')

    for step in range(max_steps):

        def closure():
            optimizer.zero_grad()
            E = mol.total_energy()
            E.backward()
            return E

        optimizer.step(closure)

        # ── diagnostic printout ───────────────────────────────────────────────
        with torch.no_grad():
            Eb, Ea, Et, Ei, Enb = mol.energy_components()
            E_tot = (Eb + Ea + Et + Ei + Enb).item()

        grad = mol.coords.grad
        fmax = grad.abs().max().item() if grad is not None else float('nan')

        if verbose and (step % 10 == 0 or step < 5):
            print(f"{step:>6}  {E_tot:>13.5f}  {Eb.item():>9.4f}  {Ea.item():>9.4f}  "
                  f"{Et.item():>9.4f}  {Ei.item():>9.4f}  {Enb.item():>12.5f}  {fmax:>9.2e}")

        if fmax < tol_grad:
            if verbose:
                print(f"\n  ✓ Converged at step {step}   |F|_max = {fmax:.2e} kcal/mol/Å")
            converged = True
            break

    if not converged and verbose:
        print(f"\n  ⚠  Not converged after {max_steps} steps.  |F|_max = {fmax:.2e} kcal/mol/Å")

    return mol.coords.detach().clone(), converged


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
# Test molecule: butane C1-C2-C3-C4  (OPLS-AA-like parameters)
# =============================================================================

def build_butane():
    """
    Butane (C4H10):  C1–C2–C3–C4 skeleton, H's numbered sequentially.

    Atom order:
      0  C1   1  C2   2  C3   3  C4
      4  H (C1)  5  H (C1)  6  H (C1)
      7  H (C2)  8  H (C2)
      9  H (C3)  10 H (C3)
      11 H (C4) 12 H (C4)  13 H (C4)
    """
    # ── initial coordinates (slightly distorted from ideal) ──────────────────
    coords = [
        [ 0.000,  0.000,  0.000],  # C1   0
        [ 1.540,  0.000,  0.000],  # C2   1
        [ 2.054,  1.431,  0.000],  # C3   2
        [ 3.594,  1.431,  0.000],  # C4   3
        [-0.380,  1.040,  0.000],  # H/C1 4
        [-0.380, -0.520,  0.900],  # H/C1 5
        [-0.380, -0.520, -0.900],  # H/C1 6
        [ 1.920, -0.520,  0.900],  # H/C2 7
        [ 1.920, -0.520, -0.900],  # H/C2 8
        [ 1.680,  1.950,  0.900],  # H/C3 9
        [ 1.680,  1.950, -0.900],  # H/C3 10
        [ 3.974,  0.390,  0.000],  # H/C4 11
        [ 3.974,  1.950,  0.900],  # H/C4 12
        [ 3.974,  1.950, -0.900],  # H/C4 13
    ]

    elements = ['C','C','C','C'] + ['H']*10

    # ── OPLS-AA partial charges ───────────────────────────────────────────────
    charges = [-0.18, -0.12, -0.12, -0.18,
                0.06,  0.06,  0.06,
                0.06,  0.06,
                0.06,  0.06,
                0.06,  0.06,  0.06]

    # ── LJ parameters: ε (kcal/mol),  σ (Å) ─────────────────────────────────
    #   CT (sp3 C):  ε=0.066,  σ=3.500
    #   HC:          ε=0.030,  σ=2.500
    eps   = [0.066]*4 + [0.030]*10
    sigma = [3.500]*4 + [2.500]*10

    # ── Bonds ─────────────────────────────────────────────────────────────────
    CC_bonds = [(0,1),(1,2),(2,3)]
    CH_bonds = [(0,4),(0,5),(0,6),(1,7),(1,8),(2,9),(2,10),(3,11),(3,12),(3,13)]
    bonds    = CC_bonds + CH_bonds
    #   k [kcal/mol/Å²],  r0 [Å]   (OPLS-AA)
    bond_k  = [222.0]*3 + [309.0]*10
    bond_r0 = [1.529]*3 + [1.090]*10

    # ── Angles ────────────────────────────────────────────────────────────────
    angles = [
        # C–C–C
        (0,1,2),(1,2,3),
        # H–C1–C2
        (4,0,1),(5,0,1),(6,0,1),
        # H–C1–H
        (4,0,5),(4,0,6),(5,0,6),
        # H–C2–C and C–C2–H
        (0,1,7),(0,1,8),(7,1,2),(8,1,2),
        # H–C2–H
        (7,1,8),
        # C–C3–H and H–C3–C
        (1,2,9),(1,2,10),(9,2,3),(10,2,3),
        # H–C3–H
        (9,2,10),
        # C–C4–H
        (2,3,11),(2,3,12),(2,3,13),
        # H–C4–H
        (11,3,12),(11,3,13),(12,3,13),
    ]

    #   OPLS-AA:  CCC  k=58.35, θ0=112.7°
    #             CCH  k=37.50, θ0=110.7°
    #             HCH  k=33.00, θ0=107.8°
    def angle_type(i, j, k):
        C_idx = set(range(4))
        nc = sum(x in C_idx for x in (i, k))
        if nc == 2:
            return (58.35, math.radians(112.7))   # CCC
        elif nc == 1:
            return (37.50, math.radians(110.7))   # CCH
        else:
            return (33.00, math.radians(107.8))   # HCH

    angle_params = [angle_type(i, j, k) for (i, j, k) in angles]
    angle_k      = [p[0] for p in angle_params]
    angle_theta0 = [p[1] for p in angle_params]

    # ── Proper torsions ───────────────────────────────────────────────────────
    #   Generated automatically from the connectivity.
    #   OPLS-AA 3-fold approximation:  V3=0.20 kcal/mol, n=3, δ=0
    #   (a full OPLS expansion uses V1/V2/V3 terms; this is the dominant one)
    torsions = []
    bond_set = {(i, j) for i, j in bonds} | {(j, i) for i, j in bonds}

    def connected(a, b):
        return (a, b) in bond_set

    for (ai, aj) in bonds:
        for ak in range(14):
            if ak == ai or not connected(aj, ak):
                continue
            for al in range(14):
                if al == aj or al == ai or not connected(ak, al):
                    continue
                torsions.append((ai, aj, ak, al))

    n_tors         = len(torsions)
    torsion_V      = [0.200] * n_tors
    torsion_n      = [3.0]   * n_tors
    torsion_delta  = [0.0]   * n_tors

    # ── Improper torsions ─────────────────────────────────────────────────────
    #   Butane has no planar centres — list is empty
    impropers     = []
    improper_k    = []
    improper_psi0 = []

    # ── Non-bonded exclusion lists ─────────────────────────────────────────────
    #   1-2 pairs (bonds) + 1-3 pairs (angles) are fully excluded
    excl_pairs = set()
    for i, j in bonds:
        excl_pairs.add((i, j))
    for i, j, k in angles:
        excl_pairs.add((min(i, k), max(i, k)))

    #   1-4 pairs (from torsions, not already excluded)
    pairs_14 = set()
    for i, j, k, l in torsions:
        pair = (min(i, l), max(i, l))
        if pair not in excl_pairs:
            pairs_14.add(pair)

    return dict(
        coords=coords, charges=charges, eps=eps, sigma=sigma,
        bonds=bonds, angles=angles, torsions=torsions, impropers=impropers,
        bond_k=bond_k, bond_r0=bond_r0,
        angle_k=angle_k, angle_theta0=angle_theta0,
        torsion_V=torsion_V, torsion_n=torsion_n, torsion_delta=torsion_delta,
        improper_k=improper_k, improper_psi0=improper_psi0,
        excl_pairs=list(excl_pairs),
        pairs_14=list(pairs_14),
    ), elements


# =============================================================================
# Main
# =============================================================================

if __name__ == '__main__':
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
