# Machine-checked proofs (Lean 4 + Mathlib)

This folder is a [Lean 4](https://lean-lang.org) project. It proves, with the
[Mathlib](https://github.com/leanprover-community/mathlib4) library, seven groups of
results used in the performance notebooks. The explanations are in
`PrestazioniRoskam/09_Rigorous_Proofs.ipynb`.

| File | Result | Roskam & Lan |
|---|---|---|
| `DragPolar.lean` | maximum of `C_L/C_D` (point E): global, unique; minimum drag | Eqns 8.18-8.19 |
| `Glide.lean` | `tan γ = D/L`; minimum glide angle and maximum glide distance only at point E | Sec. 8.2 |
| `MinPower.lean` | minimum power at `C_L = √(3 C_D0/K)` (point P); `(V_E/V_P)⁴ = 3` | Eqns 8.25-8.26, 8.75 |
| `JetClimb.lean` | jet maximum rate of climb: unique positive root, global maximum | Eqn 9.30 |
| `Breguet.lean` | weight history `W₀ e^{-ct}`, endurance and range (jet and propeller) | Ch. 11 |
| `TimeToClimb.lean` | `t = -(H/RC₀) ln(1 - h/H)`; the absolute ceiling is never reached | Eqn 9.74 |
| `Atmosphere.lean` | ISA troposphere: existence and uniqueness of `p = p₀ θ^{g/(LR)}`; `σ = θ^{g/(LR)-1}` | Ch. 1 |

## Working in VS Code

1. Install the **lean4** extension (it also installs `elan`, the Lean version manager).
2. *File -> Open Folder...* and choose this `lean/` folder (not the repository root).
3. In a terminal inside `lean/`:
   ```bash
   lake exe cache get   # download the pre-compiled Mathlib (a few GB, once)
   lake build           # check every proof
   ```
4. Open any `.lean` file: the *Lean Infoview* panel shows, at the cursor, the hypotheses
   and the goal still to be proved. Moving the cursor through a proof shows each step.

The Lean version (`lean-toolchain`) and the Mathlib version (`lakefile.toml`) are pinned,
so everybody checks the proofs with the same tools. `python tools/check_lean_proofs.py`
(from the repository root) verifies that no proof is left unfinished (`sorry`).

## Status of these files

The proofs were written without a Lean installation. All their algebraic identities were
re-checked with SymPy, but the files have **not yet been compiled**. The first
`lake build` may report small problems, typically a Mathlib lemma that was renamed or a
`field_simp` step that leaves a slightly different goal. The Infoview shows the goal at
the failing line, and `exact?` or `apply?` usually suggests the current lemma name.
