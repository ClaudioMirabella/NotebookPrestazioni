/-
# Proof 1 - The optimum of the parabolic drag polar

Roskam & Lan, Ch. 5 and Eqns 8.18-8.19 (point E of the polar); notebooks 01 and 08.

Model (the *hypotheses* of every theorem below):
* parabolic drag polar  `C_D = C_D0 + K C_L²`  with  `C_D0 > 0`,  `K > 0`;
* aerodynamic efficiency (lift-to-drag ratio)  `E = C_L / C_D`.

Results proved here:
1. `efficiency_le_eMax`   : `C_L / C_D ≤ 1 / (2 √C_D0 √K)` for EVERY `C_L`
                            (a GLOBAL maximum, not only a stationary point);
2. `efficiency_clE`       : the bound is attained at `C_L = √C_D0 / √K`;
3. `efficiency_eq_eMax_iff`: ... and ONLY there (uniqueness of point E);
4. `drag_ge_min`          : in level flight `D ≥ 2 √C_D0 √K · W`  (minimum drag);
5. `eMax_eq`              : the familiar form  `E_max = 1 / (2 √(C_D0 K))`.

Proof idea (arithmetic-geometric mean inequality): with `a = √C_D0`, `b = √K`,
    `C_D - 2ab C_L = a² + b² C_L² - 2ab C_L = (b C_L - a)² ≥ 0`.
No calculus is needed, and the argument shows at once that the optimum is global
and unique.

Reading guide for Lean newcomers
* `theorem name (hypotheses) : statement := by tactics` - `by` starts a proof.
* `Real.sqrt x` is `√x`; `Real.sq_sqrt h : √x ^ 2 = x` (needs `h : 0 ≤ x`).
* `nlinarith`, `linarith` : automatic (non)linear arithmetic;
  `linear_combination e` : proves an equality `lhs = rhs` by checking, with `ring`,
  that `lhs - rhs` equals the combination `e` of the hypotheses (an exact
  algebraic certificate; the certificates of this library were computed with
  SymPy, see notebook 09).
* `positivity` : proves goals `0 < e` / `0 ≤ e` from the structure of `e`.
-/
import Mathlib

namespace AircraftPerformance

/-- Parabolic drag polar `C_D = C_D0 + K C_L²` (Roskam & Lan, Ch. 5). -/
def dragCoeff (CD0 K CL : ℝ) : ℝ := CD0 + K * CL ^ 2

/-- Aerodynamic efficiency (lift-to-drag ratio) `E = C_L / C_D`. -/
noncomputable def efficiency (CD0 K CL : ℝ) : ℝ := CL / dragCoeff CD0 K CL

/-- Lift coefficient of point E (minimum drag):  `C_L,E = √C_D0 / √K = √(C_D0/K)`. -/
noncomputable def clE (CD0 K : ℝ) : ℝ := Real.sqrt CD0 / Real.sqrt K

/-- Maximum efficiency  `E_max = 1 / (2 √C_D0 √K)`. -/
noncomputable def eMax (CD0 K : ℝ) : ℝ := 1 / (2 * Real.sqrt CD0 * Real.sqrt K)

variable {CD0 K : ℝ}

/-- With `C_D0, K > 0` the drag coefficient is positive for every `C_L`. -/
lemma dragCoeff_pos (hCD0 : 0 < CD0) (hK : 0 < K) (CL : ℝ) : 0 < dragCoeff CD0 K CL := by
  unfold dragCoeff
  positivity

/-- The algebraic heart of the proof, for arbitrary reals `a`, `b`:
`2ab·C_L ≤ a² + b² C_L²`, because the difference is the square `(b C_L - a)²`. -/
lemma am_gm_core (a b CL : ℝ) : 2 * (a * b) * CL ≤ a ^ 2 + b ^ 2 * CL ^ 2 := by
  nlinarith [sq_nonneg (b * CL - a)]

/-- AM-GM applied to the polar:  `2 √C_D0 √K · C_L ≤ C_D(C_L)`  for every `C_L`. -/
lemma am_gm_polar (hCD0 : 0 < CD0) (hK : 0 < K) (CL : ℝ) :
    2 * (Real.sqrt CD0 * Real.sqrt K) * CL ≤ dragCoeff CD0 K CL := by
  -- instantiate the core inequality with a = √C_D0, b = √K ...
  have h := am_gm_core (Real.sqrt CD0) (Real.sqrt K) CL
  -- ... and replace (√C_D0)² by C_D0 and (√K)² by K
  rw [Real.sq_sqrt hCD0.le, Real.sq_sqrt hK.le] at h
  unfold dragCoeff
  exact h

/-- **Equality case.**  `C_D(C_L) = 2 √C_D0 √K · C_L`  holds if and only if
`C_L = √C_D0 / √K`: equality in AM-GM forces the square `(b C_L - a)²` to vanish. -/
lemma dragCoeff_eq_iff (hCD0 : 0 < CD0) (hK : 0 < K) (CL : ℝ) :
    dragCoeff CD0 K CL = 2 * (Real.sqrt CD0 * Real.sqrt K) * CL ↔ CL = clE CD0 K := by
  have ha2 : Real.sqrt CD0 ^ 2 = CD0 := Real.sq_sqrt hCD0.le
  have hb2 : Real.sqrt K ^ 2 = K := Real.sq_sqrt hK.le
  have hb : 0 < Real.sqrt K := Real.sqrt_pos.mpr hK
  unfold clE dragCoeff
  -- `CL = a / b` is the same as `CL * b = a` (b ≠ 0)
  rw [eq_div_iff hb.ne']
  constructor
  · intro h
    -- the square (b C_L - a)² is zero ...
    have hsq : (Real.sqrt K * CL - Real.sqrt CD0) ^ 2 = 0 := by
      linear_combination CL ^ 2 * hb2 + ha2 + h
    -- ... hence so is its base
    have h0 : Real.sqrt K * CL - Real.sqrt CD0 = 0 := (pow_eq_zero_iff two_ne_zero).mp hsq
    linarith
  · intro h
    -- conversely, if b C_L = a the square vanishes and C_D = 2ab C_L
    linear_combination -ha2 - CL ^ 2 * hb2 + (CL * Real.sqrt K - Real.sqrt CD0) * h

/-- **Theorem 1 (global maximum of the efficiency).**
For every lift coefficient, `C_L / C_D ≤ E_max = 1 / (2 √C_D0 √K)`. -/
theorem efficiency_le_eMax (hCD0 : 0 < CD0) (hK : 0 < K) (CL : ℝ) :
    efficiency CD0 K CL ≤ eMax CD0 K := by
  have hD := dragCoeff_pos hCD0 hK CL
  have ha : 0 < Real.sqrt CD0 := Real.sqrt_pos.mpr hCD0
  have hb : 0 < Real.sqrt K := Real.sqrt_pos.mpr hK
  have h := am_gm_polar hCD0 hK CL
  unfold efficiency eMax
  -- cross-multiply the two fractions (both denominators are positive)
  rw [div_le_div_iff₀ hD (by positivity)]
  linarith

/-- **Theorem 2 (the maximum is attained at point E).**  `E(C_L,E) = E_max`. -/
theorem efficiency_clE (hCD0 : 0 < CD0) (hK : 0 < K) :
    efficiency CD0 K (clE CD0 K) = eMax CD0 K := by
  have hD := dragCoeff_pos hCD0 hK (clE CD0 K)
  have ha : 0 < Real.sqrt CD0 := Real.sqrt_pos.mpr hCD0
  have hb : 0 < Real.sqrt K := Real.sqrt_pos.mpr hK
  have heq := (dragCoeff_eq_iff hCD0 hK (clE CD0 K)).mpr rfl
  unfold efficiency eMax
  rw [div_eq_div_iff hD.ne' (by positivity), heq]
  ring

/-- **Theorem 3 (uniqueness of point E).**  `E(C_L) = E_max ↔ C_L = C_L,E`. -/
theorem efficiency_eq_eMax_iff (hCD0 : 0 < CD0) (hK : 0 < K) (CL : ℝ) :
    efficiency CD0 K CL = eMax CD0 K ↔ CL = clE CD0 K := by
  have hD := dragCoeff_pos hCD0 hK CL
  have ha : 0 < Real.sqrt CD0 := Real.sqrt_pos.mpr hCD0
  have hb : 0 < Real.sqrt K := Real.sqrt_pos.mpr hK
  rw [← dragCoeff_eq_iff hCD0 hK CL]
  unfold efficiency eMax
  rw [div_eq_div_iff hD.ne' (by positivity)]
  constructor <;> intro h <;> linarith

/-- **Theorem 4 (minimum drag in level flight).**
In level flight `L = W`, so `D = W · C_D / C_L`; for `C_L > 0` and `W > 0`
the drag is never below `D_min = 2 √C_D0 √K · W = W / E_max`. -/
theorem drag_ge_min (hCD0 : 0 < CD0) (hK : 0 < K) {W CL : ℝ} (hW : 0 < W) (hCL : 0 < CL) :
    2 * (Real.sqrt CD0 * Real.sqrt K) * W ≤ W * (dragCoeff CD0 K CL / CL) := by
  -- divide the AM-GM inequality by C_L > 0 ...
  have h : 2 * (Real.sqrt CD0 * Real.sqrt K) ≤ dragCoeff CD0 K CL / CL := by
    rw [le_div_iff₀ hCL]
    exact am_gm_polar hCD0 hK CL
  -- ... and multiply it by W > 0
  have hW' := mul_le_mul_of_nonneg_left h hW.le
  linarith

/-- The textbook form  `E_max = 1 / (2 √(C_D0 K))`  (because `√(xy) = √x √y`, x ≥ 0). -/
theorem eMax_eq (hCD0 : 0 < CD0) : eMax CD0 K = 1 / (2 * Real.sqrt (CD0 * K)) := by
  unfold eMax
  rw [Real.sqrt_mul hCD0.le K]
  ring

end AircraftPerformance
