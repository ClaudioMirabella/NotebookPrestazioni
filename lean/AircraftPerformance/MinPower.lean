/-
# Proof 3 - Minimum power required and the speed ratio V_E / V_P = 3^(1/4)

Roskam & Lan, Eqns 8.25-8.26 (point P of the polar), 8.75 and 9.57; notebook 07.

Level flight, parabolic polar (hypotheses of Proof 1):
    `W = ½ ρ V² S C_L`,   `D = ½ ρ V² S C_D`,   power required `P = D V`.

Results proved here:
1. `power_sq_eq`     : `P² = 2 W³ C_D² / (ρ S C_L³)`: at a given weight and altitude,
                       MINIMUM POWER  <->  MAXIMUM  `C_L³ / C_D²`  (= (C_L^{3/2}/C_D)²);
2. `power_factor_le` : `16 √C_D0 √K³ C_L³ ≤ 3√3 · C_D²` for every `C_L ≥ 0` ...
3. `power_factor_clP`: ... with equality at `C_L,P = √3 √C_D0 / √K = √(3 C_D0 / K)`;
4. `cube_div_sq_le`  : hence `C_L³ / C_D² ≤ C_L,P³ / C_D(C_L,P)²` (global maximum at P);
5. `speedSq_clE`, `speed_ratio_pow_four` : `V_E² = √3 V_P²`, i.e. `(V_E / V_P)⁴ = 3`.

Proof idea. With `a = √C_D0`, `b = √K`, `r = √3` (so `r² = 3`) the polynomial identity
    `3r (a² + b² C_L²)² - 16 a b³ C_L³ = (b C_L - r a)² (3r b² C_L² + 2ab C_L + r a²)`
holds (modulo `r² = 3`): the right-hand side is a square times a positive factor.
The identity was found with SymPy (factorisation) and is re-checked here by `ring`.
-/
import AircraftPerformance.DragPolar

namespace AircraftPerformance

/-- Lift coefficient of point P (minimum power):  `C_L,P = √3 √C_D0 / √K`. -/
noncomputable def clP (CD0 K : ℝ) : ℝ := Real.sqrt 3 * Real.sqrt CD0 / Real.sqrt K

/-- Square of the level-flight speed at lift coefficient `C_L`: `V² = 2W / (ρ S C_L)`. -/
noncomputable def levelSpeedSq (W ρ S CL : ℝ) : ℝ := 2 * W / (ρ * S * CL)

/-- **Theorem 1 (power required).**  In level flight `(D V)² = 2 W³ C_D² / (ρ S C_L³)`. -/
theorem power_sq_eq {ρ S V CL CD W D : ℝ} (hρ : 0 < ρ) (hS : 0 < S) (hCL : 0 < CL)
    (hW : W = ρ * V ^ 2 * S * CL / 2) (hD : D = ρ * V ^ 2 * S * CD / 2) :
    (D * V) ^ 2 = 2 * W ^ 3 * CD ^ 2 / (ρ * S * CL ^ 3) := by
  have hρ' := hρ.ne'
  have hS' := hS.ne'
  have hCL' := hCL.ne'
  rw [hW, hD]
  field_simp <;> ring

/-- Polynomial identity behind Proof 3 (valid whenever `r² = 3`). -/
lemma power_identity {a b r CL : ℝ} (hr : r ^ 2 = 3) :
    3 * r * (a ^ 2 + b ^ 2 * CL ^ 2) ^ 2 - 16 * a * b ^ 3 * CL ^ 3
      = (b * CL - r * a) ^ 2 * (3 * r * b ^ 2 * CL ^ 2 + 2 * a * b * CL + r * a ^ 2) := by
  linear_combination (6 * a * b ^ 3 * CL ^ 3 - 3 * r * a ^ 2 * b ^ 2 * CL ^ 2 - r * a ^ 4) * hr

variable {CD0 K : ℝ}

/-- **Theorem 2.**  `16 √C_D0 √K³ C_L³ ≤ 3√3 · C_D(C_L)²` for every `C_L ≥ 0`. -/
theorem power_factor_le (hCD0 : 0 < CD0) (hK : 0 < K) {CL : ℝ} (hCL : 0 ≤ CL) :
    16 * Real.sqrt CD0 * Real.sqrt K ^ 3 * CL ^ 3 ≤ 3 * Real.sqrt 3 * dragCoeff CD0 K CL ^ 2 := by
  have ha : 0 ≤ Real.sqrt CD0 := Real.sqrt_nonneg _
  have hb : 0 ≤ Real.sqrt K := Real.sqrt_nonneg _
  have hr : 0 ≤ Real.sqrt 3 := Real.sqrt_nonneg _
  have hr2 : Real.sqrt 3 ^ 2 = 3 := Real.sq_sqrt (by norm_num)
  -- the identity with a = √C_D0, b = √K, r = √3 ...
  have hid := power_identity (a := Real.sqrt CD0) (b := Real.sqrt K) (CL := CL) hr2
  -- ... whose right-hand side is (square) × (positive factor) ≥ 0
  have hnn : 0 ≤ (Real.sqrt K * CL - Real.sqrt 3 * Real.sqrt CD0) ^ 2 *
      (3 * Real.sqrt 3 * Real.sqrt K ^ 2 * CL ^ 2 + 2 * Real.sqrt CD0 * Real.sqrt K * CL
        + Real.sqrt 3 * Real.sqrt CD0 ^ 2) :=
    mul_nonneg (sq_nonneg _) (by positivity)
  rw [Real.sq_sqrt hCD0.le, Real.sq_sqrt hK.le] at hid hnn
  unfold dragCoeff
  linarith

/-- **Theorem 3.**  Equality holds at point P:  `16 √C_D0 √K³ C_L,P³ = 3√3 · C_D(C_L,P)²`. -/
theorem power_factor_clP (hCD0 : 0 < CD0) (hK : 0 < K) :
    16 * Real.sqrt CD0 * Real.sqrt K ^ 3 * clP CD0 K ^ 3
      = 3 * Real.sqrt 3 * dragCoeff CD0 K (clP CD0 K) ^ 2 := by
  have hb : 0 < Real.sqrt K := Real.sqrt_pos.mpr hK
  have hr2 : Real.sqrt 3 ^ 2 = 3 := Real.sq_sqrt (by norm_num)
  have hid := power_identity (a := Real.sqrt CD0) (b := Real.sqrt K) (CL := clP CD0 K) hr2
  -- at C_L,P the square (b C_L - r a)² vanishes
  have hzero : Real.sqrt K * clP CD0 K - Real.sqrt 3 * Real.sqrt CD0 = 0 := by
    unfold clP
    -- √K · (√3 √C_D0 / √K) = √3 √C_D0
    rw [mul_div_assoc', mul_div_cancel_left₀ _ hb.ne', sub_self]
  rw [hzero, Real.sq_sqrt hCD0.le, Real.sq_sqrt hK.le] at hid
  unfold dragCoeff
  linarith

/-- **Theorem 4 (global maximum of `C_L³ / C_D²` at point P).** -/
theorem cube_div_sq_le (hCD0 : 0 < CD0) (hK : 0 < K) {CL : ℝ} (hCL : 0 ≤ CL) :
    CL ^ 3 / dragCoeff CD0 K CL ^ 2 ≤ clP CD0 K ^ 3 / dragCoeff CD0 K (clP CD0 K) ^ 2 := by
  have hD := dragCoeff_pos hCD0 hK CL
  have hDP := dragCoeff_pos hCD0 hK (clP CD0 K)
  have hP : 0 ≤ clP CD0 K := by unfold clP; positivity
  have hr : 0 < Real.sqrt 3 := Real.sqrt_pos.mpr (by norm_num)
  have hle := power_factor_le hCD0 hK hCL
  have heq := power_factor_clP hCD0 hK
  -- multiply Theorem 2 by C_L,P³ ≥ 0
  have h1 := mul_le_mul_of_nonneg_right hle (pow_nonneg hP 3)
  -- 3√3 · (C_L³ C_D(P)²) ≤ 3√3 · (C_L,P³ C_D(C_L)²), using Theorem 3
  have h3 : 3 * Real.sqrt 3 * (CL ^ 3 * dragCoeff CD0 K (clP CD0 K) ^ 2)
      ≤ 3 * Real.sqrt 3 * (clP CD0 K ^ 3 * dragCoeff CD0 K CL ^ 2) := by
    rw [show 3 * Real.sqrt 3 * (CL ^ 3 * dragCoeff CD0 K (clP CD0 K) ^ 2)
        = CL ^ 3 * (3 * Real.sqrt 3 * dragCoeff CD0 K (clP CD0 K) ^ 2) by ring, ← heq]
    linarith
  rw [div_le_div_iff₀ (by positivity) (by positivity)]
  exact le_of_mul_le_mul_left h3 (by positivity)

/-- **Theorem 5 (speeds of points E and P).**  `V_E² = √3 · V_P²`. -/
theorem speedSq_clE (hCD0 : 0 < CD0) (hK : 0 < K) {W ρ S : ℝ} (hρ : 0 < ρ) (hS : 0 < S) :
    levelSpeedSq W ρ S (clE CD0 K) = Real.sqrt 3 * levelSpeedSq W ρ S (clP CD0 K) := by
  have ha := (Real.sqrt_pos.mpr hCD0).ne'
  have hb := (Real.sqrt_pos.mpr hK).ne'
  have hr : Real.sqrt 3 ≠ 0 := (Real.sqrt_pos.mpr (by norm_num)).ne'
  have hρ' := hρ.ne'
  have hS' := hS.ne'
  unfold levelSpeedSq clE clP
  field_simp <;> ring

/-- **Corollary (Roskam Eqn 8.75).**  `(V_E / V_P)⁴ = 3`, i.e. `V_E = 3^{1/4} V_P ≈ 1.316 V_P`. -/
theorem speed_ratio_pow_four (hCD0 : 0 < CD0) (hK : 0 < K) {W ρ S V_E V_P : ℝ}
    (hρ : 0 < ρ) (hS : 0 < S) (hVP : 0 < V_P)
    (hE : V_E ^ 2 = levelSpeedSq W ρ S (clE CD0 K))
    (hP : V_P ^ 2 = levelSpeedSq W ρ S (clP CD0 K)) :
    (V_E / V_P) ^ 4 = 3 := by
  have hVP' := hVP.ne'
  have h1 : V_E ^ 2 = Real.sqrt 3 * V_P ^ 2 := by rw [hE, hP, speedSq_clE hCD0 hK hρ hS]
  calc (V_E / V_P) ^ 4 = (V_E ^ 2) ^ 2 / (V_P ^ 2) ^ 2 := by ring
    _ = (Real.sqrt 3 * V_P ^ 2) ^ 2 / (V_P ^ 2) ^ 2 := by rw [h1]
    _ = 3 := by
      rw [mul_pow, Real.sq_sqrt (by norm_num : (0 : ℝ) ≤ 3)]
      field_simp <;> ring

end AircraftPerformance
