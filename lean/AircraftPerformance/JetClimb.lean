/-
# Proof 4 - Maximum rate of climb of a jet (Roskam Eqn 9.30)

Roskam & Lan, Sec. 9.3 (jet airplanes); Anderson, Sec. 5.10; notebooks 03 and 07.

Model: shallow climb (`cos γ ≈ 1`, so `L ≈ W`), thrust `T` independent of speed,
parabolic polar. The drag in (quasi-)level flight is
    `D(V) = A V² + B / V²`,   `A = ½ ρ S C_D0`,   `B = 2 K W² / (ρ S)`,
and the rate of climb is `RC = (T - D) V / W`. Because `W` is a positive constant we
study `W · RC = T V - A V³ - B / V`  (definition `rcW`).

Results proved here:
1. `drag_level_flight`   : `q S C_D(W / (q S)) = q S C_D0 + K W² / (q S)` (drag of the polar);
2. `rcW_eq`              : `(T - (A V² + B/V²)) V = rcW V`;
3. `hasDerivAt_rcW`      : `d(W·RC)/dV = T - 3 A V² + B / V²`;
4. `stationary_iff`      : for `V ≠ 0` the stationarity condition is the QUADRATIC
                           `3 A x² - T x - B = 0` in `x = V²`;
5. `quadratic_root_unique`, `bestSpeedSq_root`, `other_root_neg` :
                           it has exactly one positive root,
                           `x* = (T + √(T² + 12 A B)) / (6 A)`  (Roskam Eqn 9.30),
                           the other root being negative (non-physical);
6. `rcW_le_of_stationary`, `rcW_le_best` : the stationary point is the GLOBAL maximum
                           of the rate of climb over all speeds `V > 0`.

The key identity for 6 (valid when `Vs` is stationary),
    `rcW Vs - rcW V = (Vs - V)² (A (2 Vs + V) + B / (V Vs²))`,
was obtained with SymPy (`reduced`, i.e. division by a Gröbner basis) and is
re-checked by Lean's `linear_combination`.
-/
import AircraftPerformance.DragPolar

namespace AircraftPerformance

/-- `W · RC` as a function of the true airspeed:  `T V - A V³ - B / V`. -/
noncomputable def rcW (T A B V : ℝ) : ℝ := T * V - A * V ^ 3 - B / V

/-- Positive root of the stationarity quadratic (Roskam Eqn 9.30, written for `V²`). -/
noncomputable def bestSpeedSq (T A B : ℝ) : ℝ := (T + Real.sqrt (T ^ 2 + 12 * A * B)) / (6 * A)

/-- **Theorem 1.**  With `C_L = W / (q S)` the parabolic polar gives
`D = q S C_D = q S C_D0 + K W² / (q S)`;  with `q = ½ ρ V²` this is `A V² + B / V²`. -/
theorem drag_level_flight {CD0 K W q S : ℝ} (hq : 0 < q) (hS : 0 < S) :
    q * S * dragCoeff CD0 K (W / (q * S)) = q * S * CD0 + K * W ^ 2 / (q * S) := by
  have hq' := hq.ne'
  have hS' := hS.ne'
  unfold dragCoeff
  field_simp <;> ring

/-- **Theorem 2.**  `(T - D(V)) V = rcW V`  for `V ≠ 0`. -/
theorem rcW_eq {T A B V : ℝ} (hV : V ≠ 0) :
    (T - (A * V ^ 2 + B / V ^ 2)) * V = rcW T A B V := by
  unfold rcW
  field_simp <;> ring

/-- **Theorem 3.**  `d(W·RC)/dV = T - 3 A V² + B / V²`  (for `V ≠ 0`). -/
theorem hasDerivAt_rcW {T A B V : ℝ} (hV : V ≠ 0) :
    HasDerivAt (rcW T A B) (T - 3 * A * V ^ 2 + B / V ^ 2) V := by
  -- derivative of each term: T·V, A·V³ and B·V⁻¹
  have h := (((hasDerivAt_id' V).const_mul T).sub ((hasDerivAt_pow 3 V).const_mul A)).sub
    ((hasDerivAt_inv hV).const_mul B)
  convert h using 1
  · -- the two functions coincide
    funext y
    simp only [rcW]
    ring
  · -- and so do the two expressions of the derivative (norm_num evaluates ↑3 and 3 - 1)
    norm_num <;> ring

/-- **Theorem 4.**  For `V ≠ 0`:  `T - 3 A V² + B / V² = 0  ↔  3 A (V²)² - T V² - B = 0`. -/
theorem stationary_iff {T A B V : ℝ} (hV : V ≠ 0) :
    T - 3 * A * V ^ 2 + B / V ^ 2 = 0 ↔ 3 * A * (V ^ 2) ^ 2 - T * V ^ 2 - B = 0 := by
  have h2 : V * V⁻¹ = 1 := mul_inv_cancel₀ hV
  constructor
  · intro hs
    linear_combination (-V ^ 2) * hs + B * (V * V⁻¹ + 1) * h2
  · intro hq
    linear_combination (-(V⁻¹) ^ 2) * hq - (T - 3 * A * V ^ 2) * (1 + V * V⁻¹) * h2

variable {T A B : ℝ}

/-- **Theorem 5a (uniqueness).**  A POSITIVE root of `3 A x² - T x - B = 0` (`A, B > 0`)
is necessarily `x* = (T + √(T² + 12 A B)) / (6 A)`. -/
theorem quadratic_root_unique (hA : 0 < A) (hB : 0 < B) {x : ℝ} (hx : 0 < x)
    (hq : 3 * A * x ^ 2 - T * x - B = 0) : x = bestSpeedSq T A B := by
  -- completing the square:  (6 A x - T)² = T² + 12 A B
  have hsq : (6 * A * x - T) ^ 2 = T ^ 2 + 12 * A * B := by linear_combination 12 * A * hq
  -- the base is non-negative: otherwise B = 3 A x² - T x would be negative
  have hpos : 0 ≤ 6 * A * x - T := by nlinarith [mul_pos hA (mul_pos hx hx)]
  -- hence √(T² + 12 A B) = 6 A x - T
  have hs : Real.sqrt (T ^ 2 + 12 * A * B) = 6 * A * x - T := by
    rw [← hsq]
    exact Real.sqrt_sq hpos
  unfold bestSpeedSq
  rw [hs, eq_div_iff (by positivity)]
  ring

/-- **Theorem 5b (existence).**  For `T ≥ 0`, `x*` is positive and solves the quadratic. -/
theorem bestSpeedSq_root (hA : 0 < A) (hB : 0 < B) (hT : 0 ≤ T) :
    0 < bestSpeedSq T A B ∧ 3 * A * bestSpeedSq T A B ^ 2 - T * bestSpeedSq T A B - B = 0 := by
  -- abbreviation: s = √(T² + 12 A B), with s² = T² + 12 A B
  have hs2 : Real.sqrt (T ^ 2 + 12 * A * B) ^ 2 = T ^ 2 + 12 * A * B := Real.sq_sqrt (by positivity)
  refine ⟨by unfold bestSpeedSq; positivity, ?_⟩
  -- 6 A x* = T + s
  have hx6 : 6 * A * bestSpeedSq T A B = T + Real.sqrt (T ^ 2 + 12 * A * B) := by
    unfold bestSpeedSq
    rw [mul_div_assoc', mul_div_cancel_left₀ _ (by positivity)]
  -- 12 A · (3 A x*² - T x* - B) = s² - T² - 12 A B = 0
  have h12 : 12 * A * (3 * A * bestSpeedSq T A B ^ 2 - T * bestSpeedSq T A B - B) = 0 := by
    linear_combination
      (6 * A * bestSpeedSq T A B + Real.sqrt (T ^ 2 + 12 * A * B) - T) * hx6 + hs2
  exact (mul_eq_zero.mp h12).resolve_left (by positivity)

/-- **Theorem 5c.**  The other root `(T - √(T² + 12 A B)) / (6 A)` is negative:
it has no physical meaning (V² must be positive). -/
theorem other_root_neg (hA : 0 < A) (hB : 0 < B) (hT : 0 ≤ T) :
    (T - Real.sqrt (T ^ 2 + 12 * A * B)) / (6 * A) < 0 := by
  have hlt : T < Real.sqrt (T ^ 2 + 12 * A * B) :=
    (Real.lt_sqrt hT).mpr (by nlinarith [mul_pos hA hB])
  exact div_neg_of_neg_of_pos (by linarith) (by positivity)

/-- **Theorem 6a (global maximum).**  If `Vs > 0` is a stationary point, then
`rcW V ≤ rcW Vs` for EVERY speed `V > 0`. -/
theorem rcW_le_of_stationary (hA : 0 < A) (hB : 0 < B) {V Vs : ℝ} (hV : 0 < V) (hVs : 0 < Vs)
    (hstat : T - 3 * A * Vs ^ 2 + B / Vs ^ 2 = 0) : rcW T A B V ≤ rcW T A B Vs := by
  have h1 : V * V⁻¹ = 1 := mul_inv_cancel₀ hV.ne'
  have h2 : Vs * Vs⁻¹ = 1 := mul_inv_cancel₀ hVs.ne'
  -- exact identity (certificate computed with SymPy)
  have hdiff : rcW T A B Vs - rcW T A B V
      = (Vs - V) ^ 2 * (A * (2 * Vs + V) + B / (V * Vs ^ 2)) := by
    unfold rcW
    linear_combination (Vs - V) * hstat + (-B * (Vs⁻¹) ^ 2 * (V - 2 * Vs)) * h1
      + (-B * (Vs * V⁻¹ * Vs⁻¹ + V⁻¹ - Vs⁻¹)) * h2
  -- the right-hand side is (square) × (positive factor)
  have hnn : 0 ≤ (Vs - V) ^ 2 * (A * (2 * Vs + V) + B / (V * Vs ^ 2)) :=
    mul_nonneg (sq_nonneg _) (by positivity)
  linarith

/-- **Theorem 6b (Roskam Eqn 9.30 gives the maximum rate of climb).**
With `V* = √x*`, `rcW V ≤ rcW V*` for every `V > 0`. -/
theorem rcW_le_best (hA : 0 < A) (hB : 0 < B) (hT : 0 ≤ T) {V : ℝ} (hV : 0 < V) :
    rcW T A B V ≤ rcW T A B (Real.sqrt (bestSpeedSq T A B)) := by
  obtain ⟨hx, hq⟩ := bestSpeedSq_root hA hB hT
  have hVs : 0 < Real.sqrt (bestSpeedSq T A B) := Real.sqrt_pos.mpr hx
  have hsq : Real.sqrt (bestSpeedSq T A B) ^ 2 = bestSpeedSq T A B := Real.sq_sqrt hx.le
  apply rcW_le_of_stationary hA hB hV hVs
  rw [stationary_iff hVs.ne', hsq]
  exact hq

end AircraftPerformance
