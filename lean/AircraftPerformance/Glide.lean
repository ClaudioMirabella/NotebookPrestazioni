/-
# Proof 2 - Gliding flight: minimum glide angle and maximum glide distance

Roskam & Lan, Sec. 8.2 (unpowered flight); Anderson, Sec. 5.11; notebooks 04 and 08.

Model: steady, straight, unpowered glide at a constant flight-path angle `γ`
(measured positive downwards), still air. Equilibrium along and normal to the
flight path (Roskam Eqns 8.15-8.16):
    `D = W sin γ`,   `L = W cos γ`.

Results proved here:
1. `glide_equilibrium`       : the equilibrium equations imply `γ = arctan (D / L)`,
                               i.e. `tan γ = D/L = C_D/C_L = 1/E` (exact, no small angles);
2. `minGlideAngle_le`        : `γ(C_L) ≥ γ_min = arctan (1 / E_max)` for every `C_L > 0`;
3. `glideAngle_eq_min_iff`   : the minimum is reached ONLY at `C_L = C_L,E` (point E);
4. `glideDistance_eq`        : the still-air distance from height `h` is `x = h · E`;
5. `glideDistance_le`        : `x ≤ h · E_max`, again with equality only at point E.

Everything rests on Proof 1 (`DragPolar.lean`): `arctan` is strictly increasing,
so minimising `γ` is the same as minimising `C_D / C_L`, i.e. maximising `E`.
-/
import AircraftPerformance.DragPolar

namespace AircraftPerformance

open Real

/-- **Theorem 1 (glide equilibrium).**  If `L = W cos γ`, `D = W sin γ`, `W > 0` and
`-π/2 < γ < π/2`, then `γ = arctan (D / L)`: the glide angle is fixed by the ratio
`D / L` alone, independently of the weight. -/
theorem glide_equilibrium {W L D γ : ℝ} (hW : 0 < W)
    (hγ₁ : -(π / 2) < γ) (hγ₂ : γ < π / 2)
    (hL : L = W * cos γ) (hD : D = W * sin γ) :
    γ = arctan (D / L) := by
  -- D / L = (W sin γ)/(W cos γ) = sin γ / cos γ = tan γ,  and arctan (tan γ) = γ on (-π/2, π/2)
  rw [hL, hD, mul_div_mul_left _ _ hW.ne', ← tan_eq_sin_div_cos, arctan_tan hγ₁ hγ₂]

/-- Glide angle as a function of the lift coefficient: `γ = arctan (C_D / C_L)`. -/
noncomputable def glideAngle (CD0 K CL : ℝ) : ℝ := arctan (dragCoeff CD0 K CL / CL)

/-- Minimum glide angle `γ_min = arctan (2 √C_D0 √K) = arctan (1 / E_max)`. -/
noncomputable def minGlideAngle (CD0 K : ℝ) : ℝ := arctan (2 * (Real.sqrt CD0 * Real.sqrt K))

/-- Still-air glide distance from height `h` along a path of angle `γ`: `x = h / tan γ`. -/
noncomputable def glideDistance (h γ : ℝ) : ℝ := h / tan γ

variable {CD0 K : ℝ}

/-- `1 / E_max = 2 √C_D0 √K`, so `γ_min` is indeed `arctan (1 / E_max)`. -/
lemma one_div_eMax : 1 / eMax CD0 K = 2 * (Real.sqrt CD0 * Real.sqrt K) := by
  unfold eMax
  rw [one_div_one_div]
  ring

/-- **Theorem 2 (minimum glide angle).**  For every `C_L > 0`, `γ(C_L) ≥ γ_min`. -/
theorem minGlideAngle_le (hCD0 : 0 < CD0) (hK : 0 < K) {CL : ℝ} (hCL : 0 < CL) :
    minGlideAngle CD0 K ≤ glideAngle CD0 K CL := by
  unfold minGlideAngle glideAngle
  -- arctan is increasing, so it suffices to compare the arguments ...
  apply arctan_strictMono.monotone
  -- ... i.e.  2 √C_D0 √K ≤ C_D / C_L,  which is AM-GM divided by C_L > 0
  rw [le_div_iff₀ hCL]
  exact am_gm_polar hCD0 hK CL

/-- **Theorem 3 (uniqueness).**  `γ(C_L) = γ_min` if and only if `C_L = C_L,E`. -/
theorem glideAngle_eq_min_iff (hCD0 : 0 < CD0) (hK : 0 < K) {CL : ℝ} (hCL : 0 < CL) :
    glideAngle CD0 K CL = minGlideAngle CD0 K ↔ CL = clE CD0 K := by
  unfold glideAngle minGlideAngle
  -- arctan is injective, so equal angles <-> equal tangents ...
  rw [arctan_strictMono.injective.eq_iff, div_eq_iff hCL.ne']
  -- ... <-> C_D = 2 √C_D0 √K · C_L, which is the equality case of Proof 1
  exact dragCoeff_eq_iff hCD0 hK CL

/-- **Theorem 4 (glide distance).**  From height `h`, `x = h · C_L / C_D = h · E`. -/
theorem glideDistance_eq (CD0 K h CL : ℝ) :
    glideDistance h (glideAngle CD0 K CL) = h * efficiency CD0 K CL := by
  unfold glideDistance glideAngle efficiency
  -- tan (arctan y) = y, then  h / (C_D / C_L) = h · C_L / C_D
  rw [tan_arctan, div_div_eq_mul_div, mul_div_assoc]

/-- **Theorem 5 (maximum glide distance).**  From height `h ≥ 0` the still-air glide
distance never exceeds `h · E_max`. -/
theorem glideDistance_le (hCD0 : 0 < CD0) (hK : 0 < K) {h : ℝ} (hh : 0 ≤ h) (CL : ℝ) :
    glideDistance h (glideAngle CD0 K CL) ≤ h * eMax CD0 K := by
  rw [glideDistance_eq]
  exact mul_le_mul_of_nonneg_left (efficiency_le_eMax hCD0 hK CL) hh

/-- ... and, from a positive height, the maximum is reached only at point E. -/
theorem glideDistance_eq_max_iff (hCD0 : 0 < CD0) (hK : 0 < K) {h : ℝ} (hh : 0 < h) (CL : ℝ) :
    glideDistance h (glideAngle CD0 K CL) = h * eMax CD0 K ↔ CL = clE CD0 K := by
  rw [glideDistance_eq, mul_right_inj' hh.ne']
  exact efficiency_eq_eMax_iff hCD0 hK CL

end AircraftPerformance
