/-
# Proof 6 - Time to climb with a linear rate of climb (Roskam Eqn 9.74)

Roskam & Lan, Sec. 9.6 (time to climb); notebooks 03 and 07.

Model: the maximum rate of climb decreases linearly with altitude,
    `RC(h) = RC₀ (1 - h / H)`,   `RC₀ > 0` (sea level),   `H > 0` (absolute ceiling),
and the time to climb from sea level to `h` is  `t(h) = ∫₀^h ds / RC(s)`.

Results proved here:
1. `hasDerivAt_timeFormula` : `d/dh [-(H/RC₀) ln(1 - h/H)] = 1 / RC(h)`   (for `h < H`);
2. `climbTime_eq`           : `t(h) = -(H / RC₀) ln(1 - h / H)`  for `0 ≤ h < H`
                              (Roskam Eqn 9.74), by the fundamental theorem of calculus;
3. `climbTime_unbounded`    : for every `M` there is an altitude `h < H` with `t(h) ≥ M`:
                              the absolute ceiling can NOT be reached in a finite time.
                              (This is why the SERVICE ceiling, RC = 100 ft/min, is used
                              in practice.)
-/
import Mathlib

namespace AircraftPerformance

open Real

/-- Linear model of the rate of climb: `RC(h) = RC₀ (1 - h / H)`. -/
noncomputable def rcLinear (RC0 H h : ℝ) : ℝ := RC0 * (1 - h / H)

/-- Time to climb from sea level to `h`:  `t(h) = ∫₀^h ds / RC(s)`. -/
noncomputable def climbTime (RC0 H h : ℝ) : ℝ := ∫ s in (0 : ℝ)..h, 1 / rcLinear RC0 H s

variable {RC0 H : ℝ}

/-- Below the ceiling `1 - h/H > 0`. -/
lemma one_sub_div_pos (hH : 0 < H) {h : ℝ} (hh : h < H) : 0 < 1 - h / H := by
  have : h / H < 1 := (div_lt_one hH).mpr hh
  linarith

/-- **Theorem 1.**  `F(h) = -(H/RC₀) ln(1 - h/H)` is an antiderivative of `1 / RC(h)`. -/
theorem hasDerivAt_timeFormula (hRC : 0 < RC0) (hH : 0 < H) {h : ℝ} (hh : h < H) :
    HasDerivAt (fun y => -(H / RC0) * log (1 - y / H)) (1 / rcLinear RC0 H h) h := by
  have hpos := one_sub_div_pos hH hh
  -- d/dy (1 - y/H) = -1/H
  have h1 : HasDerivAt (fun y => 1 - y / H) (-(1 / H)) h :=
    ((hasDerivAt_id' h).div_const H).const_sub 1
  -- chain rule for the logarithm, then multiply by the constant -(H/RC₀)
  have h2 := (h1.log hpos.ne').const_mul (-(H / RC0))
  convert h2 using 1
  -- remaining goal:  1 / (RC₀ u) = -(H/RC₀) · (-(1/H) / u)   with  u = 1 - h/H ≠ 0
  unfold rcLinear
  generalize 1 - h / H = u at hpos ⊢
  have hu := hpos.ne'
  have hRC' := hRC.ne'
  have hH' := hH.ne'
  field_simp <;> ring

/-- **Theorem 2 (Roskam Eqn 9.74).**  For `0 ≤ h < H`:  `t(h) = -(H / RC₀) ln(1 - h / H)`. -/
theorem climbTime_eq (hRC : 0 < RC0) (hH : 0 < H) {h : ℝ} (h0 : 0 ≤ h) (hh : h < H) :
    climbTime RC0 H h = -(H / RC0) * log (1 - h / H) := by
  -- on [0, h] we are below the ceiling
  have hbelow : ∀ y ∈ Set.uIcc 0 h, y < H := by
    intro y hy
    rw [Set.uIcc_of_le h0] at hy
    exact lt_of_le_of_lt hy.2 hh
  have hderiv : ∀ y ∈ Set.uIcc 0 h,
      HasDerivAt (fun y => -(H / RC0) * log (1 - y / H)) (1 / rcLinear RC0 H y) y :=
    fun y hy => hasDerivAt_timeFormula hRC hH (hbelow y hy)
  -- the integrand is continuous on [0, h] (RC does not vanish there)
  have hcont : ContinuousOn (fun y => 1 / rcLinear RC0 H y) (Set.uIcc 0 h) := by
    apply ContinuousOn.div continuousOn_const
    · unfold rcLinear
      fun_prop
    · intro y hy
      unfold rcLinear
      exact mul_ne_zero hRC.ne' (one_sub_div_pos hH (hbelow y hy)).ne'
  -- fundamental theorem of calculus:  t(h) = F(h) - F(0)  and  F(0) = 0
  unfold climbTime
  rw [intervalIntegral.integral_eq_sub_of_hasDerivAt hderiv hcont.intervalIntegrable]
  simp

/-- **Theorem 3 (the absolute ceiling is never reached).**  For every time `M` there is an
altitude `h`, with `0 ≤ h < H`, whose time to climb is at least `M`. -/
theorem climbTime_unbounded (hRC : 0 < RC0) (hH : 0 < H) (M : ℝ) :
    ∃ h, 0 ≤ h ∧ h < H ∧ M ≤ climbTime RC0 H h := by
  -- target time m = max M 0 ≥ 0, reached at  h = H (1 - e),  e = exp(-m RC₀ / H)
  obtain ⟨m, hm⟩ : ∃ m, m = max M 0 := ⟨_, rfl⟩
  have hm0 : 0 ≤ m := by rw [hm]; exact le_max_right M 0
  obtain ⟨e, he⟩ : ∃ e, e = exp (-(m * RC0 / H)) := ⟨_, rfl⟩
  have he0 : 0 < e := by rw [he]; exact exp_pos _
  have he1 : e ≤ 1 := by
    rw [he, exp_le_one_iff, neg_nonpos]
    exact div_nonneg (mul_nonneg hm0 hRC.le) hH.le
  have hh0 : 0 ≤ H * (1 - e) := mul_nonneg hH.le (by linarith)
  have hhH : H * (1 - e) < H := by nlinarith [mul_pos hH he0]
  refine ⟨H * (1 - e), hh0, hhH, ?_⟩
  -- 1 - h/H = e,  so  t(h) = -(H/RC₀) ln e = -(H/RC₀)(-m RC₀/H) = m ≥ M
  have hq : 1 - H * (1 - e) / H = e := by
    rw [mul_div_cancel_left₀ _ hH.ne']
    ring
  have hRC' := hRC.ne'
  have hH' := hH.ne'
  have hm_eq : -(H / RC0) * -(m * RC0 / H) = m := by
    field_simp <;> ring
  rw [climbTime_eq hRC hH hh0 hhH, hq, he, log_exp, hm_eq, hm]
  exact le_max_left M 0

end AircraftPerformance
