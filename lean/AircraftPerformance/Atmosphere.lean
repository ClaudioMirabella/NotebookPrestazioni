/-
# Proof 7 - The ISA troposphere as the unique solution of the hydrostatic equation

Roskam & Lan, Ch. 1 / App. (standard atmosphere); Anderson, Ch. 3; Capitolo1 and
notebook 07 (symbolic ISA).

Physical laws (the hypotheses):
* hydrostatic equilibrium       `dp/dh = -ρ g`;
* ideal gas                     `p = ρ R T`;
* linear temperature lapse      `T(h) = T₀ - L h`   (troposphere, `L > 0`).
Eliminating `ρ` gives the ordinary differential equation
    `dp/dh = -g p / (R (T₀ - L h))`.                                   (*)

Results proved here:
1. `ode_of_physics`   : hydrostatic + ideal gas + linear T  ==>  (*);
2. `hasDerivAt_pressureTropo` : `p(h) = p₀ (1 - L h / T₀)^{g/(L R)}` SOLVES (*) (existence);
3. `pressure_unique`  : it is the ONLY solution of (*) with `p(0) = p₀` on any interval
                        `[0, h_max]` below the altitude where `T` would vanish (uniqueness);
4. `density_ratio`    : `σ = ρ/ρ₀ = (1 - L h / T₀)^{g/(L R) - 1}`  (the exponent 4.256 of ISA).

Uniqueness proof: for any solution `p`, the function `q(h) = p(h) θ(h)^{-n}`, with
`θ = 1 - L h / T₀` and `n = g/(L R)`, has zero derivative, hence is constant (mean value
theorem, `constant_of_has_deriv_right_zero`), and `q(0) = p₀`.
-/
import Mathlib

namespace AircraftPerformance

open Real

/-- ISA troposphere pressure:  `p(h) = p₀ (1 - L h / T₀)^{g / (L R)}`. -/
noncomputable def pressureTropo (p0 T0 L g R h : ℝ) : ℝ := p0 * (1 - L * h / T0) ^ (g / (L * R))

/-- **Theorem 1.**  Hydrostatics (`p' = -ρ g`), ideal gas (`ρ = p / (R T)`) and the linear
lapse (`T = T₀ - L h`) give  `p' = -g p / (R (T₀ - L h))`. -/
theorem ode_of_physics {p ρ T : ℝ → ℝ} {T0 L g R h : ℝ}
    (hydro : HasDerivAt p (-(ρ h) * g) h) (gas : ρ h = p h / (R * T h))
    (lapse : T h = T0 - L * h) :
    HasDerivAt p (-(g / (R * (T0 - L * h))) * p h) h :=
  hydro.congr_deriv (by rw [gas, lapse]; ring)

/-- Derivative of `θ(y) = 1 - L y / T₀`:  `θ' = -(L / T₀)`. -/
lemma hasDerivAt_theta (T0 L h : ℝ) :
    HasDerivAt (fun y => 1 - L * y / T0) (-(L * 1 / T0)) h :=
  (((hasDerivAt_id' h).const_mul L).div_const T0).const_sub 1

/-- Below the altitude `T₀ / L` the temperature ratio `θ = 1 - L h / T₀` is positive. -/
lemma theta_pos {T0 L h : ℝ} (hT0 : 0 < T0) (hLh : L * h < T0) : 0 < 1 - L * h / T0 := by
  have : L * h / T0 < 1 := (div_lt_one hT0).mpr hLh
  linarith

variable {p0 T0 L g R : ℝ}

/-- **Theorem 2 (existence).**  The ISA formula solves (*) wherever `L h < T₀`. -/
theorem hasDerivAt_pressureTropo (hT0 : 0 < T0) (hL : 0 < L) (hR : 0 < R) {h : ℝ}
    (hLh : L * h < T0) :
    HasDerivAt (pressureTropo p0 T0 L g R)
      (-(g / (R * (T0 - L * h))) * pressureTropo p0 T0 L g R h) h := by
  have hθ := theta_pos hT0 hLh
  -- write the function explicitly (pressureTropo with h as the variable)
  show HasDerivAt (fun y => p0 * (1 - L * y / T0) ^ (g / (L * R))) _ h
  -- chain rule:  d/dh θ^n = θ' · n · θ^{n-1},  then multiply by p₀
  have h1 := ((hasDerivAt_theta T0 L h).rpow_const (p := g / (L * R)) (Or.inl hθ.ne')).const_mul p0
  convert h1 using 1
  -- θ^{n-1} = θ^n / θ, then pure algebra with X = θ^n
  unfold pressureTropo
  rw [rpow_sub_one hθ.ne']
  generalize (1 - L * h / T0) ^ (g / (L * R)) = X
  have hθ' := hθ.ne'
  have hT : T0 - L * h ≠ 0 := by linarith
  have hT0' := hT0.ne'
  have hL' := hL.ne'
  have hR' := hR.ne'
  field_simp <;> ring

/-- **Theorem 3 (uniqueness).**  Let `p` satisfy (*) on `[0, h_max]`, with `L h_max < T₀`,
and `p(0) = p₀`. Then `p(h) = p₀ (1 - L h / T₀)^{g/(L R)}` for every `h ∈ [0, h_max]`. -/
theorem pressure_unique {p : ℝ → ℝ} {hmax : ℝ} (hT0 : 0 < T0) (hL : 0 < L) (hR : 0 < R)
    (hceil : L * hmax < T0) (hp0 : p 0 = p0)
    (hode : ∀ h ∈ Set.Icc 0 hmax, HasDerivAt p (-(g / (R * (T0 - L * h))) * p h) h) :
    ∀ h ∈ Set.Icc 0 hmax, p h = pressureTropo p0 T0 L g R h := by
  -- θ > 0 on the whole interval
  have hθpos : ∀ h ∈ Set.Icc 0 hmax, 0 < 1 - L * h / T0 := by
    intro h hh
    apply theta_pos hT0
    -- L h ≤ L h_max < T₀
    linarith [mul_le_mul_of_nonneg_left hh.2 hL.le]
  -- q(h) = p(h) θ(h)^{-n} has zero derivative on [0, h_max]
  have hq : ∀ h ∈ Set.Icc 0 hmax,
      HasDerivAt (fun y => p y * (1 - L * y / T0) ^ (-(g / (L * R)))) 0 h := by
    intro h hh
    have hθ := hθpos h hh
    have h2 := (hasDerivAt_theta T0 L h).rpow_const (p := -(g / (L * R))) (Or.inl hθ.ne')
    have h3 := (hode h hh).mul h2
    convert h3 using 1
    rw [rpow_sub_one hθ.ne']
    generalize (1 - L * h / T0) ^ (-(g / (L * R))) = X
    have hθ' := hθ.ne'
    have hT : T0 - L * h ≠ 0 := by
      linarith [mul_le_mul_of_nonneg_left hh.2 hL.le]
    have hT0' := hT0.ne'
    have hL' := hL.ne'
    have hR' := hR.ne'
    field_simp <;> ring
  -- a function with zero derivative on [0, h_max] is constant there
  have hconst := constant_of_has_deriv_right_zero
    (fun x hx => (hq x hx).continuousAt.continuousWithinAt)
    (fun x hx => (hq x (Set.Ico_subset_Icc_self hx)).hasDerivWithinAt)
  intro h hh
  have hc := hconst h hh
  -- q(0) = p(0) · 1^{-n} = p₀
  simp only [mul_zero, zero_div, sub_zero, one_rpow, mul_one, hp0] at hc
  -- hence p(h) · (θ^n)⁻¹ = p₀, i.e. p(h) = p₀ θ^n
  have hθ := hθpos h hh
  rw [rpow_neg hθ.le] at hc
  unfold pressureTropo
  rw [← hc, inv_mul_cancel_right₀ (rpow_pos_of_pos hθ _).ne']

/-- **Theorem 4 (density ratio).**  With `ρ = p / (R T)` and `T = T₀ - L h`,
`σ = ρ(h) / ρ₀ = (1 - L h / T₀)^{g/(L R) - 1}`. -/
theorem density_ratio (hp0 : 0 < p0) (hT0 : 0 < T0) (hR : 0 < R) {h : ℝ} (hLh : L * h < T0) :
    (pressureTropo p0 T0 L g R h / (R * (T0 - L * h))) / (p0 / (R * T0))
      = (1 - L * h / T0) ^ (g / (L * R) - 1) := by
  have hθ := theta_pos hT0 hLh
  unfold pressureTropo
  rw [rpow_sub_one hθ.ne']
  generalize (1 - L * h / T0) ^ (g / (L * R)) = X
  have hθ' := hθ.ne'
  have hT : T0 - L * h ≠ 0 := by linarith
  have hT0' := hT0.ne'
  have hR' := hR.ne'
  have hp0' := hp0.ne'
  field_simp <;> ring

end AircraftPerformance
