/-
# Proof 5 - The Breguet range and endurance equations

Roskam & Lan, Ch. 11 (Breguet equations, Eqns 11.57-11.63); Anderson, Sec. 5.13; notebook 02, Capitolo9.

Model: cruise at constant aerodynamic efficiency `E = L/D`, level flight (`T = D = W/E`),
constant specific fuel consumption. The weight decreases because fuel is burnt.

Two complementary rigorous derivations are given.

**(A) The weight as a function of time.**  For a jet, `dW/dt = - c_t T = -(c_t/E) W`.
    `weight_history`  : the ONLY solution of  `W' = -c W`  is  `W(t) = W(0) e^{-c t}`
                        (uniqueness: `W(t) e^{c t}` has zero derivative, hence is constant);
    `endurance_time`  : the weight `W₁` is reached at  `t₁ = ln(W₀/W₁) / c`
                        -> jet endurance  `t = (E / c_t) ln(W₀/W₁)`;
    `breguet_range_jet`: at constant speed `x = V t`
                        -> jet range  `R = (V E / c_t) ln(W₀/W₁)`.

**(B) The distance as a function of the weight.**  For a propeller airplane,
`dx/dW = -(η_p E / c_p) / W` (distance flown per unit weight of fuel).
    `integral_const_mul_inv` : `∫_{W₁}^{W₀} k / W dW = k ln(W₀/W₁)`;
    `breguet_of_hasDerivAt`  : if `dx/dW = -k/W` on `[W₁, W₀]` then `x(W₁) - x(W₀) = k ln(W₀/W₁)`
                               (fundamental theorem of calculus)
                               -> propeller range  `R = (η_p E / c_p) ln(W₀/W₁)`;
    `log_ratio_pos`          : the range is positive whenever fuel is burnt (`W₁ < W₀`).
-/
import Mathlib

namespace AircraftPerformance

open Real

/-! ### (A) Weight history and endurance -/

/-- **Theorem 1 (unique solution of `W' = -c W`).**  If `W` is differentiable everywhere
with `W'(t) = -c W(t)`, then `W(t) = W(0) e^{-c t}`. -/
theorem weight_history {W : ℝ → ℝ} {c : ℝ} (hW : ∀ t, HasDerivAt W (-c * W t) t) (t : ℝ) :
    W t = W 0 * exp (-c * t) := by
  -- g(t) = W(t) e^{c t} has derivative  -c W e^{ct} + W e^{ct} c = 0 ...
  have hg : ∀ s, HasDerivAt (fun y => W y * exp (c * y)) 0 s := by
    intro s
    have h := (hW s).mul (((hasDerivAt_id' s).const_mul c).exp)
    convert h using 1
    ring
  -- ... so g is constant (a function with zero derivative on ℝ is constant)
  have hconst := is_const_of_deriv_eq_zero (fun s => (hg s).differentiableAt)
    (fun s => (hg s).deriv) t 0
  simp only [mul_zero, exp_zero, mul_one] at hconst
  -- hconst : W t · e^{c t} = W 0;  multiply by e^{-c t}
  rw [← hconst, mul_assoc, ← exp_add, show c * t + -c * t = 0 by ring, exp_zero, mul_one]

/-- **Theorem 2 (endurance).**  If `W' = -c W` with `c > 0`, `W(0) > 0`, and the weight
`W₁ > 0` is reached at time `t₁`, then `t₁ = ln(W(0) / W₁) / c`. -/
theorem endurance_time {W : ℝ → ℝ} {c W₁ t₁ : ℝ} (hW : ∀ t, HasDerivAt W (-c * W t) t)
    (hc : 0 < c) (hW0 : 0 < W 0) (ht : W t₁ = W₁) :
    t₁ = log (W 0 / W₁) / c := by
  rw [weight_history hW t₁] at ht
  -- e^{-c t₁} = W₁ / W₀
  have he : exp (-c * t₁) = W₁ / W 0 := by
    rw [eq_div_iff hW0.ne']
    linarith
  -- -c t₁ = ln(W₁ / W₀)
  have hl : -c * t₁ = log (W₁ / W 0) := by rw [← he, log_exp]
  -- ln(W₀ / W₁) = - ln(W₁ / W₀)
  have hinv : log (W 0 / W₁) = -log (W₁ / W 0) := by rw [← log_inv, inv_div]
  rw [eq_div_iff hc.ne', hinv, ← hl]
  ring

/-- **Corollary (jet Breguet range and endurance).**  With `c = c_t / E` (from
`dW/dt = -c_t D = -(c_t / E) W`) the endurance is `(E / c_t) ln(W₀/W₁)` and, at constant
speed `V`, the range `V t₁` is `(V E / c_t) ln(W₀/W₁)`. -/
theorem breguet_range_jet {W : ℝ → ℝ} {ct E V W₁ t₁ : ℝ}
    (hW : ∀ t, HasDerivAt W (-(ct / E) * W t) t) (hct : 0 < ct) (hE : 0 < E)
    (hW0 : 0 < W 0) (ht : W t₁ = W₁) :
    t₁ = E / ct * log (W 0 / W₁) ∧ V * t₁ = V * E / ct * log (W 0 / W₁) := by
  have h := endurance_time hW (div_pos hct hE) hW0 ht
  -- ln(W₀/W₁) / (c_t / E) = (E / c_t) ln(W₀/W₁)
  rw [div_div_eq_mul_div] at h
  constructor
  · rw [h]
    ring
  · rw [h]
    ring

/-! ### (B) Distance as a function of weight (propeller form) -/

/-- **Theorem 3 (the Breguet integral).**  `∫_{W₁}^{W₀} k / W dW = k ln(W₀ / W₁)`. -/
theorem integral_const_mul_inv (k : ℝ) {W₀ W₁ : ℝ} (h₀ : 0 < W₀) (h₁ : 0 < W₁) :
    ∫ W in W₁..W₀, k * W⁻¹ = k * log (W₀ / W₁) := by
  rw [intervalIntegral.integral_const_mul, integral_inv_of_pos h₁ h₀]

/-- **Theorem 4 (range from the differential relation).**  If the distance flown `x(W)`
satisfies `dx/dW = -k / W` for every weight between `W₁` and `W₀` (both positive), then
`x(W₁) - x(W₀) = k ln(W₀ / W₁)`. For a propeller airplane `k = η_p E / c_p`. -/
theorem breguet_of_hasDerivAt {x : ℝ → ℝ} {k W₀ W₁ : ℝ} (h₀ : 0 < W₀) (h₁ : 0 < W₁)
    (hx : ∀ W ∈ Set.uIcc W₁ W₀, HasDerivAt x (-k * W⁻¹) W) :
    x W₁ - x W₀ = k * log (W₀ / W₁) := by
  -- every weight between W₁ and W₀ is positive (so 1/W is continuous there)
  have hpos : ∀ W ∈ Set.uIcc W₁ W₀, 0 < W := by
    intro W hW
    rcases Set.mem_uIcc.mp hW with h | h <;> linarith [h.1]
  have hcont : ContinuousOn (fun W => -k * W⁻¹) (Set.uIcc W₁ W₀) :=
    continuousOn_const.mul (continuousOn_id.inv₀ fun W hW => (hpos W hW).ne')
  -- fundamental theorem of calculus:  ∫ x'(W) dW = x(W₀) - x(W₁)
  have hftc := intervalIntegral.integral_eq_sub_of_hasDerivAt (f' := fun W => -k * W⁻¹)
    hx hcont.intervalIntegrable
  rw [integral_const_mul_inv (-k) h₀ h₁] at hftc
  linarith

/-- **Theorem 5.**  If fuel is burnt (`0 < W₁ < W₀`) the logarithm, hence the range and the
endurance, are positive. -/
theorem log_ratio_pos {W₀ W₁ : ℝ} (h₁ : 0 < W₁) (h : W₁ < W₀) : 0 < log (W₀ / W₁) :=
  log_pos ((one_lt_div h₁).mpr h)

end AircraftPerformance
