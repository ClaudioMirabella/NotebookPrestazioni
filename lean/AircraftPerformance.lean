/-
# AircraftPerformance - machine-checked proofs for the aircraft performance course

Each file proves one group of results used in the `PrestazioniRoskam` notebooks;
notebook `09_Rigorous_Proofs.ipynb` explains them. Build with `lake build`
(after `lake exe cache get`, which downloads the pre-compiled Mathlib).
-/
import AircraftPerformance.DragPolar     -- 1. optimum of the parabolic polar (point E)
import AircraftPerformance.Glide         -- 2. minimum glide angle, maximum glide distance
import AircraftPerformance.MinPower      -- 3. minimum power (point P), V_E / V_P = 3^(1/4)
import AircraftPerformance.JetClimb      -- 4. jet maximum rate of climb (Roskam Eqn 9.30)
import AircraftPerformance.Breguet       -- 5. Breguet range and endurance
import AircraftPerformance.TimeToClimb   -- 6. time to climb, unreachable absolute ceiling
import AircraftPerformance.Atmosphere    -- 7. ISA troposphere: existence and uniqueness
