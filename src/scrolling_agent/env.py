"""Gymnasium environment (step 2 — not implemented yet).

Plan (draft sections 03–05, 08, 10):
    modes: LIST / WATCH (+ fixed-length transitions for click and back)
    actions: down(speed), up(speed), stay(None | i), back — with action masking
    LIST: move, update visibility, advance DDM of visible items with drift nu_i * g(|v|)
          and noise sigma * sqrt(dt); reveal (A_i, lambda_i) on first threshold hit
    WATCH: reward = (A/lambda) * exp(-lambda * w) * (1 - exp(-lambda * dt)); w += dt
    every step: reward -= step_cost; episode ends after N = T / dt steps
    observation: mode, scroll position, remaining time, visible items (id, position,
          evidence, exposure, done flag), known (A, lambda, position), w per item,
          previous action and reward — never the hidden A, lambda, nu of unknown items
"""
