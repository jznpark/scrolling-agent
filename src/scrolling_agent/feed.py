"""Stochastic feed generator (step 1 — not implemented yet).

Plan (draft section 06):
    z_{i+1} = z_i with prob. kappa, otherwise z_{i+1} ~ Bernoulli(p)
    log A_i      = a0 + a1 * z_i + sigma_A * eps
    log nu_i     = m_nu + d_nu * z_i + sigma_nu * eps
    log lambda_i = m_lambda + sigma_lambda * eps
Requirements: reproducible from a seed; a way to reorder an existing item set
(to separate supply composition from ordering); record the realised nu–A relation.
"""
