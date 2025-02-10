import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

with st.sidebar:
    random_state = st.number_input("Random seed:", 1, 1_000, 42)
    n = st.number_input("User Sessions n:", 1, 100_000, 5_000)
    st.divider()
    st.text("Configure (known) bias:")
    theta = st.slider("Position Bias $\\theta$", 0.01, 1.0, value=0.5, step=0.01)

    st.divider()
    st.text("Configure an (unknown) item:")
    a_gamma = st.number_input("Prior relevance $a_{\\gamma}$", 1, 1_000, value=1)
    b_gamma = st.number_input("Prior non-relevance $b_{\\gamma}$", 1, 1_000, value=1)
    gamma = st.slider("Relevance $\\gamma$", 0.01, 1.0, value=0.6, step=0.01)

np.random.seed(random_state)
E = np.random.binomial(1, theta, size=n)
R = np.random.binomial(1, gamma, size=n)
C = E * R
m = len(C)
s = C.sum()


def posterior(theta, gamma, a_gamma, b_gamma, n, s):
    return (gamma ** (a_gamma + s -1)) * (1 - gamma) ** (b_gamma - 1) * (1 - theta * gamma) ** (n - s)

def log_posterior(theta, gamma, a_gamma, b_gamma, n, s, eps=1e-8):
    log_post = (a_gamma + s - 1) * np.log(gamma + eps) + (b_gamma - 1) * np.log(1 - gamma + eps) + (n - s) * np.log(1 - theta * gamma + eps)
    log_post = log_post - np.max(log_post)
    return np.exp(log_post)

gamma_grid = np.linspace(0, 1, 1_000)
gamma_post = log_posterior(theta, gamma_grid, a_gamma, b_gamma, m, s)
gamma_post = gamma_post / np.trapz(gamma_post, gamma_grid)

df = pd.DataFrame({
    "gamma": gamma_grid,
    "posterior": gamma_post,
})

chart = (alt.Chart(df).mark_line().encode(
    x=alt.X("gamma", title="gamma"),
    y=alt.Y("posterior", title="Posterior Density"),
) + alt.Chart(df.sort_values("posterior", ascending=False).head(1)).mark_circle().encode(
    x=alt.X("gamma", title="gamma"),
    y=alt.Y("posterior", title="Posterior Density"),
))

st.altair_chart(chart)

st.markdown("""
* Prior: $p(\gamma) = \\frac{1}{B(a_\gamma,b_\gamma)}\,\gamma^{a_\gamma-1}(1-\gamma)^{b_\gamma-1}, \quad 0 \le \gamma \le 1$\n
* Likelihood: $L(\gamma) = (\\theta \gamma)^s (1-\\theta\gamma)^{n-s}$\n
* Posterior: $p(\gamma \mid \\text{data}) \propto \\theta^{s} \gamma^{a_\gamma+s-1} (1-\gamma)^{b_\gamma-1} (1-\\theta\gamma)^{n-s}$
""")
