from utils.engine import simulate_price_paths, portfolio_trajectory
import pandas as pd
import altair as alt


def plot_data(weights, mu, cov, horizon):
    price_paths = simulate_price_paths(
        weights,
        mu,
        cov,
        horizon_days=horizon,
        n_sims=5,
    )
    port_paths = portfolio_trajectory(price_paths, weights)
    df_plot = pd.DataFrame(
        port_paths.T, columns=[f"Sim {i + 1}" for i in range(port_paths.shape[0])]
    )
    df_plot["Day"] = range(horizon + 1)

    df_long = df_plot.melt(id_vars="Day", var_name="Simulation", value_name="Price")
    line_chart = (
        alt.Chart(df_long)
        .mark_line()
        .encode(
            x=alt.X("Day:Q", title="Trading day"),
            y=alt.Y(
                "Price:Q",
                title="Portfolio value (relative)",
                scale=alt.Scale(domain=[0.95, 1.05]),
            ),
            color=alt.Color("Simulation:N", legend=alt.Legend(title="Simulation")),
            tooltip=["Simulation", "Day", alt.Tooltip("Price:Q", format=".4f")],
        )
        .properties(width=700, height=400)
        .interactive()
    )
    return line_chart
