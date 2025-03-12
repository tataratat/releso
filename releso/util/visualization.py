import pathlib

import pandas as pd
from pandas.errors import EmptyDataError

from releso.util.module_import_raiser import ModuleImportRaiser

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
except ModuleNotFoundError:
    go = ModuleImportRaiser("plotly")
    make_subplots = ModuleImportRaiser("plotly")


plotly_colors = [
    "red",
    "blue",
    "green",
    "orange",
    "purple",
    "brown",
    "pink",
    "gray",
    "olive",
    "cyan",
]
plotly_lines = ["solid", "dash", "dot", "dashdot"]


def plot_episode_log(
    result_folders: dict[str, pathlib.Path],
    export_path: pathlib.Path,
    window: int = 5,
):
    end_episode = []

    cut_off_point = 350000
    df: list[pd.DataFrame] = []
    n_env: list[str] = list(result_folders.keys())
    unique_values = set()
    for idx, folder in enumerate(result_folders.values()):
        try:
            temp_df = pd.read_csv(folder / "episode_log.csv", index_col=0)
        except EmptyDataError as err:
            print(
                f"Error occured with data in folder {folder}. The error is {err}"
            )
            continue
        end_episode.append(
            (temp_df.loc[temp_df["total_timesteps"] < cut_off_point].index)[-1]
        )
        row = pd.to_datetime(temp_df["wall_time"])
        temp_df["wall_time"] = row
        temp_df["run_time"] = row - row.iloc[0]
        temp_df["time_delta"] = row - row.shift()
        temp_df["mean_step_reward"] = (
            temp_df["episode_reward"] / temp_df["steps_in_episode"]
        )
        unique_values.update(temp_df["episode_end_reason"].unique())
        df.append(temp_df)

    fig = make_subplots(
        rows=6, cols=1, shared_xaxes=True, vertical_spacing=0.01
    )
    first_run = True
    for idx, dataframe in enumerate(df):
        print(idx)
        fig.add_trace(
            go.Scatter(
                x=dataframe["total_timesteps"].iloc[: end_episode[idx]],
                y=dataframe["episode_reward"]
                .iloc[: end_episode[idx]]
                .rolling(window, window)
                .mean(),
                mode="lines",
                name=f"{n_env[idx]}",
                legendgroup="Tests",
                legendgrouptitle=dict(text="Tests"),
                line=dict(color=plotly_colors[idx]),
                showlegend=True,
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=dataframe["total_timesteps"].iloc[: end_episode[idx]],
                y=dataframe["steps_in_episode"]
                .iloc[: end_episode[idx]]
                .rolling(window, window)
                .mean(),
                mode="lines",
                name=f"{n_env[idx]}",
                legendgroup="Tests",
                line=dict(color=plotly_colors[idx]),
                showlegend=False,
            ),
            row=2,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=dataframe["total_timesteps"].iloc[: end_episode[idx]],
                y=dataframe["mean_step_reward"]
                .iloc[: end_episode[idx]]
                .rolling(window, window)
                .mean(),
                mode="lines",
                name=f"{n_env[idx]}",
                legendgroup="Tests",
                line=dict(color=plotly_colors[idx]),
                showlegend=False,
            ),
            row=3,
            col=1,
        )
        for marker, unique_key in zip(plotly_lines, unique_values):
            fig.add_trace(
                go.Scatter(
                    x=dataframe["total_timesteps"].iloc[: end_episode[idx]],
                    y=(dataframe["episode_end_reason"] == unique_key)
                    .iloc[: end_episode[idx]]
                    .rolling(100, 0)
                    .mean(),
                    legendgroup="end_reason",
                    legendgrouptitle=dict(text="Episode End Reasons"),
                    name=f"{unique_key}",
                    line=dict(
                        color=plotly_colors[idx],
                        dash=marker,
                        width=4,
                    ),
                    showlegend=first_run,
                ),
                row=4,
                col=1,
            )
        first_run = False
        fig.add_trace(
            go.Scatter(
                x=dataframe["total_timesteps"].iloc[: end_episode[idx]],
                y=dataframe["time_delta"]
                .iloc[: end_episode[idx]]
                .dt.total_seconds()
                / dataframe["steps_in_episode"].iloc[: end_episode[idx]],
                mode="lines",
                name=f"{n_env[idx]}",
                legendgroup="Tests",
                line=dict(color=plotly_colors[idx]),
                showlegend=False,
            ),
            row=5,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=dataframe["total_timesteps"].iloc[: end_episode[idx]],
                y=dataframe["run_time"].iloc[: end_episode[idx]],
                mode="lines",
                name=f"{n_env[idx]}",
                legendgroup="Tests",
                line=dict(color=plotly_colors[idx]),
                showlegend=False,
            ),
            row=6,
            col=1,
        )

        print(f"n_env[idx]: {n_env[idx]}")
    fig.update_traces(line=dict(width=0.6))
    fig.update_layout(height=800, width=1000)
    fig.update_xaxes(title_text="Total time steps", row=6, col=1)
    fig.update_yaxes(title_text="Episode reward", row=1, col=1)
    fig.update_yaxes(title_text="Steps<br>per episode", row=2, col=1)
    fig.update_yaxes(title_text="Mean step<br>reward", row=3, col=1)
    fig.update_yaxes(title_text="Episode<br>end reason", row=4, col=1)
    fig.update_yaxes(title_text="Seconds<br>per step [s]", row=5, col=1)
    fig.update_yaxes(title_text="Wall time", row=6, col=1)

    # Export the plot as html
    suffix = export_path.suffix
    if suffix == ".html":
        fig.write_html(export_path)
    elif suffix in ["png", "jpg", "jpeg", "webp", "svg", "pdf"]:
        fig.write_image(export_path)
    elif export_path.is_dir():
        fig.write_html(export_path / "episode_log.html")
    else:
        if not (export_path.parent / "episode_log.html").exists():
            print(
                "File path suffix is not html nor is the file path a directory. "
                f"Saving to {export_path.parent / 'episode_log.html'}"
            )
            fig.write_html(export_path.parent / "episode_log.html")
        else:
            print(
                "File path suffix is not know nor is the file path a "
                "directory. Not saving."
            )
