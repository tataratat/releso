import pathlib
from typing import Literal, Union

import numpy as np
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
    window_size: Union[tuple[int, int], Literal["auto"]] = "auto",
    cut_off_point: int = np.iinfo(int).max,
):
    """Plot one or multiple episodes to check out the training progress.

    This function can already be called once the first results are in the
     episode log file, to check out progress during training. But, in general,
     this function is normally used to compare different runs to see
     differences in the learning progress. In general all values are plotted
     over the total number of steps taken during the training. The following
     variables are shown:

    1. Episode reward - Best used to see if the training is progressing
    2. Steps per Episode - Here you can see if the number of steps to solve
     the environment decreases over time.
    3. Mean step reward - The higher the better each action actually is.
     (Calculated value episode_reward/steps_per_episode)
    4. Episode End Reason - For each run multiple lines to show why the
     episode was terminated. !ATTENTION! the window here is 100 to calculate a
     crude percentage.
    5. Seconds per Step - Time each step took to compute. Should not have much
     variation in general. But use case specific can have differences.
    6. Wall time - How much time did elapse to this time step. (Integral of
      previous plot).

    To check out the training I like to look at all the plots at once to see if
     there is anything that should concern me. For publishing most values have
     no informational gain so feel free to adapt (Remove certain sub plot, etc)
      the function.

    Author: Clemens Fricke (clemens.david.fricke@tuwien.ac.at)

    Args:
        result_folders (dict[str, pathlib.Path]): Dict of name and path to each
          experiment to visualize.
        export_path (pathlib.Path): Path to export the visualization to. If
          file name the suffix will be used to determine the file format. If a
          folder is given a file 'episode_log.html' will be created.
        window (int, optional): Windowing is used to smooth the plots.
          Defaults to 5.
        window_size (Union[tuple[int, int], Literal['auto']], optional): Size
         of the figure used. For html 'auto' will adapt the size to the browser
         or container size. Defaults to "auto".
        cut_off_point (int, optional): Plot the episode only to a certain
         number of time steps. Defaults to max int.
    """
    end_episode = []
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
    for idx, dataframe in enumerate(df):
        fig.add_trace(
            go.Scatter(
                x=dataframe["total_timesteps"].iloc[: end_episode[idx]],
                y=dataframe["episode_reward"]
                .iloc[: end_episode[idx]]
                .rolling(window, window)
                .mean(),
                mode="lines",
                legendgroup=f"{n_env[idx]}",
                name=f"{n_env[idx]}",
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
                legendgroup=f"{n_env[idx]}",
                name=f"{n_env[idx]}",
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
                legendgroup=f"{n_env[idx]}",
                name=f"{n_env[idx]}",
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
                    line=dict(
                        color=plotly_colors[idx],
                        dash=marker,
                        width=4,
                    ),
                    legendgroup=f"{n_env[idx]}",
                    name=f"{unique_key}: {n_env[idx]}",
                    showlegend=False,
                ),
                row=4,
                col=1,
            )
        fig.add_trace(
            go.Scatter(
                x=dataframe["total_timesteps"].iloc[: end_episode[idx]],
                y=dataframe["time_delta"]
                .iloc[: end_episode[idx]]
                .dt.total_seconds()
                / dataframe["steps_in_episode"].iloc[: end_episode[idx]],
                mode="lines",
                legendgroup=f"{n_env[idx]}",
                name=f"{n_env[idx]}",
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
                legendgroup=f"{n_env[idx]}",
                name=f"{n_env[idx]}",
                line=dict(color=plotly_colors[idx]),
                showlegend=False,
            ),
            row=6,
            col=1,
        )

    for marker, unique_key in zip(plotly_lines, unique_values):
        fig.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                legendgroup="end_reason",
                legendgrouptitle=dict(text="Episode End Reasons"),
                name=f"{unique_key}",
                line=dict(
                    color="black",
                    dash=marker,
                    width=4,
                ),
                mode="lines",
                showlegend=True,
            ),
            row=4,
            col=1,
        )

    fig.update_traces(line=dict(width=0.6))
    if window_size[0] == "auto":
        fig.update_layout(autosize=True)
    else:
        fig.update_layout(height=window_size[0], width=window_size[1])
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
