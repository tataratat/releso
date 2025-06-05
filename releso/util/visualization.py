import pathlib
from typing import Literal, Union

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from pandas.errors import EmptyDataError
from plotly.graph_objects import Figure
from plotly.subplots import make_subplots

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


def export_figure(
    fig: Figure,
    export_path: pathlib.Path,
    default_filename: str,
) -> None:
    """Exports a given figure to a specified path.
    The function checks the file extension of the export path and saves the
    figure accordingly. If the path is a directory, it saves the figure with
    the specified default filename in that directory. If the path contains a
    file, it saves the figure with the specified extension if supported or
    prints an error message if the extension is not recognized.

    Args:
        fig (plotly.graph_objs.Figure): The figure to be exported.
        export_path (pathlib.Path): The path where the figure should be saved.
        default_filename (str): The default filename to fall back to if the
          path is a directory or if the file extension is not recognized.
    """
    # Export the plot as the suffix or as "episode_log.html"
    suffix = export_path.suffix
    if suffix == ".html":
        fig.write_html(export_path)
    elif suffix in [".png", ".jpg", ".jpeg", ".webp", ".svg", ".pdf"]:
        fig.write_image(export_path)
    elif export_path.is_dir():
        fig.write_html(export_path / default_filename)
    else:
        if not (export_path.parent / default_filename).exists():
            print(
                "File path suffix is not known nor is the file path a directory. "
                f"Saving to {export_path.parent / default_filename}"
            )
            fig.write_html(export_path.parent / default_filename)
        else:
            print(
                "File path suffix is not know nor is the file path a "
                "directory. Not saving."
            )


def plot_episode_log(
    result_folders: dict[str, pathlib.Path],
    window: int = 5,
    window_size: Union[tuple[int, int], Literal["auto"]] = "auto",
    cut_off_point: int = np.iinfo(int).max,
) -> Figure:
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
        window (int, optional): Windowing is used to smooth the plots.
          Defaults to 5.
        window_size (Union[tuple[int, int], Literal['auto']], optional): Size
         of the figure used. For html 'auto' will adapt the size to the browser
         or container size. Defaults to "auto".
        cut_off_point (int, optional): Plot the episode only to a certain
         number of time steps. Defaults to max int.

    Returns:
        fig (plotly.graph_objects.Figure): Plotly figure object with the
         requested plots for further customization or export.
    """
    end_episode = []
    df: list[pd.DataFrame] = []
    # make a separate list for just the names of the experiments
    n_env: list[str] = list(result_folders.keys())
    # set of all episode end reasons
    unique_values = set()
    for idx, folder in enumerate(result_folders.values()):
        try:
            temp_df = pd.read_csv(folder / "episode_log.csv", index_col=0)
        except EmptyDataError as err:
            print(
                f"Error occurred with data in folder {folder}. The error is {err}"
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
    # plot each experiment into each sub plot, the idx is used to determine the
    # episode end and the episode name
    for idx, dataframe in enumerate(df):
        # first subplot
        fig.add_trace(
            go.Scatter(
                x=dataframe["total_timesteps"].iloc[: end_episode[idx]],
                y=dataframe["episode_reward"]
                .iloc[: end_episode[idx]]
                .rolling(window, window)
                .mean(),
                mode="lines",
                legendgroup=f"{n_env[idx]}",  # Used so that each experiment can be disabled
                name=f"{n_env[idx]}",  # name the legend item
                line=dict(color=plotly_colors[idx]),
                showlegend=True,  # show legend only for first subplot
                customdata=dataframe.index,
                hovertemplate="(%{x:d},%{y:.2f}) Episode: %{customdata:d}",
            ),
            row=1,
            col=1,
        )
        # second subplot
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
        # third subplot
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
        # fourth subplot - episode end reason iterate through all reasons
        # legend for this plot is created at the end
        for marker, unique_key in zip(plotly_lines, unique_values):
            fig.add_trace(
                go.Scatter(
                    x=dataframe["total_timesteps"].iloc[: end_episode[idx]],
                    # window is 100 to calculate a crude percentage approximation
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
        # fith subplot
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
        # sixth subplot
        fig.add_trace(
            go.Scatter(
                x=dataframe["total_timesteps"].iloc[: end_episode[idx]],
                y=dataframe["run_time"]
                .iloc[: end_episode[idx]]
                .dt.total_seconds()
                / 3600.0,
                mode="lines",
                legendgroup=f"{n_env[idx]}",
                name=f"{n_env[idx]}",
                line=dict(color=plotly_colors[idx]),
                showlegend=False,
            ),
            row=6,
            col=1,
        )

    # create end reason legend
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

    # adapt figure size
    fig.update_traces(line=dict(width=0.6))
    if not isinstance(window_size, list) and window_size == "auto":
        fig.update_layout(autosize=True)
    else:
        fig.update_layout(height=window_size[0], width=window_size[1])
    # update axis labels
    fig.update_xaxes(title_text="Total time steps", row=6, col=1)
    fig.update_yaxes(title_text="Episode reward", row=1, col=1)
    fig.update_yaxes(title_text="Steps<br>per episode", row=2, col=1)
    fig.update_yaxes(title_text="Mean step<br>reward", row=3, col=1)
    fig.update_yaxes(title_text="Episode<br>end reason", row=4, col=1)
    fig.update_yaxes(title_text="Seconds<br>per step [s]", row=5, col=1)
    fig.update_yaxes(title_text="Wall time [h]", row=6, col=1)

    return fig


def get_data_from_dataframe(
    df: pd.DataFrame,
    name: str,
    selector: list[tuple[str, tuple[int, int]]],
) -> pd.DataFrame:
    """Get data from a dataframe based on the provided selector.

    This function extracts specific data from a given dataframe based on the
    provided selector. The selector is a list of tuples, where each tuple
    contains a string and a tuple of two integers. The string indicates the
    type of data to extract (e.g., "obs", "info"), and the tuple of integers
    specifies the range of indices to extract. The function returns a new
    dataframe containing the extracted data.
    The function is designed to handle different types of data extraction,
    including observations and information from the dataframe. It uses
    recursion to navigate through the dataframe structure (and subsequent dict)
    and extract the desired data.

    Args:
        df (pd.DataFrame): StepDataFrame containing the data to be extracted.
        name (str): Name of field the extracted data should be stored in.
        selector (list[tuple[str, tuple[int, int]]]): List of tuples
          containing the type of data to extract and the range of indices.

    Returns:
        pd.DataFrame: DataFrame containing the extracted data.
    """

    def get_elem_from_crums(df: pd.DataFrame, crums: list[str]) -> np.ndarray:
        """Get the element from the dataframe/dictionary based on the crums."""
        if len(crums) == 1:
            return df.apply(lambda x: x[crums[0]])
        else:
            return get_elem_from_crums(
                df.apply(lambda x: x[crums[0]]), crums[1:]
            )

    objective_id = 0
    ret_df = []
    for element in selector:
        if element[0] == "obs":
            selected = np.vstack(df["obs"].values).T[
                element[1][0] : element[1][1]
            ]
            ret_df.append(
                pd.DataFrame(
                    selected.T,
                    columns=[
                        f"{name}_{objective_id + i}"
                        for i in range(len(selected))
                    ],
                )
            )
            objective_id += len(selected)
        elif element[0].startswith("infos"):
            if element[0].endswith("geometry_information"):
                selected = np.vstack(
                    df["infos"].apply(
                        lambda x: np.array(x["geometry_information"]).flatten()
                    )
                ).T[element[1][0] : element[1][1]]
            else:
                crums = element[0].split(".")
                selected = np.vstack(
                    get_elem_from_crums(df["infos"], crums[1:])
                ).T[element[1][0] : element[1][1]]
            ret_df.append(
                pd.DataFrame(
                    selected.T,
                    columns=[
                        f"{name}_{objective_id + i}"
                        for i in range(len(selected))
                    ],
                )
            )
            objective_id += len(selected)
    return pd.concat(ret_df, axis=1)


def plot_step_log(
    step_log_file: pathlib.Path,
    env_id: int,
    episode_start: int = 0,
    episode_end: int = np.iinfo(int).max,
    episode_step: int = 1,
    figure_size: Union[tuple[int, int], Literal["auto"]] = "auto",
    objective_observation: list[tuple[str, tuple[int, int]]] = [
        ("obs", (0, 1))
    ],
    design_variable: list[tuple[str, tuple[int, int]]] = [
        ("obs", (1, None)),
    ],
) -> Figure:
    """Plot the step log data of a single run for multiple episodes.

    This function is used to visualize the step log data of a single run
    with the intention of understanding the training progression and the
    policy that is learned by the agent.
    It creates an interactive plot with two subplots:

    1. The first subplot shows the reward and objective value(s) over the number
       of timesteps within the current episode.
    2. The second subplot shows the values of the design variables (observations)
       over the number of timesteps within the current episode.
       The plot is interactive, allowing users to select different episodes via a
       slider to view the corresponding data.

    Since plotly does not recompute data when the user interacts with the
    plot, the data for all episodes is precomputed and stored in the figure.
    This can result in large file sizes, especially when the optimization
    contained a lot of design parameters and when the chosen episode range is
    large. Please be aware of this when using the function.

    Author: Daniel Wolff (d.wolff@unibw.de)

    Args:
        step_log_file (pathlib.Path): Path to the step log file.
        env_id (int): ID of the environment whose results should be visualized.
          This parameter is only relevant in the case of multi-environment
          training. If the training did not use multiple environments, this
          parameter should be set to 0.
        episode_start (int, optional): First episode that should be included in
          the visualization. Defaults to 0.
        episode_end (int, optional): Last episode that should be included in
          the visualization. If None, the maximum episode number is used.
          Defaults to None.
        episode_step (int, optional): Step size for selecting episodes.
          Defaults to 1, which means that every episode between the starting
          and final episode are visualized.
        figure_size (Union[tuple[int, int], Literal['auto']], optional):
          Size of the figure. If 'auto', the size will be adjusted to fit the
          container. Defaults to 'auto'.
        objective_observation (list[tuple[str, tuple[int, int]]]): List of
          tuples containing information from which column of the steplog data
          to extract the information about the objective (each row in a column
          generally contains a list of values) and the indices which entries
          from that list are supposed to be to extracted.
        design_variable (list[tuple[str, tuple[int, int]]]): List of tuples
          containing information from which column of the steplog data to
          extract the information about the design variables (each row in a
          column generally contains a list of values) and the indices which
          entries from that list are supposed to be to extracted.
    Returns:
        plotly.graph_objects.Figure: A Plotly figure object containing the
          interactive plot for further customization or export.
    """
    # Load the step log data from the provided path
    try:
        df_raw = pd.read_json(step_log_file, lines=True)
    except RuntimeError as err:
        print(
            f"Error occurred when trying to read {step_log_file}. The error is {err}"
        )
        return

    # Process the data for the visualization

    # Extract scalar reward and observation (we use new_obs for visualization)
    # The underlying assumption of the extraction is that we are always dealing
    # with a vectorized environment, i.e., that the rewards and observations
    # corresponding to each environment are collected in a list
    df_raw["episodes"] = df_raw["episodes"].apply(lambda x: x[env_id])
    df_raw["reward"] = df_raw["rewards"].apply(lambda x: x[env_id])
    df_raw["obs"] = df_raw["new_obs"].apply(lambda x: x[env_id])
    has_info = "infos" in df_raw.columns
    if has_info:
        df_raw["infos"] = df_raw["infos"].apply(lambda x: x[env_id])

    # Convert obs vector into columns
    # obs_array = np.vstack(df_raw["obs"].values)
    # obs_df = pd.DataFrame(
    #     obs_array, columns=[f"obs_{i}" for i in range(obs_array.shape[1])]
    # )
    # Extract the objective observation
    objectives = get_data_from_dataframe(
        df_raw,
        "objective",
        objective_observation,
    )

    # Extract the design variables
    design_vars = get_data_from_dataframe(
        df_raw,
        "design_variable",
        design_variable,
    )

    # Combine everything
    df = pd.concat(
        [df_raw["episodes"], df_raw["reward"], objectives, design_vars], axis=1
    )

    # Filter only the selected episodes
    max_idx = df["episodes"].max()
    if episode_end is None or episode_end > max_idx:
        episode_end = max_idx
    # selected episode numbers do not necessarily match the filter applied. The
    # filter does not directly filter for episode number but filters the whole
    # list of episodes, which can have missing episodes, due to episodes
    # generated outside the environment id chosen.
    selected_episodes = df["episodes"].unique()[
        episode_start : (episode_end + 1) : episode_step
    ]
    df = df[df["episodes"].isin(selected_episodes)]

    # Create the interactive visualization

    # Choose which obs dimensions to show in bottom subplot
    design_var_names = df.columns[
        df.columns.str.contains("design_variable_")
    ].tolist()

    # Choose which obs dimensions to show in top subplot
    objective_names = df.columns[
        df.columns.str.contains("objective_")
    ].tolist()

    # Get all unique episodes
    episodes = df["episodes"].unique()

    # Create subplot layout
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        specs=[
            [{"secondary_y": True}],
            [{}],
        ],  # Enable secondary y-axis for row 1
        subplot_titles=("Reward and Objective", "Design variables"),
        vertical_spacing=0.1,
    )

    # Build one trace group per episode
    for ep in episodes:
        ep_data = df[df["episodes"] == ep]
        steps_per_episode = list(range(len(ep_data)))

        # First subplot: reward and objective
        for i, obs_dim in enumerate(objective_names):
            fig.add_trace(
                go.Scatter(
                    x=steps_per_episode,
                    y=ep_data[obs_dim],
                    name=f"Objective_{i} (Ep. {ep})",
                    visible=(ep == episode_start),
                    # line=dict(color="green"),
                ),
                row=1,
                col=1,
            )
        fig.add_trace(
            go.Scatter(
                x=steps_per_episode,
                y=ep_data["reward"],
                name=f"Reward (Ep. {ep})",
                visible=(ep == episode_start),
                line=dict(color="blue"),
            ),
            row=1,
            col=1,
            secondary_y=True,
        )

        # Second subplot: selected observation dimensions
        for j, dim in enumerate(design_var_names):
            fig.add_trace(
                go.Scatter(
                    x=steps_per_episode,
                    y=ep_data[dim],
                    name=f"Design Var. {j} (Ep. {ep})",
                    visible=(ep == episode_start),
                ),  # one trace per dimension
                row=2,
                col=1,
            )

    # once all traces are added, set up slider steps
    sliders = []
    for id, ep in enumerate(episodes):
        # Create slider step
        sliders.append({
            "label": f"Episode {ep}",
            "method": "update",
            "args": [
                {"visible": [False] * len(fig.data)},
                {"title": f"Steplog Evaluation — Episode {ep}"},
            ],
        })
        # Make the traces for this episode visible
        n_traces_per_episode = 1 + len(objective_names) + len(design_var_names)
        for k in range(n_traces_per_episode):
            sliders[id]["args"][0]["visible"][
                id * n_traces_per_episode + k
            ] = True

    # Set up slider control
    fig.update_layout(
        sliders=[{"active": 0, "pad": {"t": 50}, "steps": sliders}],
        title="Steplog Evaluation",
        showlegend=True,
    )

    # rescale the figure
    if not isinstance(figure_size, list) and figure_size == "auto":
        fig.update_layout(autosize=True)
    else:
        fig.update_layout(height=figure_size[0], width=figure_size[1])

    # Set y-axis labbels
    fig.update_yaxes(title_text="Objective", row=1, col=1, secondary_y=False)
    fig.update_yaxes(title_text="Reward", row=1, col=1, secondary_y=True)
    fig.update_yaxes(title_text="Observation Value", row=2, col=1)

    return fig
