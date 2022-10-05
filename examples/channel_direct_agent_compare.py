import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from pandas.errors import EmptyDataError
from matplotlib.ticker import EngFormatter, PercentFormatter
from matplotlib import gridspec

window = 100
folder_paths = list()
n_env = []
# this is the one that is to be overwritten
folder_paths.append("path-to-main-folder")
n_env.append("run description")

marker = ["-", "-", (0, (5, 10)), (0, (5, 10)), (0, (3, 10, 1, 10)), (0, (3, 10, 1, 10)), (0, (1, 10)), (0, (1, 10))]
colors = ["r", "r","orange", "orange", "b", "b", "turquoise", "turquoise"]
end_episode = []
print(len(folder_paths), len(marker), len(n_env), len(colors))

cut_off_point = 350000
df = []
for idx, folder in enumerate(folder_paths):
  try:
    temp_df = pd.read_csv(folder+"/episode_log.csv", index_col=0)
  except EmptyDataError as err:
    print(f"Error occured with data in folder {folder}. The error is {err}")
    continue
  end_episode.append((temp_df.loc[temp_df["total_timesteps"]<cut_off_point].index)[-1])
  row = pd.to_datetime(temp_df["wall_time"])
  temp_df['wall_time'] = row
  temp_df['time_delta'] = row - row.shift()
  df.append(temp_df)



fig = plt.figure()
gs = gridspec.GridSpec(3, 1, height_ratios=[1, 1, 0.5])
ax0 = plt.subplot(gs[0])
ax0.set_xlim((0,cut_off_point+cut_off_point*0.05))

for idx, dataframe in enumerate(df):
    ax0.plot(dataframe["total_timesteps"].iloc[:end_episode[idx]],dataframe["episode_reward"].iloc[:end_episode[idx]].rolling(window, window).mean(), label=f"{n_env[idx]}"
    # ,linestyle=marker[idx]
    ,c=colors[idx]
    ,linewidth=0.6)
# if idx == (hdx-1):
#   break
# ax0.set_ylim((-30,10))
# ax0.set_yticks([-25,0])
# legend = plt.legend(loc="lower left", bbox_to_anchor=(0,1), framealpha=0.3)
# legend.set_title("Trained Agents")
# for t in legend.get_texts():
#  t.set_ha('left')
ax0.set_ylabel("Episode \nreward")
# ax0.set_xlabel("Total timesteps")
formatter1 = EngFormatter(places=0, sep="")  # U+2009
ax0.xaxis.set_major_formatter(formatter1)
# ax0.set_yticks([-10,0,5])
# ax0.legend()
for tick in ax0.get_yticklabels():
    tick.set_rotation(90)


## second plot
ax1 = plt.subplot(gs[1], sharex = ax0)
for idx, dataframe in enumerate(df):
    ax1.plot(dataframe["total_timesteps"].iloc[:end_episode[idx]],dataframe["steps_in_episode"].iloc[:end_episode[idx]].rolling(window, window).mean(), label=f"{n_env[idx]}"
    # ,linestyle=marker[idx]
    ,c=colors[idx]
    ,linewidth=0.6)
# if idx == (hdx-1):
# #   break
# ax1.set_ylim((-5,50))
# ax1.set_yticks([0,20,40,20])
for tick in ax1.get_yticklabels():
    tick.set_rotation(90)
ax1.set_ylabel("Steps per\n episode")
# ax1.set_xlabel("Total timesteps")
ax1.xaxis.set_major_formatter(formatter1)
plt.setp(ax0.get_xticklabels(), visible=False)
# remove last tick label for the second subplot
yticks = ax1.yaxis.get_major_ticks()
yticks[-1].label1.set_visible(False)


## third plot
# ax2 = plt.subplot(gs[2], sharex = ax0)
# for idx, dataframe in enumerate(df):
    # ax2.plot(dataframe["total_timesteps"].iloc[:end_episode[idx]],
    # # (((dataframe.replace('goal_achieved',np.NaN)["episode_end_reason"].iloc[:end_episode[idx]].rolling(window,window).count()/window)))
    # ((dataframe.replace('ExecutionFailed-main_solver',np.NaN)["episode_end_reason"].iloc[:end_episode[idx]].rolling(window,window).count()/window)-1)*-1
    # , label=f"{n_env[idx]}"
    # # ,linestyle=marker[idx]
    # ,c=colors[idx]
    # ,linewidth=0.6)
# # if idx == (hdx-1):
# #   break
# ax2.set_ylim((-0.1,1.1))
# ax2.set_yticks([0,1,0])
# for tick in ax2.get_yticklabels():
#     tick.set_rotation(90)
# ax2.set_ylabel("Solver\n error")
# ax2.set_xlabel("Total timesteps")
# formatter2 = PercentFormatter(1)  # U+2009
# ax2.yaxis.set_major_formatter(formatter2)
# plt.setp(ax1.get_xticklabels(), visible=False)
# # remove last tick label for the second subplot
# yticks = ax2.yaxis.get_major_ticks()
# yticks[-1].label1.set_visible(False)

plt.tight_layout()
plt.subplots_adjust(hspace=.0)
# plt.savefig(f"results_icms.pgf", transparent=True)
plt.savefig(f"results_icms.png", transparent=False)
plt.close()

# plt.figure(figsize=(6.2,3))
# for idx, dataframe in enumerate(df):
#   plt.plot(((dataframe["wall_time"]-dataframe["wall_time"][0])/np.timedelta64(1, 'h')).iloc[:end_episode[idx]],dataframe["episode_reward"].iloc[:end_episode[idx]].rolling(window, window).mean(), label=f"{n_env[idx]}"
#   ,c=colors[idx]
#   ,linewidth=0.6)
#   print(n_env[idx], ((dataframe["wall_time"]-dataframe["wall_time"][0])/np.timedelta64(1, 'h')).iloc[end_episode[idx]])
# plt.legend()
# plt.xlabel(r"Training time [h]")
# plt.ylabel("Episode reward")
# for tick in plt.gca().get_yticklabels():
#     tick.set_rotation(90)

# plt.tight_layout()
# plt.savefig("compare_agents_direct_time.pgf", transparent=True)
# plt.savefig("compare_agents_direct_time.png", transparent=False)