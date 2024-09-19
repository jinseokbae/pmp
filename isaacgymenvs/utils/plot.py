import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np

#################### plot style config ####################
TITLE = "Sight Tracking Error"
X_LABEL = "Timestep"
Y_LABEL = "Tracking Error (rad)"

FONT_TITLE = {
    "family": "sans serif",
    "weight": "bold",
    "color": "white",
    "size": 40,
}
FONT_LABEL = {
    "family": "sans serif",
    "weight": "normal",
    "color": "white",
    "size": 25,
}
FONT_LEGEND = {
    "family": "sans serif",
    "weight": "normal",
    "size": 25,
}
FIG_WIDTH = 22
FIG_HEIGHT = 7
FIG_COLOR = "black"
FIG_ALPHA = 0

AXE_COLOR = "black"
AXE_ALHPA = 0
AXIS_LINEWIDTH = 3
AXIS_COLOR = "white"

TICK_LINEWIDTH = 3
TICK_LINELENGTH = 7
TICK_LABEL_SIZE = 20
TICK_COLOR = "white"

PAD_TITLE = 0
PAD_X_LABEL = 0
PAD_Y_LABEL = 20

LINE_WIDTH = 6
LINE_ALPHA = 0.9
LINE_COLOR_AMP = "red"
LINE_COLOR_OURS = "blue"
############################################################


def plot(datafile_ours, datafile_amp=None, save_video=False, video_filename="plot"):
    # load data
    data_ours = np.load(datafile_ours)
    num_frames = data_ours.shape[0]
    if datafile_amp is not None:
        data_amp = np.load(datafile_amp)
        assert data_amp.shape == data_ours.shape, f"Total frame counts are not same!"

    # initialize figure
    fig, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT))

    # set background color (fig, axe)
    fig.set_facecolor(FIG_COLOR)
    fig.set_alpha(FIG_ALPHA)
    ax.set_facecolor(AXE_COLOR)
    ax.set_alpha(AXE_ALHPA)

    # set border style
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_linewidth(AXIS_LINEWIDTH)
    ax.spines["bottom"].set_color(AXIS_COLOR)
    ax.spines["left"].set_linewidth(AXIS_LINEWIDTH)
    ax.spines["left"].set_color(AXIS_COLOR)

    # set tick style
    ax.tick_params(width=TICK_LINEWIDTH, length=TICK_LINELENGTH, labelsize=TICK_LABEL_SIZE, colors=TICK_COLOR)

    # set title & lable
    ax.set_title(TITLE, pad=PAD_TITLE, fontdict=FONT_TITLE)
    ax.set_xlabel(X_LABEL, labelpad=PAD_X_LABEL, fontdict=FONT_LABEL)
    ax.set_ylabel(Y_LABEL, labelpad=PAD_Y_LABEL, fontdict=FONT_LABEL)

    # set view limit
    ax.set_xlim(0, num_frames)
    ax.set_ylim(0, 3.5)

    # plot lines
    (line_ours,) = ax.plot(data_ours[:1], linewidth=LINE_WIDTH, alpha=LINE_ALPHA, c=LINE_COLOR_OURS)
    if datafile_amp is not None:
        (line_amp,) = ax.plot(data_amp[:1], linewidth=LINE_WIDTH, alpha=LINE_ALPHA, c=LINE_COLOR_AMP)
        ax.legend([line_amp, line_ours], ["AMP", "Ours"], loc="upper right", prop=FONT_LEGEND)

    def animate(t):
        line_ours.set_xdata(range(t))
        line_ours.set_ydata(data_ours[:t])
        if datafile_amp is not None:
            line_amp.set_xdata(range(t))
            line_amp.set_ydata(data_amp[:t])
            return (line_ours, line_amp)
        return (line_ours,)

    ani = animation.FuncAnimation(fig, animate, interval=5, frames=num_frames, blit=True, save_count=50)
    if save_video:
        writer = animation.FFMpegWriter(fps=60, metadata=dict(artist="Me"), bitrate=-1)
        ani.save(f"videos/{video_filename}.mp4", writer=writer)
    else:
        plt.show()


if __name__ == "__main__":
    from pathlib import Path

    ours_dir = Path("videos/1_locosight/locosight_ours")
    amp_dir = Path("videos/1_locosight/locosight_amp")
    video_filename_prefix = "locosight_graph"

    ours_datafiles = sorted(
        [file for file in ours_dir.glob("*.npy") if not file.name.startswith(".")],
        key=lambda file: int(file.stem.split("_")[-1][3:]),
    )
    amp_datafiles = sorted(
        [file for file in amp_dir.glob("*.npy") if not file.name.startswith(".")],
        key=lambda file: int(file.stem.split("_")[-1][3:]),
    )
    assert len(ours_datafiles) == len(amp_datafiles), "Total episode counts are not same!"

    for ours_datafile, amp_datafile in zip(ours_datafiles, amp_datafiles):
        ours_episode_idx = int(ours_datafile.stem.split("_")[-1][3:])
        amp_episode_idx = int(amp_datafile.stem.split("_")[-1][3:])
        assert ours_episode_idx == amp_episode_idx, "Episode indices are not same!"
        
        try:
            plot(
                datafile_ours=ours_datafile,
                datafile_amp=amp_datafile,
                save_video=True,
                video_filename=f"{video_filename_prefix}_epi{ours_episode_idx}",
            )
        except AssertionError as e:
            print(e)
            continue
