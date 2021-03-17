import matplotlib
import matplotlib._color_data as mcd
import matplotlib.patches as mpatch
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter

matplotlib.rcParams['ps.useafm'] = True
matplotlib.rcParams['pdf.use14corefonts'] = True
# matplotlib.rcParams['text.usetex'] = True


overlap = {name for name in mcd.CSS4_COLORS
           if "xkcd:" + name in mcd.XKCD_COLORS}

font_size = 14  # 14
colors = {
    "DBEst_1k": mcd.XKCD_COLORS['xkcd:coral'],
    "DBEst_10k": mcd.XKCD_COLORS['xkcd:orange'],  # blue
    "DBEst_100k": mcd.XKCD_COLORS['xkcd:orangered'],  # green
    "DBEst_1m": mcd.XKCD_COLORS['xkcd:red'],  # yellow
    "BlinkDB_1k": mcd.XKCD_COLORS['xkcd:lightblue'],  # red
    "BlinkDB_10k": mcd.XKCD_COLORS['xkcd:turquoise'],  # red
    "BlinkDB_100k": mcd.XKCD_COLORS['xkcd:teal'],  # cyan
    "BlinkDB_1m": mcd.XKCD_COLORS['xkcd:azure'],  # magenta
    "BlinkDB_5m": mcd.XKCD_COLORS['xkcd:blue'],  # red
    "BlinkDB_26m": mcd.XKCD_COLORS['xkcd:darkblue'],  # red
    "green1":mcd.XKCD_COLORS['xkcd:pale green'],
    "green2":mcd.XKCD_COLORS['xkcd:lime'],
    "green3":mcd.XKCD_COLORS['xkcd:neon green'],
    "lightgreen": mcd.XKCD_COLORS['xkcd:lightgreen'],
    "green": mcd.XKCD_COLORS['xkcd:green'],
    "orange": mcd.XKCD_COLORS['xkcd:orange'],
    "orangered": mcd.XKCD_COLORS['xkcd:orangered'],
    "red": mcd.XKCD_COLORS['xkcd:red'],
}
alpha = {
    "1": 0.1,
    "2": 0.3,
    "3": 0.5,
    "4": 0.7,
    "5": 0.9,
    '6': 1.0
}

def add_value_labels(ax, spacing=5,fontsize=12,b_float=True,b_percent=False):
    """Add labels to the end of each bar in a bar chart.

    Arguments:
        ax (matplotlib.axes.Axes): The matplotlib object containing the axes
            of the plot to annotate.
        spacing (int): The distance between the labels and the bars.
    """
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    # For each bar: Place a label
    for rect in ax.patches:
        # Get X and Y placement of label from rect.
        y_value = rect.get_height()
        x_value = rect.get_x() + rect.get_width() / 2

        # Number of points between bar and label. Change to your liking.
        space = spacing
        # Vertical alignment for positive values
        va = 'bottom'

        # If value of bar is negative: Place label below bar
        if y_value < 0:
            # Invert space to place label below
            space *= -1
            # Vertically align label at top
            va = 'top'

        # Use Y value as label and format number with one decimal place

        if b_float:
            if b_percent:
                label = '%.2f%%' % (y_value * 100)
            else:
                label = "{:.2f}".format(y_value)
        else:
            label = "{:d}".format(y_value)

        # Create annotation
        ax.annotate(
            label,                      # Use `label` as label
            (x_value, y_value),         # Place label at end of the bar
            xytext=(0, space),          # Vertically shift label by `space`
            textcoords="offset points", # Interpret `xytext` as offset in points
            ha='center',                # Horizontally center label
            va=va,
            fontsize=fontsize)                      # Vertically align label differently for
                                        # positive and negative values.



def plt_space():
    plt.rcParams.update({'font.size': 20})  # 20
    width = 0.25
    data = [
        [0.191, 43.4,1.7,0.0254, 0.0757],
        [145.42, 145.42, 145.42, 145.42, 145.42],
        [418724.985,418724.985,418724.985, 57219.51,57219.51],
    ]

    X = np.arange(5)

    fig, ax = plt.subplots()

    p1 = plt.bar(
        X + 0.00, data[0], color=colors["DBEst_1k"], width=width, alpha=0.9)
    p2 = plt.bar(
        X + width, data[1], color=colors["BlinkDB_1k"], width=width, alpha=0.5)
    p3 = plt.bar(
        X + 2*width, data[2], color=colors["green1"], width=width, alpha=0.7)

    plt.legend((p1[0], p2[0],p3[0]),  # , p3[0]),#, p4[0],p5[0]),
               ('DBEst++ Model', 'Sample', 'Actual Table' ), loc='upper left', prop={'size': 12})

    plt.xticks(X + 1.0 * width, ("1", '2', '3','4','5'))
    ax.set_ylabel("Space Overheads (MB)")
    ax.set_xlabel("Query Template")
    ax.set_yscale('log')
    # formatter = FuncFormatter(to_percent)
    # ax.yaxis.set_major_formatter(formatter)
    # plt.subplots_adjust(bottom=0.11)
    # plt.subplots_adjust(left=0.12)
    plt.subplots_adjust(bottom=0.15)
    plt.subplots_adjust(left=0.15)

    add_value_labels(ax, b_float=True, b_percent=False, fontsize=12)
    plt.savefig("/home/quincy/Pictures/space.pdf")
    print("figure saved.")

    plt.show()


def plt_response_time():
    plt.rcParams.update({'font.size': 14})
    width = 0.3
    data = [
        [0.1342,0.1060,9.366,0.0311,0.0306 ],
        [3315.239,3322.649,3750.774,526.732,498.802],
        # [0.021001,	0.025905,	0.020196],
    ]

    X = np.arange(5)

    fig, ax = plt.subplots()

    # p1 = plt.bar(
    #     X + 0.00, data[0], color=colors["DBEst_1k"], width=width, alpha=0.9)
    p2 = plt.bar(
        X + width, data[1], color=colors["BlinkDB_1k"], width=width, alpha=0.5)
    # p3 = plt.bar(
    #     X + 2*width, data[2], color=colors["green1"], width=width, alpha=0.7)

    # plt.legend((p1[0],p2[0]),  # , p3[0]),#, p4[0],p5[0]),
    #            ('DBEst++', 'HIVE', ), loc='center right', prop={'size': 12})

    plt.xticks(X +  0.5* width, ("1", '2', '3','4','5'))
    ax.set_ylabel("Response Time (s)")
    ax.set_xlabel("Query Template")
    ax.set_yscale('log')
    # formatter = FuncFormatter(to_percent)
    # ax.yaxis.set_major_formatter(formatter)
    # plt.subplots_adjust(bottom=0.11)
    # plt.subplots_adjust(left=0.12    ax.set_ylim([0.01,5000])
    plt.subplots_adjust(bottom=0.15)
    plt.subplots_adjust(left=0.18)

    add_value_labels(ax, b_float=True, b_percent=False, fontsize=12)
    plt.savefig("/home/quincy/Pictures/response.pdf")
    print("figure saved.")

    plt.show()

if __name__=="__main__":
    # plt_space()
    plt_response_time()