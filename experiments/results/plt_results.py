import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np
import matplotlib.pyplot as plt
import matplotlib._color_data as mcd
import matplotlib.patches as mpatch
import matplotlib 
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
def to_percent(y, pos):
    return '%.1f%%' % (y * 100)

def to_percent3(y, pos):
    return '%.3f%%' % (y * 100)


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
                label = "{:.1f}".format(y_value)
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



def plt_tpcds_universal_relative_error():
    plt.rcParams.update({'font.size': 12})
    width = 0.2
    # data = [
    #     [0.011825885, 0.011224583,  0.00359967, 0.008883379 ],  # , 0.0335],
    #     [0.01995,  0.01972,  0.00079, 0.013486667],  # , 0.0486],
    #     [0.019962,   0.020326766,  0.020300222, 0.020196329],  # , 0.0182],
    # ]

    data = [
        [0.0149, 0.0148,  0.00328, 0.0109 ],  # , 0.0335],
        [0.02134,  0.01999,  0.00123, 0.01418666],  # , 0.0486],
        [0.019962,   0.020326766,  0.020300222, 0.020196329],  # , 0.0182],
    ]


    X = np.arange(4)

    fig, ax = plt.subplots()

    p1 = plt.bar(
        X + 0.00, data[0], color=colors["DBEst_1k"], width=width, alpha=0.9)
    p2 = plt.bar(
        X + width, data[1], color=colors["BlinkDB_1k"], width=width, alpha=0.5)
    p3 = plt.bar(
        X + 2*width, data[2], color=colors["green1"], width=width, alpha=0.7)
    

    plt.legend((p1[0], p2[0], p3[0]),#, p4[0],p5[0]),
               ('DBEst++', 'DeepDB', 'VerdictDB', 'BlinkDB_100k',"haha"), loc='lower left')

    plt.xticks(X + 1 * width, ("COUNT", 'SUM', 'AVG', 'OVERALL'))
    ax.set_ylabel("Relative Error (%)")
    # ax.set_yscale('log')
    formatter = FuncFormatter(to_percent)
    ax.yaxis.set_major_formatter(formatter)
    plt.subplots_adjust(bottom=0.07)
    # plt.subplots_adjust(left=0.23)
    # plt.subplots_adjust(bottom=0.12)

    add_value_labels(ax,b_float=True,b_percent=True,fontsize=6)
    plt.savefig("/Users/scott/Pictures/accuracy_universal.pdf")
    print("figure saved.")
    plt.show()

def plt_tpcds_universal_relative_error_scalability():
    plt.rcParams.update({'font.size': 12})
    width = 0.2
    # data = [
    #     [0.011013,	0.010773,	0.008883],
    #     [0.015733,	0.013967,	0.013487],
    #     [0.021001,	0.025905,	0.020196],
    # ]

    data = [
        [0.011013,	0.010773,	0.008883],
        [0.0142  ,	0.013967,	0.013487],
        [0.021001,	0.025905,	0.020196],
    ]

    X = np.arange(3)

    fig, ax = plt.subplots()

    p1 = plt.bar(
        X + 0.00, data[0], color=colors["DBEst_1k"], width=width, alpha=0.9)
    p2 = plt.bar(
        X + width, data[1], color=colors["BlinkDB_1k"], width=width, alpha=0.5)
    p3 = plt.bar(
        X + 2*width, data[2], color=colors["green1"], width=width, alpha=0.7)
    

    plt.legend((p1[0], p2[0], p3[0]),#, p4[0],p5[0]),
               ('DBEst++', 'DeepDB', 'VerdictDB', 'BlinkDB_100k',"haha"), loc='lower left')

    plt.xticks(X + 1 * width, ("10", '100', '1000'))
    ax.set_ylabel("Relative Error (%)")
    ax.set_xlabel("TPC-DS Scaling Factor")
    # ax.set_yscale('log')
    formatter = FuncFormatter(to_percent)
    ax.yaxis.set_major_formatter(formatter)
    plt.subplots_adjust(bottom=0.07)
    # plt.subplots_adjust(left=0.23)
    plt.subplots_adjust(bottom=0.12)

    add_value_labels(ax,b_float=True,b_percent=True)
    plt.savefig("/Users/scott/Pictures/accuracy_universal_scalability.pdf")
    print("figure saved.")

    plt.show()

def plt_tpcds_universal_relative_error_scalability_count():
    plt.rcParams.update({'font.size': 16})
    width = 0.2
    # data = [
    #     [0.011013,	0.010773,	0.008883],
    #     [0.015733,	0.013967,	0.013487],
    #     [0.021001,	0.025905,	0.020196],
    # ]

    data = [
        [0.0149,	0.0108,	0.0118],
        [0.0213,	0.0168,	0.0126],
        [0.0203,	0.0259,	0.0200],
    ]


    X = np.arange(3)

    fig, ax = plt.subplots()

    p1 = plt.bar(
        X + 0.00, data[0], color=colors["DBEst_1k"], width=width, alpha=0.9)
    p2 = plt.bar(
        X + width, data[1], color=colors["BlinkDB_1k"], width=width, alpha=0.5)
    p3 = plt.bar(
        X + 2*width, data[2], color=colors["green1"], width=width, alpha=0.7)
    

    plt.legend((p1[0], p2[0], p3[0]),#, p4[0],p5[0]),
               ('DBEst++', 'DeepDB', 'VerdictDB', 'BlinkDB_100k',"haha"), loc='lower left')

    plt.xticks(X + 1 * width, ("10", '100', '1000'))
    ax.set_ylabel("Relative Error (%)")
    ax.set_xlabel("TPC-DS Scaling Factor")
    # ax.set_yscale('log')
    formatter = FuncFormatter(to_percent)
    ax.yaxis.set_major_formatter(formatter)
    plt.subplots_adjust(bottom=0.07)
    plt.subplots_adjust(left=0.17)
    plt.subplots_adjust(bottom=0.12)

    add_value_labels(ax,b_float=True,b_percent=True)
    plt.savefig("/Users/scott/Pictures/accuracy_universal_scalability_count.pdf")
    print("figure saved.")

    plt.show()

def plt_tpcds_universal_relative_error_scalability_sum():
    plt.rcParams.update({'font.size': 16})
    width = 0.2
    # data = [
    #     [0.011013,	0.010773,	0.008883],
    #     [0.015733,	0.013967,	0.013487],
    #     [0.021001,	0.025905,	0.020196],
    # ]

    data = [
        [0.0148,	0.0111,	0.0112],
        [0.0200,	0.0178,	0.0125],
        [0.0216,	0.0259,	0.0203],
    ]

    X = np.arange(3)

    fig, ax = plt.subplots()

    p1 = plt.bar(
        X + 0.00, data[0], color=colors["DBEst_1k"], width=width, alpha=0.9)
    p2 = plt.bar(
        X + width, data[1], color=colors["BlinkDB_1k"], width=width, alpha=0.5)
    p3 = plt.bar(
        X + 2*width, data[2], color=colors["green1"], width=width, alpha=0.7)
    

    plt.legend((p1[0], p2[0], p3[0]),#, p4[0],p5[0]),
               ('DBEst++', 'DeepDB', 'VerdictDB', 'BlinkDB_100k',"haha"), loc='lower left')

    plt.xticks(X + 1 * width, ("10", '100', '1000'))
    ax.set_ylabel("Relative Error (%)")
    ax.set_xlabel("TPC-DS Scaling Factor")
    # ax.set_yscale('log')
    formatter = FuncFormatter(to_percent)
    ax.yaxis.set_major_formatter(formatter)
    plt.subplots_adjust(bottom=0.07)
    plt.subplots_adjust(left=0.17)
    plt.subplots_adjust(bottom=0.12)

    add_value_labels(ax,b_float=True,b_percent=True)
    plt.savefig("/Users/scott/Pictures/accuracy_universal_scalability_sum.pdf")
    print("figure saved.")

    plt.show()


def plt_tpcds_universal_response_time_scalability():
    plt.rcParams.update({'font.size': 20})
    width = 0.3
    data = [
        [46.51,	145.60,	320.75],
        [67.67,	261.33,	409.33],
        [400.73,	997.30,	4112.51],
    ]

    X = np.arange(3)

    fig, ax = plt.subplots()

    p1 = plt.bar(
        X + 0.00, data[0], color=colors["DBEst_1k"], width=width, alpha=0.9)
    p2 = plt.bar(
        X + width, data[1], color=colors["BlinkDB_1k"], width=width, alpha=0.5)
    p3 = plt.bar(
        X + 2*width, data[2], color=colors["green1"], width=width, alpha=0.7)
    

    plt.legend((p1[0], p2[0], p3[0]),#, p4[0],p5[0]),
               ('DBEst++', 'DeepDB', 'VerdictDB', 'BlinkDB_100k',"haha"), loc='upper left')

    plt.xticks(X + 1 * width, ("10", '100', '1000'))
    ax.set_ylabel("Response Time (ms)")
    ax.set_xlabel("TPC-DS Scaling Factor")
    # ax.set_yscale('log')
    # formatter = FuncFormatter(to_percent)
    # ax.yaxis.set_major_formatter(formatter)
    # plt.subplots_adjust(bottom=0.11)
    # plt.subplots_adjust(left=0.12)
    plt.subplots_adjust(bottom=0.18)
    plt.subplots_adjust(left=0.2)

    add_value_labels(ax,b_float=True,b_percent=False,fontsize=8)
    plt.savefig("/Users/scott/Pictures/response_time_universal.pdf")
    print("figure saved.")

    plt.show()

def plt_tpcds_universal_space_scalability():
    plt.rcParams.update({'font.size': 20})
    width = 0.3
    data = [
        [83,	139,	309],
        [163122,	454012,	584020],
        [40100,	171400,	540900],
    ]

    X = np.arange(3)

    fig, ax = plt.subplots()

    p1 = plt.bar(
        X + 0.00, data[0], color=colors["DBEst_1k"], width=width, alpha=0.9)
    p2 = plt.bar(
        X + width, data[1], color=colors["BlinkDB_1k"], width=width, alpha=0.5)
    p3 = plt.bar(
        X + 2*width, data[2], color=colors["green1"], width=width, alpha=0.7)
    

    plt.legend((p1[0], p2[0], p3[0]),#, p4[0],p5[0]),
               ('DBEst++', 'DeepDB', 'VerdictDB', 'BlinkDB_100k',"haha"), loc='center left', prop={'size': 12})

    plt.xticks(X + 1 * width, ("10", '100', '1000'))
    ax.set_ylabel("Space Overheads (KB)")
    ax.set_xlabel("TPC-DS Scaling Factor")
    ax.set_yscale('log')
    # formatter = FuncFormatter(to_percent)
    # ax.yaxis.set_major_formatter(formatter)
    # plt.subplots_adjust(bottom=0.11)
    # plt.subplots_adjust(left=0.12)
    plt.subplots_adjust(bottom=0.18)
    plt.subplots_adjust(left=0.18)
    add_value_labels(ax,b_float=False,fontsize=8)

    
    plt.savefig("/Users/scott/Pictures/space_universal.pdf")
    print("figure saved.")
    plt.show()

# ---------------------------------------------------------------------------------------------->>>>>>>>>>>>>>>>>>>>>>>
def plt_tpcds_compact_relative_error():
    plt.rcParams.update({'font.size': 16})
    width = 0.3
    # data = [
    #     [0.011825885, 0.011224583,  0.00359967, 0.008883379 ],  # , 0.0335],
    #     [0.01995,  0.01972,  0.00079, 0.013486667],  # , 0.0486],
    #     # [0.019962,   0.020326766,  0.020300222, 0.020196329],  # , 0.0182],
    # ]

    data = [
        [0.0149, 0.0148,  0.00328, 0.0109],  # , 0.0335],
        [0.01835,  0.01842,  0.00214, 0.01297],  # , 0.0486],
        # [0.019962,   0.020326766,  0.020300222, 0.020196329],  # , 0.0182],
    ]

    X = np.arange(4)

    fig, ax = plt.subplots()

    p1 = plt.bar(
        X + 0.00, data[0], color=colors["DBEst_1k"], width=width, alpha=0.9)
    p2 = plt.bar(
        X + width, data[1], color=colors["BlinkDB_1k"], width=width, alpha=0.5)
    # p3 = plt.bar(
    #     X + 2*width, data[2], color=colors["green1"], width=width, alpha=0.7)
    

    plt.legend((p1[0], p2[0]),#, p3[0]),#, p4[0],p5[0]),
               ('DBEst++', 'DeepDB', 'VerdictDB', 'BlinkDB_100k',"haha"), loc='lower left')

    plt.xticks(X + 0.5 * width, ("COUNT", 'SUM', 'AVG', 'OVERALL'))
    ax.set_ylabel("Relative Error (%)")
    # ax.set_yscale('log')
    formatter = FuncFormatter(to_percent)
    ax.yaxis.set_major_formatter(formatter)
    plt.subplots_adjust(bottom=0.07)
    plt.subplots_adjust(left=0.15)
    # plt.subplots_adjust(bottom=0.12)
    add_value_labels(ax,b_float=True,b_percent=True,fontsize=10)
    plt.savefig("/Users/scott/Pictures/accuracy_compact.pdf")
    print("figure saved.")

    plt.show()

def plt_tpcds_compact_relative_error_scalability():
    plt.rcParams.update({'font.size': 16})
    width = 0.3
    data = [
        [0.011013,	0.010773,	0.008883],
        [0.01297 ,	0.013003,	0.013487],
        # [0.021001,	0.025905,	0.020196],
    ]

    X = np.arange(3)

    fig, ax = plt.subplots()

    p1 = plt.bar(
        X + 0.00, data[0], color=colors["DBEst_1k"], width=width, alpha=0.9)
    p2 = plt.bar(
        X + width, data[1], color=colors["BlinkDB_1k"], width=width, alpha=0.5)
    # p3 = plt.bar(
    #     X + 2*width, data[2], color=colors["green1"], width=width, alpha=0.7)
    

    plt.legend((p1[0], p2[0]),#, p3[0]),#, p4[0],p5[0]),
               ('DBEst++', 'DeepDB', 'VerdictDB', 'BlinkDB_100k',"haha"), loc='lower left')

    plt.xticks(X + 0.5 * width, ("10", '100', '1000'))
    ax.set_ylabel("Relative Error (%)")
    ax.set_xlabel("TPC-DS Scaling Factor")
    # ax.set_yscale('log')
    formatter = FuncFormatter(to_percent)
    ax.yaxis.set_major_formatter(formatter)
    plt.subplots_adjust(bottom=0.07)
    plt.subplots_adjust(left=0.15)
    plt.subplots_adjust(bottom=0.12)

    add_value_labels(ax,b_float=True,b_percent=True,fontsize=10)
    plt.savefig("/Users/scott/Pictures/accuracy_compact_scalability.pdf")
    print("figure saved.")

    plt.show()

def plt_tpcds_compact_response_time_scalability():
    plt.rcParams.update({'font.size': 20})
    width = 0.3
    data = [
        [46.51,	145.60,	320.75],
        [25.67,	42.33,	78.67],
        # [0.021001,	0.025905,	0.020196],
    ]

    X = np.arange(3)

    fig, ax = plt.subplots()

    p1 = plt.bar(
        X + 0.00, data[0], color=colors["DBEst_1k"], width=width, alpha=0.9)
    p2 = plt.bar(
        X + width, data[1], color=colors["BlinkDB_1k"], width=width, alpha=0.5)
    # p3 = plt.bar(
    #     X + 2*width, data[2], color=colors["green1"], width=width, alpha=0.7)
    

    plt.legend((p1[0], p2[0]),#, p3[0]),#, p4[0],p5[0]),
               ('DBEst++', 'DeepDB', 'VerdictDB', 'BlinkDB_100k',"haha"), loc='upper left')

    plt.xticks(X + 0.5 * width, ("10", '100', '1000'))
    ax.set_ylabel("Response Time (ms)")
    ax.set_xlabel("TPC-DS Scaling Factor")
    # ax.set_yscale('log')
    # formatter = FuncFormatter(to_percent)
    # ax.yaxis.set_major_formatter(formatter)
    # plt.subplots_adjust(bottom=0.11)
    # plt.subplots_adjust(left=0.12)
    plt.subplots_adjust(bottom=0.18)
    plt.subplots_adjust(left=0.2)

    add_value_labels(ax,b_float=True,b_percent=False,fontsize=10)
    plt.savefig("/Users/scott/Pictures/response_time_compact.pdf")
    print("figure saved.")

    plt.show()

def plt_tpcds_compact_space_scalability():
    plt.rcParams.update({'font.size': 12})
    width = 0.3
    data = [
        [83,	139,	309],
        [1301,	1412,	1508],
        # [0.021001,	0.025905,	0.020196],
    ]

    X = np.arange(3)

    fig, ax = plt.subplots()

    p1 = plt.bar(
        X + 0.00, data[0], color=colors["DBEst_1k"], width=width, alpha=0.9)
    p2 = plt.bar(
        X + width, data[1], color=colors["BlinkDB_1k"], width=width, alpha=0.5)
    # p3 = plt.bar(
    #     X + 2*width, data[2], color=colors["green1"], width=width, alpha=0.7)
    

    plt.legend((p1[0], p2[0]),#, p3[0]),#, p4[0],p5[0]),
               ('DBEst++', 'DeepDB', 'VerdictDB', 'BlinkDB_100k',"haha"), loc='center left')

    plt.xticks(X + 0.5 * width, ("10", '100', '1000'))
    ax.set_ylabel("Space Overheads (KB)")
    ax.set_xlabel("TPC-DS Scaling Factor")
    # ax.set_yscale('log')
    # formatter = FuncFormatter(to_percent)
    # ax.yaxis.set_major_formatter(formatter)
    # plt.subplots_adjust(bottom=0.11)
    # plt.subplots_adjust(left=0.12)
    plt.subplots_adjust(bottom=0.18)
    plt.subplots_adjust(left=0.15)

    add_value_labels(ax,b_float=False,b_percent=False,fontsize=10)
    plt.savefig("/Users/scott/Pictures/space_compact.pdf")
    print("figure saved.")
    

    plt.show()

def plt_3d_chart():
    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.pyplot as plt
    import numpy as np
    # %matplotlib inline

    data = np.array([
    [0,1,0,0,1,0,0,1,0],
    [0,3,0,0,3,0,0,3,0],
    [6,1,1,6,1,1,6,1,1],
    ])

    column_names = ['COUNT_10g','SUM_10g','AVF_10g','COUNT_100g','SUM_100g','AVF_100g','COUNT_1t','SUM_1t','AVF_1t']
    row_names = ['DBEst++','DeepDB','VerdictDB']

    fig = plt.figure()
    ax = Axes3D(fig)

    lx= len(data[0])            # Work out matrix dimensions
    ly= len(data[:,0])
    xpos = np.arange(0,lx,1)    # Set up a mesh of positions
    ypos = np.arange(0,ly,1)
    xpos, ypos = np.meshgrid(xpos-1.25, ypos-1.25)

    xpos = xpos.flatten()   # Convert positions to 1D array
    ypos = ypos.flatten()
    zpos = np.zeros(lx*ly)

    dx = 0.5 * np.ones_like(zpos)
    dy =  dx.copy()
    dz = data.flatten()

    cs = ['r', 'g', 'b', 'r', 'g', 'b','r', 'g', 'b'] * ly

    ax.bar3d(xpos,ypos,zpos, dx, dy, dz, color=cs)

    #sh()
    ax.w_xaxis.set_ticklabels(column_names)
    ax.w_yaxis.set_ticklabels(row_names)
    ax.set_xlabel('Aggregates')
    ax.set_ylabel('AQP Engine')
    ax.set_zlabel('Average Relative Error (\%)')
    ax.set_xlim(-1,8)
    ax.set_ylim(-1,3)

    plt.show()


# ---------------------------------------------------------------------------------------------->>>>>>>>>>>>>>>>>>>>>>>
def plt_flight_relative_error():
    plt.rcParams.update({'font.size': 12})
    width = 0.2
    data = [
        [0.01300,	0.01310,	0.00022,	0.00877],
        [0.01131,	0.01132,	0.00008,	0.00757],
        [0.00674,	0.00678,	0.00016,	0.00456],
        [0.00517,	0.00516,	0.000075,	0.00369],
    ]


    X = np.arange(4)

    fig, ax = plt.subplots()

    p1 = plt.bar(
        X + 0.00, data[0], color=colors["DBEst_1k"], width=width, alpha=0.9)
    p2 = plt.bar(
        X + width, data[1], color=colors["BlinkDB_1k"], width=width, alpha=0.5)
    p3 = plt.bar(
        X + 2*width, data[2], color=colors["DBEst_10k"], width=width, alpha=0.7)
    p4 = plt.bar(
        X + 3*width, data[3], color=colors["BlinkDB_10k"], width=width, alpha=0.3)
    # p5 = plt.bar(
    #     X + 4*width, data[4], color=colors["green1"], width=width, alpha=alpha['6'])

    plt.legend((p1[0], p2[0], p3[0], p4[0]),#,p5[0]),
               ('DBEst++_1m', 'DeepDB_1m', 'DBEst++_5m', 'DeepDB_5m',"haha"), loc='best', prop={'size': 8})

    plt.xticks(X + 1 * width, ("COUNT", 'SUM', 'AVG', 'OVERALL'))
    ax.set_ylabel("Relative Error (%)")
    # ax.set_yscale('log')
    formatter = FuncFormatter(to_percent)
    ax.yaxis.set_major_formatter(formatter)
    plt.subplots_adjust(bottom=0.14) #0.07
    # plt.subplots_adjust(left=0.23)

    add_value_labels(ax,b_float=True,b_percent=True,fontsize=6)
    plt.savefig("/Users/scott/Pictures/flight_accuracy.pdf")
    print("figure saved.")

    plt.show()

def plt_flight_relative_error_overall():
    plt.rcParams.update({'font.size': 12})
    width = 0.3
    data = [
        [0.01300,	0.01310,	0.00022,	0.00877],
        [0.01131,	0.01132,	0.00008,	0.00757],
        [0.00674,	0.00678,	0.00016,	0.00456],
        [0.00517,	0.00516,	0.000075,	0.00369],
    ]

    data = [
        [0.00877, 0.00456],
        [0.00757, 0.00369],
    ]


    X = np.arange(2)

    fig, ax = plt.subplots()

    p1 = plt.bar(
        X + 0.00, data[0], color=colors["DBEst_1k"], width=width, alpha=0.9)
    p2 = plt.bar(
        X + width, data[1], color=colors["BlinkDB_1k"], width=width, alpha=0.5)
    # p3 = plt.bar(
    #     X + 2*width, data[2], color=colors["DBEst_10k"], width=width, alpha=0.7)
    # p4 = plt.bar(
    #     X + 3*width, data[3], color=colors["BlinkDB_10k"], width=width, alpha=0.3)
    # p5 = plt.bar(
    #     X + 4*width, data[4], color=colors["green1"], width=width, alpha=alpha['6'])

    plt.legend((p1[0], p2[0]),# p3[0], p4[0]),#,p5[0]),
               ('DBEst++', 'DeepDB', 'DBEst++_5m', 'DeepDB_5m',"haha"), loc='lower left', prop={'size': 12})

    plt.xticks(X + 0.5 * width, ("1m", '5m', 'AVG', 'OVERALL'))
    ax.set_xlabel("Sample Size")
    ax.set_ylabel("Relative Error (%)")
    # ax.set_yscale('log')
    formatter = FuncFormatter(to_percent)
    ax.yaxis.set_major_formatter(formatter)
    plt.subplots_adjust(bottom=0.25) #0.14
    # plt.subplots_adjust(left=0.23)

    add_value_labels(ax,b_float=True,b_percent=True,fontsize=8)
    plt.savefig("/Users/scott/Pictures/flight_accuracy.pdf")
    print("figure saved.")

    plt.show()

def plt_flight_response_time():
    plt.rcParams.update({'font.size': 20})
    width = 0.3
    data = [
        [32.67,	33.50],
        [27.1,	29],
        # [0.021001,	0.025905,	0.020196],
    ]

    X = np.arange(2)

    fig, ax = plt.subplots()

    p1 = plt.bar(
        X + 0.00, data[0], color=colors["DBEst_1k"], width=width, alpha=0.9)
    p2 = plt.bar(
        X + width, data[1], color=colors["BlinkDB_1k"], width=width, alpha=0.5)
    # p3 = plt.bar(
    #     X + 2*width, data[2], color=colors["green1"], width=width, alpha=0.7)
    

    plt.legend((p1[0], p2[0]),#, p3[0]),#, p4[0],p5[0]),
               ('DBEst++', 'DeepDB', 'VerdictDB', 'BlinkDB_100k',"haha"), loc='lower left')

    plt.xticks(X + 0.5 * width, ("1m", '5m'))
    ax.set_ylabel("Response Time (ms)")
    ax.set_xlabel("Sample Size")
    # ax.set_yscale('log')
    # formatter = FuncFormatter(to_percent)
    # ax.yaxis.set_major_formatter(formatter)
    # plt.subplots_adjust(bottom=0.11)
    # plt.subplots_adjust(left=0.12)
    plt.subplots_adjust(bottom=0.18)
    plt.subplots_adjust(left=0.15)

    add_value_labels(ax,b_float=True,b_percent=False,fontsize=12)
    plt.savefig("/Users/scott/Pictures/flight_response.pdf")
    print("figure saved.")

    plt.show()

def plt_flight_space():
    plt.rcParams.update({'font.size': 12}) #20
    width = 0.3
    data = [
        [170,	35],
        # [1200,	1200],
        [3502, 4314],
    ]

    X = np.arange(2)

    fig, ax = plt.subplots()

    p1 = plt.bar(
        X + 0.00, data[0], color=colors["DBEst_1k"], width=width, alpha=0.9)
    p2 = plt.bar(
        X + width, data[1], color=colors["BlinkDB_1k"], width=width, alpha=0.5)
    # p3 = plt.bar(
    #     X + 2*width, data[2], color=colors["green1"], width=width, alpha=0.7)
    

    plt.legend((p1[0], p2[0]),#, p3[0]),#, p4[0],p5[0]),
               ('DBEst++', 'DeepDB', 'VerdictDB', 'BlinkDB_100k',"haha"), loc='center left')

    plt.xticks(X + 0.5 * width, ("1m", '5m'))
    ax.set_ylabel("Space Overheads (KB)")
    ax.set_xlabel("Sample Size")
    # ax.set_yscale('log')
    # formatter = FuncFormatter(to_percent)
    # ax.yaxis.set_major_formatter(formatter)
    # plt.subplots_adjust(bottom=0.11)
    # plt.subplots_adjust(left=0.12)
    # plt.subplots_adjust(bottom=0.18)
    # plt.subplots_adjust(left=0.2)

    add_value_labels(ax,b_float=False,b_percent=False,fontsize=12)
    plt.savefig("/Users/scott/Pictures/flight_space.pdf")
    print("figure saved.")
    

    plt.show()


def plt_distributed():
    plt.rcParams.update({'font.size': 12})
    x=[1,4,8,12,16,20]
    y1=[13.38,	3.5967,	1.659,	1.1032,	0.9601,	0.78]
    y2=[7,7,7,7,7,7]
    plt.plot(x,y1,'-g', marker='o', linewidth=3, label='DBEst++')#color='tab:blue', marker='o')
    plt.plot(x,y2,':b', marker='x', linewidth=3, label='DeepDB')#color='g', marker='*')
    plt.xlabel("Degree of Parallelism")
    plt.ylabel("Time Cost (s)")
    plt.xlim([0,20])
    plt.xticks(np.arange(0, 20, 2.0))
    # plt.yscale('log')
    # plt.xscale('log')
    plt.legend()
    plt.subplots_adjust(bottom=0.25)

    plt.savefig("/Users/scott/Pictures/distributed.pdf")
    print("figure saved.")
    plt.show()


if __name__ == "__main__":
    # plt_tpcds_universal_relative_error()
    # plt_tpcds_universal_relative_error_scalability()
    # plt_tpcds_universal_response_time_scalability()
    # plt_tpcds_universal_space_scalability()

    # plt_tpcds_compact_relative_error()
    # plt_tpcds_compact_relative_error_scalability()
    # plt_tpcds_compact_response_time_scalability()
    # plt_tpcds_compact_space_scalability()
    # plt_3d_chart()
    # plt_tpcds_universal_relative_error_scalability_count()
    # plt_tpcds_universal_relative_error_scalability_sum()

    plt_flight_relative_error_overall()
    # plt_flight_relative_error()
    # plt_flight_response_time()
    # plt_flight_space()


    # plt_distributed()