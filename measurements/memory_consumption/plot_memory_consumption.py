import scipy
import scipy.sparse as sp
import numpy as np
import time
import numba
import matplotlib.pyplot as plt


if __name__ == '__main__':

    # set fontsize
    plt.rcParams.update({'font.size': 13})

# cuda:0: Memory Total 85.097971712 GB or 79.253662109375 GiB
# cuda:0: Memory Free 50.89492992 GB or 47.39959716796875 GiB
# cuda:0: Memory Used 34.203041792 GB or 31.85406494140625 GiB
# cuda:1: Memory Total 85.097971712 GB or 79.253662109375 GiB
# cuda:1: Memory Free 51.98708736 GB or 48.416748046875 GiB
# cuda:1: Memory Used 33.110884352 GB or 30.8369140625 GiB

# cuda:1: Memory Total 85.097971712 GB or 79.253662109375 GiB
# cuda:1: Memory Free 62.75596288 GB or 58.446044921875 GiB
# cuda:1: Memory Used 22.342008832 GB or 20.8076171875 GiB
# cuda:2: Memory Total 85.097971712 GB or 79.253662109375 GiB
# cuda:2: Memory Free 61.81224448 GB or 57.567138671875 GiB
# cuda:2: Memory Used 23.285727232 GB or 21.6865234375 GiB
# cuda:0: Memory Total 85.097971712 GB or 79.253662109375 GiB
# cuda:0: Memory Free 62.261952512 GB or 57.9859619140625 GiB
# cuda:0: Memory Used 22.8360192 GB or 21.2677001953125 GiB

# cuda:2: Memory Total 85.097971712 GB or 79.253662109375 GiB
# cuda:2: Memory Free 70.469287936 GB or 65.629638671875 GiB
# cuda:2: Memory Used 14.628683776 GB or 13.6240234375 GiB
# cuda:3: Memory Total 85.097971712 GB or 79.253662109375 GiB
# cuda:3: Memory Free 67.2858112 GB or 62.664794921875 GiB
# cuda:3: Memory Used 17.812160512 GB or 16.5888671875 GiB
# cuda:1: Memory Total 85.097971712 GB or 79.253662109375 GiB
# cuda:1: Memory Free 66.371452928 GB or 61.813232421875 GiB
# cuda:1: Memory Used 18.726518784 GB or 17.4404296875 GiB
# cuda:0: Memory Total 85.097971712 GB or 79.253662109375 GiB
# cuda:0: Memory Free 66.792259584 GB or 62.20513916015625 GiB
# cuda:0: Memory Used 18.305712128 GB or 17.04852294921875 GiB

# cuda:0: Memory Used 12.356354048 GB or 11.50775146484375 GiB
# cuda:6: Memory Used 9.339666432 GB or 8.6982421875 GiB
# cuda:2: Memory Used 5.98212608 GB or 5.5712890625 GiB
# cuda:3: Memory Used 9.597616128 GB or 8.9384765625 GiB
# cuda:4: Memory Used 9.79894272 GB or 9.1259765625 GiB
# cuda:5: Memory Used 9.22222592 GB or 8.5888671875 GiB
# cuda:1: Memory Used 8.261730304 GB or 7.6943359375 GiB
# cuda:7: Memory Used 9.270460416 GB or 8.6337890625 GiB

    data = [61.68, 31.85406494140625, 30.8369140625,
            13.6240234375, 16.5888671875, 17.4404296875, 17.04852294921875,
            11.50775146484375, 8.6982421875, 5.5712890625, 8.9384765625, 9.1259765625, 8.5888671875, 7.6943359375, 8.6337890625]


    x = [1, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4]

    xs = [0, 2, 3, 5, 6, 7, 8, 10, 11, 12, 13, 14, 15, 16, 17]


    ticks = ["$rank_$"]

    # bar plot for all the data
    # with different colors for the different x
    colors = ["midnightblue", "mediumblue", "dodgerblue", "lightskyblue"]


    fig, ax = plt.subplots()
    plt.figure(figsize=(3, 3))

    for i, (x_, data_) in enumerate(zip(x, data)):
        if i == 0:
            plt.bar(xs[i], data_, color=colors[x_-1], label = "One")
        elif i == 1:
            plt.bar(xs[i], data_, color=colors[x_-1], label = "Two")
        elif i == 3:
            plt.bar(xs[i], data_, color=colors[x_-1], label = "Three")
        elif i == 6:
            plt.bar(xs[i], data_, color=colors[x_-1], label = "Four")
        elif i == 12:
            plt.bar(xs[i], data_, color=colors[x_-1], label = "Eight")
        else:
            plt.bar(xs[i], data_, color=colors[x_-1])

    plt.ylabel("Memory [GiB]")
    plt.xticks([0, 2.5, 6.5, 13.5], ["1", "2", "4", "8"])
    plt.xlabel("#Slices ($N_t$)")


    plt.savefig("memory.png", dpi=700, bbox_inches='tight')
    plt.close("all")
    