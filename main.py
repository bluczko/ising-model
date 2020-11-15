import numpy as np

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as plt_anim

from ising.model import IsingModel

if __name__ == "__main__":
    ising = IsingModel(size=128, init_tempetature=1)
    stats = np.zeros(shape=(3, 512))

    matplotlib.rcParams["toolbar"] = "None"
    fig = plt.figure(figsize=(12, 6), constrained_layout=True)
    fig.canvas.set_window_title("Wizualizacja modelu Isinga")
    spec = fig.add_gridspec(3, 2)

    system_plot = fig.add_subplot(spec[:, 0])
    temp_plot = fig.add_subplot(spec[0, 1])
    magnet_plot = fig.add_subplot(spec[1, 1])
    heat_plot = fig.add_subplot(spec[2, 1])

    system_plot.get_xaxis().set_visible(False)
    system_plot.get_yaxis().set_visible(False)

    temp_plot.get_xaxis().set_visible(False)
    magnet_plot.get_xaxis().set_visible(False)
    heat_plot.get_xaxis().set_visible(False)

    def anim_seq(i):
        global ising, stats

        ising.temperature = 0.1 + np.clip(4 * np.sin(i / 20), 0, 3)

        ising.step(2 * np.prod(ising.system.shape))

        system_plot.clear()
        # system_plot.set_title("Wizualizacja spinów układu")
        system_plot.imshow(ising.system, cmap="seismic")

        stats = np.roll(stats, -1, axis=1)

        stats[0, -1] = ising.temperature
        stats[1, -1] = ising.magnetization
        stats[2, -1] = ising.heat_capacity

        temp_plot.clear()
        temp_plot.set_title("Temperatura $ [ \\frac{J}{K_b} ] $")
        temp_plot.plot(stats[0, :])

        magnet_plot.clear()
        magnet_plot.set_title("Magnetyzacja $ [ \\mu ] $")
        magnet_plot.plot(stats[1, :])

        heat_plot.clear()
        heat_plot.set_title("Pojemność cieplna $ [ \\frac{J}{K_{b}^{2}} ] $")
        heat_plot.plot(stats[2, :])

    anim = plt_anim.FuncAnimation(fig, anim_seq, interval=1, blit=False)
    plt.show()
