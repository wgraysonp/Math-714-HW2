import os
import matplotlib.pyplot as plt


def plot_error(E, H, problem=None):
    fig = plt.figure(figsize=(7, 2 / 3 * 7))
    ax = plt.subplot()

    ax.plot(H, E, marker='o', color='black')

    ax.set_xscale('log')
    ax.set_yscale('log')

    ax.set_xlabel(r'Grid spacing $h$')
    ax.set_ylabel(r'Error')

    directory = os.path.join(os.getcwd(), 'pic')
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, 'problem{}_error.png'.format(problem))

    plt.savefig(path)

    plt.close()


