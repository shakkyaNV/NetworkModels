import matplotlib.pyplot as plt


X = 10
def plotWrapper(x, y_list, labels=None, xlabel='X-axis', ylabel='Y-axis', title='Plot'):

    for i, y in enumerate(y_list):
        plt.plot(x, y, label=labels[i] if labels else f"Series {i+1}")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)

    if labels:
        plt.legend()

    # return current plot
    return plt.gcf()