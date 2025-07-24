import matplotlib.pyplot as plt

def plotWrapper(x, y_list, labels=None, xlabel='X-axis', ylabel='Y-axis', title='Plot'):
    # fig, ax = plt.subplots()
    for i, y in enumerate(y_list):
        plt.plot(x, y, label=labels[i] if labels else f"Series {i+1}")
    plt.xlabel(xlabel)  # Use ax to set labels and title
    plt.ylabel(ylabel)
    plt.title(title)

    if labels:
        plt.legend()

    # return current plot
    return plt.gcf()