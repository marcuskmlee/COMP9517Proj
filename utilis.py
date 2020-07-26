import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from skimage.future import graph

def crop(img_mat, black_bg=False):
    img = cv.cvtColor(img_mat, cv.COLOR_BGR2GRAY)

    rows, cols = img.shape

    val = 255

    if black_bg:
        val = 0

    # print(img)
    
    def left():
        for x in range(cols):
            for y in range(rows):
                if img[y, x] != val:
                    return x
    
    left = left()
    # cv.circle(img, (int(x), int(y)), 8, (0, 255, 255), thickness=-1, lineType=cv.FILLED)

    def top():
        for y in range(rows):
            for x in range(cols):
                if img[y, x] != val:
                    return y
    
    top = top()
    # cv.circle(img, (int(x), int(y)), 8, (0, 255, 255), thickness=-1, lineType=cv.FILLED)

    def bottom():
        for y in reversed(range(rows)):
            for x in range(cols):
                if img[y, x] != val:
                    return y

    bottom = bottom()
    # cv.circle(img, (int(x), int(y)), 8, (0, 255, 255), thickness=-1, lineType=cv.FILLED)

    def right():
        for x in reversed(range(cols)):
            for y in range(rows):
                if img[y, x] != val:
                    return x

    right = right()

    # cv.circle(img, (int(left+1), int(top+1)), 8, (0, 255, 255), thickness=-1, lineType=cv.FILLED)

    # cv.imshow("Cropped", img[top:bottom, left:right])
    # cv.waitKey(0)
    # cv.destroyAllWindows()

    return img_mat[top:bottom, left:right]

def aggregate(mappings, labels):
    rows, cols = labels.shape

    for x in range(rows):
        for y in range(cols):
            mappings[labels[x, y]] += 1

    # mappings = {k: v for k, v in sorted(mappings.items(), key=lambda item: item[1])}

def set_label(cluster, labels, val):
    rows, cols = labels.shape

    for x in range(rows):
        for y in range(cols):
            if labels[x,y] == cluster:
                labels[x,y] = val

def plot_three_images(figure_title, image1, label1,
                      image2, label2, image3, label3):
    fig = plt.figure()
    fig.suptitle(figure_title)

    # Display the first image
    fig.add_subplot(1, 3, 1)
    plt.imshow(image1)
    plt.axis('off')
    plt.title(label1)

    # Display the second image
    fig.add_subplot(1, 3, 2)
    plt.imshow(image2)
    plt.axis('off')
    plt.title(label2)

    # Display the third image
    fig.add_subplot(1, 3, 3)
    plt.imshow(image3)
    plt.axis('off')
    plt.title(label3)

    plt.tight_layout()
    plt.show()

def plot_two(figure_title, image1, label1, image2, label2):
    # Display the first image
    fig = plt.figure()
    fig.suptitle(figure_title)

    fig.add_subplot(1, 2, 1)
    plt.imshow(image1)
    plt.axis('off')
    plt.title(label1)

    # Display the second image
    fig.add_subplot(1, 2, 2)
    plt.imshow(image2)
    plt.axis('off')
    plt.title(label2)

    plt.show()

def plot_four_images(figure_title, image1, label1, image2, label2, image3, label3, image4, label4):
    _, ax_arr = plt.subplots(2, 2, sharex=True, sharey=True, figsize=(10, 10))
    ax1, ax2, ax3, ax4 = ax_arr.ravel()

    ax1.imshow(image1)
    ax1.set_title(label1)

    ax2.imshow(image2)
    ax2.set_title(label2)

    ax3.imshow(image3)
    ax3.set_title(label3)

    ax4.imshow(image4)
    ax4.set_title(label4)

    for ax in ax_arr.ravel():
        ax.set_axis_off()

    plt.tight_layout()
    plt.show()

def weight_mean_color(graph, src, dst, n):
    """Callback to handle merging nodes by recomputing mean color.

    The method expects that the mean color of `dst` is already computed.

    Parameters
    ----------
    graph : RAG
        The graph under consideration.
    src, dst : int
        The vertices in `graph` to be merged.
    n : int
        A neighbor of `src` or `dst` or both.

    Returns
    -------
    data : dict
        A dictionary with the `"weight"` attribute set as the absolute
        difference of the mean color between node `dst` and `n`.
    """

    diff = graph.nodes[dst]['mean color'] - graph.nodes[n]['mean color']
    diff = np.linalg.norm(diff)
    return {'weight': diff}


def merge_mean_color(graph, src, dst):
    """Callback called before merging two nodes of a mean color distance graph.

    This method computes the mean color of `dst`.

    Parameters
    ----------
    graph : RAG
        The graph under consideration.
    src, dst : int
        The vertices in `graph` to be merged.
    """
    graph.nodes[dst]['total color'] += graph.nodes[src]['total color']
    graph.nodes[dst]['pixel count'] += graph.nodes[src]['pixel count']
    graph.nodes[dst]['mean color'] = (graph.nodes[dst]['total color'] /
                                      graph.nodes[dst]['pixel count'])