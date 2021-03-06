from Helper_func import *


def plot_history(history, parameter, path):
    """
    Plot the loss and the evaluation metric for the train and validation set
    :param history: The history of the model
    :param parameter: The evaluation metric
    :param path: Directory path to save the history graphs
    """
    plt.figure(figsize=(30, 5))
    plt.subplot(121)
    plt.plot(history.history[parameter])
    plt.plot(history.history['val_' + parameter])
    plt.title('Model ' + parameter + "\n best train:" + str(max(history.history[parameter])) + "\n best val:" + str(
        max(history.history['val_' + parameter])))
    plt.ylabel(parameter)
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.savefig(path + "_" + parameter + '-history.png')


def plot_row(fig, num_rows, num_cols, number, title, img, img_seg):
    """
    Plot original image and the corresponding mask for each class
    :param fig: Image that include few rows of examples
    :param num_rows:
    :param num_cols:
    :param number: The position of the current row
    :param title: To mention original/predicated in the title of the row
    :param img: image (H,W,1)
    :param img_seg: mask (H,W,4) 3 classes and 1 background
    :return:
    """
    ax = fig.add_subplot(num_rows, num_cols, number)
    ax.set_ylabel(title, rotation=0, labelpad=50, fontsize=16)
    ax.imshow(img)
    for pos, i in enumerate(range(number + 1, number + 5)):
        ax = fig.add_subplot(num_rows, num_cols, i)
        ax.imshow(img_seg[:, :, pos])
    return i


def plot_examples(paths, name_set, num_examples, img_dim, interpolation, model, path):
    """
    Plot examples of the results of the model, the original image and mask and below is the predicated mask.
    :param paths: The paths of the images
    :param name_set: Train/ valid/ test set
    :param num_examples: Number of examples
    :param img_dim: The new image dimension
    :param interpolation: the interpolation method for resize
    :param model:
    :param path: Directory path to save the example images
    """
    fig = plt.figure(figsize=(20, 40))
    plt.suptitle(name_set + "-examples", fontsize=23)
    num_rows = num_examples * 2
    num_cols = 5
    image_gen = generator_images(paths, paths.dropna().path)

    for number in range(1, num_examples * 10 + 1, 10):
        img, img_seg, id_img = next(image_gen)

        img = cv2.resize(img, img_dim, interpolation=interpolation)
        img_seg = cv2.resize(img_seg, img_dim, interpolation=interpolation)

        x = img.reshape(1, 256, 256, 1)

        y = np.round(model.predict(x))

        i = plot_row(fig, num_rows, num_cols, number, "original", img, img_seg)
        plot_row(fig, num_rows, num_cols, i + 1, "predicated", img, y[0, :, :, :])
    plt.tight_layout(pad=2)
    fig.subplots_adjust(top=0.96)
    plt.savefig(path + "_" + name_set + '-examples.png')

