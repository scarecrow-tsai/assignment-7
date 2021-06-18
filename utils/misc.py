import numpy as np
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt


from torch.utils.data import DataLoader
from torchvision.utils import make_grid


def show_image(img):
    img_np = img.numpy()
    plt.imshow(np.transpose(img_np, (1, 2, 0)), interpolation="nearest")


def show_image_grid(dataset, num_samples, num_cols):
    dataloader = DataLoader(dataset, batch_size=num_samples, shuffle=False)
    images, _ = next(iter(dataloader))
    image_grid = make_grid(images, nrow=num_cols)
    plt.imshow(image_grid.permute(1, 2, 0))
    plt.grid(False)
    plt.axis("off")


def viz_transform(original_data, transformed_data, num_samples):

    images_original = [original_data[i][0] for i in range(num_samples)]
    images_transformed = [transformed_data[i][0] for i in range(num_samples)]

    plt.suptitle("Original vs Transformed Images")

    fig, axes = plt.subplots(figsize=(30, 10), nrows=2, ncols=num_samples)

    for i in range(num_samples):
        axes[0, i].imshow(images_original[i].permute(1, 2, 0))
        axes[0, i].title.set_text("OG")

    for i in range(num_samples):
        axes[1, i].imshow(images_transformed[i].permute(1, 2, 0))
        axes[1, i].title.set_text("TF")

    for ax in fig.axes:
        ax.axis("off")
        ax.grid("False")


def visualize_misclassified_images(
    y_true_list, y_pred_list, dataset, idx_to_class, num_samples
):
    misclassified_info = [
        {"idx": i, "pred": pred, "true": actual}
        for i, (pred, actual) in enumerate(zip(y_pred_list, y_true_list))
        if pred != actual
    ]

    fig, axes = plt.subplots(figsize=(30, 5), nrows=1, ncols=num_samples)
    plt.suptitle("Misclassified Images")
    for i in range(num_samples):
        axes[i].imshow(dataset[misclassified_info[i]["idx"]][0].permute(1, 2, 0))
        axes[i].title.set_text(
            f"True: {idx_to_class[misclassified_info[i]['true']]}\nPred: {idx_to_class[misclassified_info[i]['pred']]}"
        )

    for ax in fig.axes:
        ax.axis("off")
        ax.grid("False")


def calc_data_stats(dataset):
    np_train_dataset = dataset.data / 255

    mean_1, mean_2, mean_3 = (
        np_train_dataset[:, :, :, 0].mean(),
        np_train_dataset[:, :, :, 1].mean(),
        np_train_dataset[:, :, :, 2].mean(),
    )

    std_1, std_2, std_3 = (
        np_train_dataset[:, :, :, 0].std(),
        np_train_dataset[:, :, :, 1].std(),
        np_train_dataset[:, :, :, 2].std(),
    )

    return {"mean": [mean_1, mean_2, mean_3], "std": [std_1, std_2, std_3]}
