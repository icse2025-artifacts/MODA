import os
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm

from dataset_loader import load_mnist_dataset, load_cifar10_dataset
from model_modularizer import calculate_modular_layer_masks
from models.model_utils import print_model_summary, get_runtime_device
from trainer import create_cnn_model

import seaborn as sns
import matplotlib.pyplot as plt

from models.modular_utils import get_activation_rate_during_inference

DEVICE = get_runtime_device()


def visualize_modular_layer_masks(modular_masks_path, num_classes):
    all_modular_layer_masks = torch.load(modular_masks_path)
    layer_idx = 1
    all_class_layer_feature_maps = {class_name: masks[layer_idx] for class_name, masks in
                                    all_modular_layer_masks.items()}
    merged_layer_feature_maps = [torch.sum(torch.stack(l_mask), dim=0, dtype=torch.float) for l_mask
                                 in zip(*all_class_layer_feature_maps.values())]
    # merged_layer_feature_maps = [torch.tensor(merged_layer_feature_maps).reshape(7,12)]
    # print(merged_layer_feature_maps)
    # merged_layer_feature_maps = [torch.tensor(merged_layer_feature_maps)]
    num_cols = 4
    num_rows = len(merged_layer_feature_maps) // num_cols

    fig, axes = plt.subplots(num_rows, num_cols,
                             figsize=(7 * num_cols, 7 * num_rows))

    flat_axs = axes.flatten()
    for i, f_map in enumerate(merged_layer_feature_maps):
        f_map[f_map == 0] = float('nan')
        sns.heatmap(f_map, cmap='copper_r', annot=True,  # annot_kws={"fontsize": 5},
                    linewidths=.5, linecolor="black", cbar=False, ax=flat_axs[i],
                    vmin=1e-3, vmax=10).set_title(f'Feature map {i}')
        # plt.title(f'Feature Map {i + 1}')

    plt.tight_layout(pad=3)
    plt.show()


def visualize_overlaps_of_modular_layer_masks(modular_masks_path):
    all_modular_layer_masks = torch.load(modular_masks_path)
    merged_layer_feature_map_overlap_dict = {}
    for l_idx in range(15):
        all_class_layer_feature_maps = {class_name: torch.flatten(masks[l_idx]) for class_name, masks in
                                        all_modular_layer_masks.items()}
        feature_map_overlaps = [torch.sum(torch.stack(l_masks, dim=0))
                                for l_masks in zip(*all_class_layer_feature_maps.values())]
        merged_layer_feature_map_overlap_dict[l_idx] = torch.stack(feature_map_overlaps).numpy()

    fig, ax = plt.subplots(figsize=(15, 10))
    # plt.grid()
    fig.set_facecolor("w")
    sns.set_style('whitegrid')
    colormap = plt.colormaps.get_cmap('Set3')

    plot_df = pd.DataFrame(columns=['layer_index', *list(range(4))])
    for k, v in merged_layer_feature_map_overlap_dict.items():
        his_count = np.histogram(v, bins=11, range=(0, 11))[0]
        his_prob = his_count / his_count.sum()
        his_prob = [his_prob[0], his_prob[1], his_prob[2], sum(his_prob[3:])]
        plot_df.loc[len(plot_df)] = [str(f"layer_{k}"), *his_prob]

    x_titles = plot_df["layer_index"].to_list()
    stack_bottom = np.zeros(len(x_titles))
    colors = ["#BAB3C5", "#ffa600", "#CD85CC", "#3B889D"]
    for i in range(4):
        plot_data = plot_df[i].to_numpy()
        ax.bar(x_titles, plot_data, bottom=stack_bottom,
               color=colors[i], alpha=0.8,
               label=f"value {i}")
        stack_bottom += plot_data

    for c in ax.containers:
        ax.bar_label(c, fmt=lambda x: f'{x:.3f}' if round(x, 3) > 0 else '',
                     label_type='center', fontsize=12)

    plt.ylim(0, 1)
    legend_prefix = "act_by_n_modules"
    plt.legend([f"{legend_prefix}=0",
                f"{legend_prefix}=1",
                f"{legend_prefix}=2",
                f"{legend_prefix}>2"], ncol=1, loc='upper right',
               columnspacing=2.0, labelspacing=1,
               handletextpad=0.5, handlelength=1.5,
               fancybox=True, shadow=True, bbox_to_anchor=(1.13, 1))

    plt.title(modular_masks_path)
    plt.ylabel('Percent', fontsize=15, labelpad=15)
    plt.xlabel('Layers', fontsize=15, labelpad=15)
    plt.show()


def main():
    batch_size = 256
    activation_rate_threshold = 0.9
    num_classes = 10

    model_type = "vgg16"
    # model_type = "resnet18"
    # model_type = "lenet5"

    model_checkpoint_dir = f"./data/model_checkpoints/cnn/{model_type}"

    # model_checkpoint_name = "original_model"
    # model_checkpoint_name = "modularization_model"
    # model_checkpoint_name = "modularization_model.L1Reg"
    # model_checkpoint_name = "modularization_model.GroupL1Reg"
    # model_checkpoint_name = "modularization_model.avgPool"
    model_checkpoint_name = "modularization_model.avgPool.L1Reg"

    checkpoint_path = os.path.join(model_checkpoint_dir, model_checkpoint_name + ".pt")
    print(activation_rate_threshold, checkpoint_path)

    # modular_masks_save_path = f"./data/model_checkpoints/cnn/{model_type}/masks/" \
    #                           f"modular_layer_masks__{model_checkpoint_name}__{activation_rate_threshold}.full.pt"
    modular_masks_save_path = f"./data/model_checkpoints/cnn/{model_type}/masks/" \
                              f"modular_layer_masks__{model_checkpoint_name}__{activation_rate_threshold}.pt"

    raw_model = create_cnn_model(model_type=model_type, num_classes=num_classes,
                                 save_activation_values=True, masked_model=False)
    raw_model.load_pretrained_weights(checkpoint_path)
    print_model_summary(raw_model)
    if not os.path.exists(modular_masks_save_path):
        _, train_loader, test_loader = load_cifar10_dataset(batch_size=batch_size, num_workers=2)
        # _, train_loader, test_loader = load_mnist_dataset(batch_size=batch_size, num_workers=2)
        calculate_modular_layer_masks(model=raw_model, data_loader=train_loader, num_classes=num_classes,
                                      save_path=modular_masks_save_path,
                                      activation_rate_threshold=activation_rate_threshold)
    # visualize_modular_layer_masks(modular_masks_path=modular_masks_save_path, num_classes=num_classes)
    visualize_overlaps_of_modular_layer_masks(modular_masks_path=modular_masks_save_path)


if __name__ == '__main__':
    main()
