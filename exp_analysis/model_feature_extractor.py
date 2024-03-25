import os
import pickle
import sys
from pathlib import Path

import torch
from torch import nn
from torchvision.models.feature_extraction import create_feature_extractor

from dataset_loader import load_dataset
from models import create_modular_model
from models.model_evaluation_utils import evaluate_model, _get_model_outputs
from models.model_utils import get_runtime_device, get_model_leaf_layers

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # handle PYTHONPATH env

DEVICE = get_runtime_device()

if __name__ == '__main__':
    model_type = "vgg16"

    # dataset_type = "cifar10"
    dataset_type = "cifar100"
    # dataset_type = "svhn"

    # checkpoint_path = f"./temp/{model_type}_{dataset_type}/model__bs128__ep200__lr0.05__coh1.0_coup1.0_comp0.3/model.pt"
    # checkpoint_path = f"./temp/{model_type}_{dataset_type}/model__bs128__ep200__lr0.05__coh1.0_coup1.0_comp0.0/model.pt"
    checkpoint_path = f"./temp/{model_type}_{dataset_type}/model__bs128__ep200__lr0.05__coh0.0_coup0.0_comp0.0/model.pt"

    nth_last_relu_layer_index = -2

    num_classes, train_loader, test_loader = load_dataset(dataset_type=dataset_type, batch_size=128, num_workers=2)
    model = create_modular_model(model_type=model_type, num_classes=num_classes, modular_training_mode=False)
    model.load_pretrained_weights(checkpoint_path=checkpoint_path)

    layer_tuples = [(l_name, l) for (l_name, l) in get_model_leaf_layers(model, return_with_layer_name=True)
                    if isinstance(l, nn.ReLU)]
    nth_last_relu_layer = layer_tuples[nth_last_relu_layer_index]
    nth_last_relu_layer_name = nth_last_relu_layer[0]

    feature_extractor = create_feature_extractor(model, return_nodes={nth_last_relu_layer_name: "out"})
    old_forward_method = feature_extractor.forward
    feature_extractor.forward = lambda x: old_forward_method(x)["out"]

    outputs, labels = _get_model_outputs(feature_extractor, train_loader, device=DEVICE,
                                         num_classes=num_classes,
                                         show_progress=True)
    result = {nth_last_relu_layer_name: outputs.detach().cpu(), "labels": labels.detach().cpu()}
    # with open(os.path.join(".", f"layer_act_values.{model_type}_{dataset_type}.CohesionCouplingCompactness.pt"), "wb") as output_file:
    # with open(os.path.join(".", f"layer_act_values.{model_type}_{dataset_type}.CohesionCoupling.pt"), "wb") as output_file:
    with open(os.path.join(".", f"layer_act_values.{model_type}_{dataset_type}.STD.pt"), "wb") as output_file:
        torch.save(result, output_file)

