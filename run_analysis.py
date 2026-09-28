import os
import json
import test
import tqdm
import torch
import numpy as np
from flags import parser
from utils import load_args, set_seed
from dataset import CompositionDataset
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
from model.configure_model import configure_model
from analysis.disentanglement_analysis import disentanglement_analysis
from analysis.flops_analysis import analyze_flops


device = 'cuda' if torch.cuda.is_available() else 'cpu'


if __name__ == '__main__':
    args = parser.parse_args()
    if args.config:
        load_args(args.config, args)
    print(args)

    dataset_path = args.dataset_path
    set_seed(args.seed)

    train_dataset = CompositionDataset(dataset_path, phase='train', split='compositional-split-natural', open_world=args.open_world)
    val_dataset = CompositionDataset(dataset_path, phase='val', split='compositional-split-natural', open_world=args.open_world)
    test_dataset = CompositionDataset(dataset_path, phase='test', split='compositional-split-natural', open_world=args.open_world)

    model, optimizer = configure_model(args, train_dataset)
    model.to(device)
    
    train_loader = DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=True)
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.eval_batch_size,
        shuffle=False
    )
   
    print("--- loading best validation checkpoint for final test evaluation ---")
    model.load_state_dict(torch.load(os.path.join(args.save_path, "val_best.pt")))

    analyze_flops(model, device)
    # model.eval()
    # analysis_results = disentanglement_analysis(model, train_loader, test_loader, device)
