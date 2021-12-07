import json
import argparse
import random
from pathlib import Path

from tqdm import tqdm
import wandb
import numpy as np
import torch
from torch.utils.data import DataLoader

from models.my_model import MyModel
from utils.load_dataset import load_dataset, split_dataset
from utils.logging import init_logging, log_visualizations, log_losses, log_final
from utils.visualization import create_visualizations, save_visualizations

def run(cfg):

    # PREP
    best_val_loss = float("inf")
    best_model_path = str((Path(cfg.out_dir) / 'best_model.pth').resolve())
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    init_logging(cfg)

    # DATA
    dataset = load_dataset(cfg)
    train_data, val_data, test_data = split_dataset(cfg, dataset)
    train_loader = DataLoader(train_data, batch_size=cfg.batch_size, shuffle=True, num_workers=4, drop_last=True)
    val_loader = DataLoader(val_data, batch_size=1, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=1, shuffle=False)

    # WandB
    if not cfg.no_wandb:
        wandb.init(config=cfg, project=cfg.project_name)

    # MODEL AND OPTIMIZER
    model = MyModel(cfg)
    optimizer, optimizer_scheduler = None, None
    if not cfg.no_train:
        optimizer = torch.optim.Adam(params=model.parameters(), lr=cfg.lr)
        optimizer_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5,
                                                                         min_lr=cfg.lr / 128.0, verbose=True)

    # MAIN LOOP
    for epoch in range(cfg.epochs):

        # train
        print(f'\nTraining (epoch: {epoch+1} of {cfg.epochs})')
        if not cfg.no_train:
            # use prediction model's own training loop if available
            if callable(getattr(model, "train_iter", None)):
                model.train_iter(cfg, train_loader, optimizer, epoch)
            else:
                train_iter(cfg, train_loader, model, optimizer)
        else:
            print("Skipping trianing loop.")

        # validation
        print("Validating...")
        val_losses, indicator_loss = val_iter(cfg, val_loader, model)
        if not cfg.no_train:
            optimizer_scheduler.step(indicator_loss)
        print("Validation losses (mean over entire validation set):")
        for k, v in val_losses.items():
            print(f" - {k}: {v}")

        # save model if validation loss improved
        cur_val_loss = indicator_loss.item()
        if best_val_loss > cur_val_loss:
            best_val_loss = cur_val_loss
            torch.save(model, best_model_path)
            print(f"Minimum indicator loss reduced -> model saved!")

        # visualize current model performance every Nth epoch, using eval mode and validation data
        if not cfg.no_vis and (epoch+1) % cfg.vis_every == 0:
            print("Saving visualizations...")
            visualizations = create_visualizations(cfg, val_data, model, cfg.num_vis)
            log_visualizations(cfg, visualizations)
            save_visualizations(cfg, visualizations)

        # final bookkeeping
        log_losses(cfg, val_losses)

    # TESTING
    print("\nTraining done, testing best model...")
    test_metrics = val_iter(cfg, test_loader, model, test=True)
    log_final(cfg, test_metrics)

    print("Done. bye bye!")
    return test_metrics


def train_iter(cfg, train_loader, model, optimizer):

    loop = tqdm(train_loader)
    for batch_idx, data in enumerate(loop):

        # prepare input
        input = NotImplemented

        # fwd
        output, model_losses = model(input)

        # loss
        loss = NotImplemented
        total_loss = loss
        for value in model_losses.values():
            total_loss += value

        # bwd
        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm(model.parameters(), 100)
        optimizer.step()

        # bookkeeping
        loop.set_postfix(loss=total_loss.item())
        # loop.set_postfix(mem=torch.cuda.memory_allocated())

def val_iter(cfg, val_loader, model, test=False):

    model.eval()
    loop = tqdm(val_loader)
    all_losses = []
    indicator_losses = []

    with torch.no_grad():
        for batch_idx, data in enumerate(loop):

            # prepare input
            input = NotImplemented

            # fwd
            output, model_losses = model(input)

            # metrics
            cur_metrics = {"loss": NotImplemented}
            if model_losses is not None:
                for k, v in model_losses.items():
                    cur_metrics[k] = v
            all_losses.append(cur_metrics)
            indicator_losses.append(cur_metrics[cfg.val_criterion])

    indicator_loss = torch.stack(indicator_losses).mean()
    all_losses = {
        k: torch.stack([loss_values[k] for loss_values in all_losses]).mean().item() for k in all_losses[0].keys()
    }

    model.train()
    return all_losses, indicator_loss

# ===========================================================

if __name__ == '__main__':

    with open("config.json", "w") as config_file:
        config = json.load(config_file)

    parser = argparse.ArgumentParser(description=config["PROJECT_NAME"])
    parser.add_argument("--a", type=float, default=config["a"], help="parameter a")
    parser.add_argument("--b", type=str, default=config["b"], help="parameter b")
    parser.add_argument("--c", type=bool, default=config["c"], action="store_true", help="parameter c")
    args = parser.parse_args()
    args.project_name = config["PROJECT_NAME"]
    run(args)