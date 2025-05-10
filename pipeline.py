import os
import time
import logging
import pandas as pd
from cfg import cfg
from utils import *
from flow import flow

import sys

best_val_loss = float('inf')
patience_counter = 0

log = []
timestamp = time.strftime("%Y_%m%d_%H%M")
folder_name = cfg.codedir + f"weights/{cfg.model_type}_{timestamp}_{cfg.tag}"
os.makedirs(folder_name, exist_ok=True)  # Create directory to save weights

model, optimizer = load_best_model_and_optimizer(cfg.model, cfg.optimizer, weights_dir=cfg.weight_dir)

from dataset.load_data_gpu import train_loader, val_loader, test_loader, logger

logger.setLevel(logging.WARNING)

for epoch in range(cfg.num_epochs * (1 - cfg.test) + cfg.test):
    # Validation only, without training
    if epoch == 0 and cfg.test == 0:
        avg_val_loss, _, _ = flow(model, val_loader, 'Validating')
        log.append({'epoch': 'Validation Only', 'val_loss': avg_val_loss})
        print(f'Validation Loss (Without Training): {avg_val_loss:.4f}')

    # Training
    loss, optimizer = flow(model, train_loader, 'Training', optimizer)

    # Validation
    avg_val_loss, _, _ = flow(model, val_loader, 'Validating')
    log.append({
        'epoch': epoch + 1,
        'train_loss': loss.item() if isinstance(loss, torch.Tensor) else loss,
        'val_loss': avg_val_loss})

    # Early stopping check
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        patience_counter = 0
        message = f'{cfg.model_type}_val_loss_{best_val_loss:.4f}_{timestamp}'
        torch.save(model.state_dict(), f'{folder_name}/{message}.pth')  # Save best model weights
        torch.save(optimizer.state_dict(), f'{folder_name}/{message}_optimizer.pth')  # Save optimizer state
        print(f"--Saving weights and optimizer: {folder_name}")
    else:
        patience_counter += 1
        if patience_counter >= cfg.patience:
            print("Early stopping triggered, terminating training.")
            break

    print(
        f'Epoch {epoch + 1}: Training Loss: {loss.item() if isinstance(loss, torch.Tensor) else loss:.4f}, Validation Loss: {avg_val_loss:.4f}, {patience_counter}/{cfg.patience}')

# Testing
model, _ = load_best_model_and_optimizer(model, weights_dir=folder_name, forceload=1)
mse, mae, smape = flow(model, test_loader, 'Testing')
rmse = torch.sqrt(torch.tensor(mse, dtype=torch.float))
print(f'MAE: {mae:.3f}, RMSE: {rmse:.3f}, SMAPE: {smape:.3f}')
log.append({'epoch': 'test', 'val_loss': rmse})

delete_too_much_file('weights')

# Save log to CSV
log_df = pd.DataFrame(log)
log_file = cfg.codedir + f'logs/{cfg.model_type}_{timestamp}_{cfg.tag}_{rmse:.4f}.csv'
log_df.to_csv(log_file, index=False)
print(f'Log saved to {log_file}')
print(cfg.message)
