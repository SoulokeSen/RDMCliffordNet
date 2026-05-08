#import sidechainnet as scn

import engineer

from engineer.schedulers.cosine import CosineAnnealingLR
import torch
import numpy as np
#import h5py
import utils
torch.set_default_dtype(torch.float64)

def main(config):
#    print("config", config)
    dataset_config = config["dataset"]
    batch_size = dataset_config["batch_size"]
    n_train = dataset_config["n_train"]
    dataset = engineer.load_module(dataset_config.pop("module"))(**dataset_config)
    train_loader = dataset.train_loader()
    val_loader = dataset.val_loader()
    test_loader = dataset.test_loader()
    
    traindebug_loader = dataset.traindebug_loader()
# =============================================================================
#     count=0
#     print("priting from graph data class")
#     for k in train_loader :
#         print(k.loc.shape,k.vel.shape, k.edge_attr.shape, k.charges.shape, k.y.shape, k.edge_index.shape)
#         count += 1
#     print("number of batches",count)    
# =============================================================================
    
#    exit()
    model_config = config["model"]
    model = engineer.load_module(model_config.pop("module"))(**model_config)

#    model = model.cuda()
    optimizer_config = config["optimizer"]
    wlr = optimizer_config.pop("wlr")
    min_lrs = optimizer_config.pop("min_lrs")
    decay_steps = optimizer_config.pop("decay_steps")
#    print("optimizer config",optimizer_config)
#    exit()
#    print("optimizer_config", optimizer_config)
#    exit()

    if model_config["energyOnly"]:
        optimizer = engineer.load_module(optimizer_config.pop("module"))(
            model.parameters(), **optimizer_config
            )
    else:    
        optimizer = engineer.load_module(optimizer_config.pop("module"))(
            get_param_groups(model,wlr), **optimizer_config
            )

    # scheduler_config = config['scheduler']
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    #     optimizer, 
    #     mode=scheduler_config["mode"], 
    #     factor=scheduler_config["factor"], 
    #     patience=scheduler_config["patience"])


    # for name, param in model.named_parameters():
    #     print(name, param.shape, param.requires_grad)
    # name_to_param = dict(model.named_parameters())
    # for group_idx, group in enumerate(optimizer.param_groups):
    #     l = group["lr"]
    #     wd = group["weight_decay"]

    #     for p in group["params"]:
    #         for name, param in name_to_param.items():
    #             if p is param:
    #                 print(f"{name}: lr = {l} w = {wd} (group {group_idx})")
                    

    # exit()
    prntinterval = int(n_train/batch_size)
    print("printinteral", prntinterval)
    loss_config = config["loss"]
    loss_config["printstep"] = prntinterval
    loss = engineer.load_module(loss_config.pop("module"))(**loss_config)

    steps = config["trainer"]["max_steps"]
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, steps)
    scheduler = CosineAnnealingLR(
        optimizer,
        steps,
        warmup_steps=int(1 / 32 * steps),
        decay_steps = int(decay_steps * steps),
        min_lrs=min_lrs
#        decay_steps=int(1 / 4 * steps),
#        decay_steps  = int(0.75 * steps)
    )
#    scheduler=None

    trainer_module = engineer.load_module(config["trainer"].pop("module"))

    trainer_config = config["trainer"]
    trainer_config['run_dir'] = config['run_dir']
    trainer_config['use_wandb'] = 'wandb' in config
    
    trainer_config['val_check_interval'] = prntinterval
    trainer_config['log_interval'] = prntinterval
    
    trainer = trainer_module(
        **trainer_config,
    )
    train_loss, val_loss = trainer.fit(model, optimizer, loss, train_loader, scheduler, val_loader, test_loader=test_loader, debug_loader=traindebug_loader)

    ### write to file and save plot
#    with h5py.File("training_loss.h5", "w") as f:
    loss_matrix_train = torch.stack([train_loss[k] for k in sorted(train_loss.keys())])
#    f.create_dataset("loss", data=loss_matrix.numpy())
#    with h5py.File("validation_loss.h5", "w") as g:
    loss_matrix_val = torch.stack([val_loss[k] for k in sorted(val_loss.keys())])
    # Plot
    utils.plottrainvalepoch(loss_matrix_train[:,-1],loss_matrix_val[:,-1], config['run_dir'])
#    g.create_dataset("loss", data=loss_matrix.numpy())    

def get_param_groups_wd(model):
    decay = []
    no_decay = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        # Biases and normalization weights
#        if name in ["lossfn.a_w1", "lossfn.z_raw"] :
        if name in ["lossfn.log_w1_uu", "lossfn.log_w2_uu", "lossfn.log_w1_ud", "lossfn.log_w2_ud","lossfn.log_w1_dd","lossfn.log_w2_dd"]:
            no_decay.append(param)
        else:
            decay.append(param)

    return [
        {"params": decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]

def get_param_groups_lr(model):
    lr1 = []
    lr2 = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        # Biases and normalization weights
#        if name in ["lossfn.a_w1", "lossfn.z_raw"] :
        if name in ["lossfn.log_w1_uu", "lossfn.log_w2_uu", "lossfn.log_w1_ud", "lossfn.log_w2_ud","lossfn.log_w1_dd","lossfn.log_w2_dd"]:
            lr1.append(param)
        else:
            lr2.append(param)

    return [
        {"params": lr2},
        {"params": lr1, "lr": 1e-5},
    ]

def get_param_groups(model,losswlr):
    para1 = []
    para2 = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        # Biases and normalization weights
#        if name in ["lossfn.a_w1", "lossfn.z_raw"] :
        if name in ["lossfn.log_w1_uu", "lossfn.log_w2_uu", "lossfn.log_w1_ud", "lossfn.log_w2_ud","lossfn.log_w1_dd","lossfn.log_w2_dd"]:
#        if name in ["lossfn.log_w1_ud", "lossfn.log_w2_ud"]:
            para1.append(param)
        else:
            para2.append(param) 

    return [
        {"params": para2},
        {"params": para1, "lr": losswlr, "weight_decay": 0.0},
    ]

if __name__ == "__main__":
    engineer.fire(main)
