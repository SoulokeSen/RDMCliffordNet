import datetime
import os
import subprocess
import time
import warnings
from typing import Any

import torch
import torch.distributed as dist
from torch import nn
from torch.utils.data import DataLoader
import numpy as np


from .. import gradient_clip, metrics, callbacks, checkpointer, loggers, plot

class Queue():
    def __init__(self, max_len=50):
        self.items = []
        self.max_len = max_len

    def __len__(self):
        return len(self.items)

    def add(self, item):
        self.items.insert(0, item)
        if len(self) > self.max_len:
            self.items.pop()

    def mean(self):
        return np.mean(self.items)

    def std(self):
        return np.std(self.items)

def human_format(num: float):
    num = float("{:.3g}".format(num))
    magnitude = 0
    while abs(num) >= 1000:
        magnitude += 1
        num /= 1000.0
    return "{}{}".format(
        "{:f}".format(num).rstrip("0").rstrip("."), ["", "K", "M", "B", "T"][magnitude]
    )


def count_parameters(module: nn.Module):
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def to_device(input, device, detach=True):
    if isinstance(input, tuple):
        input = list(input)
    if isinstance(input, list):
        keys = range(len(input))
    elif isinstance(input, dict):
        keys = input.keys()
        return input
    elif isinstance(input, torch.Tensor):
        input = input.to(device)
        if detach:
            input = input.detach()
        return input
    else:
        return input
    for k in keys:
        input[k] = to_device(input[k], device)
    return input


def run_bash_command(command: str) -> str:
    result = subprocess.run(
        command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, shell=True
    )

    if result.returncode == 0:
        output = result.stdout.strip()
        return output
    else:
        error = result.stderr.strip()
        raise RuntimeError(f"Error executing command: {error}")


def parse_time_components(time_string: str):
    days, hours, minutes, seconds = 0, 0, 0, 0

    # Splitting days if present.
    if "-" in time_string:
        try:
            days_str, time_string = time_string.split("-")
        except:
            raise ValueError(f"Invalid time format {time_string}.")
        days = int(days_str)

    # Splitting hours, minutes, and seconds.
    time_components = time_string.split(":")
    num_components = len(time_components)

    if num_components == 3:
        hours, minutes, seconds = map(int, time_components)
    elif num_components == 2:
        minutes, seconds = map(int, time_components)
    elif num_components == 1:
        seconds = int(time_components[0])
    else:
        raise ValueError(f"Invalid time format {time_string}.")

    return days, hours, minutes, seconds


def parse_slurm_time(time_string) -> datetime.timedelta:
    days, hours, minutes, seconds = parse_time_components(time_string)
    return datetime.timedelta(days=days, hours=hours, minutes=minutes, seconds=seconds)


def _parse_max_time(time):
    if time is None:
        return

    if time is None and "SLURM_JOB_ID" in os.environ:
        time = run_bash_command(
            "squeue -j $SLURM_JOB_ID -h --Format TimeLimit"
        ).splitlines()
        if len(time) > 1:
            warnings.warn(
                "More than one job found (array job?). Using the first one for setting the time limit."
            )
        time = time[0]

    max_time = parse_slurm_time(time)
    return max_time


class Trainer:
    def __init__(
        self,
        run_dir=None,
        max_steps: int = 0,
        max_time: str = None,
        limit_val_batches: int = float("inf"),
        val_check_interval: int = 35,
        print_interval: int = 1,
        log_interval=35,
        use_wandb: bool = False,
        test_only=False,
        checkpoint=True,
        test_callbacks=list(),
        train_metrics=(),
        test_metrics=(),
        gradclip=None,
        energyOnly=False
    ):
        if use_wandb:
            logger = loggers.WANDBLogger()
        else:
            logger = loggers.ConsoleLogger()

        self.starting_time = datetime.datetime.now()
        self.max_time = _parse_max_time(max_time)
        self.run_dir = run_dir
        self.max_steps = max_steps
        self.limit_val_batches = limit_val_batches
        self.val_check_interval = val_check_interval
        self.logger = logger
        self.print_interval = print_interval
        self.log_interval = log_interval
        self.test_only = test_only
        self.checkpoint = checkpoint
        self.is_distributed = dist.is_initialized()
        
        self._should_raise = None
        self._should_stop = False

        self.train_metrics = metrics.setup_metrics(
            list(train_metrics) + ["mean_gradnorm", "max_gradnorm"]
        )
        self.test_metrics = metrics.setup_metrics(list(test_metrics))

        self.test_callbacks = callbacks.setup_callbacks(
            list(test_callbacks), logger=self.logger
        )

        if self.checkpoint:
            self.checkpointer = checkpointer.Checkpointer(
                self.run_dir, self.test_metrics
            )

        self.train_state = self._setup_train_state()

        self.gradclip = gradient_clip.setup_gradclip(gradclip)
        self.fitenergy=energyOnly
        if self.fitenergy:
            self.ncompdebug = 2
            self.ncompfit = 2
        else:    
            self.ncompdebug = 18
            self.ncompfit = 10
        

    def _setup_train_state(self):
        return {
            "global_step": 0,
            "last_global_step": 0,
            "current_epoch": 0,
            "batch_index": 0,
            "device": None,
        }

    def _add_prefix(self, metrics, prefix: str):
        return {f"{prefix}/{k}": v for k, v in metrics.items()}


    def train_step(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        loss: nn.Module,
        batch: Any,
    ):
        prefix = "train"
        model.train()
        batch = to_device(batch, self.device)  #do we need it here and inside the forward function ?

        loss, outputs, loss_comp, _ = model(batch, self.train_state["global_step"], prefix, loss, CalcNrep=False)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()

        gradient_norm = self.gradclip(model)

        outputs["mean_gradnorm"] = gradient_norm.unsqueeze(0)
        outputs["max_gradnorm"] = gradient_norm.unsqueeze(0)

        optimizer.step()

        if torch.isnan(loss):
            self._should_raise = ValueError("Loss is NaN.")

        if self.is_distributed:
            raise NotImplementedError
            model.module.train_metrics.update(**outputs)  # type: ignore
        else:
            self.train_metrics.update(**outputs)  # type: ignore

        if self.train_state["global_step"] % self.print_interval == 0:
            print(
                f"Step: {self.train_state['global_step']} (Training) Loss: {loss:.8f}", [f"{x:.8f}" for x in loss_comp]
            )

        return loss_comp 
    
    @torch.no_grad()
    def test_loop(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        loss:nn.Module,
        test_loader: DataLoader,
        validation=False,
#        test=False,
#        debugtrain=False
    ):
        model.eval()

        num_iterations = int(min(len(test_loader), self.limit_val_batches))
        t0 = time.time()

        if self.is_distributed:
            raise NotImplementedError
            assert model.module.test_metrics.empty()  # type: ignore
        else:
            assert self.test_metrics.empty()  # type: ignore
        if validation:
            print_str = "Validation"
            prefix = "val"
        # elif test:
        #     print_str = "Testing"
        #     prefix = "test"
        else:
            print_str = "Testing"
            prefix = "test"
         
        loss_components = torch.zeros(self.ncompfit, dtype=torch.float64) 
#        num_batch_val = len(test_loader)
        num_sample = 0
#        tot_energies=[]   
#        batch_outputs = []
        for batch_idx, batch in enumerate(test_loader):
            if batch_idx >= self.limit_val_batches:
                break
            
            batch = to_device(batch, self.device)
            batch_size = batch.ptr.shape[0] - 1
            _, outputs, loss_comp, _ = model(batch, batch_idx, prefix, loss, CalcNrep=False)
 #           _, outputs, _ , _ = model(batch, batch_idx, prefix)
#            tot_energies.append(e)
#            batch_outputs.extend(outputs["loss"].cpu().tolist())
            # _, outputs = model(batch, batch_idx)
            loss_components += batch_size*loss_comp
            num_sample += batch_size 

            if self.is_distributed:
                raise NotImplementedError
                model.module.test_metrics.update(**outputs)  # type: ignore
            else:
                self.test_metrics.update(**outputs)  # type: ignore

            if batch_idx % self.print_interval == 0:
                print(
                    f"Step: {self.train_state['global_step']} ({print_str}) Batch: {batch_idx} / {num_iterations}",
                    "Loss comp:", [f"{x:.4f}" for x in loss_comp]
                )
  
            # if batch_idx % self.print_interval == 0:
            #     print(
            #         f"Step: {self.train_state['global_step']} ({print_str}) Batch: {batch_idx} / {num_iterations}"
            #     )      
        loss_components /= num_sample
#        print("loss components", loss_components)
#        print("batch outputs mean", np.mean(batch_outputs))
        t1 = time.time()
        s_it = (t1 - t0) / num_iterations

        if self.is_distributed:
            raise NotImplementedError
            metrics = model.module.test_metrics.compute()
            model.module.test_metrics.reset()
        else:
            metrics = self.test_metrics.compute()  # type: ignore
            self.test_metrics.reset()  # type: ignore
        metrics[f"s_it"] = s_it

        metrics = self._add_prefix(metrics, prefix)

        if self.logger:
            self.logger.log_metrics(metrics, step=self.train_state["global_step"])

        for callback in self.test_callbacks:
            callback.on_test_end(
                model=model,
                batch=batch,
                batch_idx=batch_idx,
                step=self.train_state["global_step"],
                mode=prefix,
            )

#        if test: plot.dumpandplot(loss_components,tot_energies,batch_outputs,"test")
        
#        if debugtrain: plot.dumpandplot(loss_components,tot_energies,batch_outputs,"train_debug")
        
        if validation and self.checkpoint:
            validation_improved = self.checkpointer.on_test_end(
                self.train_state, model, optimizer, metrics, prefix
            )
#            if validation_improved:
#                plot.dumpandplot(loss_components,tot_energies,batch_outputs,"val")
            return validation_improved,loss_components 
#            return validation_improved

# =============================================================================
    @torch.no_grad()
    def train_debug(self,
                    model: nn.Module,
                    loss : nn.Module,
                    debug_loader: DataLoader,
                    datastr: str):
        
        model.eval()
#        print(f"DataLoader num_workers: {debug_loader.num_workers}")
#        print(" is it training mode", model.training)
        loss_components = torch.zeros(self.ncompdebug, dtype=torch.float64) 
#        num_batch_val = len(test_loader)
        num_sample = 0
        tot_energies=[]   
        batch_outputs = []
        for batch_idx, batch in enumerate(debug_loader):
            if batch_idx >= self.limit_val_batches:
                break
            
            batch = to_device(batch, self.device)
            batch_size = batch.ptr.shape[0] - 1
            _, outputs, loss_comp, e = model(batch, batch_idx, "debug", loss, CalcNrep=True)
            tot_energies.append(e)
            batch_outputs.extend(outputs["loss"].tolist())
            # _, outputs = model(batch, batch_idx)
            loss_components += batch_size*loss_comp
            num_sample += batch_size 

        
        loss_components /= num_sample
        plot.dumpandplot(loss_components,tot_energies,batch_outputs, datastr, self.run_dir)
# =============================================================================

    @property
    def should_stop(self):

        if self._should_stop:
            return True

        if (
            self.max_time is not None
            and self.max_time < datetime.datetime.now() - self.starting_time
        ):
            print("Stopping due to max_time.")
            self._should_stop = True
            return True
        if (
            self.max_steps is not None
            and self.train_state["global_step"] >= self.max_steps
        ):
            print("Stopping due to max_steps.")
            self._should_stop = True
            return True

        return False

    def fit(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        loss : nn.Module,
        train_loader,
        scheduler=None,
        val_loader=None,
        test_loader=None,
        debug_loader=None
    ):
        if hasattr(model, "device"):
            device = torch.device(model.device)
        else:
            device = next(model.parameters()).device
        self.device = device

        if torch.cuda.is_available() and not device.type == "cuda":
            warnings.warn("CUDA is available but not being used.")

        if not hasattr(train_loader, "sampler"):
            warnings.warn("Not using a conventional PyTorch data loader.")
        else:
            if not isinstance(
                train_loader.sampler, torch.utils.data.sampler.RandomSampler
            ):
                warnings.warn("Training loader has a non-random sampler!")

        print("\nModel Summary\n---")
        print(model)
        total_parameters = count_parameters(model)
        print(f"Total parameters: {human_format(total_parameters)}\n")
        print("i s distributed", self.is_distributed)
        # if self.checkpoint:
        #     self.checkpoint.restore(self, model, optimizer)

        if self.test_only:
            print(f"Testing mode.")
            with torch.no_grad():
                self.test_loop(model, optimizer, test_loader, validation=False)  # type: ignore
            return

        t0 = time.time()

        last_global_step = self.train_state["global_step"]
        
        ##instantitate loss module
        traning_loss_comp = {}
        validation_loss_comp = {}
#        loss_comp_train = torch.zeros(self.ncomp, dtype=torch.float64)
#        num_batch_train = len(train_loader)
        
        while not self.should_stop:
            if self.is_distributed:
                train_loader.sampler.set_epoch(self.train_state["current_epoch"])
  
            for batch in train_loader:
                loss_comp_train = self.train_step(model, optimizer, loss, batch)
#                exit()
                if scheduler is not None:
                    scheduler.step()

                lr = optimizer.param_groups[0]["lr"]

                if self.train_state["global_step"] % self.log_interval == 0:
                    t1 = time.time()
                    if self.is_distributed:
                        raise NotImplementedError
                        train_metrics = model.module.train_metrics.compute()  # type: ignore
                        model.module.train_metrics.reset()  # type: ignore
                    else:
                        train_metrics = self.train_metrics.compute()
                        self.train_metrics.reset()
                    s_it = (t1 - t0) / (
                        self.train_state["global_step"] + 1 - last_global_step
                    )
                    train_metrics["s_it"] = s_it
                    train_metrics["lr"] = lr
                    train_metrics["epoch"] = self.train_state["current_epoch"]
                    train_metrics["total_parameters"] = total_parameters

                    if self.logger:
                        train_metrics = self._add_prefix(train_metrics, "train")
                        self.logger.log_metrics(
                            train_metrics, step=self.train_state["global_step"]
                        )

                    t0 = time.time()
                    last_global_step = self.train_state["global_step"]
                    
                    traning_loss_comp[self.train_state["current_epoch"]] = loss_comp_train
                   

                should_validate = (
                    self.train_state["global_step"] % self.val_check_interval == 0
                )

                if should_validate:
                    should_test = True
                    if val_loader is not None and self.limit_val_batches > 0:
                        with torch.no_grad():
                            validation_improved, loss_comp_val = self.test_loop(
                                model, optimizer, loss, val_loader, validation=True
                            )
                            should_test = validation_improved

                    t0 = time.time()
                    last_global_step = self.train_state["global_step"]
                    if should_test:
                        if test_loader is not None:
                            with torch.no_grad():
                                self.test_loop(
                                    model, optimizer, loss, test_loader, validation=False
                                )
                        # if debug_loader is not None:
                        #     with torch.no_grad():
                        #         self.train_debug(model, debug_loader)                                
# =============================================================================
#                     if validation_improved : 
#                             with torch.no_grad():
#                                 self.train_debug(
#                                     model, optimizer, train_loader
#                                 )
# 
# =============================================================================
                self.train_state["global_step"] += 1
                if self._should_raise is not None:
                    raise self._should_raise

                if self.should_stop:
                    break
            
            validation_loss_comp[self.train_state["current_epoch"]] = loss_comp_val
#            loss_comp_val.zero_()
            self.train_state["current_epoch"] += 1
            
        if debug_loader is not None:
            bestmodel = self.checkpointer.best_paths["loss"]
 #           print("bestmodel", bestmodel)
            checkpoint = torch.load(bestmodel, map_location=self.device, weights_only=False)
            model.load_state_dict(checkpoint["model"])
            with torch.no_grad():
                self.train_debug(model, loss, debug_loader, "train_debug")
                self.train_debug(model, loss, val_loader,"val")
                self.train_debug(model, loss, test_loader,"test")
                
                        
        return  traning_loss_comp,   validation_loss_comp