#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 14 12:23:06 2025

@author: souloke
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
from pathlib import Path

def dumpandplot(loss, energy, total_loss_batch, prefix, rundir):
    
##dump errors for 2rdm
    outputdir = Path(rundir)
    name1 = "losses"+prefix+".txt"
    np.savetxt(outputdir / name1, loss.cpu().numpy(), fmt="%.8f")

    name2 = "total_loss_fullbatch_2rdm_"+prefix+".txt"
    loss_batch = np.array(total_loss_batch)
    np.savetxt(outputdir / name2, loss_batch, fmt="%.8f")
    
    mean = loss_batch.mean()
    median = np.median(loss_batch)
    max_val = loss_batch.max()
    max_index = np.argmax(loss_batch)
#    p90 = np.percentile(loss, 90)

    threshold = np.percentile(loss_batch, 95)   # top 5%
    spike_indices = np.where(loss_batch >= threshold)[0]

    name3 = "errors_2rdm_"+prefix+".txt"
    with open(outputdir / name3, "w") as f:
        f.write(f"Mean: {mean:.8f}\n")
        f.write(f"Median: {median:.8f}\n")
        f.write(f"Max: {max_val:.8f}\n")
        f.write(f"Max_index: {max_index}\n")
        f.write(f"Spike Indices: {spike_indices.tolist()}\n")


##dump errors for energy

    n_batch = len(energy)
    e_pred=[]
    e_fci=[]
    for i in range(n_batch):
        e_pred.append(energy[i][0])
        e_fci.append(energy[i][1])
        
    e_pred_2plot = torch.cat(e_pred, dim=0)
    e_fci_2plot = torch.cat(e_fci, dim=0)     
    
    # Create a figure
    plt.figure(figsize=(8, 6))

#calculate mean energy error and write to file
    energy_error_mae = torch.abs(e_pred_2plot-e_fci_2plot)

    name4 = "total_loss_fullbatch_energy_"+prefix+".txt"
    loss_batch_energy = energy_error_mae.cpu().numpy()
    np.savetxt(outputdir / name4, loss_batch_energy, fmt="%.8f")
    
    mean = loss_batch_energy.mean()
    median = np.median(loss_batch_energy)
    max_val = loss_batch_energy.max()
    max_index = np.argmax(loss_batch_energy)
#    p90 = np.percentile(loss, 90)

    threshold = np.percentile(loss_batch_energy, 95)   # top 5%
    spike_indices = np.where(loss_batch_energy >= threshold)[0]

    name5 = "errors_energy_"+prefix+".txt"
    with open(outputdir / name5, "w") as f:
        f.write(f"Mean: {mean:.8f}\n")
        f.write(f"Median: {median:.8f}\n")
        f.write(f"Max: {max_val:.8f}\n")
        f.write(f"Max_index: {max_index}\n")
        f.write(f"Spike Indices: {spike_indices.tolist()}\n")




    mae = energy_error_mae.mean()
    name6 = "mae_energy_"+prefix+".txt"
    np.savetxt(outputdir / name6, np.atleast_1d(mae.cpu().numpy()), fmt="%.8f")    
    
# Plot both arrays
    
    plt.plot(e_fci_2plot.detach().cpu().numpy(), label="fci", color="black", marker='o')
    plt.plot(e_pred_2plot.detach().cpu().numpy(), label="ML", color="red", marker='s')

# Add title, labels, legend
    plt.title("fci vs ML")
#    plt.ylim(-1.175, -0.95)
#    plt.ylim(-2.95, -2.60)
#    plt.ylim(-1.175, -0.60)
#    plt.ylim(-7.90, -7.60)
#    plt.ylim(-5.75, -5.65)
#    plt.ylim(-7.90, -7.60)
#    plt.ylim(-107.75, -105.50)
#    plt.ylim(-14.68, -14.56)
#    plt.ylim(-109.10, -108.40)
#    plt.ylim(-109.2, -107.5)
#    plt.ylim(-157.308, -157.291)
#    plt.ylim(-77.2, -76.4)
#    plt.ylim(-157.308, -157.291)
    plt.ylim(-25.075, -24.850)
#    plt.xlabel("x")
    plt.ylabel("energies (in Hartrees")
    plt.legend()
#    plt.grid(True)

# Save figure as PDF
    name="pred_fci_"+prefix
    plt.savefig(outputdir / name)  # overwrites if exists
    plt.close()  # close figure to free memory