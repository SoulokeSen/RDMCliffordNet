import abc
import functools
from typing import Callable, Optional, Tuple

import torch
import numpy as np
import torch.nn.functional as F
import utils
import torch.nn as nn
#from entmax import entmax15,sparsemax
import math

def compute_G_blocks (RDM1s, RDM2s):
    
    norbs = RDM1s[0].shape[1]
    batch_size = RDM1s[0].shape[0]

    I = torch.eye(norbs).unsqueeze(0).repeat(batch_size, 1, 1)
    # G^{αα, αα}
    G_aaaa = torch.einsum('bik,bjl->bijkl', RDM1s[0], I) \
            - RDM2s[0].permute(0, 1, 4, 3, 2)
#   G_blocks['alphaalpha_alphaalpha'] = G_aaaa

    # G^{ββ, ββ}
    G_bbbb = torch.einsum('bik,bjl->bijkl', RDM1s[1], I) \
            - RDM2s[2].permute(0,1, 4, 3, 2)
    
    G_aabb =  RDM2s[1].permute(0,1, 3, 4, 2)
#   G_blocks['betabeta_betabeta'] = G_bbbb

    # G^{αβ, αβ}
    G_bbaa =  RDM2s[1].permute(0, 2, 4, 3, 1)
    
  
    G_abab = torch.einsum('bik,bjl->bijkl', RDM1s[0], I) \
            - RDM2s[1].permute(0, 1, 4, 3, 2)  
  
    G_baba = torch.einsum('bik,bjl->bijkl', RDM1s[1], I) \
            - RDM2s[1].permute(0, 4, 1, 2, 3) 


    return   {"G_aaaa":G_aaaa,"G_bbbb":G_bbbb,"G_aabb":G_aabb,
              "G_bbaa":G_bbaa,"G_abab":G_abab,"G_baba":G_baba}
    
    
def compute_G_blocks_4H2 (RDM1s, RDM2s):
    
    norbs = RDM1s[0].shape[1]
    batch_size = RDM1s[0].shape[0]

    I = torch.eye(norbs).unsqueeze(0).repeat(batch_size, 1, 1)
    # G^{αα, αα}
    G_aaaa = torch.einsum('bik,bjl->bijkl', RDM1s[0], I) 
            
#   G_blocks['alphaalpha_alphaalpha'] = G_aaaa

    # G^{ββ, ββ}
    G_bbbb = torch.einsum('bik,bjl->bijkl', RDM1s[1], I) 
            
    
    G_aabb =  RDM2s.permute(0,1, 3, 4, 2)
#   G_blocks['betabeta_betabeta'] = G_bbbb

    # G^{αβ, αβ}
    G_bbaa =  RDM2s.permute(0, 2, 4, 3, 1)
    
  
    G_abab = torch.einsum('bik,bjl->bijkl', RDM1s[0], I) \
            - RDM2s.permute(0, 1, 4, 3, 2)  
  
    G_baba = torch.einsum('bik,bjl->bijkl', RDM1s[1], I) \
            - RDM2s.permute(0, 4, 1, 2, 3) 


    return   {"G_aaaa":G_aaaa,"G_bbbb":G_bbbb,"G_aabb":G_aabb,
              "G_bbaa":G_bbaa,"G_abab":G_abab,"G_baba":G_baba}



def compute_Q_blocks (RDM1s, RDM2s):   
    
    norbs = RDM1s[0].shape[1]
    batch_size = RDM1s[0].shape[0]
    
#    Q_aaaa = torch.zeros_like(RDM2s[0])
#    Q_bbbb = torch.zeros_like(RDM2s[2])
#    Q_abab = torch.zeros_like(RDM2s[1])
    
  #Q_aaaa
    I = torch.eye(norbs).unsqueeze(0).repeat(batch_size, 1, 1)
    
    Q_aaaa = torch.einsum('bik,bjl->bijkl', I, I) \
                -torch.einsum('bil,bjk->bijkl', I, I) \
                - torch.einsum('bik,bjl->bijkl', I, RDM1s[0]) \
                + torch.einsum('bil,bjk->bijkl', I, RDM1s[0]) \
                + torch.einsum('bjk,bil->bijkl', I, RDM1s[0]) \
                - torch.einsum('bjl,bik->bijkl', I, RDM1s[0]) \
                + RDM2s[0].permute(0, 3, 4, 1, 2)
    
#Q_bbbb
        
    Q_bbbb = torch.einsum('bik,bjl->bijkl', I, I) \
             -torch.einsum('bil,bjk->bijkl', I, I) \
             -torch.einsum('bik,bjl->bijkl', I, RDM1s[1]) \
             + torch.einsum('bil,bjk->bijkl', I, RDM1s[1]) \
             + torch.einsum('bjk,bil->bijkl', I, RDM1s[1]) \
             - torch.einsum('bjl,bik->bijkl', I, RDM1s[1]) \
             + RDM2s[2].permute(0, 3, 4, 1, 2)  
    
    
#Q_abab

    Q_abab = torch.einsum('bik,bjl->bijkl', I, I) \
            - torch.einsum('bik,bjl->bijkl', I, RDM1s[1]) \
            - torch.einsum('bjl,bik->bijkl', I, RDM1s[0]) \
            + RDM2s[1].permute(0, 3, 4, 1, 2)       
    
    return [Q_aaaa, Q_abab, Q_bbbb]
 

def compute_Q_blocks_4H2 (RDM1s, RDM2s):   
    
    norbs = RDM1s[0].shape[1]
    batch_size = RDM1s[0].shape[0]
    
#    Q_aaaa = torch.zeros_like(RDM2s[0])
#    Q_bbbb = torch.zeros_like(RDM2s[2])
#    Q_abab = torch.zeros_like(RDM2s[1])
    
  #Q_aaaa
    I = torch.eye(norbs).unsqueeze(0).repeat(batch_size, 1, 1)
    
    Q_aaaa = torch.einsum('bik,bjl->bijkl', I, I) \
                -torch.einsum('bil,bjk->bijkl', I, I) \
                - torch.einsum('bik,bjl->bijkl', I, RDM1s[0]) \
                + torch.einsum('bil,bjk->bijkl', I, RDM1s[0]) \
                + torch.einsum('bjk,bil->bijkl', I, RDM1s[0]) \
                - torch.einsum('bjl,bik->bijkl', I, RDM1s[0]) 
    
#Q_bbbb
        
    Q_bbbb = torch.einsum('bik,bjl->bijkl', I, I) \
             -torch.einsum('bil,bjk->bijkl', I, I) \
             -torch.einsum('bik,bjl->bijkl', I, RDM1s[1]) \
             + torch.einsum('bil,bjk->bijkl', I, RDM1s[1]) \
             + torch.einsum('bjk,bil->bijkl', I, RDM1s[1]) \
             - torch.einsum('bjl,bik->bijkl', I, RDM1s[1])  
    
    
#Q_abab

    Q_abab = torch.einsum('bik,bjl->bijkl', I, I) \
            - torch.einsum('bik,bjl->bijkl', I, RDM1s[1]) \
            - torch.einsum('bjl,bik->bijkl', I, RDM1s[0]) \
            + RDM2s.permute(0, 3, 4, 1, 2)       
    
    return [Q_aaaa, Q_abab, Q_bbbb]

    

def softened_l1_norm(x, epsilon= 1e-6):
    
    return torch.sqrt(epsilon**2 + x**2) - epsilon
        

def mse_loss(pred,target):
    return F.mse_loss(pred, target, reduction='none')

def trace_spin_block (RDM2s):
    return torch.einsum('bijij->b', RDM2s)

def trace_spin_block_1rdm(rdm1s):
    return torch.einsum('bii->b', rdm1s)

def compute_penalty_function (M, b, tensortype, tol=1e-12):
    
    if tensortype == "D" or tensortype == "Q":
        
        eig_ud = utils.diag(M[1])[0]
        eig_u = utils.diag (M[0],True)[0]
        eig_d = utils.diag (M[2],True)[0]
        
        
        loss_penalty = torch.where(eig_ud < -tol, eig_ud**2, torch.zeros_like(eig_ud)).sum(dim=1)
        
        loss_penalty =  loss_penalty + torch.where(eig_u < -tol, eig_u**2, torch.zeros_like(eig_u)).sum(dim=1)
        loss_penalty = loss_penalty + torch.where(eig_d < -tol, eig_d**2, torch.zeros_like(eig_d)).sum(dim=1)

        return loss_penalty

    else:

        
        M_uudd = torch.cat([
            torch.cat([utils.create_matrix_full(M["G_aaaa"]), utils.create_matrix_full(M["G_aabb"])], dim=2),
            torch.cat([utils.create_matrix_full(M["G_bbaa"]), utils.create_matrix_full(M["G_bbbb"])], dim=2),
            ], dim=1)        

        eig_uudd, _ = torch.linalg.eigh(M_uudd)
        eig_udud = utils.diag(M["G_abab"])[0]
        eig_dudu = utils.diag(M["G_baba"])[0]
        

        loss_penalty = torch.where(eig_uudd < -tol, eig_uudd**2, torch.zeros_like(eig_uudd)).sum(dim=1)
        loss_penalty = loss_penalty + torch.where(eig_udud < -tol, eig_udud**2, torch.zeros_like(eig_udud)).sum(dim=1)
        loss_penalty = loss_penalty + torch.where(eig_dudu < -tol, eig_dudu**2, torch.zeros_like(eig_dudu)).sum(dim=1)

        return loss_penalty
        

class RDM2sLoss(nn.Module): #SS

    def __init__(self, printstep, energyOnly, weight_reg, regtype, weights=None,target_ratio=5.25, lambda_ratio=0.5): #weights for the multiobjective loss function
        super().__init__()
#        self.mse = nn.MSELoss()
        self.weights = weights
        self.printstep = printstep
        self.fitenergyonly = energyOnly

# =============================================================================
#         self.ratio_target = 5.0
#         self.register_buffer("log_ratio_target",
#                               torch.tensor(float(torch.log(torch.tensor(self.ratio_target)))))
# 
#         self.a_w1 = nn.Parameter(torch.tensor(0.0))   
# #        self.a_w1 = torch.tensor(0.0)   
#         z_raw_init = torch.log(torch.tensor(self.ratio_target - 1.0))
#         z_raw_init +=  0.1 * torch.randn(())
#         self.z_raw = nn.Parameter(z_raw_init.clone())
# 
# 
#         self.b = 0.05
# 
# =============================================================================
#        self.log_b = nn.Parameter(torch.tensor(math.log(0.05)))
      
#        torch.nn.init.normal_(self.a, mean=0.0, std=0.01)
        
#        torch.nn.init.normal_(self.z_raw, mean=self.log_ratio_target, std=0.01)
#        torch.nn.init.normal_(self.log_b, mean=torch.tensor(math.log(0.03)), std=0.01)

#        self.alpha = 8.0
#        self.beta =  0.16


# =============================================================================
#         self.log_w1 = nn.Parameter(torch.tensor(0.0))
#         self.log_w2 = nn.Parameter(torch.log(torch.tensor(target_ratio)))

#         self.target_ratio = target_ratio
#         self.lambda_ratio = lambda_ratio
# #        self.gamma_scale = gamma_scale
# =============================================================================
        if not self.fitenergyonly:
            self.log_w1_uu = nn.Parameter(torch.tensor(0.0))
            self.log_w2_uu = nn.Parameter(torch.tensor(0.0))
            self.log_w1_ud = nn.Parameter(torch.tensor(0.0))
            self.log_w2_ud = nn.Parameter(torch.tensor(0.0))
            self.log_w1_dd = nn.Parameter(torch.tensor(0.0))
            self.log_w2_dd = nn.Parameter(torch.tensor(0.0))
            self.lambda_log = weight_reg
            self.regtype = regtype
        # self.log_w1 = nn.Parameter(torch.tensor(0.0))
        # self.log_w2 = nn.Parameter(torch.tensor(0.0))

    def forward(self, e_pred, pred, batch, step, metric, calcnrepconidtion):


    
        RDM1s_t = [batch.rdm1s_u, batch.rdm1s_d]
        
        target = [batch.rdm2s_uu, batch.rdm2s_ud, batch.rdm2s_dd]
        eris = [batch.eris_uu, batch.eris_ud, batch.eris_dd]
        e_twobody_t = batch.e_twobody
        fci_energy = batch.fci_energy
        batch_size = batch.ptr.shape[0] - 1
        components=[]
#        print("shape of n_eleces", batch.n_elecs.shape)

        N_u = batch.n_elecs[:,0]
        N_d = batch.n_elecs[:,1] 
        


# =============================================================================
# =============================================================================
#=============================================================================


#        loss_ud, loss_nz, loss_z = self._masked_MAEloss_pw2(pred[1], target[1])
        if not self.fitenergyonly:
            loss_uu, loss_nz_uu, loss_z_uu, w_nz_uu, w_z_uu = self._masked_MAEloss_pw3(pred[0], utils.create_matrix_upper(target[0]), param=[self.log_w1_uu,self.log_w2_uu])
            loss_ud, loss_nz_ud, loss_z_ud, w_nz_ud, w_z_ud = self._masked_MAEloss_pw3(pred[1], target[1], param=[self.log_w1_ud,self.log_w2_ud])
            loss_dd, loss_nz_dd, loss_z_dd, w_nz_dd, w_z_dd = self._masked_MAEloss_pw3(pred[2], utils.create_matrix_upper(target[2]), param=[self.log_w1_dd,self.log_w2_dd])


        # loss_uu, loss_nz_uu, loss_z_uu, w_nz_uu, w_z_uu = self._masked_MAEloss_pw3(pred[0], utils.create_matrix_upper(target[0]))
        # loss_ud, loss_nz_ud, loss_z_ud, w_nz_ud, w_z_ud = self._masked_MAEloss_pw3(pred[1], target[1])
        # loss_dd, loss_nz_dd, loss_z_dd, w_nz_dd, w_z_dd = self._masked_MAEloss_pw3(pred[2], utils.create_matrix_upper(target[2]))

        # loss_uu = F.l1_loss(pred[0], utils.create_matrix_upper(target[0]), reduction='none').mean(dim=(1,2))
        # loss_ud = F.l1_loss(pred[1], target[1], reduction='none').mean(dim=(1,2,3,4))
        # loss_dd = F.l1_loss(pred[2], utils.create_matrix_upper(target[2]),reduction='none').mean(dim=(1,2))



        # loss_uu, loss_nz_uu, loss_z_uu,_,_ = self._masked_MAEloss_pw4(pred[0], utils.create_matrix_upper(target[0]), utils.create_matrix_upper(eris[0]),param=[self.log_w1_uu,self.log_w2_uu])
        # loss_ud, loss_nz_ud, loss_z_ud,_,_ = self._masked_MAEloss_pw4(pred[1], target[1], eris[1],param=[self.log_w1_ud,self.log_w2_ud])
        # loss_dd, loss_nz_dd, loss_z_dd,_,_ = self._masked_MAEloss_pw4(pred[2], utils.create_matrix_upper(target[2]), utils.create_matrix_upper(eris[2]),param=[self.log_w1_dd,self.log_w2_dd])



        # loss_uu, loss_nz_uu, loss_z_uu = self._masked_MAEloss_pw3(pred[0], utils.create_matrix_upper(target[0]))
        # loss_ud, loss_nz_ud, loss_z_ud = self._masked_MAEloss_pw3(pred[1], target[1])
        # loss_dd, loss_nz_dd, loss_z_dd = self._masked_MAEloss_pw3(pred[2], utils.create_matrix_upper(target[2]))



        # loss_uu, loss_nz_uu, loss_z_uu,_,_ = self._masked_MAEloss_pw3(pred[0], utils.create_matrix_upper(target[0]), param=[self.log_w1,self.log_w2])
        # loss_ud, loss_nz_ud, loss_z_ud,_,_ = self._masked_MAEloss_pw3(pred[1], target[1], param=[self.log_w1,self.log_w2])
        # loss_dd, loss_nz_dd, loss_z_dd,_,_ = self._masked_MAEloss_pw3(pred[2], utils.create_matrix_upper(target[2]),param=[self.log_w1,self.log_w2])

        # loss_uu = eri_weighted_MAEloss(pred[0], utils.create_matrix_upper(target[0]), utils.create_matrix_upper(eris[0]))
        # loss_ud = eri_weighted_MAEloss(pred[1], target[1], eris[1])
        # loss_dd = eri_weighted_MAEloss(pred[2], utils.create_matrix_upper(target[2]), utils.create_matrix_upper(eris[2]))



#        loss_ud = F.mse_loss(pred[1], target[1], reduction='none').mean(dim=(1,2,3,4))
#        loss_ud = eri_weighted_MAEloss(pred[1], target[1], eris[1])
# 
# #        total = loss_uu + loss_dd + loss_ud

            total = loss_uu + loss_ud + loss_dd
# #        total = loss_ud

            components.append(loss_uu.mean())
            components.append(loss_nz_uu.mean())
            components.append(loss_z_uu.mean())
        
            components.append(loss_ud.mean())
            components.append(loss_nz_ud.mean())  
            components.append(loss_z_ud.mean())
        
            components.append(loss_dd.mean())
            components.append(loss_nz_dd.mean())  
            components.append(loss_z_dd.mean())        
        


            if calcnrepconidtion:
            
                tr_uu = trace_spin_block(utils.expand_symmetric_matrix_to_tensor(pred[0], metric[0]))
                loss_tr_uu = F.l1_loss(tr_uu, N_u*(N_u-1), reduction='none')
                components.append(loss_tr_uu.mean()) 

                tr_ud = trace_spin_block(pred[1])
                loss_tr_ud = F.l1_loss(tr_ud, N_u*N_d, reduction='none')
                components.append(loss_tr_ud.mean()) 
        
                tr_dd = trace_spin_block(utils.expand_symmetric_matrix_to_tensor(pred[2],metric[1]))
                loss_tr_dd = F.l1_loss(tr_dd, N_d*(N_d-1), reduction='none')
                components.append(loss_tr_dd.mean())         



#        RDM1s_p = utils.rdm2s_to_rdm1s_4H2(pred[1])
                RDM1s_p = utils.rdm2s_to_rdm1s([utils.expand_symmetric_matrix_to_tensor(pred[0], metric[0]),
                                            pred[1],
                                            utils.expand_symmetric_matrix_to_tensor(pred[2], metric[1])])        

                  
                n = RDM1s_p[0].size(-1)
                loss_u_diag = F.l1_loss(RDM1s_p[0].diagonal(dim1=-2, dim2=-1),RDM1s_t[0].diagonal(dim1=-2, dim2=-1),reduction='none').mean(dim=1)
                loss_u_off_diag = torch.abs(RDM1s_p[0] - torch.diag_embed(RDM1s_p[0].diagonal(dim1=-2, dim2=-1))).sum(dim=(-2,-1))/(n * (n - 1))
#        total = total + (loss_u_diag + loss_u_off_diag)

                
                loss_d_diag = F.l1_loss(RDM1s_p[1].diagonal(dim1=-2, dim2=-1),RDM1s_t[1].diagonal(dim1=-2, dim2=-1),reduction='none').mean(dim=1)        
                loss_d_off_diag = torch.abs(RDM1s_p[1] - torch.diag_embed(RDM1s_p[1].diagonal(dim1=-2, dim2=-1))).sum(dim=(-2,-1))/(n * (n - 1))
#        total = total + (loss_d_diag + loss_d_off_diag)

                components.append((loss_u_diag+loss_d_diag).mean())
                components.append((loss_u_off_diag+loss_d_off_diag).mean())
  

                tr_u = trace_spin_block_1rdm(RDM1s_p[0])
                tr_d = trace_spin_block_1rdm(RDM1s_p[1])
                loss_tr_u = F.l1_loss(tr_u, N_u, reduction='none')
                loss_tr_d = F.l1_loss(tr_d, N_d, reduction='none')
#        total = total + (loss_tr_u + loss_tr_d)
                components.append((loss_tr_u+loss_tr_d).mean())        
#

                G_block = compute_G_blocks(RDM1s_p, [utils.expand_symmetric_matrix_to_tensor(pred[0], metric[0]),
                                                 pred[1],
                                                 utils.expand_symmetric_matrix_to_tensor(pred[2], metric[1])])      
                Q_block = compute_Q_blocks(RDM1s_p, [utils.expand_symmetric_matrix_to_tensor(pred[0], metric[0]),
                                                 pred[1],
                                                 utils.expand_symmetric_matrix_to_tensor(pred[2], metric[1])])   
       #        loss_penalty_D = compute_penalty_function(pred, batch_size, "D")
                loss_penalty_Q = compute_penalty_function(Q_block, batch_size, "Q")       
                loss_penalty_G = compute_penalty_function(G_block, batch_size, "G")

                components.append(loss_penalty_Q.mean()) 
                components.append(loss_penalty_G.mean())           


#       total = total + (loss_penalty_Q + loss_penalty_G)
#        components.append((loss_penalty_Q + loss_penalty_G).mean())         
# =============================================================================
# =============================================================================

# =============================================================================
#         
# =============================================================================
#=============================================================================
# =============================================================================
# additionally total energy loss

            e_twobody_p =  (1 / 4 * torch.einsum("bijkl,bijkl->b", eris[0], utils.expand_symmetric_matrix_to_tensor(pred[0], metric[0])) 
                            + torch.einsum("bijkl,bijkl->b", eris[1], pred[1]) 
                            + 1 / 4 * torch.einsum("bijkl,bijkl->b", eris[2], utils.expand_symmetric_matrix_to_tensor(pred[2],metric[1])))
        
#        e_twobody_p = torch.einsum("bijkl,bijkl->b", eris[1], pred[1])

        else:
            e_twobody_p = e_pred 
            loss_etwobody = F.l1_loss(e_twobody_p, e_twobody_t, reduction='none')

            total = loss_etwobody
            components.append(loss_etwobody.mean())
        
        components.append(total.mean())
        ####print all loss components meaned over batch####
        if step % self.printstep == 0:
            torch.set_printoptions(precision=8)
            print("########### Loss components ##################################")
            print ("                                                             ")
#            print("1RDM MSE loss (diag, off-diag):", (loss_diag + loss_off_diag).mean(), loss_diag.mean(), loss_off_diag.mean())

##            print("2RDM Trace loss               :", loss_tr_ud.mean())
##            print("2RDM wMSE loss                 :", loss_ud.mean())
##            print("trace of 2-RDM                 :", tr_ud.mean()  )
            if not self.fitenergyonly:
                print("2-RDM  loss uu                 :", loss_uu.mean())
                print("2-RDM  nreg                    :", loss_nz_uu.mean())
                print("2-RDM zero                     :", loss_z_uu.mean())
                print("  ")
                print("2-RDM  loss ud                 :", loss_ud.mean())
                print("2-RDM  nreg                    :", loss_nz_ud.mean())
                print("2-RDM zero                     :", loss_z_ud.mean())   
                print("  ")
                print("2-RDM  loss dd                 :", loss_dd.mean())
                print("2-RDM  nreg                    :", loss_nz_dd.mean())
                print("2-RDM zero                     :", loss_z_dd.mean())              
                print(" ")
                print(" ")
                print(" the weights :", w_nz_uu, w_z_uu, w_nz_ud, w_z_ud, w_nz_dd, w_z_dd)
                print(" ")
            else:
                print("two-body energy           :", loss_etwobody.mean())    

            # print("w_reg                          :", w1)
            # print("w_pen                          :", w2)
            # print("w_reg/w_pen                    :", w1/w2)
#            print(" prior b                       :",hyper_b)
#            print("2-rdm kl loss                  :", loss_kl.mean())
#            print("1-RDM wMAE loss                :", (loss_u + loss_d).mean())
#            print("1-RDM diagonal loss            :", (loss_u_diag+loss_d_diag).mean())
#            print("1-RDM off-diagonal loss        :", (loss_u_off_diag+loss_d_off_diag).mean())
#            print("2-RDM loss zero                :", loss_z.mean())
#            print("1RDM wMSE loss                 :", (loss_u + loss_d).mean())
#            print("1-RDM trace loss              :", (loss_tr_u + loss_tr_d).mean())

#         print(" 2RDM MSE loss            :", (mse_uu + mse_dd + mse_ud).mean())
#            print("penalties from Q , G         :", (loss_penalty_Q + loss_penalty_G).mean())
#            print("two-body energy           :", loss_etwobody.mean())
            print("------------------------------------------")
            print("Total                     :", total.mean())
            print("########### Loss components ##################################")
            print("                                                              ")

        
        return total, components, [(e_twobody_p+batch.e_onebody+batch.e_nuclear_repulsion),fci_energy]
    
    def  _masked_MAEloss_uw(self, pred, target, eps=1e-8, tau=1e-12):


        if pred.shape != target.shape:
            raise ValueError(f"pred and target must have the same shape, got {pred.shape} vs {target.shape}")

        reduce_axes = tuple(range(1, pred.ndim))

        mask = (target.abs() > eps).float()
        
        s_pen = self.s_reg - F.softplus(self.delta)
        w_nz = 0.5 * torch.exp(-self.s_reg) 
        w_z = 0.5 * torch.exp(-s_pen)

        mask1 = (target.abs() > 1e-4).float() 
        mask2 = mask - mask1 
    #    weights = 0.5*mask2+1.0*mask1    
              
        MAE_nz = (mask * torch.abs(pred - target)).sum(dim=reduce_axes)/(mask.sum(dim=reduce_axes) + eps)    
        zero_penalty = ((1 - mask) * torch.abs(pred)).sum(dim=reduce_axes)/((1 - mask).sum(dim=reduce_axes) + eps)
#        MAE_nz = torch.abs(pred - target).mean(dim=reduce_axes)
#        zero_penalty = torch.abs(pred).mean(dim=reduce_axes)
        
        ratio = torch.exp(F.softplus(self.delta))
        prior = self.beta * (torch.log(ratio) - torch.log(torch.tensor(self.ratio_prior)))**2
#    return (w_l*loss_nz_l+w_s*loss_nz_s+w_z*zero_penalty), loss_nz_l, loss_nz_s, zero_penalty
        return (w_nz*MAE_nz + w_z*zero_penalty + 0.5 * (self.s_reg + s_pen) + prior), MAE_nz, zero_penalty, w_nz, w_z


    def  _masked_MAEloss_pw3(self, pred, target, param=None, eps=1e-8, tau=1e-12):


        if pred.shape != target.shape:
            raise ValueError(f"pred and target must have the same shape, got {pred.shape} vs {target.shape}")

        reduce_axes = tuple(range(1, pred.ndim))

        mask = (target.abs() > eps).float()
        
        if param is not None:       

            w_nz = torch.exp(param[0])
            w_z = torch.exp(param[1])
        
        else:
            # w_nz = 1.0
            # w_z = 5.25

            w_nz = 1.0
            w_z = 1.0

        
        MAE_nz = (mask * torch.abs(pred - target)).sum(dim=reduce_axes)/(mask.sum(dim=reduce_axes) + eps)    
        zero_penalty = ((1 - mask) * torch.abs(pred)).sum(dim=reduce_axes)/((1 - mask).sum(dim=reduce_axes) + eps)

        weighted_loss = w_nz*MAE_nz + w_z*zero_penalty


        if param is not None:  
            if self.regtype == "L1" :
                scale_penalty = torch.abs(param[0]) + torch.abs(param[1])
            else:   
                scale_penalty = param[0]**2 + param[1]**2

        if param is not None:
#            return weighted_loss , MAE_nz, zero_penalty, w_nz.detach(), w_z.detach()
            return (weighted_loss + self.lambda_log * scale_penalty) , MAE_nz, zero_penalty, w_nz.detach(), w_z.detach()
        else:
            return weighted_loss , MAE_nz, zero_penalty, w_nz, w_z

 
    def  _masked_MAEloss_pw4(self, pred, target, eris, param=None,  eps=1e-8, tau=1e-12):


        if pred.shape != target.shape:
            raise ValueError(f"pred and target must have the same shape, got {pred.shape} vs {target.shape}")

        reduce_axes = tuple(range(1, pred.ndim))

        mask = (target.abs() > eps).float()
        nz_weights = torch.abs(eris) * mask
 
        if param is not None:       

            w_nz = torch.exp(param[0])
            w_z = torch.exp(param[1])
        
        else:
            w_nz = 1.0
            w_z = 5.25
        
        MAE_nz = (nz_weights * torch.abs(pred - target)).sum(dim=reduce_axes)/(nz_weights.sum(dim=reduce_axes) + eps)    
        zero_penalty = ((1 - mask) * torch.abs(pred)).sum(dim=reduce_axes)/((1 - mask).sum(dim=reduce_axes) + eps)

        weighted_loss = w_nz*MAE_nz + w_z*zero_penalty
        scale_penalty = torch.abs(param[0]) + torch.abs(param[1])

        if param is not None:
            # return weighted_loss , MAE_nz, zero_penalty, w_nz.detach(), w_z.detach()
             return (weighted_loss + self.lambda_log * scale_penalty) , MAE_nz, zero_penalty, w_nz.detach(), w_z.detach()
        else:
             return weighted_loss , MAE_nz, zero_penalty

   
# =============================================================================
    def  _masked_MAEloss_pw(self, pred, target, eps=1e-8, tau=1e-12):


        if pred.shape != target.shape:
            raise ValueError(f"pred and target must have the same shape, got {pred.shape} vs {target.shape}")

        reduce_axes = tuple(range(1, pred.ndim))

        mask = (target.abs() > eps).float()
        
        
#        w_nz = 1.0
#        w_z = 5.25
        w_nz = torch.exp(self.log_w1)
        w_z = torch.exp(self.log_w2)


        
        MAE_nz = (mask * torch.abs(pred - target)).sum(dim=reduce_axes)/(mask.sum(dim=reduce_axes) + eps)    
        zero_penalty = ((1 - mask) * torch.abs(pred)).sum(dim=reduce_axes)/((1 - mask).sum(dim=reduce_axes) + eps)
#        MAE_nz = torch.abs(pred - target).mean(dim=reduce_axes)
#        zero_penalty = torch.abs(pred).mean(dim=reduce_axes)
        weighted_loss = w_nz*MAE_nz + w_z*zero_penalty
        ratio_penalty = torch.abs(
            self.log_w2 - self.log_w1
            - torch.log(torch.tensor(self.target_ratio, device=MAE_nz.device))
        )
        
#        scale_penalty = torch.abs(self.log_w1 + self.log_w2)
#        return (w_nz*MAE_nz + w_z*zero_penalty + norm_loss + ratio_prior), MAE_nz, zero_penalty, w_nz.detach(), w_z.detach(), self.b
        return (weighted_loss + self.lambda_ratio * ratio_penalty), MAE_nz, zero_penalty
# =============================================================================


    def  _masked_MAEloss_pw2(self, pred, target, eps=1e-8, tau=1e-12):


        if pred.shape != target.shape:
            raise ValueError(f"pred and target must have the same shape, got {pred.shape} vs {target.shape}")

        reduce_axes = tuple(range(1, pred.ndim))

        mask = (target.abs() > eps).float()
        
        
        w_nz = 1.0
        w_z = 5.25

# =============================================================================
#         w_nz = torch.exp(self.a_w1)
#         z = F.softplus(self.z_raw)        # ensures z > 0
#         w_z = torch.exp(self.a_w1 + z)
#         norm_loss = -torch.log(w_nz) - torch.log(w_z)
# #        b = torch.exp(self.log_b)
#         ratio_prior = (z - self.log_ratio_target).abs() / self.b
# #        precision = 1.0 / b
# #        gamma_prior = (
# #            self.beta * precision
# #           - (self.alpha - 1.0) * torch.log(precision)
# #        )
# 
# =============================================================================
#        mask1 = (target.abs() > 1e-4).float() 
#        mask2 = mask - mask1 
    #    weights = 0.5*mask2+1.0*mask1    
#        attn_weighted_MAE = masked_weightedonloss_regression_loss(pred, target, mask, method='softmax')
    #    z_weights = torch.zeros_like(target)
    
        
        MAE_nz = (mask * torch.abs(pred - target)).sum(dim=reduce_axes)/(mask.sum(dim=reduce_axes) + eps)    
        zero_penalty = ((1 - mask) * torch.abs(pred)).sum(dim=reduce_axes)/((1 - mask).sum(dim=reduce_axes) + eps)
#        MAE_nz = torch.abs(pred - target).mean(dim=reduce_axes)
#        zero_penalty = torch.abs(pred).mean(dim=reduce_axes)
 
#        return (w_nz*MAE_nz + w_z*zero_penalty + norm_loss + ratio_prior), MAE_nz, zero_penalty, w_nz.detach(), w_z.detach(), self.b
        return (w_nz*MAE_nz + w_z*zero_penalty), MAE_nz, zero_penalty


    def  _masked_MAEloss_pw1(self, pred, target, eps=1e-8, tau=1e-12):


        if pred.shape != target.shape:
            raise ValueError(f"pred and target must have the same shape, got {pred.shape} vs {target.shape}")

        reduce_axes = tuple(range(1, pred.ndim))

        mask = (target.abs() > eps).float()
        
        
        w_nz = 1.0
        w_z = 1 + 5 * torch.sigmoid(self.alpha)
#        prior = self.beta * (torch.log(w_z) - torch.log(torch.tensor(5.0)))**2
#        prior = (self.alpha - self.mu).pow(2) / (2 * self.sigma**2)


#        mask1 = (target.abs() > 1e-4).float() 
#        mask2 = mask - mask1 
    #    weights = 0.5*mask2+1.0*mask1    
              
        MAE_nz = (mask * torch.abs(pred - target)).sum(dim=reduce_axes)/(mask.sum(dim=reduce_axes) + eps)    
        zero_penalty = ((1 - mask) * torch.abs(pred)).sum(dim=reduce_axes)/((1 - mask).sum(dim=reduce_axes) + eps)
        
 
#    return (w_l*loss_nz_l+w_s*loss_nz_s+w_z*zero_penalty), loss_nz_l, loss_nz_s, zero_penalty
        return (w_nz*MAE_nz + w_z*zero_penalty), MAE_nz, zero_penalty, w_nz, w_z.detach()

def nmse_per_sample(pred, target, eps=1e-6):
    """
    Compute per-sample normalized MSE.

    pred: [B, ...] predicted tensor
    target: [B, ...] target tensor
    returns: scalar loss averaged over batch
    """
#    B = pred.shape[0]
    if pred.shape != target.shape:
        raise ValueError(f"pred and target must have the same shape, got {pred.shape} vs {target.shape}")

    reduce_axes = tuple(range(1, pred.ndim))
    # flatten each sample to [B, N]
#    pred_flat   = pred.view(B, -1)
#    target_flat = target.view(B, -1)

    # per-sample MSE
    mse_per_sample = F.mse_loss(pred, target, reduction='none')
#    mse_per_sample = torch.sum((pred - target) ** 2, dim=reduce_axes)
    mse_per_sample = mse_per_sample.mean(dim=reduce_axes)  # [B]

    # per-sample normalization
#    target_norm = (target ** 2).mean(dim=reduce_axes)  # [B]
 #   nmse_per_sample = mse_per_sample / (target_norm + eps)

    # average over batch
    return mse_per_sample

def loss_diag_1rdm(pred, target, eps=1e-12):
        d_pred = torch.diagonal(pred, dim1=-2, dim2=-1)
        d_true = torch.diagonal(target, dim1=-2, dim2=-1)
        return F.mse_loss(d_pred, d_true, reduction='none').mean(dim=1) # MSE
    
def loss_off_diag_1rdm(pred, alpha=1.0):
    # ----- 3. Off-diagonal penalty -----
    B, N, _ = pred.shape
    
    # mask with 0 on diagonal, 1 on off-diagonals
    eye = torch.eye(N, device=pred.device).unsqueeze(0)  # (1, N, N)
    off_diag_mask = 1.0 - eye  # (1, N, N)

    # apply mask, square, mean
    off_diag_values = pred * off_diag_mask
    loss_offdiag = (off_diag_values ** 2).mean(dim=(1,2))   
    return alpha*loss_offdiag

def weighted_MSELoss(pred, target, eps=1e-12, s=0.0625):
    

    if pred.shape != target.shape:
        raise ValueError(f"pred and target must have the same shape, got {pred.shape} vs {target.shape}")

    reduce_axes = tuple(range(1, pred.ndim))

    mask = (target.abs() > eps).float()

# Mask for non-zeros
#    nz_frac = mask.flatten(1).mean(dim=1)  # shape: (B,)
#    z_frac  = 1.0 - nz_frac
    nz_frac = s
    z_frac  = 1.0 - nz_frac
    
#    nz_frac = torch.clamp(nz_frac, min=1e-12)
#    z_frac  = torch.clamp(z_frac, min=1e-12)
    w_nz = 1.0/nz_frac
    w_z = 1.0/ z_frac
    # Expand weights to match tensor shape
    # weight = w_nz * mask + w_z * (1-mask)
#    w_nz = (1.0 / nz_frac).view(-1, *([1]*(target.dim()-1)))   # shape broadcastable to y
#    w_z  = (1.0 / z_frac).view(-1, *([1]*(target.dim()-1)))
    weights = mask * w_nz + (1 - mask) * w_z

    # Weighted MSE
    wMSEloss = (weights * (pred - target) ** 2).sum(dim=reduce_axes)/weights.sum(dim=reduce_axes)
    
    return wMSEloss


def masked_MAE_BCEloss(pred, mask_pred, target, eps=1e-8, tau=1e-12, pos_weight=15.67):


    if pred.shape != target.shape:
        raise ValueError(f"pred and target must have the same shape, got {pred.shape} vs {target.shape}")

    reduce_axes = tuple(range(1, pred.ndim))

    target_mask = (target.abs() > eps).float()

#    mask_pred_soft = soft_mask_from_values(
#        pred,
#        threshold=eps
#    )

#    weights = torch.clamp(target.abs() ** beta, min=w_min)
#    bce = F.binary_cross_entropy(
#        mask_pred,
#        target_mask,
#        reduction="none"
#    )    

    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]),reduction="none")
    weighted_bce = criterion(mask_pred,target_mask)
#    weighted_bce = bce * (1.0 + weights * target_mask)
#    weighted_bce = bce * (1 + (pos_weight - 1) * target_mask)
#    w_neg=1.0

#    weights = (
#    w_neg * (1.0 - target_mask)
#    + pos_weight * weights_nz * target_mask
#    )
#    weighted_bce = weights * bce
    mask_loss = weighted_bce.sum(dim=reduce_axes) / (1 + (pos_weight - 1) * target_mask).sum(dim=reduce_axes)
#    mask_loss = weighted_bce.sum(dim=reduce_axes) / weights.sum(dim=reduce_axes)
#    mask_loss = bce.mean(dim=reduce_axes)
#    mag_weights = torch.zeros_like(target)
#    mag_weights[mask.bool()] = mag_weights_fn(target[mask.bool()])
    
    w1 = 10.0
    w2 = 1.0
#    w1_switched = 10.0
#    w2_switched = 0.0



#    loss_nz = mask * mag_weights * torch.abs(pred - target)
##    loss_nz = target_mask * torch.abs(pred - target)
#    wMAE_nz = loss_nz.sum(dim=reduce_axes)/mag_weights.sum(dim=reduce_axes)
##    MAE_nz = loss_nz.sum(dim=reduce_axes)/target_mask.sum(dim=reduce_axes)  
#    zero_penalty = ((1 - target_mask) * torch.abs(pred)).sum(dim=reduce_axes)/(1 - mask).sum(dim=reduce_axes)
#    w1, w2 = dynamic_weights(zero_penalty.detach())
#    if zero_penalty.mean() <  1e-5 :
#        w1, w2 = 10.0, 1.0  #we do a hard swap
#    wmask = (zero_penalty < 1e-5).float()        # 1 if switch, 0 if not
#    w1 = w1_init*(1 - wmask) + w1_switched*wmask
#    w2 = w2_init*(1 - wmask) + w2_switched*wmask
#    w1 = MAE_nz.detach() / (MAE_nz.detach() + zero_penalty.detach())
#    w2 = zero_penalty.detach() / (MAE_nz.detach() + zero_penalty.detach())

    mask1 = (target.abs() > 1e-4).float() 
    mask2 = target_mask - mask1 
    weights_nz = 0.1*mask2+1.0*mask1 
          
    loss_nz = weights_nz * torch.abs(pred - target)
#    wMAE_nz = loss_nz.sum(dim=reduce_axes)/mag_weights.sum(dim=reduce_axes)
    MAE_nz = loss_nz.sum(dim=reduce_axes)/weights_nz.sum(dim=reduce_axes)      
     

    return (w1*MAE_nz+w2*mask_loss), MAE_nz, mask_loss


def soft_mask_from_values(pred_values, threshold=1e-4, alpha=50.0):
    """
    Differentiable approximation to |pred| > threshold
    """
    return torch.sigmoid(alpha * (pred_values.abs() - threshold)).float()

def masked_MAEloss(pred, target, eps=1e-8, tau=1e-12):


    if pred.shape != target.shape:
        raise ValueError(f"pred and target must have the same shape, got {pred.shape} vs {target.shape}")

    reduce_axes = tuple(range(1, pred.ndim))

    mask = (target.abs() > eps).float()

#    rho_t = mask.mean(dim=reduce_axes)
#    mag_weights = torch.zeros_like(target)
#    mag_weights[mask.bool()] = mag_weights_fn(target[mask.bool()])

#    w_l = 1.0
#    w_s = 1.0
    w_nz = 1.0
    w_z = 7.0
#    w_kl = 0.05
#    w_z = get_cardinalityweights(target,reduce_axes)
# calc sparsity-based weighting for each individual points

#    w1_switched = 10.0
#    w2_switched = 0.0



#    loss_nz = mask * mag_weights * torch.abs(pred - target)
#    mag_weights = torch.zeros_like(target)
#    mag_weights[mask.bool()] = mag_weights_fn(target[mask.bool()])
    mask1 = (target.abs() > 1e-4).float() 
    mask2 = mask - mask1 
#    weights = 0.5*mask2+1.0*mask1    
          
    
#    loss_nz_l = (mask1 * torch.abs(pred - target)).sum(dim=reduce_axes)/(mask1.sum(dim=reduce_axes)+ eps)
#    loss_nz_s = (mask2 * torch.abs(pred - target)).sum(dim=reduce_axes)/(mask2.sum(dim=reduce_axes) + eps)
#    print("loss_nz_l", "loss_nz_s", loss_nz_l,loss_nz_s)
#    wMAE_nz = loss_nz.sum(dim=reduce_axes)/mag_weights.sum(dim=reduce_axes)
#    MAE_nz = loss_nz.mean(dim=reduce_axes)   
#    MAE_nz = loss_nz.sum(dim=reduce_axes)/wei.sum(dim=reduce_axes)  
#    weights = torch.zeros_like(target)
#    weights[mask.bool()] = step_weight(target[mask.bool()])

    MAE_nz = (mask * torch.abs(pred - target)).sum(dim=reduce_axes)/(mask.sum(dim=reduce_axes) + eps)    

#    attn_weighted_MAE = get_attn_weights(pred - target, mask)

    zero_penalty = ((1 - mask) * torch.abs(pred)).sum(dim=reduce_axes)/((1 - mask).sum(dim=reduce_axes) + eps)

###kl-divergence error
# =============================================================================
#     prob_active = rho_hat_logscale(pred)
#     rho_hat = prob_active.mean(dim=reduce_axes)  # [B]
#     rho_hat = rho_hat.clamp(min=1e-6, max=1-1e-6)
#     # KL per sample
#     kl = rho_t * torch.log((rho_t + eps) / (rho_hat + eps)) + \
#          (1 - rho_t) * torch.log((1 - rho_t + eps) / (1 - rho_hat + eps))
# #    MAE_nz = (mask * F.smooth_l1_loss(pred, target, reduction='none')).sum(dim=reduce_axes)/(mask.sum(dim=reduce_axes) + eps)    
# =============================================================================
#    zero_penalty = ((1 - mask) * F.smooth_l1_loss(pred, torch.zeros_like(target), reduction='none')).sum(dim=reduce_axes)/((1 - mask).sum(dim=reduce_axes) + eps)
    


#    zero_penalty = torch.abs(pred[(1-mask).bool()]).mean(dim=reduce_axes)
#    w1, w2 = dynamic_weights(zero_penalty.detach())
#    if zero_penalty.mean() <  1e-5 :
#        w1, w2 = 10.0, 1.0  #we do a hard swap
#    wmask = (zero_penalty < 1e-5).float()        # 1 if switch, 0 if not
#    w1 = w1_init*(1 - wmask) + w1_switched*wmask
#    w2 = w2_init*(1 - wmask) + w2_switched*wmask
#    w1 = MAE_nz.detach() / (MAE_nz.detach() + zero_penalty.detach())
#    w2 = zero_penalty.detach() / (MAE_nz.detach() + zero_penalty.detach())
    
     

#    return (w_l*loss_nz_l+w_s*loss_nz_s+w_z*zero_penalty), loss_nz_l, loss_nz_s, zero_penalty
    return (w_nz*MAE_nz + w_z*zero_penalty), MAE_nz, zero_penalty


# def entmax_weighted_mae(pred, target, alpha=1.5, dim=-1, eps=1e-8,temperature=0.01):
#     # 1. Compute importance weights from target
#     w1=1.0
#     w2=0.02
#     B=target.shape[0]
#     target_flat = target.reshape(B,-1)
#     pred_flat = pred.reshape(B,-1)
# #    with torch.no_grad():  # important: block gradients
#     weights = entmax15(target_flat.abs(), dim=dim)
# #        weights = sparsemax(target_flat.abs(), dim=dim)
# #        logits = torch.log1p(target_flat.abs())
# #        weights = F.softmax( logits/ temperature, dim=dim)
#     # 2. Compute MAE
#     abs_err = torch.abs(pred_flat - target_flat)

#     # 3. Weighted MAE (normalized)
#     loss_reg = (weights * abs_err).sum(dim=dim) / (weights.sum(dim=dim) + eps)
#     loss_entmax = entmax_kl_loss(pred_flat,target_flat)
    
#     return (w1*loss_reg + w2*loss_entmax), loss_reg, loss_entmax
    

# def entmax_kl_loss(y_pred, y_true, dim=-1, eps=1e-8):
#     p = entmax15(y_pred.abs(), dim=dim)
#     with torch.no_grad():
#         q = entmax15((y_true.abs() > 1e-8).float() * y_true.abs(), dim=dim)

#     loss = (q * (torch.log(q + eps) - torch.log(p + eps))).sum(dim=dim)
#     return loss


def range_weight(x, low_tail=1e-8, high_tail=1e-4, low_bulk=1e-4, high_bulk=1e-1):
    # Smooth bump weights using sigmoids
    alpha = 50.0  # controls sharpness
    w_tail = torch.sigmoid(alpha * (x - low_tail)) * torch.sigmoid(alpha * (high_tail - x))
    w_bulk = torch.sigmoid(alpha * (x - low_bulk)) * torch.sigmoid(alpha * (high_bulk - x))
 
    # scale weights
    return 0.1 * w_tail + 1.0 * w_bulk  # λ_tail=0.1, λ_bulk=1.0


def dynamic_weights(L0, tau0=1e-5, 
                            alpha_max=10.0, beta_max=10.0, 
                            L_min=1e-5, L_max=0.1, 
                            k_min=50, k_max=5e5):



    """
    Compute per-example dynamic weights alpha and beta for zero/non-zero loss.

    Args:
        L0 (torch.Tensor): Zero-loss per example, shape (batch_size,)
        tau0 (float): Zero-loss threshold for shifting focus
        alpha_max (float): Maximum weight for non-zero regression
        beta_max (float): Maximum weight for zero penalty
        L_min, L_max (float): Expected min/max zero-loss for scaling k
        k_min, k_max (float): Min/max slope for sigmoid
    Returns:
        alpha, beta (torch.Tensor): Per-example weights, shape (batch_size,)
    """
    # Ensure L_min and L_max are tensors on the same device/dtype as L0
    L_min_tensor = torch.tensor(L_min, dtype=L0.dtype, device=L0.device)
    L_max_tensor = torch.tensor(L_max, dtype=L0.dtype, device=L0.device)

    # Clamp L0 to avoid log(0) issues
    L0_clamped = L0.clamp(min=L_min_tensor, max=L_max_tensor)

    # Logarithmic scaling for k
    scale = (torch.log10(L0_clamped / L_min_tensor)) / (torch.log10(L_max_tensor / L_min_tensor))
    k = k_min + (k_max - k_min) * scale  # shape: (batch_size,)

    # Sigmoid weighting (use detached L0 to avoid backprop through alpha/beta)
    sigma = 1 / (1 + torch.exp(-k * (tau0 - L0.detach())))

    # Compute per-example weights
    alpha = alpha_max * sigma
    beta  = beta_max * (1 - sigma)

    return alpha, beta





def eri_weighted_masked_MAEloss(pred, target, eris, eps=1e-8):


    if pred.shape != target.shape:
        raise ValueError(f"pred and target must have the same shape, got {pred.shape} vs {target.shape}")

    reduce_axes = tuple(range(1, pred.ndim))

    mask = (target.abs() > eps).float()

    weights = torch.abs(eris)    
    nz_weights = torch.zeros_like(target)
#    z_weights = torch.zeros_like(target)
    nz_weights[mask.bool()] = weights[mask.bool()]
#    z_weights[(1-mask).bool()] =  weights[(1-mask).bool()]   

    w1=1.0
    w2=10.0  #this gives best energy test error so far
     
    loss_nz = nz_weights * torch.abs(pred - target)
#    loss_nz = mask * torch.abs(pred - target)
    MAE_nz = loss_nz.sum(dim=reduce_axes)/nz_weights.sum(dim=reduce_axes)
#    MAE_nz = loss_nz.sum(dim=reduce_axes)/mask.sum(dim=reduce_axes)  
#    loss_z = z_weights*torch.abs(pred)
    zero_penalty = ((1 - mask) * torch.abs(pred)).sum(dim=reduce_axes)/(1 - mask).sum(dim=reduce_axes)   
#    zero_penalty = loss_z.sum(dim=reduce_axes)/z_weights.sum(dim=reduce_axes)

#    if zero_penalty.mean() <  1e-5 :
#        w1, w2 = 10.0, 1.0  #we do a hard swap


    return (w1*MAE_nz+w2*zero_penalty), MAE_nz, zero_penalty


def eri_weighted_MAEloss(pred, target, eris):
    
    if pred.shape != target.shape:
        raise ValueError(f"pred and target must have the same shape, got {pred.shape} vs {target.shape}")

    reduce_axes = tuple(range(1, pred.ndim))
    weights = torch.abs(eris)
    loss =  (weights*torch.abs(pred - target)).sum(dim=reduce_axes)/weights.sum(dim=reduce_axes)
    return loss


def log_MAEloss(pred, target, eps=1e-12):


    if pred.shape != target.shape:
        raise ValueError(f"pred and target must have the same shape, got {pred.shape} vs {target.shape}")

    reduce_axes = tuple(range(1, pred.ndim))

#    mask = (target.abs() > eps).float()
    
#    mag_weights = torch.zeros_like(target)
#    mag_weights[mask.bool()] = mag_weights_fn(target[mask.bool()])
    
#    w1=1.0
#    w2=5.50
     
#    loss_nz = mask * mag_weights * torch.abs(pred - target)
#    loss_nz = mask * torch.abs(pred - target)
#    wMAE_nz = loss_nz.sum(dim=reduce_axes)/mag_weights.sum(dim=reduce_axes)
    loss = torch.abs(signed_log(pred) - signed_log(target))
    logMAE = loss.mean(dim=reduce_axes)
#    zero_penalty = ((1 - mask) * torch.abs(pred)).sum(dim=reduce_axes)/(1 - mask).sum(dim=reduce_axes)

#    if zero_penalty.mean() <  1e-5 :
#        w1, w2 = 10.0, 1.0  #we do a hard swap


    return logMAE



def aug_weighted_Loss(pred, target, eps=1e-12, s=0.0625):
    

    if pred.shape != target.shape:
        raise ValueError(f"pred and target must have the same shape, got {pred.shape} vs {target.shape}")

    reduce_axes = tuple(range(1, pred.ndim))

    mask = (target.abs() > eps).float()

# Mask for non-zeros
    nz_frac = mask.flatten(1).mean(dim=1)  # shape: (B,)
    z_frac  = 1.0 - nz_frac
#    nz_frac = s
#    z_frac  = 1.0 - nz_frac
    
#    nz_frac = torch.clamp(nz_frac, min=1e-12)
#    z_frac  = torch.clamp(z_frac, min=1e-12)
    w_nz = 1.0/nz_frac
    w_z = 1.0/ z_frac
    # Expand weights to match tensor shape
    # weight = w_nz * mask + w_z * (1-mask)
#
    w_nz = (1.0 / nz_frac).view(-1, *([1]*(target.dim()-1)))   # shape broadcastable to y
    w_z  = (1.0 / z_frac).view(-1, *([1]*(target.dim()-1)))
    weights = mask * w_nz + (1 - mask) * w_z

  
    mag_weights = torch.zeros_like(target)
    mag_weights[mask.bool()] = mag_weights_fn(target[mask.bool()])
    weights = weights * mag_weights



    # Weighted MSE
#    if mse == True: 
#        wMSEloss = (weights * (pred - target) ** 2).sum(dim=reduce_axes)/weights.sum(dim=reduce_axes)
#    else:
    wMSEloss = (weights * torch.abs(pred - target)).sum(dim=reduce_axes)/weights.sum(dim=reduce_axes)
    
    return wMSEloss

def mag_weights_fn(x,eps=1e-4, max_weight=1e6):
    # Example: weight small values more
    # x is a non-zero tensor
    # Here we weight inversely proportional to magnitude
    weights = 1.0 / (x.abs()+eps)
    
    # Optional: cap extreme weights
 #   if max_weight is not None:
 #       weights = torch.clamp(weights, max=max_weight)
    
    return weights

def signed_log(x, eps=1e-8):
    return torch.sign(x) * torch.log1p(torch.abs(x))

def signed_log_hard(x, eps=1e-8):
    return torch.sign(x) * torch.log(torch.abs(x) + eps)

def signed_log_soft(x, eps=1e-8):
    return torch.sign(x) * torch.log1p(torch.abs(x) / eps)

def asinh_transform(x, eps=1e-8):
    return torch.asinh(x / eps)

def sigmoid_weight(x, k=10.0, eps=1e-12):
    z = torch.log10(torch.abs(x) + eps)
    return torch.sigmoid(k * (z + 5.0))

def step_weight(x, w1=0.1, w2=1.0, thresh=1e-4):
    return torch.where(torch.abs(x) < thresh, w1, w2)

def get_cardinalityweights(x,reduce_axes):
    num_zeros = (x.abs() < 1e-12).sum(dim=reduce_axes)    
    num_nonzeros = x[0].numel() - num_zeros
    prior_card = (num_nonzeros / torch.clamp(num_zeros, min=1)) * 80.0
    return prior_card

def rho_hat_logscale(preds, alpha=5.0, tau=-18.50, eps=1e-12):
    log_mag = torch.log(preds.abs() + eps)
    activity = torch.sigmoid(alpha * (log_mag - tau))
    return activity

def get_attn_weights(x,mask,error=None, temperature=1.0, detach_attention=False, eps=1e-8):
    abs_x =  torch.abs(x) 
    B = abs_x.shape[0]
    x_flat = abs_x.reshape(B, -1)       # (B, n^4)
    mask_flat = mask.reshape(B, -1)
#    errors_norm = error_flat / (error_flat.mean(dim=1, keepdim=True) + 1e-8)
    logits = x_flat / temperature
    
    logits = logits - logits.max(dim=1, keepdim=True)[0]  # stability
    exp_logits = torch.exp(logits) * mask_flat

    denom = exp_logits.sum(dim=1, keepdim=True) + eps
    attn = exp_logits / denom

    if detach_attention:
        attn = attn.detach()
    if error is not None:    
        return (attn * error.reshape(B, -1)).sum(dim=1)  
    else:
        return (attn * x_flat).sum(dim=1)  
    
# def masked_weighted_regression_loss(
#     pred: torch.Tensor,
#     target: torch.Tensor,
#     mask: torch.Tensor,
#     method: str = "entmax",   # "entmax" or "softmax"
#     eps: float = 1e-8,
# ):
#     """
#     Applies masked softmax/entmax weights and computes masked regression loss.

#     Args:
#         pred:   (B, ...)
#         target: (B, ...)
#         mask:   (B, ...) boolean or {0,1}
#     Returns:
#         loss: scalar
#         weights: same shape as pred
#     """

#     assert pred.shape == target.shape == mask.shape

#     B = pred.shape[0]

#     # ---------- flatten non-batch dims ----------
#     pred_f   = pred.reshape(B, -1)
#     target_f = target.reshape(B, -1)
#     mask_f   = mask.reshape(B, -1).bool()

#     # ---------- compute scores ----------

#     scores = torch.abs(target_f)


#     # ---------- mask scores ----------
#     neg_inf = torch.finfo(scores.dtype).min
#     scores = torch.where(mask_f, scores, neg_inf)

#     # ---------- normalize ----------
#     if method == "softmax":
#         weights_f = F.softmax(scores, dim=-1)
#     elif method == "entmax":
#         weights_f = entmax15(scores, dim=-1)
#     else:
#         raise ValueError("method must be 'softmax' or 'entmax'")

#     # ---------- regression loss ----------
#     per_entry_loss = torch.abs(pred_f - target_f)

#     # weighted, masked loss
#     loss_per_sample = (weights_f * per_entry_loss).sum(dim=-1)



#     return loss_per_sample

# def masked_weightedonloss_regression_loss(
#     pred: torch.Tensor,
#     target: torch.Tensor,
#     mask: torch.Tensor,
#     method: str = "entmax",   # "entmax" or "softmax"
#     eps: float = 1e-8,
# ):
#     """
#     Applies masked softmax/entmax weights and computes masked regression loss.

#     Args:
#         pred:   (B, ...)
#         target: (B, ...)
#         mask:   (B, ...) boolean or {0,1}
#     Returns:
#         loss: scalar
#         weights: same shape as pred
#     """

#     assert pred.shape == target.shape == mask.shape

#     B = pred.shape[0]

#     # ---------- flatten non-batch dims ----------
#     pred_f   = pred.reshape(B, -1)
#     target_f = target.reshape(B, -1)
#     mask_f   = mask.reshape(B, -1).bool()

#     # ---------- compute scores ----------

# #    scores = torch.abs(target_f)
#     scores = torch.abs(pred_f - target_f)

#     # ---------- mask scores ----------
#     neg_inf = torch.finfo(scores.dtype).min
#     scores = torch.where(mask_f, scores, neg_inf)

#     # ---------- normalize ----------
#     if method == "softmax":
#         weights_f = F.softmax(scores, dim=-1)
#     elif method == "entmax":
#         weights_f = entmax15(scores, dim=-1)
#     else:
#         raise ValueError("method must be 'softmax' or 'entmax'")

#     # ---------- regression loss ----------
# #    per_entry_loss = torch.abs(pred_f - target_f)

#     # weighted, masked loss
#     loss_per_sample = (weights_f.detach() * scores).sum(dim=-1)



#     return loss_per_sample