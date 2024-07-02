# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
from functools import partial
from typing import Any, Dict, List, Optional

import torch
from pytorch_lightning import LightningModule
from pytorch_lightning.cli import instantiate_class

from pdearena import utils
from pdearena.data.utils import PDEDataConfig
from pdearena.modules.loss import CustomMSELoss, ScaledLpLoss, PearsonCorrelationScore
from pdearena.rollout import cond_rollout2d


from .registry import COND_MODEL_REGISTRY

logger = utils.get_logger(__name__)

import kornia.filters as filters
import torch.nn as nn

def get_model(args, pde):
    if args.name in COND_MODEL_REGISTRY:
        _model = COND_MODEL_REGISTRY[args.name].copy()
        _model["init_args"].update(
            dict(
                n_input_scalar_components=pde.n_scalar_components*2,
                n_output_scalar_components=pde.n_scalar_components,
                n_input_vector_components=pde.n_vector_components,
                n_output_vector_components=pde.n_vector_components,
                time_history=args.time_history,
                time_future=args.time_future,
                activation=args.activation,
                param_conditioning=args.param_conditioning,
                n_dims=pde.n_spatial_dim,
            )
        )
        model = instantiate_class(tuple(), _model)
    else:
        raise NotImplementedError(f"Model {args.name} not found in registry.")

    return model


class SemicondPDEModel(LightningModule):
    def __init__(
        self,
        name: str,
        time_history: int,
        time_future: int,
        time_gap: int,
        max_num_steps: int,
        activation: str,
        criterion: str,
        lr: float,
        pdeconfig: PDEDataConfig,
        model: Optional[Dict] = None,
        param_conditioning: Optional[str] = None,
        padding_mode: str = "zeros",
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore="pdeconfig")
        self.pde = pdeconfig
        # Set padding for convolutions globally.
        print(self.pde.n_spatial_dim)
        if (self.pde.n_spatial_dim) == 3:
            self._mode = "3D"
            nn.Conv3d = partial(nn.Conv3d, padding_mode=self.hparams.padding_mode)
        elif (self.pde.n_spatial_dim) == 2:
            self._mode = "2D"
            nn.Conv2d = partial(nn.Conv2d, padding_mode=self.hparams.padding_mode)
        elif (self.pde.n_spatial_dim) == 1:
            self._mode = "1D"
            nn.Conv1d = partial(nn.Conv1d, padding_mode=self.hparams.padding_mode)
        else:
            raise NotImplementedError(f"{self.pde}")

        self.model = get_model(self.hparams, self.pde)
        if criterion == "mse":
            self.train_criterion = CustomMSELoss()
        elif criterion == "scaledl2":
            self.train_criterion = ScaledLpLoss()
        else:
            raise NotImplementedError(f"Criterion {criterion} not implemented yet")

        self.val_criterions = {"mse": CustomMSELoss(), "scaledl2": ScaledLpLoss()}
        self.rollout_criterions = {"mse": torch.nn.MSELoss(reduction="none"), "corr": PearsonCorrelationScore()}
        time_resolution = self.pde.trajlen
        # Max number of previous points solver can eat
        reduced_time_resolution = time_resolution - self.hparams.time_history
        # Number of future points to predict
        self.max_start_time = (
            reduced_time_resolution - self.hparams.time_future * self.hparams.max_num_steps - self.hparams.time_gap
        )
        self.max_start_time = max(0, self.max_start_time)

        lr_ratio = 4
        hr_ratio = 1
        self.pooling = torch.nn.AvgPool1d(kernel_size=lr_ratio, stride=lr_ratio, padding=0)
        self.half_pooling = torch.nn.AvgPool1d(kernel_size=hr_ratio, stride=hr_ratio, padding=0)

        self.hr_encoder = HR_Encoder(in_channels=1, out_channels=1, coarsening_factor=lr_ratio//hr_ratio)
        self.pre_lr_encoder = Pre_LR_Encoder(in_channels=1, out_channels=1, coarsening_factor=2)

        """model_ckpt = "/mnt/SSD2/constantin/pdearena/outputs/ks1d-test-lowres32/ckpts/last.ckpt" #ks1d-test-lowres64/ckpts/last-v3.ckpt" #ks1d-test/ckpts/last-v8.ckpt" #ks1d-test-lowres64/ckpts/last.ckpt"
        pretrained_dict = torch.load(model_ckpt)['state_dict']
        self.model.load_state_dict({k.replace("model.",""): v for k, v in pretrained_dict.items() if k.startswith("model")})
        """

    def forward(self, x, cond):
        pre_lr_encoding = self.pre_lr_encoder(x, cond)
        #pred_lr_encoding = self.
        return 0.3* self.model(torch.cat((x, pre_lr_encoding), dim=2), z=cond) + x
        #return 0.3* self.model(x, z=cond) + x

    def forward_modified_hr_encoder(self, x, cond, highres_x):
        hr_encoding = self.hr_encoder(highres_x, cond)
        #return 0.3*hr_encoding + x
        return 0.3* self.model(torch.cat((x, hr_encoding), dim=2), z=cond) + x        

    def train_step(self, batch):
        x, y, cond = batch
        highres_x = x #self.half_pooling(x[:,0]).unsqueeze(1)
        x = self.pooling(x[:,0]).unsqueeze(1)
        y = self.pooling(y[:,0]).unsqueeze(1)
        #pred = self.forward(x, cond)
        pred = torch.cat((self.forward_modified_hr_encoder(x, cond, highres_x), self.forward(x, cond)), dim=0)
        #pred = self.forward_modified_hr_encoder(x, cond, highres_x)
        y = torch.cat((y, y), dim=0)
        loss = self.train_criterion(pred, y)
        return loss, pred, y

    def eval_step(self, batch):
        x, y, cond = batch
        x = self.pooling(x[:,0]).unsqueeze(1)
        y = self.pooling(y[:,0]).unsqueeze(1)
        pred = self.forward(x, cond)
        loss = {k: vc(pred, y) for k, vc in self.val_criterions.items()}
        return loss, pred, y

    def training_step(self, batch, batch_idx: int):
        loss, preds, targets = self.train_step(batch)

        if self._mode == "1D" or self._mode == "2D":
            scalar_loss = self.train_criterion(
                preds[:, :, 0 : self.pde.n_scalar_components, ...],
                targets[:, :, 0 : self.pde.n_scalar_components, ...],
            )

            if self.pde.n_vector_components > 0:
                vector_loss = self.train_criterion(
                    preds[:, :, self.pde.n_scalar_components :, ...],
                    targets[:, :, self.pde.n_scalar_components :, ...],
                )
            else:
                vector_loss = torch.tensor(0.0)
            self.log("train/loss", loss)
            self.log("train/scalar_loss", scalar_loss)
            self.log("train/vector_loss", vector_loss)
            return {
                "loss": loss,
                "scalar_loss": scalar_loss.detach(),
                "vector_loss": vector_loss.detach(),
            }
        else:
            raise NotImplementedError(f"{self._mode}")

    def training_epoch_end(self, outputs: List[Any]):
        # `outputs` is a list of dicts returned from `training_step()`
        for key in outputs[0].keys():
            if "loss" in key:
                loss_vec = torch.stack([outputs[i][key] for i in range(len(outputs))])
                mean, std = utils.bootstrap(loss_vec, 64, 1)
                self.log(f"train/{key}_mean", mean)
                self.log(f"train/{key}_std", std)

    def compute_rolloutloss(self, batch: Any):
        (u, v, cond, grid) = batch
        
        u = self.pooling(u[:,:,0]).unsqueeze(2)

        losses = {k: [] for k in self.rollout_criterions.keys()}

        for start in range(
            0,
            self.max_start_time + 1,
            self.hparams.time_future + self.hparams.time_gap,
        ):
            end_time = start + self.hparams.time_history
            target_start_time = end_time + self.hparams.time_gap
            target_end_time = target_start_time + self.hparams.time_future * self.hparams.max_num_steps

            init_u = u[:, start:end_time, ...]
            if self.pde.n_vector_components > 0:
                init_v = v[:, start:end_time, ...]
            else:
                init_v = None
            targ_u = u[:, target_start_time:target_end_time, ...]
            if self.pde.n_vector_components > 0:
                targ_v = v[:, target_start_time:target_end_time, ...]
                targ_traj = torch.cat((targ_u, targ_v), dim=2)
            else:
                targ_traj = targ_u

            pred_traj = cond_rollout2d(
                self,
                init_u,
                init_v,
                None,
                cond,
                grid,
                self.pde,
                self.hparams.time_history,
                min(targ_u.shape[1], self.hparams.max_num_steps),
            )
            """from matplotlib import pyplot as plt

            fig, axes = plt.subplots(1, figsize=(10, 10))
            axes.imshow(u[0, :, 0].cpu(), aspect='auto')
            axes.set_title('Ground truth')
            fig.suptitle("Raw Unetmod-1d-64 Rollouts (Res. 32)")
            plt.savefig('groundtruth_rollout_resolution_32.jpg')
            print('a')
            return"""
            """from scipy.fft import fft
            import numpy as np
            times = [0, 9, 19, 39, 59, 99]
            fig, axes = plt.subplots(1, len(times), figsize=(15, 6))
            for i in range(len(times)):
                axes[i].plot(np.abs(fft(pred_traj[0, times[i], 0].cpu().numpy()))[1:32], label="PDE Model")
                axes[i].plot(np.abs(fft(u[0, times[i]+1, 0].cpu().numpy()))[1:32], label="Groundtruth")
                axes[i].set_yscale('log')
                axes[i].set_title(f'{times[i]+1} Timesteps')


            plt.legend()
            fig.suptitle('Raw Unet-1d-64 (Res. 64), Power Spectrum')
            plt.savefig('Unet-1d-64 prediction spectrum Res 64.jpg')
            return """
            for k, criterion in self.rollout_criterions.items():
                loss = criterion(pred_traj, targ_traj)
                loss = loss.mean(dim=(0,) + tuple(range(2, loss.ndim)))
                losses[k].append(loss)
        loss_vecs = {k: sum(v) / max(1, len(v)) for k, v in losses.items()}
        return loss_vecs

    def validation_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0):
        if dataloader_idx == 0:
            # one-step loss
            loss_mse, preds, targets = self.eval_step(batch)
            if self._mode == "1D" or self._mode == "2D":
                loss_mse["scalar_mse"] = self.val_criterions["mse"](
                    preds[:, :, 0 : self.pde.n_scalar_components, ...],
                    targets[:, :, 0 : self.pde.n_scalar_components, ...],
                )
                loss_mse["vector_mse"] = self.val_criterions["mse"](
                    preds[:, :, self.pde.n_scalar_components :, ...],
                    targets[:, :, self.pde.n_scalar_components :, ...],
                )

                for k in loss_mse.keys():
                    self.log(f"valid/loss/{k}", loss_mse[k])
                return {f"{k}_loss": v for k, v in loss_mse.items()}

            else:
                raise NotImplementedError(f"{self._mode}")

        elif dataloader_idx == 1:
            # rollout loss

            if self._mode == "1D" or self._mode == "2D":
                loss_vecs = self.compute_rolloutloss(batch)
            else:
                raise NotImplementedError(f"{self._mode}")
            # summing across "time axis"
            loss_mse = loss_vecs["mse"].sum()
            loss_mse_t = loss_vecs["mse"].cumsum(0)
            chan_avg_loss = loss_mse / (self.pde.n_scalar_components + self.pde.n_vector_components)
            self.log("valid/unrolled_loss", loss_mse)
            return {
                "unrolled_loss": loss_mse,
                "loss_timesteps": loss_mse_t,
                "unrolled_chan_avg_loss": chan_avg_loss,
                "corr": loss_vecs["corr"],
            }

    def validation_epoch_end(self, outputs: List[Any]):
        if len(outputs) > 1:
            if len(outputs[0]) > 0:
                for key in outputs[0][0].keys():
                    if "loss" in key:
                        loss_vec = torch.stack([outputs[0][i][key] for i in range(len(outputs[0]))])
                        mean, std = utils.bootstrap(loss_vec, 64, 1)
                        self.log(f"valid/{key}_mean", mean)
                        self.log(f"valid/{key}_std", std)
            
            if len(outputs[1]) > 0:
                unrolled_loss = torch.stack([outputs[1][i]["unrolled_loss"] for i in range(len(outputs[1]))])
                loss_timesteps_B = torch.stack([outputs[1][i]["loss_timesteps"] for i in range(len(outputs[1]))])
                loss_timesteps = loss_timesteps_B.mean(0)

                log_timesteps = range(0, loss_timesteps.shape[0], max(1, loss_timesteps.shape[0] // 10))

                for i in log_timesteps:
                    self.log(f"valid/intime_{i}_loss", loss_timesteps[i])

                mean, std = utils.bootstrap(unrolled_loss, 64, 1)
                self.log("valid/unrolled_loss_mean", mean)
                self.log("valid/unrolled_loss_std", std)

                # Correlation
                corr_timesteps_B = torch.stack([outputs[1][i]["corr"] for i in range(len(outputs[1]))], dim=0)
                corr_timesteps = corr_timesteps_B.mean(0)
                for threshold in [0.8, 0.9, 0.95]:
                    self.log(f"valid/time_till_corr_lower_{threshold}", (corr_timesteps > threshold).float().sum())
                for t in log_timesteps:
                    self.log(f"valid/corr_at_{t}", corr_timesteps[t])

    def test_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0):
        if dataloader_idx == 0:
            loss, preds, targets = self.eval_step(batch)
            if self._mode == "2D":
                loss["scalar_mse"] = self.val_criterions["mse"](
                    preds[:, :, 0 : self.pde.n_scalar_components, ...],
                    targets[:, :, 0 : self.pde.n_scalar_components, ...],
                )
                loss["vector_mse"] = self.val_criterions["mse"](
                    preds[:, :, self.pde.n_scalar_components :, ...],
                    targets[:, :, self.pde.n_scalar_components :, ...],
                )

                self.log("test/loss", loss)
                return {f"{k}_loss": v for k, v in loss.items()}
            else:
                raise NotImplementedError(f"{self._mode}")

        elif dataloader_idx == 1:
            if self._mode == "2D":
                loss_vec = self.compute_rolloutloss2D(batch)
            else:
                raise NotImplementedError(f"{self._mode}")
            # summing across "time axis"
            loss_mse = loss_vecs["mse"].sum()
            loss_mse_t = loss_vecs["mse"].cumsum(0)
            chan_avg_loss = loss_mse / (self.pde.n_scalar_components + self.pde.n_vector_components)
            self.log("valid/unrolled_loss", loss_mse)
            return {
                "unrolled_loss": loss_mse,
                "loss_timesteps": loss_mse_t,
                "unrolled_chan_avg_loss": chan_avg_loss,
                "corr": loss_vecs["corr"],
            }

    def test_epoch_end(self, outputs: List[Any]):
        assert len(outputs) > 1
        if len(outputs[0]) > 0:
            for key in outputs[0][0].keys():
                if "loss" in key:
                    loss_vec = torch.stack([outputs[0][i][key] for i in range(len(outputs[0]))])
                    mean, std = utils.bootstrap(loss_vec, 64, 1)
                    self.log(f"test/{key}_mean", mean)
                    self.log(f"test/{key}_std", std)
        if len(outputs[1]) > 0:
            unrolled_loss = torch.stack([outputs[1][i]["unrolled_loss"] for i in range(len(outputs[1]))])
            loss_timesteps_B = torch.stack([outputs[1][i]["loss_timesteps"] for i in range(len(outputs[1]))])
            loss_timesteps = loss_timesteps_B.mean(0)
            log_timesteps = range(0, loss_timesteps.shape[0], max(1, loss_timesteps.shape[0] // 10))
            for i in log_timesteps:
                self.log(f"test/intime_{i}_loss", loss_timesteps[i])

            mean, std = utils.bootstrap(unrolled_loss, 64, 1)
            self.log("test/unrolled_loss_mean", mean)
            self.log("test/unrolled_loss_std", std)

            # Correlation
            corr_timesteps_B = torch.stack([outputs[1][i]["corr"] for i in range(len(outputs[1]))], dim=0)
            corr_timesteps = corr_timesteps_B.mean(0)
            for threshold in [0.8, 0.9, 0.95]:
                self.log(f"tests/time_till_corr_lower_{threshold}", (corr_timesteps > threshold).float().sum())
            for t in log_timesteps:
                self.log(f"tests/corr_at_{t}", corr_timesteps[t])

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.hparams.lr)
        return optimizer

from pdearena.modules.conditioned.oned_unet import Unet
from pdearena.modules.conditioned.oned_resnet import ResNet, OneD_DilatedBasicBlock

class HR_Encoder(nn.Module):

    def __init__(self, in_channels=2, out_channels=2, init_features=16, coarsening_factor=2):
        super(HR_Encoder, self).__init__()
        n_features = 128
        self.coarsening_factor = coarsening_factor
        """self.model = Unet(n_input_scalar_components=1,
                            n_input_vector_components=0,
                            n_output_scalar_components=1,
                            n_output_vector_components=0,
                            time_history=1,
                            time_future=1, 
                            hidden_channels=8, 
                            norm=True, 
                            activation='gelu', 
                            param_conditioning= "scalar_2",
                            n_dims=1)"""
        
        self.model = ResNet(n_input_scalar_components=in_channels,
                            n_input_vector_components=0,
                            n_output_scalar_components=out_channels,
                            n_output_vector_components=0,
                            time_history=1,
                            time_future=1, 
                            hidden_channels=16, 
                            norm=False, 
                            block=OneD_DilatedBasicBlock, 
                            num_blocks=[1, 1, 1, 1],
                            param_conditioning= "scalar_2",
                            n_dims=1)
        
        self.pooling = torch.nn.AvgPool1d(kernel_size=self.coarsening_factor, stride=self.coarsening_factor, padding=0)
                
    def forward(self, x, cond):
        return self.pooling(self.model(x, z = cond)[:,:, 0]).unsqueeze(1)
    

"""class HR_Encoder(nn.Module):

    def __init__(self, in_channels=2, out_channels=2, init_features=16, coarsening_factor=2):
        super(HR_Encoder, self).__init__()
        n_features = 128
        self.coarsening_factor = coarsening_factor
        self.model = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=2, stride=2, padding=0)
                        
    def forward(self, x, cond):
        pred = self.model(x[:,0]).unsqueeze(1)
        return pred"""


"""class Pre_LR_Encoder(nn.Module):

    def __init__(self, in_channels=2, out_channels=2, init_features=16, coarsening_factor=4):
        super(Pre_LR_Encoder, self).__init__()
        n_features = 128
        self.coarsening_factor = coarsening_factor
        self.model = Unet(n_input_scalar_components=1,
                            n_input_vector_components=0,
                            n_output_scalar_components=1,
                            n_output_vector_components=0,
                            time_history=1,
                            time_future=1, 
                            hidden_channels=64, 
                            norm=True, 
                            activation='gelu', 
                            param_conditioning= "scalar_2",
                            n_dims=1)

        self.model = ResNet(n_input_scalar_components=in_channels,
                            n_input_vector_components=0,
                            n_output_scalar_components=out_channels,
                            n_output_vector_components=0,
                            time_history=1,
                            time_future=1, 
                            hidden_channels=128, 
                            norm=False, 
                            block=OneD_DilatedBasicBlock, 
                            num_blocks=[1, 1, 1, 1],
                            param_conditioning= "scalar_2",
                            n_dims=1)

        self.tanh = nn.Tanh()
        
    def forward(self, x, cond):
        return self.model(x, z=cond)"""


class Pre_LR_Encoder(nn.Module):

    def __init__(self, in_channels=2, out_channels=2, init_features=16, coarsening_factor=4):
        super(Pre_LR_Encoder, self).__init__()
        n_features = 128
        self.coarsening_factor = coarsening_factor
        self.model = ResNet(n_input_scalar_components=in_channels,
                            n_input_vector_components=0,
                            n_output_scalar_components=out_channels,
                            n_output_vector_components=0,
                            time_history=1,
                            time_future=1, 
                            hidden_channels=16, 
                            norm=False, 
                            block=OneD_DilatedBasicBlock, 
                            num_blocks=[1, 1, 1, 1],
                            param_conditioning= "scalar_2",
                            n_dims=1)
        
    def forward(self, x, cond):
        return self.model(x, z=cond)