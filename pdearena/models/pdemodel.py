# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
from typing import Any, Dict, List, Optional

import torch
from pytorch_lightning import LightningModule
from pytorch_lightning.cli import instantiate_class

from pdearena import utils
from pdearena.data.utils import PDEDataConfig
from pdearena.modules.loss import CustomMSELoss, ScaledLpLoss, PearsonCorrelationScore
from pdearena.rollout import rollout2d, rollout3d_maxwell


from .registry import MODEL_REGISTRY

logger = utils.get_logger(__name__)

import kornia.filters as filters
import torch.nn as nn

def get_model(args, pde):
    print(args.name)
    if args.name in MODEL_REGISTRY:
        _model = MODEL_REGISTRY[args.name].copy()
        if "Maxwell" in args.name:
            _model["init_args"].update(
                dict(
                    time_history=args.time_history,
                    time_future=args.time_future,
                    activation=args.activation,
                )
            )
        elif "GCA" in args.name:
            _model["init_args"].update(
                dict(
                    in_channels=args.time_history,
                    out_channels=args.time_future,
                )
            )
        else:
            _model["init_args"].update(
                dict(
                    n_input_scalar_components=pde.n_scalar_components,
                    n_output_scalar_components=pde.n_scalar_components,
                    n_input_vector_components=pde.n_vector_components,
                    n_output_vector_components=pde.n_vector_components,
                    time_history=args.time_history,
                    time_future=args.time_future,
                    activation=args.activation,
                )
            )
        model = instantiate_class(tuple(), _model)
    else:
        logger.warning("Model not found in registry. Using fallback. Best to add your model to the registry.")
        if hasattr(args, "time_history") and args.model["init_args"]["time_history"] != args.time_history:
            logger.warning(
                f"Model time_history ({args.model['init_args']['time_history']}) does not match data time_history ({pde.time_history})."
            )
        if hasattr(args, "time_future") and args.model["init_args"]["time_future"] != args.time_future:
            logger.warning(
                f"Model time_future ({args.model['init_args']['time_future']}) does not match data time_future ({pde.time_future})."
            )
        model = instantiate_class(tuple(), args.model)

    return model


class PDEModel(LightningModule):
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
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore="pdeconfig")
        self.pde = pdeconfig
        if (self.pde.n_spatial_dim) == 3:
            self._mode = "3DMaxwell"
            assert self.pde.n_scalar_components == 0
            assert self.pde.n_vector_components == 2
        elif (self.pde.n_spatial_dim) == 2:
            self._mode = "2D"

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
        self.smoothing_factor = 5
        self.stride = 2
        self.upsample_factor = 2
        self.multi_resolution = False
        #n_components = self.pde.n_scalar_components + 2*self.pde.n_vector_components
        n_components = 2
        self.teacher_model = HR_Encoder(in_channels=n_components, out_channels=n_components, stride=self.stride, smoothing_factor=self.smoothing_factor)
        self.std_correction = 0.13

        self.teacher_ckpt = '/mnt/SSD2/constantin/pdearena/outputs/kolmogorov2d-0624/ckpts/last-v21.ckpt'
        pretrained_dict = torch.load(self.teacher_ckpt)['state_dict']
        self.teacher_model.load_state_dict({k.replace("model.","model."): v for k, v in pretrained_dict.items() if k.startswith("model")})
        for param in self.teacher_model.parameters():
            param.requires_grad = False
        
    def forward_teacher(self, x, highres_x=None):
        hr_encoding = self.teacher_model(highres_x)
        return x + self.std_correction * filters.box_blur(hr_encoding[:,0], kernel_size=self.smoothing_factor).unsqueeze(1)

    def forward_student(self, x):
        pre_lr_encoding = self.model(x)
        return x + self.std_correction * pre_lr_encoding

    def forward(self, *args):
        return self.forward_student(args[0])

    def train_step(self, batch):
        x, y = batch
        highres_x = x
        x = filters.box_blur(x[:,0], kernel_size=self.smoothing_factor).unsqueeze(1)
        y = filters.box_blur(y[:,0], kernel_size=self.smoothing_factor).unsqueeze(1)            
            
        pred_student = self.forward_student(x)
        pred_teacher = self.forward_teacher(x, highres_x)
        
        loss = self.train_criterion(pred_student, pred_teacher) + self.train_criterion(pred_student, y)
        return loss, pred_student, y

    def eval_step(self, batch):
        x, y = batch
        highres_x = x
        x = filters.box_blur(x[:,0], kernel_size=self.smoothing_factor).unsqueeze(1)
        y = filters.box_blur(y[:,0], kernel_size=self.smoothing_factor).unsqueeze(1)            
            
        pred = self.forward(x)

        loss = {k: vc(pred, y) for k, vc in self.val_criterions.items()}
        return loss, pred, y

    def training_step(self, batch, batch_idx: int):
        loss, preds, targets = self.train_step(batch)

        if self._mode == "2D":
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

    def compute_rolloutloss2D(self, batch: Any):
        (u, v, cond, grid) = batch
        #print(u.shape)
        #u = u[:, :, :2, ::2, ::2]
        if self.multi_resolution:
            u = filters.blur_pool2d(u.reshape(
                (u.shape[0], u.shape[1]*u.shape[2], u.shape[3], u.shape[4])), kernel_size=self.smoothing_factor, stride=self.stride).reshape((u.shape[0], u.shape[1], u.shape[2], u.shape[3]//self.stride, u.shape[4]//self.stride))
        else:
            u = filters.box_blur(u.reshape(
                (u.shape[0], u.shape[1]*u.shape[2], u.shape[3], u.shape[4])), kernel_size=self.smoothing_factor).reshape((u.shape[0], u.shape[1], u.shape[2], u.shape[3], u.shape[4]))
        

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

            pred_traj = rollout2d(
                self,
                init_u,
                init_v,
                grid,
                self.pde,
                self.hparams.time_history,
                self.hparams.max_num_steps,
            )
            targ_u = u[:, target_start_time:target_end_time, ...]
            if self.pde.n_vector_components > 0:
                targ_v = v[:, target_start_time:target_end_time, ...]
                targ_traj = torch.cat((targ_u, targ_v), dim=2)
            else:
                targ_traj = targ_u
            for k, criterion in self.rollout_criterions.items():
                loss = criterion(pred_traj, targ_traj)
                loss = loss.mean(dim=(0,) + tuple(range(2, loss.ndim)))
                losses[k].append(loss)
        loss_vecs = {k: sum(v) / max(1, len(v)) for k, v in losses.items()}
        return loss_vecs

    def validation_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0):
        if dataloader_idx == 0:
            # one-step loss
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

                for k in loss.keys():
                    self.log(f"valid/loss/{k}", loss[k])
                return {f"{k}_loss": v for k, v in loss.items()}

            else:
                raise NotImplementedError(f"{self._mode}")

        elif dataloader_idx == 1:
            # rollout loss
            """self.log("valid/unrolled_loss", 0)
            return {
                "unrolled_loss": torch.zeros(1),
                "loss_timesteps": torch.zeros((5,1)),
                "unrolled_chan_avg_loss": torch.zeros(1),
            }"""
            
            if self._mode == "2D":
                loss_vecs = self.compute_rolloutloss2D(batch)
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
        self.log("valid/unrolled_loss_mean", 0)
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

                for i in range(self.hparams.max_num_steps):
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


class Maxwell3DPDEModel(PDEModel):
    def compute_rolloutloss3D(self, batch: Any):
        d, h, _ = batch
        losses = []
        for start in range(
            0,
            self.max_start_time + 1,
            self.hparams.time_future + self.hparams.time_gap,
        ):
            end_time = start + self.hparams.time_history
            target_start_time = end_time + self.hparams.time_gap
            target_end_time = target_start_time + self.hparams.time_future * self.hparams.max_num_steps
            init_d = d[:, start:end_time]
            init_h = h[:, start:end_time]
            pred_traj = rollout3d_maxwell(
                self.model,
                init_d,
                init_h,
                self.hparams.time_history,
                self.hparams.max_num_steps,
            )
            targ_d = d[:, target_start_time:target_end_time]
            targ_h = h[:, target_start_time:target_end_time]
            targ_traj = torch.cat((targ_d, targ_h), dim=2)  # along channel
            loss = self.rollout_criterion(pred_traj, targ_traj).mean(dim=(0, 2, 3, 4, 5))
            losses.append(loss)
        loss_vec = torch.stack(losses, dim=0).mean(dim=0)
        return loss_vec

    def training_step(self, batch, batch_idx: int):
        loss, preds, targets = self.train_step(batch)

        if self._mode == "3DMaxwell":
            d_loss = self.train_criterion(preds[:, :, :3, ...], targets[:, :, :3, ...])
            h_loss = self.train_criterion(preds[:, :, 3:, ...], targets[:, :, 3:, ...])
            self.log("train/loss", loss)
            self.log("train/d_loss", d_loss)
            self.log("train/h_loss", h_loss)
            return {
                "loss": loss,
                "d_loss": d_loss,
                "h_loss": h_loss,
            }
        else:
            raise NotImplementedError(f"{self._mode}")

    def validation_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0):
        if dataloader_idx == 0:
            # one-step loss
            loss, preds, targets = self.eval_step(batch)
            if self._mode == "3DMaxwell":
                loss["d_field_mse"] = self.val_criterions["mse"](preds[:, :, :3, ...], targets[:, :, :3, ...])
                loss["h_field_mse"] = self.val_criterions["mse"](preds[:, :, 3:, ...], targets[:, :, 3:, ...])

                for k in loss.keys():
                    self.log("valid/loss", loss[k])
                return {f"{k}_loss": v for k, v in loss.items()}
            else:
                raise NotImplementedError(f"{self._mode}")

        elif dataloader_idx == 1:
            # rollout loss
            if self._mode == "3DMaxwell":
                loss_vec = self.compute_rolloutloss3D(batch)
            else:
                raise NotImplementedError(f"{self._mode}")
            # summing across "time axis"
            loss = loss_vec.sum()
            loss_t = loss_vec.cumsum(0)
            chan_avg_loss = loss / (self.pde.n_scalar_components + self.pde.n_vector_components)
            self.log("valid/unrolled_loss", loss)
            return {
                "unrolled_loss": loss,
                "loss_timesteps": loss_t,
                "unrolled_chan_avg_loss": chan_avg_loss,
            }

    def test_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0):
        if dataloader_idx == 0:
            loss, preds, targets = self.eval_step(batch)
            if self._mode == "3DMaxwell":
                d_loss = self.val_criterions["mse"](preds[:, :, :3, ...], targets[:, :, :3, ...])
                h_loss = self.val_criterions["mse"](preds[:, :, 3:, ...], targets[:, :, 3:, ...])
                self.log("test/loss", loss)
                self.log("test/d_loss", d_loss)
                self.log("test/h_loss", h_loss)
                return {
                    "loss": loss,
                    "d_loss": d_loss,
                    "h_loss": h_loss,
                }
            else:
                raise NotImplementedError(f"{self._mode}")

        elif dataloader_idx == 1:
            if self._mode == "3DMaxell":
                loss_vec = self.compute_rolloutloss3D(batch)
            else:
                raise NotImplementedError(f"{self._mode}")
            # summing across "time axis"
            loss = loss_vec.sum()
            loss_t = loss_vec.cumsum(0)
            self.log("test/unrolled_loss", loss)
            # self.log("valid/normalized_unrolled_loss", loss)
            return {
                "unrolled_loss": loss,
                "loss_timesteps": loss_t,
            }

from pdearena.modules.twod_unet import Unet
from pdearena.modules.twod_resnet import ResNet
from pdearena.modules.twod_resnet import DilatedBasicBlock

class HR_Encoder(nn.Module):

    def __init__(self, in_channels=2, out_channels=2, init_features=16, stride=2, smoothing_factor=4):
        super(HR_Encoder, self).__init__()
        n_features=128
        self.stride = stride
        self.smoothing_factor = smoothing_factor
        self.model = ResNet(n_input_scalar_components=in_channels,
                            n_input_vector_components=0,
                            n_output_scalar_components=out_channels,
                            n_output_vector_components=0,
                            time_history=1,
                            time_future=1, hidden_channels=64, norm=False, block=DilatedBasicBlock, num_blocks=[1, 1, 1, 1])
        
        """Unet(n_input_scalar_components=in_channels,
                            n_input_vector_components=0,
                            n_output_scalar_components=out_channels,
                            n_output_vector_components=0,
                            time_history=1,
                            time_future=1,
                            hidden_channels=64,
                            #norm=True,
                            #ch_mults=[1,1,2,4],
                            #norm=True,
                            activation='gelu')"""
        
        self.tanh = nn.Tanh()
        
    def forward(self, x):
        return self.model(x)
        #return filters.blur_pool2d(self.model(x)[:,0], kernel_size=self.smoothing_factor, stride=self.stride).unsqueeze(1)
        #return nn.AvgPool2d(self.coarsening_factor)(filters.box_blur(self.model(x)[:,0], (self.coarsening_factor, self.coarsening_factor))).unsqueeze(1)


class Pre_LR_Encoder(nn.Module):

    def __init__(self, in_channels=2, out_channels=2, init_features=16, coarsening_factor=4):
        super(Pre_LR_Encoder, self).__init__()
        n_features=128
        self.coarsening_factor = coarsening_factor
        self.model = ResNet(n_input_scalar_components=in_channels,
                            n_input_vector_components=0,
                            n_output_scalar_components=out_channels,
                            n_output_vector_components=0,
                            time_history=1,
                            time_future=1, hidden_channels=64, norm=False, block=DilatedBasicBlock, num_blocks=[1, 1, 1, 1])
        
        self.tanh = nn.Tanh()
        
    def forward(self, x):
        return self.model(x)
        #return nn.AvgPool2d(self.coarsening_factor)(filters.box_blur(self.model(x)[:,0], (self.coarsening_factor, self.coarsening_factor))).unsqueeze(1)