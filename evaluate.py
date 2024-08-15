import os
# import pickle
import shutil
import sys
# from enum import Enum, auto

import torch
# import torchaudio
from hyperpyyaml import load_hyperpyyaml
from pesq import pesq

import speechbrain as sb
# from speechbrain.dataio.sampler import ReproducibleWeightedRandomSampler
# from speechbrain.nnet.loss.stoi_loss import stoi_loss
from speechbrain.processing.features import spectral_magnitude
from speechbrain.utils.distributed import run_on_main
from speechbrain.utils.metric_stats import MetricStats

from train import pesq_eval, comp_eval

class MGKBrain(sb.Brain):
    def compute_forward(self, batch, stage):
        if stage != sb.Stage.TEST:
            raise NotImplementedError("This module is only for evaluation of discriminator.")

        batch = batch.to(self.device)

        noisy_wav, lens = batch.noisy_sig
        noisy_spec = self.compute_feats(noisy_wav)

        mask = self.modules.generator(noisy_spec, lengths=lens)
        mask = mask.clamp(min=self.hparams.min_mask).squeeze(2)
        est_spec = torch.mul(mask, noisy_spec)
        predict_wav = self.hparams.resynth(
            torch.expm1(est_spec), noisy_wav
        )

        return predict_wav
        
    def compute_objectives(self, predict_wav, batch, stage):
        if stage != sb.Stage.TEST:
            raise NotImplementedError("This module is only for evaluation of discriminator.")
        
        batch = batch.to(self.device)

        noisy_wav, lens = batch.noisy_sig
        noisy_spec = self.compute_feats(noisy_wav)
        clean_wav, lens = batch.clean_sig
        clean_spec = self.compute_feats(clean_wav)
        est_spec = self.compute_feats(predict_wav)

        nc = self.est_score(noisy_spec, clean_spec)
        cc = self.est_score(clean_spec, clean_spec)
        ec = self.est_score(est_spec, clean_spec)

        pesq_metric = MetricStats(metric=pesq_eval, n_jobs=hparams["n_jobs"], batch_eval=False)
        pesq_metric.append(batch.id, predict=noisy_wav, target=clean_wav, lengths=lens)
        target_nc = pesq_metric.summarize("average")

        pesq_metric = MetricStats(metric=pesq_eval, n_jobs=hparams["n_jobs"], batch_eval=False)
        pesq_metric.append(batch.id, predict=predict_wav, target=clean_wav, lengths=lens)
        target_ec = pesq_metric.summarize("average")

        # target_nc = pesq_eval(noisy_wav, clean_wav)
        # target_ec = pesq_eval(predict_wav, clean_wav)

        self.nc_metric.append(batch.id, predictions=nc, targets=target_nc, reduction="batch")
        self.ec_metric.append(batch.id, predictions=ec, targets=target_ec, reduction="batch")
        self.cc_metric.append(batch.id, predictions=cc, targets=torch.ones(cc.shape), reduction="batch")

        return nc + cc + ec

    def compute_feats(self, wavs):
        """Feature computation pipeline"""
        feats = self.hparams.compute_STFT(wavs)
        feats = spectral_magnitude(feats, power=0.5)
        feats = torch.log1p(feats)
        return feats


    def est_score(self, deg_spec, ref_spec):
        """Returns score as estimated by discriminator

        Arguments
        ---------
        deg_spec : torch.Tensor
            The spectral features of the degraded utterance
        ref_spec : torch.Tensor
            The spectral features of the reference utterance

        Returns
        -------
        est_score : torch.Tensor
        """
        combined_spec = torch.cat(
            [deg_spec.unsqueeze(1), ref_spec.unsqueeze(1)], 1
        )
        ret = self.modules.discriminator(combined_spec)
        if not torch.is_tensor(ret):
            ret = torch.tensor(ret).unsqueeze(0).unsqueeze(0).unsqueeze(0)
        elif ret.dim() == 0:
            ret = ret.unsqueeze(0).unsqueeze(0).unsqueeze(0)
        elif ret.dim() == 1:
            ret = ret.unsqueeze(0).unsqueeze(0)
        elif ret.dim() == 2:
            ret = ret.unsqueeze(0)
        
        return ret

    def fit_batch(self, batch):
        raise NotImplementedError("This module is only for evaluation of discriminator.")

    def on_stage_start(self, stage, epoch=None):
        """
        Gets called at the beginning of each epoch
        """
        if stage != sb.Stage.TEST:
            raise NotImplementedError("This module is only for evaluation of discriminator.")

        self.cc_metric = MetricStats(metric=sb.nnet.losses.mse_loss)
        self.nc_metric = MetricStats(metric=sb.nnet.losses.mse_loss)
        self.ec_metric = MetricStats(metric=sb.nnet.losses.mse_loss)

    def on_stage_end(self, stage, stage_loss, epoch=None):
        "Called at the end of each stage to summarize progress"
        if stage != sb.Stage.TEST:
            raise NotImplementedError("This module is only for evaluation of discriminator.")

        stats = {
            "clean-clean": self.cc_metric.summarize("average"),
            "enhanced-clean": self.ec_metric.summarize("average"),
            "noisy-clean": self.nc_metric.summarize("average"),
            "total loss:": stage_loss,
        }

        self.hparams.train_logger.log_stats(
            {"Epoch loaded": self.hparams.epoch_counter.current},
            test_stats=stats,
        )

# Define audio pipelines
@sb.utils.data_pipeline.takes("noisy_wav", "clean_wav")
@sb.utils.data_pipeline.provides("noisy_sig", "clean_sig")
def audio_pipeline(noisy_wav, clean_wav):
    yield sb.dataio.dataio.read_audio(noisy_wav)
    yield sb.dataio.dataio.read_audio(clean_wav)


# # For historical data
# @sb.utils.data_pipeline.takes("enh_wav", "clean_wav")
# @sb.utils.data_pipeline.provides("enh_sig", "clean_sig")
# def enh_pipeline(enh_wav, clean_wav):
#     yield sb.dataio.dataio.read_audio(enh_wav)
#     yield sb.dataio.dataio.read_audio(clean_wav)


def dataio_prep(hparams):
    """This function prepares the datasets to be used in the brain class."""

    # Define datasets
    datasets = {}
    data_info = {
        "train": hparams["train_annotation"],
        "valid": hparams["valid_annotation"],
        "test": hparams["test_annotation"],
    }
    for dataset in data_info:
        datasets[dataset] = sb.dataio.dataset.DynamicItemDataset.from_json(
            json_path=data_info[dataset],
            replacements={"data_root": hparams["data_folder"]},
            dynamic_items=[audio_pipeline],
            output_keys=["id", "noisy_sig", "clean_sig", "clean_wav"],
        )

    return datasets


def create_folder(folder):
    if not os.path.isdir(folder):
        os.makedirs(folder)


if __name__ == "__main__":
    # Load hyperparameters file with command-line overrides
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])
    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin, overrides)

    # Initialize ddp (useful only for multi-GPU DDP training)
    sb.utils.distributed.ddp_init_group(run_opts)

    # Data preparation
    from voicebank_prepare import prepare_voicebank  # noqa

    run_on_main(
        prepare_voicebank,
        kwargs={
            "data_folder": hparams["data_folder"],
            "save_folder": hparams["data_folder"],
            "skip_prep": hparams["skip_prep"],
        },
    )

    # Create dataset objects
    datasets = dataio_prep(hparams)

    # Create experiment directory
    sb.create_experiment_directory(
        experiment_directory=hparams["output_folder"],
        hyperparams_to_save=hparams_file,
        overrides=overrides,
    )

    if hparams["use_tensorboard"]:
        from speechbrain.utils.train_logger import TensorboardLogger

        hparams["tensorboard_train_logger"] = TensorboardLogger(
            hparams["tensorboard_logs"]
        )

    # Create the folder to save enhanced files (+ support for DDP)
    run_on_main(create_folder, kwargs={"folder": hparams["enhanced_folder"]})

    se_brain = MGKBrain(
        modules=hparams["modules"],
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=hparams["checkpointer"],
    )
    se_brain.train_set = datasets["train"]
    se_brain.historical_set = {}
    se_brain.noisy_scores = {}
    se_brain.batch_size = hparams["dataloader_options"]["batch_size"]

    if not os.path.isfile(hparams["historical_file"]):
        shutil.rmtree(hparams["MetricGAN_KAN_folder"])
    run_on_main(create_folder, kwargs={"folder": hparams["MetricGAN_KAN_folder"]})

    # Load best checkpoint for evaluation
    test_stats = se_brain.evaluate(
        test_set=datasets["test"],
        max_key=hparams["target_metric"],
        test_loader_kwargs=hparams["dataloader_options"],
    )
