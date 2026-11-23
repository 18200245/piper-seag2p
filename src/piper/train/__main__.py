import logging
import os
import shutil
from pathlib import Path

import torch
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.cli import LightningCLI

from .vits.dataset import VitsDataModule
from .vits.lightning import VitsModel

_LOGGER = logging.getLogger(__package__)

# Checkpoint-selection signal. The raw mel L1 ("val_mel") tracks reconstruction
# fidelity and trends down as the model improves, so it is a reasonable key for
# keeping the best checkpoints. It is NOT, however, used to stop training: mel
# L1 saturates early in VITS while the adversarial (GAN) losses keep removing
# audible artifacts, so an early-stop on val_mel fires well before the audio is
# clean. Instead we train on a fixed/open-ended schedule and let the user pick a
# final checkpoint by listening (and/or by "val_mos"). Override from the
# LightningCLI config if you want different behavior.
_MONITOR = "val_mel"


class _OptionalModelCheckpoint(ModelCheckpoint):
    """ModelCheckpoint that silently skips saving when its monitored metric was not logged or keep_last_k is active."""

    def _save_topk_checkpoint(self, trainer, monitor_candidates) -> None:
        keep_last_k = getattr(
            getattr(trainer.lightning_module, "hparams", None), "keep_last_k", None
        )
        if keep_last_k and keep_last_k > 0:
            return
        if self.monitor and (self.monitor not in monitor_candidates):
            return
        super()._save_topk_checkpoint(trainer, monitor_candidates)


class _LastAndRollingCheckpoint(ModelCheckpoint):
    """Writes last.ckpt every epoch, and when model.keep_last_k > 0 maintains the N most recent epoch=XXXX.ckpt files."""

    def on_train_epoch_end(self, trainer, pl_module) -> None:
        keep_last_k = getattr(getattr(pl_module, "hparams", None), "keep_last_k", None)
        ckpt_dir = Path(self.dirpath) if self.dirpath else None
        last_path = (ckpt_dir / "last.ckpt") if ckpt_dir else None

        # Detach last.ckpt from the previous epoch's hardlink before Lightning writes the new last.ckpt
        if keep_last_k and keep_last_k > 0 and last_path and last_path.exists():
            try:
                if last_path.stat().st_nlink > 1:
                    last_path.unlink()
            except OSError:
                pass

        super().on_train_epoch_end(trainer, pl_module)

        if not (keep_last_k and keep_last_k > 0):
            return

        ckpt_dir = Path(self.dirpath) if self.dirpath else None
        if not ckpt_dir:
            return
        last_path = ckpt_dir / "last.ckpt"
        if not last_path.exists():
            return

        epoch_path = ckpt_dir / f"epoch={trainer.current_epoch:04d}.ckpt"
        try:
            if epoch_path.exists():
                epoch_path.unlink()
            try:
                os.link(last_path, epoch_path)
            except OSError:
                shutil.copy2(last_path, epoch_path)
        except OSError:
            _LOGGER.debug("Could not create rolling epoch checkpoint", exc_info=True)

        # Keep only the N most recent epoch=XXXX.ckpt files
        rolling = sorted(
            p for p in ckpt_dir.glob("epoch=*.ckpt") if "-val_" not in p.name
        )
        for old_ckpt in rolling[: max(0, len(rolling) - int(keep_last_k))]:
            try:
                old_ckpt.unlink()
            except OSError:
                pass


def _default_callbacks() -> list:
    """Fresh checkpoint callbacks; each holds per-run state, so never share."""
    return [
        _OptionalModelCheckpoint(
            monitor=_MONITOR,
            mode="min",
            save_top_k=5,
            save_last=False,
            filename="epoch={epoch}-val_mel={val_mel:.4f}",
            auto_insert_metric_name=False,
        ),
        # Also keep the best-by-perceptual-quality checkpoints. "val_mos" (UTMOS)
        # tracks audible artifacts that mel L1 misses, so its winner is often the
        # better-sounding model. If the MOS predictor is disabled or can't be
        # loaded, val_mos is never logged and this callback simply skips saving.
        _OptionalModelCheckpoint(
            monitor="val_mos",
            mode="max",
            save_top_k=5,
            save_last=False,
            filename="epoch={epoch}-val_mos={val_mos:.4f}",
            auto_insert_metric_name=False,
        ),
        # last.ckpt, rewritten every epoch. save_last on a monitored callback is
        # not this: Lightning only writes last.ckpt when that callback has just
        # saved a top-k checkpoint, so last.ckpt froze at the final epoch that
        # improved val_mel and every epoch after it was lost -- epochs 1954-1999
        # of a 2,000-epoch run, the ones at the bottom of the LR anneal. With no
        # monitor and save_top_k=1 the file is simply the latest epoch.
        _LastAndRollingCheckpoint(
            monitor=None,
            save_top_k=1,
            every_n_epochs=1,
            filename="last",
            enable_version_counter=False,
        ),
    ]


class VitsLightningCLI(LightningCLI):
    def add_arguments_to_parser(self, parser):
        parser.link_arguments("data.batch_size", "model.batch_size")
        parser.link_arguments("data.num_symbols", "model.num_symbols")
        parser.link_arguments("model.num_speakers", "data.num_speakers")
        parser.link_arguments("model.sample_rate", "data.sample_rate")
        parser.link_arguments("model.filter_length", "data.filter_length")
        parser.link_arguments("model.hop_length", "data.hop_length")
        parser.link_arguments("model.win_length", "data.win_length")
        parser.link_arguments("model.segment_size", "data.segment_size")


def main():
    logging.basicConfig(level=logging.INFO)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.deterministic = False
    _cli = VitsLightningCLI(  # noqa: ignore=F841
        VitsModel,
        VitsDataModule,
        trainer_defaults={"max_epochs": -1, "callbacks": _default_callbacks()},
    )


# -----------------------------------------------------------------------------


if __name__ == "__main__":
    main()
