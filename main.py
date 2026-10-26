import argparse, os, sys, datetime, glob, importlib, csv
import numpy as np
import time
import torch
import torchvision
import pytorch_lightning as pl

from packaging import version
from omegaconf import OmegaConf
from torch.utils.data import random_split, DataLoader, Dataset, Subset
from functools import partial
from PIL import Image

from pytorch_lightning import seed_everything
from pytorch_lightning.trainer import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, Callback, LearningRateMonitor
# from pytorch_lightning.utilities.distributed import rank_zero_only
from pytorch_lightning.utilities.rank_zero import rank_zero_only
from pytorch_lightning.utilities import rank_zero_info

from ldm.data.base import Txt2ImgIterableBaseDataset
from ldm.util import instantiate_from_config, instantiate_from_config_sr
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger

import warnings
warnings.filterwarnings("ignore")


def get_parser(**parser_kwargs):
    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ("yes", "true", "t", "y", "1"):
            return True
        elif v.lower() in ("no", "false", "f", "n", "0"):
            return False
        else:
            raise argparse.ArgumentTypeError("Boolean value expected.")

    parser = argparse.ArgumentParser(**parser_kwargs)
    parser.add_argument(
        "-n",
        "--name",
        type=str,
        const=True,
        default="",
        nargs="?",
        help="postfix for logdir",
    )
    parser.add_argument(
        "-r",
        "--resume",
        type=str,
        const=True,
        default="",
        nargs="?",
        help="resume from logdir or checkpoint in logdir",
    )
    parser.add_argument(
        "-b",
        "--base",
        nargs="*",
        metavar="base_config.yaml",
        help="paths to base configs. Loaded from left-to-right. "
             "Parameters can be overwritten or added with command-line options of the form `--key value`.",
        default=list(),
    )
    parser.add_argument(
        "-t",
        "--train",
        type=str2bool,
        const=True,
        default=False,
        nargs="?",
        help="train",
    )
    parser.add_argument(
        "--no-test",
        type=str2bool,
        const=True,
        default=False,
        nargs="?",
        help="disable test",
    )
    parser.add_argument(
        "-p",
        "--project",
        help="name of new or path to existing project"
    )
    parser.add_argument(
        "-d",
        "--debug",
        type=str2bool,
        nargs="?",
        const=True,
        default=False,
        help="enable post-mortem debugging",
    )
    parser.add_argument(
        "-s",
        "--seed",
        type=int,
        default=23,
        help="seed for seed_everything",
    )
    parser.add_argument(
        "-f",
        "--postfix",
        type=str,
        default="",
        help="post-postfix for default name",
    )
    parser.add_argument(
        "-l",
        "--logdir",
        type=str,
        default="./logs",
        help="directory for logging dat shit",
    )
    parser.add_argument(
        "--scale_lr",
        type=str2bool,
        nargs="?",
        const=True,
        default=False,
        help="scale base-lr by ngpu * batch_size * n_accumulate",
    )

    """ 添加代码：修Bug """
    # ========== 新增：手动添加 Trainer 常用参数 ==========
    # 训练相关
    parser.add_argument("--max_epochs", type=int, default=1000, help="Maximum number of epochs")
    parser.add_argument("--min_epochs", type=int, default=1, help="Minimum number of epochs")
    parser.add_argument("--max_steps", type=int, default=-1, help="Maximum number of steps")
    # GPU/设备相关
    parser.add_argument("--gpus", type=str, default=None, help="GPU ids (e.g., '0,1')")
    parser.add_argument("--accelerator", type=str, default="gpu", help="Accelerator type (gpu, cpu, etc.)")
    parser.add_argument("--devices", type=int, default=1, help="Number of devices")
    parser.add_argument("--num_nodes", type=int, default=1, help="Number of nodes")
    # 精度相关
    parser.add_argument("--precision", type=str, default="32", help="Training precision (32, 16, bf16, etc.)")
    # 梯度累积
    parser.add_argument("--accumulate_grad_batches", type=int, default=1, help="Accumulate gradients over k batches")
    # 检查点和日志
    parser.add_argument("--default_root_dir", type=str, default=None, help="Default root directory for logs and weights")
    parser.add_argument("--log_every_n_steps", type=int, default=50, help="Log every n steps")      # 10
    parser.add_argument("--check_val_every_n_epoch", type=int, default=1, help="Run validation every n epochs")
    # 梯度裁剪
    parser.add_argument("--gradient_clip_val", type=float, default=0.0, help="Gradient clipping value")
    parser.add_argument("--gradient_clip_algorithm", type=str, default="norm", help="Gradient clipping algorithm")
    # 其他常用参数
    parser.add_argument("--limit_train_batches", type=float, default=1.0, help="Limit training batches")
    parser.add_argument("--limit_val_batches", type=float, default=1.0, help="Limit validation batches")
    parser.add_argument("--limit_test_batches", type=float, default=1.0, help="Limit test batches")
    parser.add_argument("--fast_dev_run", type=str2bool, default=False, help="Run a fast development run")
    parser.add_argument("--benchmark", type=str2bool, default=False, help="Enable cudnn benchmarking")
    parser.add_argument("--deterministic", type=str2bool, default=False, help="Enable deterministic mode")
    # 分布式训练
    parser.add_argument("--strategy", type=str, default="auto", help="Training strategy (ddp, deepspeed, etc.)")

    return parser


def nondefault_trainer_args(opt):
    import inspect
    try:
        if hasattr(Trainer, "add_argparse_args"):
            parser = argparse.ArgumentParser()
            parser = Trainer.add_argparse_args(parser)
            args = parser.parse_args([])  # get defaults
            # 比较 opt 与 parser 默认值
            keys = sorted(k for k in vars(args) if getattr(opt, k, None) != getattr(args, k, None))
            return keys
    except Exception:
        # 如果任何一步失败，就回退到 fallback 实现
        pass
        # 2) 回退实现：使用 inspect.signature(Trainer.__init__) 获取默认值（尽量兼容 new/old PL）
    try:
        sig = inspect.signature(Trainer.__init__)
    except Exception:
        # 如果无法获取 signature，就返回空列表以避免阻塞程序（最保守策略）
        return []
    defaults = {}
    for name, param in sig.parameters.items():
        # 跳过 self、*args 和 **kwargs
        if name == 'self' or param.kind in (param.VAR_POSITIONAL, param.VAR_KEYWORD):
            continue
        # 如果没有默认值，设为 None 作为占位
        if param.default is inspect._empty:
            defaults[name] = None
        else:
            defaults[name] = param.default
    # 比较 opt 中可能存在的同名属性与 defaults 中的默认值
    changed = []
    for k, default_val in defaults.items():
        if hasattr(opt, k):
            opt_val = getattr(opt, k)
            # 简单比较（注意：复杂对象的比较可能不是完全准确，但对常用 trainer 参数足够）
            if opt_val != default_val:
                changed.append(k)
    return sorted(changed)
    # parser = argparse.ArgumentParser()
    # parser = Trainer.add_argparse_args(parser)
    # args = parser.parse_args([])
    # return sorted(k for k in vars(args) if getattr(opt, k) != getattr(args, k))


class WrappedDataset(Dataset):
    """Wraps an arbitrary object with __len__ and __getitem__ into a pytorch dataset"""

    def __init__(self, dataset):
        self.data = dataset

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def worker_init_fn(_):
    worker_info = torch.utils.data.get_worker_info()

    dataset = worker_info.dataset
    worker_id = worker_info.id

    if isinstance(dataset, Txt2ImgIterableBaseDataset):
        split_size = dataset.num_records // worker_info.num_workers
        # reset num_records to the true number to retain reliable length information
        dataset.sample_ids = dataset.valid_ids[worker_id * split_size:(worker_id + 1) * split_size]
        current_id = np.random.choice(len(np.random.get_state()[1]), 1)
        return np.random.seed(np.random.get_state()[1][current_id] + worker_id)
    else:
        return np.random.seed(np.random.get_state()[1][0] + worker_id)


class DataModuleFromConfig(pl.LightningDataModule):
    def __init__(self, batch_size, train=None, validation=None, test=None, predict=None,
                 wrap=False, num_workers=None, shuffle_test_loader=False, use_worker_init_fn=False,
                 shuffle_val_dataloader=False):
        super().__init__()
        self.batch_size = batch_size
        self.dataset_configs = dict()
        self.num_workers = num_workers if num_workers is not None else batch_size * 2
        self.use_worker_init_fn = use_worker_init_fn
        if train is not None:
            self.dataset_configs["train"] = train
            self.train_dataloader = self._train_dataloader
        if validation is not None:
            self.dataset_configs["validation"] = validation
            self.val_dataloader = partial(self._val_dataloader, shuffle=shuffle_val_dataloader)
        if test is not None:
            self.dataset_configs["test"] = test
            self.test_dataloader = partial(self._test_dataloader, shuffle=shuffle_test_loader)
        if predict is not None:
            self.dataset_configs["predict"] = predict
            self.predict_dataloader = self._predict_dataloader
        self.wrap = wrap

    def prepare_data(self):
        for data_cfg in self.dataset_configs.values():
            instantiate_from_config_sr(data_cfg)

    def setup(self, stage=None):
        self.datasets = dict(
            (k, instantiate_from_config_sr(self.dataset_configs[k]))
            for k in self.dataset_configs)
        if self.wrap:
            for k in self.datasets:
                self.datasets[k] = WrappedDataset(self.datasets[k])

    def _train_dataloader(self):
        is_iterable_dataset = isinstance(self.datasets['train'], Txt2ImgIterableBaseDataset)
        if is_iterable_dataset or self.use_worker_init_fn:
            init_fn = worker_init_fn
        else:
            init_fn = None
        return DataLoader(self.datasets["train"], batch_size=self.batch_size,
                          num_workers=self.num_workers, shuffle=False if is_iterable_dataset else True,
                          worker_init_fn=init_fn)

    def _val_dataloader(self, shuffle=False):
        if isinstance(self.datasets['validation'], Txt2ImgIterableBaseDataset) or self.use_worker_init_fn:
            init_fn = worker_init_fn
        else:
            init_fn = None
        return DataLoader(self.datasets["validation"],
                          batch_size=self.batch_size,
                          num_workers=self.num_workers,
                          worker_init_fn=init_fn,
                          shuffle=shuffle)

    def _test_dataloader(self, shuffle=False):
        is_iterable_dataset = isinstance(self.datasets['train'], Txt2ImgIterableBaseDataset)
        if is_iterable_dataset or self.use_worker_init_fn:
            init_fn = worker_init_fn
        else:
            init_fn = None

        # do not shuffle dataloader for iterable dataset
        shuffle = shuffle and (not is_iterable_dataset)

        return DataLoader(self.datasets["test"], batch_size=self.batch_size,
                          num_workers=self.num_workers, worker_init_fn=init_fn, shuffle=shuffle)

    def _predict_dataloader(self, shuffle=False):
        if isinstance(self.datasets['predict'], Txt2ImgIterableBaseDataset) or self.use_worker_init_fn:
            init_fn = worker_init_fn
        else:
            init_fn = None
        return DataLoader(self.datasets["predict"], batch_size=self.batch_size,
                          num_workers=self.num_workers, worker_init_fn=init_fn)


class SetupCallback(Callback):
    def __init__(self, resume, now, logdir, ckptdir, cfgdir, config, lightning_config):
        super().__init__()
        self.resume = resume
        self.now = now
        self.logdir = logdir
        self.ckptdir = ckptdir
        self.cfgdir = cfgdir
        self.config = config
        self.lightning_config = lightning_config

    def on_keyboard_interrupt(self, trainer, pl_module):
        if trainer.global_rank == 0:
            print("Summoning checkpoint.")
            ckpt_path = os.path.join(self.ckptdir, "last.ckpt")
            trainer.save_checkpoint(ckpt_path)

    def on_pretrain_routine_start(self, trainer, pl_module):
        if trainer.global_rank == 0:
            # Create logdirs and save configs
            os.makedirs(self.logdir, exist_ok=True)
            os.makedirs(self.ckptdir, exist_ok=True)
            os.makedirs(self.cfgdir, exist_ok=True)

            if "callbacks" in self.lightning_config:
                if 'metrics_over_trainsteps_checkpoint' in self.lightning_config['callbacks']:
                    os.makedirs(os.path.join(self.ckptdir, 'trainstep_checkpoints'), exist_ok=True)
            print("Project config")
            print(OmegaConf.to_yaml(self.config))
            OmegaConf.save(self.config,
                           os.path.join(self.cfgdir, "{}-project.yaml".format(self.now)))

            print("Lightning config")
            print(OmegaConf.to_yaml(self.lightning_config))
            OmegaConf.save(OmegaConf.create({"lightning": self.lightning_config}),
                           os.path.join(self.cfgdir, "{}-lightning.yaml".format(self.now)))

        else:
            # ModelCheckpoint callback created log directory --- remove it
            if not self.resume and os.path.exists(self.logdir):
                dst, name = os.path.split(self.logdir)
                dst = os.path.join(dst, "child_runs", name)
                os.makedirs(os.path.split(dst)[0], exist_ok=True)
                try:
                    os.rename(self.logdir, dst)
                except FileNotFoundError:
                    pass


class ImageLogger(Callback):
    def __init__(self, batch_frequency, max_images, clamp=True, increase_log_steps=True,
                 rescale=True, disabled=False, log_on_batch_idx=False, log_first_step=False,
                 log_images_kwargs=None):
        super().__init__()
        self.rescale = rescale
        self.batch_freq = batch_frequency
        self.max_images = max_images
        self.logger_log_images = {
            pl.loggers.TensorBoardLogger: self._TensorBoardLogger,
        }
        self.log_steps = [2 ** n for n in range(int(np.log2(self.batch_freq)) + 1)]
        if not increase_log_steps:
            self.log_steps = [self.batch_freq]
        self.clamp = clamp
        self.disabled = disabled
        self.log_on_batch_idx = log_on_batch_idx
        self.log_images_kwargs = log_images_kwargs if log_images_kwargs else {}
        self.log_first_step = log_first_step

    @rank_zero_only
    def _testtube(self, pl_module, images, batch_idx, split):
        for k in images:
            grid = torchvision.utils.make_grid(images[k])
            grid = (grid + 1.0) / 2.0  # -1,1 -> 0,1; c,h,w

            tag = f"{split}/{k}"
            pl_module.logger.experiment.add_image(
                tag, grid,
                global_step=pl_module.global_step)

    @rank_zero_only
    def log_local(self, save_dir, split, images,
                  global_step, current_epoch, batch_idx):
        root = os.path.join(save_dir, "images", split)
        for k in images:
            grid = torchvision.utils.make_grid(images[k], nrow=4)
            if self.rescale:
                grid = (grid + 1.0) / 2.0  # -1,1 -> 0,1; c,h,w
            grid = grid.transpose(0, 1).transpose(1, 2).squeeze(-1)
            grid = grid.numpy()
            grid = (grid * 255).astype(np.uint8)
            filename = "{}_gs-{:06}_e-{:06}_b-{:06}.png".format(
                k,
                global_step,
                current_epoch,
                batch_idx)
            path = os.path.join(root, filename)
            os.makedirs(os.path.split(path)[0], exist_ok=True)
            Image.fromarray(grid).save(path)

    def log_img(self, pl_module, batch, batch_idx, split="train"):
        check_idx = batch_idx if self.log_on_batch_idx else pl_module.global_step
        if (self.check_frequency(check_idx) and  # batch_idx % self.batch_freq == 0
                hasattr(pl_module, "log_images") and
                callable(pl_module.log_images) and
                self.max_images > 0):
            logger = type(pl_module.logger)

            is_train = pl_module.training
            if is_train:
                pl_module.eval()

            with torch.no_grad():
                images = pl_module.log_images(batch, split=split, **self.log_images_kwargs)

            for k in images:
                N = min(images[k].shape[0], self.max_images)
                images[k] = images[k][:N]
                if isinstance(images[k], torch.Tensor):
                    images[k] = images[k].detach().cpu()
                    if self.clamp:
                        images[k] = torch.clamp(images[k], -1., 1.)

            self.log_local(pl_module.logger.save_dir, split, images,
                           pl_module.global_step, pl_module.current_epoch, batch_idx)

            logger_log_images = self.logger_log_images.get(logger, lambda *args, **kwargs: None)
            logger_log_images(pl_module, images, pl_module.global_step, split)

            if is_train:
                pl_module.train()

    def check_frequency(self, check_idx):
        if ((check_idx % self.batch_freq) == 0 or (check_idx in self.log_steps)) and (
                check_idx > 0 or self.log_first_step):
            try:
                self.log_steps.pop(0)
            except IndexError as e:
                print(e)
                pass
            return True
        return False

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, *args, **kwargs):
        dataloader_idx = kwargs.get("dataloader_idx", None)
        if dataloader_idx is None and len(args) > 0:
            dataloader_idx = args[0]
        if not self.disabled and (pl_module.global_step > 0 or self.log_first_step):
            self.log_img(pl_module, batch, batch_idx, split="train")

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, *args, **kwargs):
        """ 添加代码 """
        dataloader_idx = kwargs.get('dataloader_idx', None)
        if dataloader_idx is None and len(args) > 0:
            dataloader_idx = args[0]
        if not self.disabled and pl_module.global_step > 0:
            self.log_img(pl_module, batch, batch_idx, split="val")
        if hasattr(pl_module, 'calibrate_grad_norm'):
            if (pl_module.calibrate_grad_norm and batch_idx % 25 == 0) and batch_idx > 0:
                self.log_gradients(trainer, pl_module, batch_idx=batch_idx)

    def _TensorBoardLogger(self, pl_module, images, global_step, split):
        # TensorBoard 的 SummaryWriter 支持 add_images(name, tensor, global_step)
        tb_writer = getattr(pl_module.logger, "experiment", None) if hasattr(pl_module, "logger") else None
        if tb_writer is None:
            return
        try:
            for k in images:
                img = images[k]  # Tensor (N,C,H,W)
                # SummaryWriter.add_images 可以接受 numpy 或 Tensor（(N,C,H,W)）
                tb_writer.add_images(f"{split}/{k}", img,
                                     global_step=int(global_step) if global_step is not None else 0)
        except Exception as e:
            import warnings
            warnings.warn(f"ImageLogger: 写入 TensorBoard 失败: {e}")


class CUDACallback(Callback):
    # see https://github.com/SeanNaren/minGPT/blob/master/mingpt/callback.py
    def on_train_epoch_start(self, trainer, pl_module):
        # # Reset the memory use counter
        # torch.cuda.reset_peak_memory_stats(trainer.root_gpu)
        # torch.cuda.synchronize(trainer.root_gpu)
        # self.start_time = time.time()
        """ 添加代码 """
        # 1) 先尝试新的 API
        root_device = None
        if hasattr(trainer, "strategy") and hasattr(trainer.strategy, "root_device"):
            root_device = trainer.strategy.root_device  # 通常是 torch.device('cuda', idx) 或 torch.device('cpu')
        # 2) 回退到旧的属性（如果你在更老的 PL 里还在用 root_gpu）
        if root_device is None and hasattr(trainer, "root_gpu"):
            # 老属性通常是 GPU id (int) 或 None
            root_gpu = trainer.root_gpu
            if isinstance(root_gpu, int):
                root_device = torch.device("cuda", root_gpu)
            elif isinstance(root_gpu, torch.device):
                root_device = root_gpu
        # 3) 最终决定是否对 CUDA 调用 reset_peak_memory_stats
        if root_device is not None and getattr(root_device, "type", None) == "cuda":
            torch.cuda.reset_peak_memory_stats(root_device)
        else:
            # 如果你希望在任何可用 CUDA 上都 reset（即使没法从 trainer 得到 device），可以解除下面注释：
            # if torch.cuda.is_available():
            #     torch.cuda.reset_peak_memory_stats()
            pass

    # def on_train_epoch_end(self, trainer, pl_module, *args, **kwargs):
    #     outputs = kwargs.get("outputs", None)
    #     if outputs is None and len(args) > 0:
    #         # 有些版本会把 outputs 作为第一个额外位置参数传入
    #         outputs = args[0]
    #     torch.cuda.synchronize(trainer.root_gpu)
    #     max_memory = torch.cuda.max_memory_allocated(trainer.root_gpu) / 2 ** 20
    #     epoch_time = time.time() - self.start_time
    #
    #     try:
    #         max_memory = trainer.training_type_plugin.reduce(max_memory)
    #         epoch_time = trainer.training_type_plugin.reduce(epoch_time)
    #
    #         rank_zero_info(f"Average Epoch time: {epoch_time:.2f} seconds")
    #         rank_zero_info(f"Average Peak memory {max_memory:.2f}MiB")
    #     except AttributeError:
    #         pass
    def on_train_epoch_end(self, trainer, pl_module):
        """
        兼容旧版 trainer.root_gpu 和新版 trainer.strategy.root_device。
        只在检测到 CUDA device 时调用 torch.cuda.synchronize(...)。
        """
        try:
            root_device = None

            # 新 API（推荐）
            if hasattr(trainer, "strategy") and hasattr(trainer.strategy, "root_device"):
                root_device = trainer.strategy.root_device  # 通常是 torch.device('cuda', idx) 或 torch.device('cpu')

            # 某些 PL 版本或自定义 trainer 可能有 device 属性
            if root_device is None and hasattr(trainer, "device"):
                root_device = trainer.device

            # 兼容旧 API（可能是 int 或 torch.device）
            if root_device is None and hasattr(trainer, "root_gpu"):
                rg = trainer.root_gpu
                if isinstance(rg, int):
                    root_device = torch.device("cuda", rg)
                elif isinstance(rg, torch.device):
                    root_device = rg

            # 如果最终确定是 cuda 设备则同步；否则在可用时同步当前设备
            if root_device is not None and getattr(root_device, "type", None) == "cuda":
                # torch.cuda.synchronize 接受 torch.device 或 int 或 None
                torch.cuda.synchronize(root_device)
            else:
                # 若找不到明确设备，但系统上有 CUDA，可同步当前设备（单 GPU 情况下安全）
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
        except Exception as e:
            # 不要让同步失败阻止训练继续；打印警告以便排查
            print(f"[Warning] CUDA synchronize failed in on_train_epoch_end: {e}")


if __name__ == "__main__":
    now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    sys.path.append(os.getcwd())

    # parser = get_parser()
    # parser = Trainer.add_argparse_args(parser)

    parser = get_parser()
    opt, unknown = parser.parse_known_args()

    if opt.name and opt.resume:
        raise ValueError(
            "-n/--name and -r/--resume cannot be specified both."
            "If you want to resume training in a new log folder, "
            "use -n/--name in combination with --resume_from_checkpoint"
        )
    if opt.resume:
        if not os.path.exists(opt.resume):
            raise ValueError("Cannot find {}".format(opt.resume))
        if os.path.isfile(opt.resume):
            paths = opt.resume.split("/")
            logdir = "/".join(paths[:-2])
            ckpt = opt.resume
        else:
            assert os.path.isdir(opt.resume), opt.resume
            logdir = opt.resume.rstrip("/")
            ckpt = os.path.join(logdir, "checkpoints", "last.ckpt")
        opt.resume_from_checkpoint = ckpt
        base_configs = sorted(glob.glob(os.path.join(logdir, "configs/*.yaml")))
        opt.base = base_configs + opt.base
        _tmp = logdir.split("/")
        nowname = _tmp[-1]
    else:
        if opt.name:
            name = "_" + opt.name
        elif opt.base:
            cfg_fname = os.path.split(opt.base[0])[-1]
            cfg_name = os.path.splitext(cfg_fname)[0]
            name = "_" + cfg_name
        else:
            name = ""
        nowname = now + name + opt.postfix
        logdir = os.path.join(opt.logdir, nowname)

    """ 添加代码：修Bug """
    # 处理 gpus 参数
    if hasattr(opt, 'gpus') and opt.gpus:
        gpus_str = opt.gpus.strip().rstrip(',')

        if gpus_str:
            # 有 GPU 指定
            gpu_ids = [int(x.strip()) for x in gpus_str.split(',') if x.strip()]
            devices = gpu_ids[0] if len(gpu_ids) == 1 else gpu_ids
            accelerator = 'gpu'
        else:
            # 空字符串 -> CPU
            devices = 'auto'
            accelerator = 'cpu'
    else:
        # 未指定 -> 默认配置
        devices = 'auto'
        accelerator = 'auto'
    # 构建 trainer_kwargs
    trainer_kwargs = {
        'accelerator': accelerator,
        'devices': devices,
        'max_epochs': getattr(opt, 'max_epochs', 1000),
        'default_root_dir': getattr(opt, 'logdir', './logs'),
    }
    # 添加其他可选参数
    optional_params = [
        'min_epochs', 'max_steps', 'num_nodes', 'precision',
        'accumulate_grad_batches', 'log_every_n_steps',
        'check_val_every_n_epoch', 'gradient_clip_val',
        'gradient_clip_algorithm', 'limit_train_batches',
        'limit_val_batches', 'limit_test_batches',
        'benchmark', 'deterministic', 'strategy'
    ]
    for param in optional_params:
        if hasattr(opt, param):
            value = getattr(opt, param)
            if value is not None and value != -1:
                trainer_kwargs[param] = value
    # 特殊处理 fast_dev_run
    if hasattr(opt, 'fast_dev_run') and opt.fast_dev_run:
        trainer_kwargs['fast_dev_run'] = True
    # 创建 Trainer
    trainer = Trainer(**trainer_kwargs)


    ckptdir = os.path.join(logdir, "checkpoints")
    cfgdir = os.path.join(logdir, "configs")
    seed_everything(opt.seed)

    # try:
    # init and save configs
    configs = [OmegaConf.load(cfg) for cfg in opt.base]
    cli = OmegaConf.from_dotlist(unknown)
    config = OmegaConf.merge(*configs, cli)
    lightning_config = config.pop("lightning", OmegaConf.create())
    # merge trainer cli with config
    trainer_config = lightning_config.get("trainer", OmegaConf.create())
    # default to ddp
    trainer_config["accelerator"] = "ddp"
    for k in nondefault_trainer_args(opt):
        trainer_config[k] = getattr(opt, k)
    if not "gpus" in trainer_config:
        del trainer_config["accelerator"]
        cpu = True
    else:
        gpuinfo = trainer_config["gpus"]
        print(f"Running on GPUs {gpuinfo}")
        cpu = False
    trainer_opt = argparse.Namespace(**trainer_config)
    lightning_config.trainer = trainer_config

    # model
    model = instantiate_from_config(config.model)

    model.configs = config

    # trainer and callbacks
    trainer_kwargs = dict()

    # default logger configs
    default_logger_cfgs = {
        "wandb": {
            "target": "pytorch_lightning.loggers.WandbLogger",
            "params": {
                "name": nowname,
                "save_dir": logdir,
                "offline": opt.debug,
                "id": nowname,
            }
        },
        "testtube": {
            "target": "pytorch_lightning.loggers.TensorBoardLogger",
            "params": {
                "name": "testtube",
                "save_dir": logdir,
            }
        },
    }
    # We use wandb by default. Change to testtube if you do not want to use wandb
    default_logger_cfg = default_logger_cfgs["wandb"]
    os.makedirs(os.path.join(logdir, 'wandb'), exist_ok=True)
    if "logger" in lightning_config:
        logger_cfg = lightning_config.logger
    else:
        logger_cfg = OmegaConf.create()
    logger_cfg = OmegaConf.merge(default_logger_cfg, logger_cfg)
    trainer_kwargs["logger"] = instantiate_from_config(logger_cfg)

    # modelcheckpoint - use TrainResult/EvalResult(checkpoint_on=metric) to
    # specify which metric is used to determine best models
    default_modelckpt_cfg = {
        "target": "pytorch_lightning.callbacks.ModelCheckpoint",
        "params": {
            "dirpath": ckptdir,
            "filename": "{epoch:06}",
            "verbose": True,
            "save_last": True,
        }
    }
    if hasattr(model, "monitor"):
        print(f"Monitoring {model.monitor} as checkpoint metric.")
        default_modelckpt_cfg["params"]["monitor"] = model.monitor
        default_modelckpt_cfg["params"]["save_top_k"] = 20

    if "modelcheckpoint" in lightning_config:
        modelckpt_cfg = lightning_config.modelcheckpoint
    else:
        modelckpt_cfg =  OmegaConf.create()
    modelckpt_cfg = OmegaConf.merge(default_modelckpt_cfg, modelckpt_cfg)
    print(f"Merged modelckpt-cfg: \n{modelckpt_cfg}")
    if version.parse(pl.__version__) < version.parse('1.4.0'):
        trainer_kwargs["checkpoint_callback"] = instantiate_from_config(modelckpt_cfg)

    # add callback which sets up log directory
    default_callbacks_cfg = {
        "setup_callback": {
            "target": "main.SetupCallback",
            "params": {
                "resume": opt.resume,
                "now": now,
                "logdir": logdir,
                "ckptdir": ckptdir,
                "cfgdir": cfgdir,
                "config": config,
                "lightning_config": lightning_config,
            }
        },
        "image_logger": {
            "target": "main.ImageLogger",
            "params": {
                "batch_frequency": 750,
                "max_images": 4,
                "clamp": True
            }
        },
        "learning_rate_logger": {
            "target": "main.LearningRateMonitor",
            "params": {
                "logging_interval": "step",
                # "log_momentum": True
            }
        },
        "cuda_callback": {
            "target": "main.CUDACallback"
        },
    }
    if version.parse(pl.__version__) >= version.parse('1.4.0'):
        default_callbacks_cfg.update({'checkpoint_callback': modelckpt_cfg})

    if "callbacks" in lightning_config:
        callbacks_cfg = lightning_config.callbacks
    else:
        callbacks_cfg = OmegaConf.create()

    if 'metrics_over_trainsteps_checkpoint' in callbacks_cfg:
        print(
            'Caution: Saving checkpoints every n train steps without deleting. This might require some free space.')
        default_metrics_over_trainsteps_ckpt_dict = {
            'metrics_over_trainsteps_checkpoint':
                {"target": 'pytorch_lightning.callbacks.ModelCheckpoint',
                 'params': {
                     "dirpath": os.path.join(ckptdir, 'trainstep_checkpoints'),
                     "filename": "{epoch:06}-{step:09}",
                     "verbose": True,
                     'save_top_k': -1,
                     'every_n_train_steps': 10000,
                     'save_weights_only': True
                 }
                 }
        }
        default_callbacks_cfg.update(default_metrics_over_trainsteps_ckpt_dict)

    callbacks_cfg = OmegaConf.merge(default_callbacks_cfg, callbacks_cfg)
    if 'ignore_keys_callback' in callbacks_cfg and hasattr(trainer_opt, 'resume_from_checkpoint'):
        callbacks_cfg.ignore_keys_callback.params['ckpt_path'] = trainer_opt.resume_from_checkpoint
    elif 'ignore_keys_callback' in callbacks_cfg:
        del callbacks_cfg['ignore_keys_callback']

    trainer_kwargs["callbacks"] = [instantiate_from_config(callbacks_cfg[k]) for k in callbacks_cfg]

    # trainer = Trainer.from_argparse_args(trainer_opt, **trainer_kwargs)
    if hasattr(Trainer, "from_argparse_args"):
        # 旧版 Lightning（带此方法），优先使用原有行为
        trainer = Trainer.from_argparse_args(trainer_opt, **trainer_kwargs)
    else:
        # 新版 Lightning：没有 from_argparse_args，手动把 trainer_opt 的字段合并到 trainer_kwargs（但不覆盖已显式设置的键）
        opt_dict = vars(trainer_opt) if trainer_opt is not None else {}
        for k, v in opt_dict.items():
            # 跳过 None 或已经在 trainer_kwargs 中显式设置的参数
            if v is None or k in trainer_kwargs:
                continue
            # 有些字段在新版本中名字不同或不适用，这里做最保守的直接传递
            trainer_kwargs[k] = v
        # 最终用标准构造器实例化
        trainer = Trainer(**trainer_kwargs)
    trainer.logdir = logdir  ###

    # data
    data = instantiate_from_config(config.data)
    # NOTE according to https://pytorch-lightning.readthedocs.io/en/latest/datamodules.html
    # calling these ourselves should not be necessary but it is.
    # lightning still takes care of proper multiprocessing though
    data.prepare_data()
    data.setup()
    print("#### Data #####")
    for k in data.datasets:
        print(f"{k}, {data.datasets[k].__class__.__name__}, {len(data.datasets[k])}")

    # configure learning rate
    bs, base_lr = config.data.params.batch_size, config.model.base_learning_rate
    if not cpu:
        ngpu = len(lightning_config.trainer.gpus.strip(",").split(','))
    else:
        ngpu = 1
    if 'accumulate_grad_batches' in lightning_config.trainer:
        accumulate_grad_batches = lightning_config.trainer.accumulate_grad_batches
    else:
        accumulate_grad_batches = 1
    print(f"accumulate_grad_batches = {accumulate_grad_batches}")
    lightning_config.trainer.accumulate_grad_batches = accumulate_grad_batches
    if opt.scale_lr:
        model.learning_rate = accumulate_grad_batches * ngpu * bs * base_lr
        print(
            "Setting learning rate to {:.2e} = {} (accumulate_grad_batches) * {} (num_gpus) * {} (batchsize) * {:.2e} (base_lr)".format(
                model.learning_rate, accumulate_grad_batches, ngpu, bs, base_lr))
    else:
        model.learning_rate = base_lr
        print("++++ NOT USING LR SCALING ++++")
        print(f"Setting learning rate to {model.learning_rate:.2e}")
    # allow checkpointing via USR1
    def melk(*args, **kwargs):
        # run all checkpoint hooks
        if trainer.global_rank == 0:
            print("Summoning checkpoint.")
            ckpt_path = os.path.join(ckptdir, "last.ckpt")
            trainer.save_checkpoint(ckpt_path)
    def divein(*args, **kwargs):
        if trainer.global_rank == 0:
            import pudb;
            pudb.set_trace()

    import signal
    def safe_register_signal(name, handler):
        """尝试按优先级注册信号处理器，不支持则忽略。"""
        # 优先候选按优先级排列
        candidates = []
        if name == "USR1":
            candidates = ["SIGUSR1", "SIGBREAK", "SIGTERM"]
        elif name == "USR2":
            candidates = ["SIGUSR2", "SIGBREAK", "SIGINT"]
        else:
            candidates = [name]
        for signame in candidates:
            if hasattr(signal, signame):
                sig = getattr(signal, signame)
                try:
                    signal.signal(sig, handler)
                    print(f"Registered handler for signal: {signame}")
                    return True
                except Exception as e:
                    print(f"Failed to register handler for {signame}: {e}")
                    # 若注册失败，尝试下一个
                    continue
        # 如果都没注册成功，返回 False（但不抛错）
        print(f"No supported signal found for {name}; continuing without it.")
        return False
    # 你的 handler 函数（示例）
    def melk(*args, **kwargs):
        if trainer.global_rank == 0:
            print("Summoning checkpoint.")
            ckpt_path = os.path.join(ckptdir, "last.ckpt")
            trainer.save_checkpoint(ckpt_path)
    def divein(*args, **kwargs):
        if trainer.global_rank == 0:
            import pudb
            pudb.set_trace()
    # 注册
    safe_register_signal("USR1", melk)
    safe_register_signal("USR2", divein)

    # run
    if opt.train:
        try:
            trainer.fit(model, data)
        except Exception:
            melk()
            raise
    if not opt.no_test and not trainer.interrupted:
        trainer.test(model, data)
