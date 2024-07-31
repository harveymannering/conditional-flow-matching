# Inspired from https://github.com/w86763777/pytorch-ddpm/tree/master.

# Authors: Kilian Fatras
#          Alexander Tong

import copy
import os

import torch
from absl import app, flags
from torchdyn.core import NeuralODE
from torchvision import datasets, transforms
from tqdm import trange
from utils_cifar import ema, generate_samples, generate_conditional_samples, infiniteloop, sample_plan, ClassSeparatedCIFAR10, NonOverlappingClassSampler, sample_conditional_pt

from torchcfm.conditional_flow_matching import (
    ConditionalFlowMatcher,
    ExactOptimalTransportConditionalFlowMatcher,
    TargetConditionalFlowMatcher,
    VariancePreservingConditionalFlowMatcher,
)
from torchcfm.models.unet.unet import UNetModelWrapper

FLAGS = flags.FLAGS

flags.DEFINE_string("model", "otcfm", help="flow matching model type")
flags.DEFINE_string("output_dir", "./results/", help="output_directory")
# UNet
flags.DEFINE_integer("num_channel", 128, help="base channel of UNet")

# Training
flags.DEFINE_float("lr", 2e-4, help="target learning rate")  # TRY 2e-4
flags.DEFINE_float("grad_clip", 1.0, help="gradient norm clipping")
flags.DEFINE_integer(
    "total_steps", 400001, help="total training steps"
)  # Lipman et al uses 400k but double batch size
flags.DEFINE_integer("warmup", 5000, help="learning rate warmup")
flags.DEFINE_integer("batch_size", 128, help="batch size")  # Lipman et al uses 128
flags.DEFINE_integer("num_workers", 4, help="workers of Dataloader")
flags.DEFINE_float("ema_decay", 0.9999, help="ema decay rate")
flags.DEFINE_bool("parallel", False, help="multi gpu training")

# Evaluation
flags.DEFINE_integer(
    "save_step",
    20000,
    help="frequency of saving checkpoints, 0 to disable during training",
)


use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")


def warmup_lr(step):
    return min(step, FLAGS.warmup) / FLAGS.warmup


def train(argv):
    print(
        "lr, total_steps, ema decay, save_step:",
        FLAGS.lr,
        FLAGS.total_steps,
        FLAGS.ema_decay,
        FLAGS.save_step,
    )

    # DATASETS/DATALOADER
    transform=transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    if FLAGS.model == "mmotcfm":
        # Test the dataloader
        dataset = ClassSeparatedCIFAR10(root='./data', train=True, transform=transform)
        sampler = NonOverlappingClassSampler(dataset, FLAGS.batch_size)
        dataloader = torch.utils.data.DataLoader(dataset, batch_sampler=sampler, num_workers=FLAGS.num_workers)
    else:
        dataset = datasets.CIFAR10(
            root="./data",
            train=True,
            download=False,
            transform=transform,
        )
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=FLAGS.batch_size,
            shuffle=True,
            num_workers=FLAGS.num_workers,
            drop_last=True,
        )

    datalooper = infiniteloop(dataloader)

    # MODELS
    if FLAGS.model == "cc-otcfm" or FLAGS.model == "mmotcfm" :
        net_model = UNetModelWrapper(
            dim=(3, 32, 32),
            num_res_blocks=2,
            num_channels=FLAGS.num_channel,
            channel_mult=[1, 2, 2, 2],
            num_heads=4,
            class_cond=True,
            num_classes=10,
            num_head_channels=64,
            attention_resolutions="16",
            dropout=0.1,
        ).to(
            device
        )  # new dropout + bs of 128
    else:
        net_model = UNetModelWrapper(
            dim=(3, 32, 32),
            num_res_blocks=2,
            num_channels=FLAGS.num_channel,
            channel_mult=[1, 2, 2, 2],
            num_heads=4,
            num_head_channels=64,
            attention_resolutions="16",
            dropout=0.1,
        ).to(
            device
        )  # new dropout + bs of 128

    ema_model = copy.deepcopy(net_model)
    optim = torch.optim.Adam(net_model.parameters(), lr=FLAGS.lr)
    sched = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda=warmup_lr)
    if FLAGS.parallel:
        print(
            "Warning: parallel training is performing slightly worse than single GPU training due to statistics computation in dataparallel. We recommend to train over a single GPU, which requires around 8 Gb of GPU memory."
        )
        net_model = torch.nn.DataParallel(net_model)
        ema_model = torch.nn.DataParallel(ema_model)

    # show model size
    model_size = 0
    for param in net_model.parameters():
        model_size += param.data.nelement()
    print("Model params: %.2f M" % (model_size / 1024 / 1024))

    #################################
    #            OT-CFM
    #################################

    sigma = 0.0
    if FLAGS.model == "mmotcfm":
        FM = ExactOptimalTransportConditionalFlowMatcher(sigma=sigma)
    elif FLAGS.model == "cc-otcfm":
        FM = ExactOptimalTransportConditionalFlowMatcher(sigma=sigma)
    elif FLAGS.model == "otcfm":
        FM = ExactOptimalTransportConditionalFlowMatcher(sigma=sigma)
    elif FLAGS.model == "icfm":
        FM = ConditionalFlowMatcher(sigma=sigma)
    elif FLAGS.model == "fm":
        FM = TargetConditionalFlowMatcher(sigma=sigma)
    elif FLAGS.model == "si":
        FM = VariancePreservingConditionalFlowMatcher(sigma=sigma)
    else:
        raise NotImplementedError(
            f"Unknown model {FLAGS.model}, must be one of ['mmotcfm', 'cc-otcfm', 'otcfm', 'icfm', 'fm', 'si']"
        )

    savedir = FLAGS.output_dir + FLAGS.model + "/"
    os.makedirs(savedir, exist_ok=True)

    with trange(FLAGS.total_steps, dynamic_ncols=True) as pbar:
        for step in pbar:
            optim.zero_grad()
            x1, y1 = next(datalooper)
            x0 = torch.randn_like(x1)
            if FLAGS.model == "mmotcfm" :
                # mmot sinkhorn
                x0, x1, x2, _, y1, y2 = sample_plan(
                    x0[:FLAGS.batch_size].numpy(), 
                    x1[:FLAGS.batch_size].numpy(), 
                    x1[FLAGS.batch_size:].numpy(), 
                    None, 
                    y1[:FLAGS.batch_size],
                    y1[FLAGS.batch_size:]
                )
                
                x0, x1, x2 = torch.tensor(x0), torch.tensor(x1), torch.tensor(x2)
                x1 = torch.concat([x1[:FLAGS.batch_size//2], x2[FLAGS.batch_size//2:]], axis=0)
                y1 = torch.concat([y1[:FLAGS.batch_size//2], y2[FLAGS.batch_size//2:]], axis=0)
                t = torch.rand(x0.shape[0]).type_as(x0)
                xt = sample_conditional_pt(x0, x1, t, sigma=sigma)
                ut = x1 - x0
                t, xt, y1, ut = t.to(device), xt.to(device), y1.to(device), ut.to(device)
                vt = net_model(t, xt, y1)
            elif FLAGS.model == "cc-otcfm": 
                t, xt, ut, _, y1 = FM.guided_sample_location_and_conditional_flow(x0, x1, None, y1)
                vt = net_model(t, xt, y1)
            else:
                t, xt, ut = FM.sample_location_and_conditional_flow(x0, x1)
                vt = net_model(t, xt)
            loss = torch.mean((vt - ut) ** 2)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net_model.parameters(), FLAGS.grad_clip)  # new
            optim.step()
            sched.step()
            ema(net_model, ema_model, FLAGS.ema_decay)  # new

            # sample and Saving the weights
            if FLAGS.save_step > 0 and step % FLAGS.save_step == 0:
                if FLAGS.model == "cc-otcfm" or FLAGS.model == "mmotcfm":
                    generate_conditional_samples(net_model, FLAGS.parallel, savedir, step, net_="normal")
                    generate_conditional_samples(ema_model, FLAGS.parallel, savedir, step, net_="ema")
                else:
                    generate_samples(net_model, FLAGS.parallel, savedir, step, net_="normal")
                    generate_samples(ema_model, FLAGS.parallel, savedir, step, net_="ema")

                torch.save(
                    {
                        "net_model": net_model.state_dict(),
                        "ema_model": ema_model.state_dict(),
                        "sched": sched.state_dict(),
                        "optim": optim.state_dict(),
                        "step": step,
                    },
                    savedir + f"{FLAGS.model}_cifar10_weights_step_{step}.pt",
                )


if __name__ == "__main__":
    app.run(train)
