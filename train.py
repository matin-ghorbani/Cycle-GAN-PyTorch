from os import makedirs
from os.path import join

import torch
from torch import nn, Tensor
from torch import optim
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from torchvision.utils import save_image
from tqdm import tqdm

from dataset import HorseZebraDataset
from models import Generator, Discriminator
import utils as ul
import config

makedirs('saved_images', exist_ok=True)


def train(
        h_dis: Discriminator,
        z_dis: Discriminator,
        h_gen: Generator,
        z_gen: Generator,
        loader: DataLoader,
        dis_optimizer: optim.Adam,
        gen_optimizer: optim.Adam,
        l1_loss: nn.L1Loss,
        mse_loss: nn.MSELoss,
        dis_scaler: GradScaler,
        gen_scaler: GradScaler
):
    horse: Tensor
    zebra: Tensor

    loop = tqdm(loader)
    for idx, batch in enumerate(loop):
        zebra, horse = map(lambda img: img.to(config.DEVICE), batch)

        # region Train Discriminator

        # Necessary for float 16
        with autocast():
            # For Horse
            fake_horse: Tensor = h_gen(zebra)

            dis_horse_real: Tensor = h_dis(horse)
            dis_horse_fake: Tensor = h_dis(fake_horse.detach())

            # Each of them should be real for a greate model
            # Real is 1 and fake is 0
            dis_horse_real_loss = mse_loss(dis_horse_real, torch.ones_like(dis_horse_real))
            dis_horse_fake_loss = mse_loss(dis_horse_fake, torch.zeros_like(dis_horse_fake))
            dis_horse_loss = dis_horse_real_loss + dis_horse_fake_loss

            # For Zebra
            fake_zebra: Tensor = h_gen(horse)

            dis_zebra_real: Tensor = z_dis(zebra)
            dis_zebra_fake: Tensor = z_dis(fake_zebra.detach())

            # Each of them should be real for a greate model
            # Real is 1 and fake is 0
            dis_zebra_real_loss = mse_loss(dis_zebra_real, torch.ones_like(dis_zebra_real))
            dis_zebra_fake_loss = mse_loss(dis_zebra_fake, torch.zeros_like(dis_zebra_fake))
            dis_zebra_loss = dis_zebra_real_loss + dis_zebra_fake_loss

            # Put them together
            dis_loss = (dis_horse_loss + dis_zebra_loss) / 2

        dis_optimizer.zero_grad()
        dis_scaler.scale(dis_loss).backward()
        dis_scaler.step(dis_optimizer)
        dis_scaler.update()

        # endregion

        # region Train Generator

        with autocast():
            # region Adversarial loss for both generators
            dis_horse_fake = h_dis(fake_horse)
            dis_zebra_fake = z_dis(fake_zebra)
            gen_loss_horse = mse_loss(dis_horse_fake, torch.ones_like(dis_horse_fake))
            gen_loss_zebra = mse_loss(dis_zebra_fake, torch.ones_like(dis_zebra_fake))
            # endregion

            # region Cycle loss for both generators
            # Try to generate a zebra from a fake horse
            cycle_zebra = z_gen(fake_horse)

            # Try to generate a horse from a fake zebra
            cycle_horse = z_gen(fake_zebra)

            # Calculate the losses
            cycle_zebra_loss = l1_loss(zebra, cycle_zebra)
            cycle_horse_loss = l1_loss(horse, cycle_horse)

            # endregion

            # region Identity loss for both generators

            # Send a zebra to the zebra generator
            identity_zebra = z_gen(zebra)
            identity_zebra_loss = l1_loss(zebra, identity_zebra)

            # Send a horse to the horse generator
            identity_horse = h_gen(horse)
            identity_horse_loss = l1_loss(horse, identity_horse)

            # endregion

            # Put all together

            gen_loss = (
                    gen_loss_zebra
                    + gen_loss_horse
                    + cycle_zebra_loss * config.LAMBDA_CYCLE
                    + cycle_horse_loss * config.LAMBDA_CYCLE
                    + identity_zebra_loss * config.LAMBDA_IDENTITY
                    + identity_horse_loss * config.LAMBDA_IDENTITY
            )

        gen_optimizer.zero_grad()
        gen_scaler.scale(gen_loss).backward()
        gen_scaler.step(gen_optimizer)
        gen_scaler.update()
        # endregion

        if idx % 200 == 0:
            # To reverse the normalization
            fake_horse_img = fake_horse * .5 + .5
            fake_zebra_img = fake_zebra * .5 + .5
            save_image(fake_horse_img, f'saved_images/fake_horse_{idx}.png')
            save_image(fake_zebra_img, f'saved_images/fake_zebra_{idx}.png')


def main() -> None:
    h_dis = Discriminator(in_channels=3).to(config.DEVICE)
    h_gen = Generator(img_channels=3, num_res_blocks=9).to(config.DEVICE)

    z_dis = Discriminator(in_channels=3).to(config.DEVICE)
    z_gen = Generator(img_channels=3, num_res_blocks=9).to(config.DEVICE)

    dis_optimizer = optim.Adam(
        list(h_dis.parameters()) + list(z_dis.parameters()),
        lr=config.LEARNING_RATE,
        betas=(.5, .999)
    )
    gen_optimizer = optim.Adam(
        list(h_gen.parameters()) + list(z_gen.parameters()),
        lr=config.LEARNING_RATE,
        betas=(.5, .999)
    )

    # For cycle consistency loss and identity loss
    l1_loss = nn.L1Loss()
    # Adversarial loss
    mse_loss = nn.MSELoss()

    if config.LOAD_MODEL:
        ul.load_checkpoint(
            config.CHECKPOINT_GEN_H,
            h_gen,
            gen_optimizer,
            config.LEARNING_RATE,
        )
        ul.load_checkpoint(
            config.CHECKPOINT_GEN_Z,
            z_gen,
            gen_optimizer,
            config.LEARNING_RATE,
        )
        ul.load_checkpoint(
            config.CHECKPOINT_CRITIC_H,
            h_dis,
            dis_optimizer,
            config.LEARNING_RATE,
        )
        ul.load_checkpoint(
            config.CHECKPOINT_CRITIC_Z,
            z_dis,
            dis_optimizer,
            config.LEARNING_RATE,
        )

    dataset = HorseZebraDataset(
        root_zebra=join(config.TRAIN_DIR, 'zebras'),
        root_horse=join(config.TRAIN_DIR, 'horses'),
        transform=config.transforms
    )
    loader = DataLoader(
        dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=True
    )

    gen_scaler = GradScaler()
    dis_scaler = GradScaler()

    for epoch in range(1, config.NUM_EPOCHS + 1):
        train(
            h_dis,
            z_dis,
            h_gen,
            z_gen,
            loader,
            dis_optimizer,
            gen_optimizer,
            l1_loss,
            mse_loss,
            dis_scaler,
            gen_scaler
        )

        if config.SAVE_MODEL:
            ul.save_checkpoint(h_gen, gen_optimizer, filename=config.CHECKPOINT_GEN_H)
            ul.save_checkpoint(z_gen, gen_optimizer, filename=config.CHECKPOINT_GEN_Z)

            ul.save_checkpoint(h_dis, dis_optimizer, filename=config.CHECKPOINT_CRITIC_H)
            ul.save_checkpoint(z_dis, dis_optimizer, filename=config.CHECKPOINT_CRITIC_Z)


if __name__ == '__main__':
    main()
