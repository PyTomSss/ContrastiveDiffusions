import jax
import jax.numpy as jnp
import einops
from diffuse.unet import UNet
from diffuse.score_matching import score_match_loss
from diffuse.sde import SDE, LinearSchedule
from functools import partial
import numpy as np
import optax
from tqdm import tqdm
import wandb
import gzip
import struct


#data = jnp.load("dataset/mnist.npz")
def load_mnist_images(path):
    open_fn = gzip.open if path.endswith(".gz") else open
    with open_fn(path, "rb") as f:
        magic, n, rows, cols = struct.unpack(">IIII", f.read(16))
        if magic != 2051:
            raise ValueError(f"Unexpected magic number for images: {magic}")
        data = np.frombuffer(f.read(), dtype=np.uint8)
        data = data.reshape(n, rows, cols)
    return data


def load_mnist_labels(path):
    open_fn = gzip.open if path.endswith(".gz") else open
    with open_fn(path, "rb") as f:
        magic, n = struct.unpack(">II", f.read(8))
        if magic != 2049:
            raise ValueError(f"Unexpected magic number for labels: {magic}")
        data = np.frombuffer(f.read(), dtype=np.uint8)
    return data


x_train = load_mnist_images("/vols/bitbucket/kebl8577/datasets/data/MNIST/train-images-idx3-ubyte.gz")
y_train = load_mnist_labels("/vols/bitbucket/kebl8577/datasets/data/MNIST/train-labels-idx1-ubyte.gz")

# normalisation
x_train = x_train.astype(np.float32) / 255.0

# conversion JAX
xs = jnp.array(x_train)
ys = jnp.array(y_train)
key = jax.random.PRNGKey(0)
#xs = data["X"]

batch_size = 256
n_epochs = 3500
n_t = 256
tf = 2.0
dt = tf / n_t
lr = 2e-4

xs = jax.random.permutation(key, xs, axis=0)
data = einops.rearrange(xs, "b h w -> b h w 1")
shape_sample = data.shape[1:]

beta = LinearSchedule(b_min=0.02, b_max=5.0, t0=0.0, T=2.0)
sde = SDE(beta)

nn_unet = UNet(dt, 64, upsampling="pixel_shuffle")
init_params = nn_unet.init(
    key, jnp.ones((batch_size, *shape_sample)), jnp.ones((batch_size,))
)


def weight_fun(t):
    int_b = sde.beta.integrate(t, 0).squeeze()
    return 1 - jnp.exp(-int_b)


loss = partial(score_match_loss, lmbda=jax.vmap(weight_fun), network=nn_unet)

nsteps_per_epoch = data.shape[0] // batch_size
until_steps = int(0.95 * n_epochs) * nsteps_per_epoch

schedule = optax.cosine_decay_schedule(
    init_value=lr, decay_steps=until_steps, alpha=1e-2
)
optimizer = optax.adam(learning_rate=schedule)
optimizer = optax.chain(optax.clip_by_global_norm(1.0), optimizer)
ema_kernel = optax.ema(0.99)


@jax.jit
def step(key, params, opt_state, ema_state, batch):
    val_loss, g = jax.value_and_grad(loss)(params, key, batch, sde, n_t, tf)
    updates, opt_state = optimizer.update(g, opt_state, params)
    params = optax.apply_updates(params, updates)
    ema_params, ema_state = ema_kernel.update(params, ema_state)
    return params, opt_state, ema_state, val_loss, ema_params


# -----------------------
# W&B init
# -----------------------
wandb.init(
    project="mnist-diffusion",
    name="unet_score_matching",
    config={
        "batch_size": batch_size,
        "n_epochs": n_epochs,
        "n_t": n_t,
        "tf": tf,
        "dt": dt,
        "lr": lr,
        "b_min": 0.02,
        "b_max": 5.0,
        "ema_decay": 0.99,
        "grad_clip": 1.0,
        "optimizer": "adam",
        "scheduler": "cosine_decay",
    },
)

params = init_params
opt_state = optimizer.init(params)
ema_state = ema_kernel.init(params)

global_step = 0

for epoch in range(n_epochs):
    subkey, key = jax.random.split(key)
    idx = jax.random.choice(
        subkey, data.shape[0], (nsteps_per_epoch, batch_size), replace=False
    )

    p_bar = tqdm(range(nsteps_per_epoch), desc=f"Epoch {epoch}")
    list_loss = []

    for i in p_bar:
        subkey, key = jax.random.split(key)
        params, opt_state, ema_state, val_loss, ema_params = step(
            subkey, params, opt_state, ema_state, data[idx[i]]
        )

        loss_value = float(val_loss)
        current_lr = float(schedule(global_step))

        p_bar.set_postfix({"loss": loss_value, "lr": current_lr})
        list_loss.append(loss_value)

        # log par step
        wandb.log(
            {
                "train/loss_step": loss_value,
                "train/lr": current_lr,
                "epoch": epoch,
                "global_step": global_step,
            },
            step=global_step,
        )

        global_step += 1

    mean_loss = float(np.mean(list_loss))
    print(f"epoch=: {epoch} | mean_loss=: {mean_loss}")

    # log par epoch
    wandb.log(
        {
            "train/loss_epoch": mean_loss,
            "epoch": epoch,
        },
        step=global_step,
    )

    if (epoch + 1) % 500 == 0:
        np.savez(f"ann_{epoch}.npz", params=params, ema_params=ema_params)

np.savez("ann_end.npz", params=params, ema_params=ema_params)
wandb.finish()