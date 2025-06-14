# train_single.py  ─────────────────────────────────────────
import os, gc, json, argparse, h5py as h5, numpy as np, tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# ──────────────────────────────────────────────────────────

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    print("🏎️  TensorFlow usará la(s) GPU:", gpus)
else:
    raise RuntimeError(
        "🚫  No se detectó GPU. "
        "Comprueba driver, CUDA/cuDNN y la instalación de TensorFlow-GPU."
    )
import utils                      # ← tus utilidades de siempre
from CaloScore          import CaloScore
from CaloScore_distill  import CaloScore_distill
from WGAN               import WGAN
import math 

tf.random.set_seed(1234)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"          # menos ruido

# ──────────────── 1. CLI ─────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--data_folder", required=True)
parser.add_argument("--config",      required=True)
parser.add_argument("--model",       default="CaloScore",
                    choices=["CaloScore","WGAN"])
parser.add_argument("--nevts", type=float, default=-1)
parser.add_argument("--frac",  type=float, default=0.8)
parser.add_argument("--distill", action="store_true")
parser.add_argument("--factor",  type=int, default=1)
parser.add_argument("--load",   action="store_true")
args = parser.parse_args()

# ──────────────── 2. Carga cfg + datos ──────────────────
cfg = utils.LoadJson(args.config)
voxels, layers, energies = [], [], []

for fname in cfg["FILES"]:
    v, l, e = utils.DataLoader(
        os.path.join(args.data_folder, fname),
        cfg["SHAPE"], args.nevts,
        emax = cfg["EMAX"], emin = cfg["EMIN"],
        max_deposit = cfg["MAXDEP"],
        logE = cfg["logE"],
        use_1D = cfg["DATASET"] == 1,
    )

    voxels.append(v); layers.append(l); energies.append(e)

voxels  = np.reshape(voxels, cfg["SHAPE"])
layers  = np.concatenate(layers)
energies= np.concatenate(energies)

if cfg["PARTICLE"] == 'pion':
    voxels = utils.ApplyPreprocessing(voxels, f"preprocessing_{cfg['DATASET']}_voxel_pions.json")
    layers = utils.ApplyPreprocessing(layers, f"preprocessing_{cfg['DATASET']}_layer_pions.json")
else:
    voxels  = utils.ApplyPreprocessing(voxels, f"preprocessing_{cfg['DATASET']}_voxel.json")
    layers  = utils.ApplyPreprocessing(layers, f"preprocessing_{cfg['DATASET']}_layer.json")

N        = voxels.shape[0]
num_cond = energies.shape[1]
num_layer= layers.shape[1]

ds_vox   = tf.data.Dataset.from_tensor_slices(voxels)
ds_layer = tf.data.Dataset.from_tensor_slices(layers)
ds_e     = tf.data.Dataset.from_tensor_slices(energies)
dataset  = tf.data.Dataset.zip((ds_vox, ds_layer, ds_e))

train_ds, val_ds = utils.split_data(dataset, N, args.frac)
BATCH  = cfg["BATCH"]
train_ds = train_ds.batch(BATCH).prefetch(tf.data.AUTOTUNE)
val_ds   = val_ds.batch(BATCH).prefetch(tf.data.AUTOTUNE)

# ──────────────── 3. Construye modelo ───────────────────
suffix = "_teacher" if not args.distill else "_student"
ckpt_dir = f"checkpoints/{cfg['CHECKPOINT_NAME']}_{args.model}{suffix}"

os.makedirs(ckpt_dir, exist_ok=True)

if args.model.lower() == "wgan":
    model = WGAN(cfg["SHAPE"][1:], num_cond, cfg, num_noise=cfg.get("NOISE_DIM",128))
    opt_g = tf.optimizers.RMSprop(cfg["LR"])
    opt_d = tf.optimizers.RMSprop(cfg["LR"])
    model.compile(d_optimizer=opt_d, g_optimizer=opt_g)

else:                                  # CaloScore / distill
    model = CaloScore(num_layer=num_layer, config=cfg)
    if args.distill:
        teacher = CaloScore(num_layer=num_layer, config=cfg)
        if args.load:
            teacher_ckpt = os.path.join(ckpt_dir, "checkpoint")
            teacher.load_weights(teacher_ckpt).expect_partial()
        model = CaloScore_distill(teacher.ema_layer, teacher.ema_voxel,
                                  factor=args.factor, num_layer=num_layer, config=cfg)
    elif args.load:
        model.load_weights(os.path.join(ckpt_dir, "checkpoint")).expect_partial()

    steps = int(N * args.frac / BATCH)
    lr_sched = tf.keras.experimental.CosineDecay(cfg["LR"], cfg["MAXEPOCH"] * steps)
    opt      = tf.keras.optimizers.Adamax(lr_sched)
    model.compile(optimizer=opt)
steps_per_epoch  = math.ceil(int(N * args.frac) / BATCH)
validation_steps = math.ceil(int(N * (1 - args.frac)) / BATCH)
# ──────────────── 4. Callbacks & entrenamiento ───────────
callbacks = [
    EarlyStopping(patience=cfg["EARLYSTOP"], restore_best_weights=True),
    ModelCheckpoint(os.path.join(ckpt_dir, "checkpoint"),
                    save_best_only=True, save_weights_only=True)
]

with tf.device('/GPU:0'):
    history = model.fit(
        train_ds,
        validation_data = val_ds,
        epochs          = cfg["MAXEPOCH"],
        steps_per_epoch = steps_per_epoch,
        validation_steps= validation_steps,
        verbose         = 1,
        callbacks       = callbacks
    )
# ────────────────── 6. Guarda configuración ──────────────
# ──────────────── 5. Guarda resultados ──────────────────
model.save(os.path.join(ckpt_dir, "caloscore_model.keras"))  # incluye todo

