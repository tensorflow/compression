# Copyright 2020 Google LLC. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""HiFiC model code."""
import collections
import glob
import itertools

from compare_gan.gans import loss_lib as compare_gan_loss_lib
import tensorflow.compat.v1 as tf
import tensorflow_datasets as tfds

from hific import archs
from hific import helpers
from hific.helpers import ModelMode
from hific.helpers import ModelType
from hific.helpers import TFDSArguments


# How many dataset preprocessing processes to use.
DATASET_NUM_PARALLEL = 8

# How many batches to prefetch.
DATASET_PREFETCH_BUFFER = 20

# How many batches to fetch for shuffling.
DATASET_SHUFFLE_BUFFER = 10

BppPair = collections.namedtuple(
    "BppPair", ["total_nbpp", "total_qbpp"])


Nodes = collections.namedtuple(
    "Nodes",                    # Expected ranges for RGB:
    ["input_image",             # [0, 255]
     "input_image_scaled",      # [0, 1]
     "reconstruction",          # [0, 255]
     "reconstruction_scaled",   # [0, 1]
     "latent_quantized"])       # Latent post-quantization.


class _LossScaler(object):
  """Helper class to manage losses."""

  def __init__(self, hific_config, ignore_schedules: bool):
    # Set to true by model if training or validation.
    self._ignore_schedules = ignore_schedules
    self._config = hific_config

  def get_rd_loss(self, distortion_loss, bpp_pair: BppPair, step):
    """Get R, D part of loss."""
    loss_config = self._config.loss_config

    weighted_distortion_loss = self._get_weighted_distortion_loss(
        loss_config, distortion_loss)
    weighted_rate = self._get_weighted_rate_loss(
        loss_config, bpp_pair, step)

    tf.summary.scalar("components/weighted_R", weighted_rate)
    tf.summary.scalar("components/weighted_D", weighted_distortion_loss)

    return weighted_rate + weighted_distortion_loss

  def _get_weighted_distortion_loss(self, loss_config, distortion_loss):
    """Get weighted D."""
    return distortion_loss * loss_config.CD * loss_config.C

  def _get_weighted_rate_loss(self, loss_config, bpp_pair, step):
    """Get weighted R."""
    total_nbpp, total_qbpp = bpp_pair
    lmbda_a = self._get_scheduled_param(
        loss_config.lmbda_a, self._config.lambda_schedule, step, "lmbda_a")
    # For a target rate R_target, implement constrained optimization:
    # { 1/lambda_a * R    if R > R_target
    #   1/lambda_b * R    else
    # We assume lambda_a < lambda_b, and thus 1/lambda_a > 1/lambda_b,
    # i.e., if the rate R is too large, we want a larger factor on it.
    if loss_config.lmbda_a >= loss_config.lmbda_b:
      raise ValueError("Expected lmbda_a < lmbda_b, got {} >= {}".format(
          loss_config.lmbda_a, loss_config.lmbda_b))

    target_bpp = self._get_scheduled_param(
        loss_config.target, loss_config.target_schedule, step, "target_bpp")
    lmbda_b = self._get_scheduled_param(
        loss_config.lmbda_b, self._config.lambda_schedule, step, "lmbda_b")
    lmbda_inv = tf.where(total_qbpp > target_bpp,
                         1 / lmbda_a,
                         1 / lmbda_b)

    tf.summary.scalar("lmbda_inv", lmbda_inv)
    return lmbda_inv * total_nbpp * loss_config.C

  def get_scaled_g_loss(self, g_loss):
    """Get scaled version of GAN loss."""
    with tf.name_scope("scaled_g_loss"):
      return g_loss * self._config.loss_config.CP

  def _get_scheduled_param(self, param, param_schedule, global_step, name):
    if (not self._ignore_schedules and param_schedule.vals
        and any(step > 0 for step in param_schedule.steps)):
      param = _scheduled_value(param, param_schedule, global_step, name)
      tf.summary.scalar(name, param)
    return param


def _pad(input_image, image_shape, factor):
  """Pad `input_image` such that H and W are divisible by `factor`."""
  with tf.name_scope("pad"):
    height, width = image_shape[0], image_shape[1]
    pad_height = (factor - (height % factor)) % factor
    pad_width = (factor - (width % factor)) % factor
    return tf.pad(input_image,
                  [[0, 0], [0, pad_height], [0, pad_width], [0, 0]],
                  "REFLECT")


class HiFiC(object):
  """HiFiC Model class."""

  def __init__(self,
               config,
               mode: ModelMode,
               lpips_weight_path=None,
               auto_encoder_ckpt_dir=None,
               create_image_summaries=True):
    """Instantiate model.

    Args:
      config: A config, see configs.py
      mode: Model mode.
      lpips_weight_path: path to where LPIPS weights are stored or should be
        stored. See helpers.ensure_lpips_weights_exist.
      auto_encoder_ckpt_dir: If given, instantiate auto-encoder and probability
        model from latest checkpoint in this folder.
      create_image_summaries: Whether to create image summaries. Turn off to
        save disk space.
    """
    self._mode = mode
    self._config = config
    self._model_type = config.model_type
    self._create_image_summaries = create_image_summaries

    if not isinstance(self._model_type, ModelType):
      raise ValueError("Invalid model_type: [{}]".format(
          self._config.model_type))

    self._auto_encoder_ckpt_path = None
    self._auto_encoder_savers = None
    if auto_encoder_ckpt_dir:
      latest_ckpt = tf.train.latest_checkpoint(auto_encoder_ckpt_dir)
      if not latest_ckpt:
        raise ValueError(f"Did not find checkpoint in {auto_encoder_ckpt_dir}!")
      self._auto_encoder_ckpt_path = latest_ckpt

    if self.training and not lpips_weight_path:
      lpips_weight_path = "lpips_weight__net-lin_alex_v0.1.pb"

    self._lpips_weight_path = lpips_weight_path
    self._transform_layers = []
    self._entropy_layers = []
    self._layers = None
    self._encoder = None
    self._decoder = None
    self._discriminator = None
    self._gan_loss_function = None
    self._lpips_loss_weight = None
    self._lpips_loss = None
    self._entropy_model = None
    self._optimize_entropy_vars = True
    self._global_step_disc = None  # global_step used for D training

    self._setup_discriminator = (
        self._model_type == ModelType.COMPRESSION_GAN
        and (self.training or self.validation))  # No disc for evaluation.

    if self._setup_discriminator:
      self._num_steps_disc = self._config.num_steps_disc
      if self._num_steps_disc == 0:
        raise ValueError("model_type=={} but num_steps_disc == 0.".format(
            self._model_type))
      tf.logging.info(
          "GAN Training enabled. Training discriminator for {} steps.".format(
              self._num_steps_disc))
    else:
      self._num_steps_disc = 0

    self.input_spec = {
        "input_image":
            tf.keras.layers.InputSpec(
                dtype=tf.uint8,
                shape=(None, None, None, 3))}

    if self._setup_discriminator:
      # This is an optional argument to build_model. If training a
      # discriminator, this is expected to contain multiple sub-batches.
      # See build_input for details.
      self.input_spec["input_images_d_steps"] = tf.keras.layers.InputSpec(
          dtype=tf.uint8,
          shape=(None, None, None, 3))

      self._gan_loss_function = compare_gan_loss_lib.non_saturating

    self._loss_scaler = _LossScaler(
        self._config,
        ignore_schedules=not self.training and not self.validation)

    self._train_op = None
    self._hooks = []

  @property
  def training(self):
    """True if in training mode."""
    return self._mode == ModelMode.TRAINING

  @property
  def validation(self):
    """True if in validation mode."""
    return self._mode == ModelMode.VALIDATION

  @property
  def evaluation(self):
    """True if in evaluation mode."""
    return self._mode == ModelMode.EVALUATION

  @property
  def train_op(self):
    return self._train_op

  @property
  def hooks(self):
    return self._hooks

  def _add_hook(self, hook):
    self._hooks.append(hook)

  @property
  def num_steps_disc(self):
    return self._num_steps_disc

  def build_input(self,
                  batch_size,
                  crop_size,
                  images_glob=None,
                  tfds_arguments: TFDSArguments = None):
    """Build input dataset."""
    if not (images_glob or tfds_arguments):
      raise ValueError("Need images_glob or tfds_arguments!")

    if self._setup_discriminator:
      # Unroll dataset for GAN training. If we unroll for N steps,
      # we want to fetch (N+1) batches for every step, where 1 batch
      # will be used for G training, and the remaining N batches for D training.
      batch_size *= (self._num_steps_disc + 1)

    if self._setup_discriminator:
      # Split the (N+1) batches into two arguments for build_model.
      def _batch_to_dict(batch):
        num_sub_batches = self._num_steps_disc + 1
        sub_batch_size = batch_size // num_sub_batches
        splits = [sub_batch_size, sub_batch_size * self._num_steps_disc]
        input_image, input_images_d_steps = tf.split(batch, splits)
        return dict(input_image=input_image,
                    input_images_d_steps=input_images_d_steps)
    else:
      def _batch_to_dict(batch):
        return dict(input_image=batch)

    dataset = self._get_dataset(batch_size, crop_size,
                                images_glob, tfds_arguments)
    return dataset.map(_batch_to_dict)

  def _get_dataset(self, batch_size, crop_size,
                   images_glob, tfds_arguments: TFDSArguments):
    """Build TFDS dataset.

    Args:
      batch_size: int, batch_size.
      crop_size: int, will random crop to this (crop_size, crop_size)
      images_glob:
      tfds_arguments: argument for TFDS.

    Returns:
      Instance of tf.data.Dataset.
    """

    crop_size_float = tf.constant(crop_size, tf.float32) if crop_size else None
    smallest_fac = tf.constant(0.75, tf.float32)
    biggest_fac = tf.constant(0.95, tf.float32)

    with tf.name_scope("tfds"):
      if images_glob:
        images = sorted(glob.glob(images_glob))
        tf.logging.info(
            f"Using images_glob={images_glob} ({len(images)} images)")
        filenames = tf.data.Dataset.from_tensor_slices(images)
        dataset = filenames.map(lambda x: tf.image.decode_png(tf.read_file(x)))
      else:
        tf.logging.info(f"Using TFDS={tfds_arguments}")
        builder = tfds.builder(
            tfds_arguments.dataset_name, data_dir=tfds_arguments.downloads_dir)
        builder.download_and_prepare()
        split = "train" if self.training else "validation"
        dataset = builder.as_dataset(split=split)

      def _preprocess(features):
        if images_glob:
          image = features
        else:
          image = features[tfds_arguments.features_key]
        if not crop_size:
          return image
        tf.logging.info("Scaling down %s and cropping to %d x %d", image,
                        crop_size, crop_size)
        with tf.name_scope("random_scale"):
          # Scale down by at least `biggest_fac` and at most `smallest_fac` to
          # remove JPG artifacts. This code also handles images that have one
          # side  shorter than crop_size. In this case, we always upscale such
          # that this side becomes the same as `crop_size`. Overall, images
          # returned will never be smaller than `crop_size`.
          image_shape = tf.cast(tf.shape(image), tf.float32)
          height, width = image_shape[0], image_shape[1]
          smallest_side = tf.math.minimum(height, width)
          # The smallest factor such that the downscaled image is still bigger
          # than `crop_size`. Will be bigger than 1 for images smaller than
          # `crop_size`.
          image_smallest_fac = crop_size_float / smallest_side
          min_fac = tf.math.maximum(smallest_fac, image_smallest_fac)
          max_fac = tf.math.maximum(min_fac, biggest_fac)
          scale = tf.random_uniform([],
                                    minval=min_fac,
                                    maxval=max_fac,
                                    dtype=tf.float32,
                                    seed=42,
                                    name=None)
          image = tf.image.resize_images(
              image, [tf.ceil(scale * height),
                      tf.ceil(scale * width)])
        with tf.name_scope("random_crop"):
          image = tf.image.random_crop(image, [crop_size, crop_size, 3])
        return image

      dataset = dataset.map(
          _preprocess, num_parallel_calls=DATASET_NUM_PARALLEL)
      dataset = dataset.batch(batch_size, drop_remainder=True)

      if not self.evaluation:
        # Make sure we don't run out of data
        dataset = dataset.repeat()
        dataset = dataset.shuffle(buffer_size=DATASET_SHUFFLE_BUFFER)
      dataset = dataset.prefetch(buffer_size=DATASET_PREFETCH_BUFFER)

      return dataset

  def build_model(self, input_image, input_images_d_steps=None):
    """Build model and losses and train_ops.

    Args:
      input_image: A single (B, H, W, C) image, in [0, 255]
      input_images_d_steps: If training a discriminator, this is expected to
        be a (B*N, H, W, C) stack of images, where N=number of sub batches.
        See build_input.

    Returns:
      output_image and bitstrings if self.evaluation else None.
    """
    if input_images_d_steps is None:
      input_images_d_steps = []
    else:
      input_images_d_steps.set_shape(
          self.input_spec["input_images_d_steps"].shape)
      input_images_d_steps = tf.split(input_images_d_steps, self.num_steps_disc)

    if self.evaluation and input_images_d_steps:
      raise ValueError("Only need input_image for eval! {}".format(
          input_images_d_steps))

    input_image.set_shape(self.input_spec["input_image"].shape)

    self.build_transforms()

    if self.training:
      self._lpips_loss = LPIPSLoss(self._lpips_weight_path)
      self._lpips_loss_weight = self._config.loss_config.lpips_weight

    if self._setup_discriminator:
      self.build_discriminator()

    # Global step needs to be created for train, val and eval.
    global_step = tf.train.get_or_create_global_step()

    # Compute output graph.
    nodes_gen, bpp_pair, bitstrings = \
      self._compute_compression_graph(input_image)

    if self.evaluation:
      tf.logging.info("Evaluation mode: build_model done.")
      reconstruction = tf.clip_by_value(nodes_gen.reconstruction, 0, 255.)
      return reconstruction, bitstrings

    nodes_disc = []  # list of Nodes, one for every sub-batch of disc
    for i, sub_batch in enumerate(input_images_d_steps):
      with tf.name_scope("sub_batch_disc_{}".format(i)):
        nodes, _, _ = self._compute_compression_graph(
            sub_batch, create_summaries=False)
        nodes_disc.append(nodes)

    if self._auto_encoder_ckpt_path:
      self._prepare_auto_encoder_restore()

    # The following is inspired by compare_gan/gans/modular_gan.py:
    # Let's say we want to train the discriminator for D steps for every 1 step
    # of generator training. We do the unroll_graph=True options:
    # The features given to the model_fn are split into
    # D + 1 sub-batches. The code then creates D train_ops for the
    # discriminator, each feeding a different sub-batch of features
    # into the discriminator.
    # The train_op for the generator then depends on all these D train_ops
    # and uses the last (D+1 th) sub-batch.
    # Note that the graph is only created once.

    d_train_ops = []
    if self._setup_discriminator:
      tf.logging.info("Unrolling graph for discriminator")
      self._global_step_disc = tf.get_variable(
          "global_step_disc", [], dtype=global_step.dtype, trainable=False)
      with tf.name_scope("steps"):
        tf.summary.scalar("global_step", global_step)
        tf.summary.scalar("global_step_disc", self._global_step_disc)

      # Create optimizer once, and then call minimize on it multiple times
      # within self._train_discriminator.
      disc_optimizer = self._make_discriminator_optimizer(
          self._global_step_disc)
      for i, nodes in enumerate(nodes_disc):
        with tf.name_scope("train_disc_{}".format(i + 1)):
          with tf.control_dependencies(d_train_ops):
            d_train_ops.append(
                self._train_discriminator(
                    nodes, disc_optimizer, create_summaries=(i == 0)))

    # Depend on `d_train_ops`, which ensures all `self._num_steps_disc` steps of
    # the discriminator will run before the generator training op.
    with tf.control_dependencies(d_train_ops):
      train_op = self._train_generator(nodes_gen, bpp_pair, global_step)

    if self.training:
      self._train_op = train_op

  def prepare_for_arithmetic_coding(self, sess):
    """Run's the update op of the EntropyBottleneck."""
    update_op = self._entropy_model.updates[0]
    sess.run(update_op)

  def restore_trained_model(self, sess, ckpt_dir):
    """Restore a trained model for evaluation."""
    saver = tf.train.Saver()
    latest_ckpt = tf.train.latest_checkpoint(ckpt_dir)
    tf.logging.info("Restoring %s...", latest_ckpt)
    saver.restore(sess, latest_ckpt)

  def restore_autoencoder(self, sess):
    """Restore encoder, decoder and probability model from checkpoint."""
    assert self._auto_encoder_savers
    for saver in self._auto_encoder_savers:
      tf.logging.info("Restoring %s...", saver)
      saver.restore(sess, self._auto_encoder_ckpt_path)

  def _prepare_auto_encoder_restore(self):
    """Prepare the savers needed to restore encoder, decoder, entropy_model."""
    assert self._auto_encoder_savers is None
    self._auto_encoder_savers = []
    for name, layer in [
        ("entropy_model", self._entropy_model),
        ("encoder", self._encoder),
        ("decoder", self._decoder)]:
      self._auto_encoder_savers.append(
          tf.train.Saver(layer.variables, name=f"restore_{name}"))

  def build_transforms(self):
    """Instantiates all transforms used by this model."""
    self._encoder = archs.Encoder()
    self._decoder = archs.Decoder()
    self._transform_layers.append(self._encoder)
    self._transform_layers.append(self._decoder)

    self._entropy_model = archs.Hyperprior()
    self._transform_layers.extend(self._entropy_model.transform_layers)
    self._entropy_layers.extend(self._entropy_model.entropy_layers)

    self._layers = self._transform_layers + self._entropy_layers

  def build_discriminator(self):
    """Instantiates discriminator."""
    self._discriminator = archs.Discriminator()

  def _compute_compression_graph(self, input_image, create_summaries=True):
    """Compute a forward pass through encoder and decoder.

    Args:
      input_image: Input image, range [0, 255]
      create_summaries: Whether to create summaries

    Returns:
      tuple Nodes, BppPair
    """
    with tf.name_scope("image_shape"):
      image_shape = tf.shape(input_image)[1:-1]  # Get H, W.

    if self.evaluation:
      num_downscaling = self._encoder.num_downsampling_layers
      factor = 2 ** num_downscaling
      tf.logging.info("Padding to {}".format(factor))
      input_image = _pad(input_image, image_shape, factor)

    with tf.name_scope("scale_down"):
      input_image_scaled = \
          tf.cast(input_image, tf.float32) / 255.

    info = self._get_encoder_out(input_image_scaled, image_shape)
    decoder_in = info.decoded
    total_nbpp = info.total_nbpp
    total_qbpp = info.total_qbpp
    bitstream_tensors = info.bitstream_tensors

    reconstruction, reconstruction_scaled = \
        self._compute_reconstruction(
            decoder_in, image_shape, input_image_scaled.shape)

    if create_summaries and self._create_image_summaries:
      tf.summary.image(
          "input_image", tf.saturate_cast(input_image, tf.uint8), max_outputs=1)
      tf.summary.image(
          "reconstruction",
          tf.saturate_cast(reconstruction, tf.uint8),
          max_outputs=1)

    nodes = Nodes(input_image, input_image_scaled,
                  reconstruction, reconstruction_scaled,
                  latent_quantized=decoder_in)
    return nodes, BppPair(total_nbpp, total_qbpp), bitstream_tensors

  def _get_encoder_out(self,
                       input_image_scaled,
                       image_shape) -> archs.HyperInfo:
    """Compute encoder transform."""
    encoder_out = self._encoder(input_image_scaled, training=self.training)
    return self._entropy_model(encoder_out,
                               image_shape=image_shape,
                               mode=self._mode)

  def _compute_reconstruction(self, decoder_in, image_shape, output_shape):
    """Compute pass through decoder.

    Args:
      decoder_in: Input to decoder transform.
      image_shape: Tuple (height, width) of the image_shape
      output_shape: Desired output shape.

    Returns:
      Tuple (reconstruction (in [0, 255],
             reconstruction_scaled (in [0, 1]),
             residual_scaled (in [-1, 1]) if it exists else None).
    """
    reconstruction_scaled = self._decoder(
        decoder_in, training=self.training)

    with tf.name_scope("undo_padding"):
      height, width = image_shape[0], image_shape[1]
      reconstruction_scaled = reconstruction_scaled[:, :height, :width, :]

    reconstruction_scaled.set_shape(output_shape)
    with tf.name_scope("re_scale"):
      reconstruction = reconstruction_scaled * 255.

    return reconstruction, reconstruction_scaled

  def _create_rd_loss(self, nodes: Nodes, bpp_pair: BppPair, step):
    """Computes noisy/quantized rd-loss and creates summaries."""

    with tf.name_scope("loss"):
      distortion_loss = self._compute_distortion_loss(nodes)
      rd_loss = self._loss_scaler.get_rd_loss(distortion_loss, bpp_pair, step)

      tf.summary.scalar("distortion_loss", distortion_loss)
      tf.summary.scalar("rd_loss", rd_loss)

      return rd_loss

  def _compute_distortion_loss(self, nodes: Nodes):
    input_image, reconstruction = nodes.input_image, nodes.reconstruction
    with tf.name_scope("distortion"):
      input_image = tf.cast(input_image, tf.float32)
      reconstruction = tf.cast(reconstruction, tf.float32)
      sq_err = tf.math.squared_difference(input_image, reconstruction)
      distortion_loss = tf.reduce_mean(sq_err)
      return distortion_loss

  def _compute_perceptual_loss(self, nodes: Nodes):
    input_image_scaled = nodes.input_image_scaled
    reconstruction_scaled = nodes.reconstruction_scaled
    # First the fake images, then the real! Otherwise no gradients.
    return self._lpips_loss(reconstruction_scaled,
                            input_image_scaled)

  def _create_gan_loss(self,
                       d_out: archs.DiscOutSplit,
                       create_summaries=True,
                       mode="g_loss"):
    """Create GAN loss using compare_gan."""
    if mode not in ("g_loss", "d_loss"):
      raise ValueError("Invalid mode: {}".format(mode))
    assert self._gan_loss_function is not None

    # Called within either train_disc or train_gen namescope.
    with tf.name_scope("gan_loss"):
      d_loss, _, _, g_loss = compare_gan_loss_lib.get_losses(
          # Note: some fn's need other args.
          fn=self._gan_loss_function,
          d_real=d_out.d_real,
          d_fake=d_out.d_fake,
          d_real_logits=d_out.d_real_logits,
          d_fake_logits=d_out.d_fake_logits)
      loss = d_loss if mode == "d_loss" else g_loss
      if create_summaries:
        tf.summary.scalar("d_loss", d_loss)
        tf.summary.scalar("g_loss", g_loss)

    return loss

  def _train_discriminator(self, nodes: Nodes, optimizer, create_summaries):
    """Creates a train_op for the discriminator.

    Args:
      nodes: Instance of Nodes, the nodes of the model to feed to D.
      optimizer: Discriminator optimizer. Passed in because it will be re-used
        in the different discriminator steps.
      create_summaries: If True, create summaries.

    Returns:
      A training op if training, else no_op.
    """
    d_out = self._compute_discriminator_out(
        nodes,
        create_summaries,
        gradients_to_generator=False)  # Only train discriminator!
    d_loss = self._create_gan_loss(d_out, create_summaries, mode="d_loss")

    if not self.training:
      return tf.no_op()

    self._add_hook(tf.train.NanTensorHook(d_loss))

    # Getting the variables here because they don't exist before calling
    # _compute_discriminator_out for the first time!
    disc_vars = self._discriminator.trainable_variables

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
      with tf.name_scope("min_d"):
        train_op_d = optimizer.minimize(
            d_loss, self._global_step_disc, disc_vars)
        return train_op_d

  def _train_generator(self, nodes: Nodes, bpp_pair: BppPair, step):
    """Create training op for generator.

    This also create the optimizers for the encoder/decoder and entropy
    layers.

    Args:
      nodes: The output of the model to create a R-D loss and feed to D.
      bpp_pair: Instance of BppPair.
      step: the global step of G.

    Returns:
      A training op if training, else None
    """
    rd_loss = self._create_rd_loss(nodes, bpp_pair, step)

    with tf.name_scope("train_gen"):
      if self._setup_discriminator:
        d_outs = self._compute_discriminator_out(nodes,
                                                 create_summaries=False,
                                                 gradients_to_generator=True)
        g_loss = self._create_gan_loss(d_outs, create_summaries=True,
                                       mode="g_loss")
        scaled_g_loss = self._loss_scaler.get_scaled_g_loss(g_loss)
        tf.summary.scalar("scaled_g_loss", scaled_g_loss)
        loss_enc_dec_entropy = rd_loss + scaled_g_loss

      else:
        loss_enc_dec_entropy = rd_loss

      if self._lpips_loss is not None:
        tf.logging.info("Using LPIPS...")
        perceptual_loss = self._compute_perceptual_loss(nodes)
        weighted_perceptual_loss = \
            self._lpips_loss_weight * perceptual_loss
        tf.summary.scalar("weighted_lpips",
                          weighted_perceptual_loss)
        loss_enc_dec_entropy += weighted_perceptual_loss

      tf.summary.scalar("loss_enc_dec_entropy", loss_enc_dec_entropy)

      if self.training:
        self._add_hook(tf.train.NanTensorHook(loss_enc_dec_entropy))

      if self.validation:
        return None

      entropy_vars, transform_vars, _ = self._get_and_check_variables()

      # Train G.
      with tf.name_scope("min_g"):
        train_op = self._make_enc_dec_entropy_train_op(
            step, loss_enc_dec_entropy, entropy_vars, transform_vars)

      return train_op

  def _compute_discriminator_out(self,
                                 nodes: Nodes,
                                 create_summaries,
                                 gradients_to_generator=True
                                ) -> archs.DiscOutSplit:
    """Get discriminator outputs."""
    with tf.name_scope("disc"):
      input_image = nodes.input_image_scaled
      reconstruction = nodes.reconstruction_scaled

      if not gradients_to_generator:
        reconstruction = tf.stop_gradient(reconstruction)

      discriminator_in = tf.concat([input_image, reconstruction], axis=0)

      # Condition D.
      latent = tf.stop_gradient(nodes.latent_quantized)
      latent = tf.concat([latent, latent], axis=0)

      discriminator_in = (discriminator_in, latent)

      disc_out_all = self._discriminator(discriminator_in,
                                         training=self.training)

    d_real, d_fake = tf.split(disc_out_all.d_all, 2)
    d_real_logits, d_fake_logits = tf.split(disc_out_all.d_all_logits, 2)
    disc_out_split = archs.DiscOutSplit(d_real, d_fake,
                                        d_real_logits, d_fake_logits)

    if create_summaries:
      tf.summary.scalar("d_real", tf.reduce_mean(disc_out_split.d_real))
      tf.summary.scalar("d_fake", tf.reduce_mean(disc_out_split.d_fake))

    return disc_out_split

  def _get_and_check_variables(self):
    """Make sure we train the right variables."""
    entropy_vars = list(
        itertools.chain.from_iterable(
            x.trainable_variables for x in self._entropy_layers))
    transform_vars = list(
        itertools.chain.from_iterable(x.trainable_variables
                                      for x in self._transform_layers))
    # Just getting these for book-keeping
    transform_vars_non_trainable = list(
        itertools.chain.from_iterable(x.variables
                                      for x in self._transform_layers))
    disc_vars = (self._discriminator.trainable_variables
                 if self._setup_discriminator
                 else [])

    # Check that we didn't miss any variables.
    all_trainable = set(tf.trainable_variables())
    all_known = set(transform_vars + entropy_vars + disc_vars)
    if ((all_trainable != all_known) and
        all_trainable != set(transform_vars_non_trainable) | all_known):
      all_known |= set(transform_vars_non_trainable)
      missing_in_trainable = all_known - all_trainable
      missing_in_known = all_trainable - all_known
      non_trainable_vars_str = \
          "\n".join(sorted(v.name for v in transform_vars_non_trainable))
      raise ValueError("Did not capture all variables! " +
                       " Missing in trainable: " + str(missing_in_trainable) +
                       " Missing in known: " + str(missing_in_known) +
                       " \n\nNon trainable transform vars: " +
                       non_trainable_vars_str)

    return entropy_vars, transform_vars, disc_vars

  def _make_enc_dec_entropy_train_op(self,
                                     step,
                                     loss,
                                     entropy_vars,
                                     transform_vars):
    """Create optimizers for encoder/decoder and entropy model."""
    minimize_ops = []
    assert len(self._entropy_model.losses) == 1
    for i, (name, vs, l) in enumerate(
        [("transform", transform_vars, loss),
         ("entropy", entropy_vars, loss),
         ("aux", entropy_vars, self._entropy_model.losses[0])
         ]):
      optimizer = tf.train.AdamOptimizer(
          learning_rate=_scheduled_value(
              self._config.lr,
              self._config.lr_schedule,
              step,
              "lr_" + name,
              summary=True),
          name="adam_" + name)
      minimize = optimizer.minimize(
          l, var_list=vs,
          global_step=step if i == 0 else None)  # Only update step once.
      minimize_ops.append(minimize)
    return tf.group(minimize_ops, name="enc_dec_ent_train_op")

  def _make_discriminator_optimizer(self, step):
    if not self.training:
      return None
    return tf.train.AdamOptimizer(
        learning_rate=_scheduled_value(
            self._config.lr,
            self._config.lr_schedule,
            step,
            "lr_disc",
            summary=True),
        name="adam_disc")


class LPIPSLoss(object):
  """Calcualte LPIPS loss."""

  def __init__(self, weight_path):
    helpers.ensure_lpips_weights_exist(weight_path)

    def wrap_frozen_graph(graph_def, inputs, outputs):
      def _imports_graph_def():
        tf.graph_util.import_graph_def(graph_def, name="")

      wrapped_import = tf.compat.v1.wrap_function(_imports_graph_def, [])
      import_graph = wrapped_import.graph
      return wrapped_import.prune(
          tf.nest.map_structure(import_graph.as_graph_element, inputs),
          tf.nest.map_structure(import_graph.as_graph_element, outputs))

    # Pack LPIPS network into a tf function
    graph_def = tf.compat.v1.GraphDef()
    with open(weight_path, "rb") as f:
      graph_def.ParseFromString(f.read())
    self._lpips_func = tf.function(
        wrap_frozen_graph(
            graph_def, inputs=("0:0", "1:0"), outputs="Reshape_10:0"))

  def __call__(self, fake_image, real_image):
    """Assuming inputs are in [0, 1]."""
    # Move inputs to [-1, 1] and NCHW format.
    def _transpose_to_nchw(x):
      return tf.transpose(x, (0, 3, 1, 2))
    fake_image = _transpose_to_nchw(fake_image * 2 - 1.0)
    real_image = _transpose_to_nchw(real_image * 2 - 1.0)
    loss = self._lpips_func(fake_image, real_image)
    return tf.reduce_mean(loss)  # Loss is N111, take mean to get scalar.


def _scheduled_value(value, schedule, step, name, summary=False):
  """Create a tensor whose value depends on global step.

  Args:
    value: The value to adapt.
    schedule: Dictionary. Expects 'steps' and 'vals'.
    step: The global_step to find to.
    name: Name of the value.
    summary: Boolean, whether to add a summary for the scheduled value.

  Returns:
    tf.Tensor.
  """
  with tf.name_scope("schedule_" + name):
    if len(schedule["steps"]) + 1 != len(schedule["vals"]):
      raise ValueError("Schedule expects one more value than steps.")
    steps = [int(s) for s in schedule["steps"]]
    steps = tf.stack(steps + [step + 1])
    idx = tf.where(step < steps)[0, 0]
    value = value * tf.convert_to_tensor(schedule["vals"])[idx]
  if summary:
    tf.summary.scalar(name, value)
  return value
