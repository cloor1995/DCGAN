import tensorflow as tf


def ReLU():
    return tf.keras.layers.ReLU()


def leaky_ReLU(leak=0.2):
    return tf.keras.layers.LeakyReLU(alpha=leak)


def batch_norm(momentum, epsilon, axis=-1):
    return tf.keras.layers.BatchNormalization(momentum=momentum,
                                              epsilon=epsilon,
                                              axis=axis)


def flatten():
    return tf.keras.layers.Flatten()


def linear(units, w_init, b_init, use_bias=False):
    return tf.keras.layers.Dense(units=units,
                                 kernel_initializer=w_init,
                                 use_bias=use_bias,
                                 bias_initializer=b_init)


def conv_layer(filters, kernel_size, strides, w_init, b_init, use_bias=False):
    return tf.keras.layers.Conv2D(filters=filters,
                                  kernel_size=kernel_size,
                                  strides=strides,
                                  padding='same',
                                  data_format='channels_last',
                                  kernel_initializer=w_init,
                                  use_bias=use_bias,
                                  bias_initializer=b_init)


def transpose_conv_layer(filters, kernel_size, strides, w_init, b_init, use_bias=False):
    return tf.keras.layers.Conv2DTranspose(filters=filters,
                                           kernel_size=kernel_size,
                                           strides=strides,
                                           padding='same',
                                           data_format='channels_last',
                                           kernel_initializer=w_init,
                                           use_bias=use_bias,
                                           bias_initializer=b_init)


class ConvBlock(tf.keras.Model):
    def __init__(self, filters, momentum, epsilon, w_init, b_init):
        super(ConvBlock, self).__init__(name='conv_block')
        self.kernel_size = (5, 5)
        self.strides = (2, 2)
        self.filters = filters
        self.conv = conv_layer(self.filters, self.kernel_size, self.strides, w_init, b_init)
        self.batch_norm = batch_norm(momentum, epsilon)
        self.leaky_ReLU = leaky_ReLU()

    def call(self, inputs, training=False, **kwargs):
        inputs = self.conv(inputs)
        inputs = self.batch_norm(inputs, training=training)
        output = self.leaky_ReLU(inputs)
        return output

    def get_config(self):
        pass


class TransposeConvBlock(tf.keras.Model):
    def __init__(self, filters, momentum, epsilon, w_init, b_init):
        super(TransposeConvBlock, self).__init__(name='transpose_conv_block')
        self.kernel_size = (5, 5)
        self.strides = (2, 2)
        self.filters = filters
        self.transpose_conv = transpose_conv_layer(self.filters, self.kernel_size, self.strides, w_init, b_init)
        self.batch_norm = batch_norm(momentum, epsilon)
        self.ReLU = ReLU()

    def call(self, inputs, training=False, **kwargs):
        inputs = self.transpose_conv(inputs)
        inputs = self.batch_norm(inputs, training=training)
        output = self.ReLU(inputs)
        return output

    def get_config(self):
        pass


class Generator(tf.keras.Model):
    """Class of the Generator in the DCGAN.

    Attributes:
      gen_filters
      img_shape
      projection_factor
      linear
      batch_norm
      ReLU
      transpose_conv_blocks
      transpose_conv
    """
    def __init__(self, gen_filters, projection_factor, img_shape, momentum, epsilon, w_init, b_init):
        super(Generator, self).__init__(name='Generator')
        self.gen_filters = gen_filters
        self.img_shape = img_shape
        self.projection_factor = projection_factor
        self.linear = linear(self.projection_factor * self.projection_factor * self.gen_filters[0], w_init, b_init)
        self.batch_norm = batch_norm(momentum, epsilon)
        self.ReLU = ReLU()
        self.transpose_conv_blocks = [TransposeConvBlock(self.gen_filters[i], momentum, epsilon, w_init, b_init) for i
                                      in
                                      range(1, len(self.gen_filters), 1)]
        self.transpose_conv = transpose_conv_layer(self.img_shape[-1], (5, 5), (2, 2), w_init, b_init,
                                                   use_bias=True)

    def call(self, inputs, training=False, **kwargs):
        inputs = self.linear(inputs)
        inputs = tf.reshape(inputs, [-1, self.projection_factor, self.projection_factor, self.gen_filters[0]])
        inputs = self.batch_norm(inputs, training=training)
        inputs = self.ReLU(inputs)
        for transpose_conv_block in self.transpose_conv_blocks:
            inputs = transpose_conv_block(inputs, training=training)
        inputs = self.transpose_conv(inputs)
        output = tf.nn.tanh(inputs)
        return output

    def get_config(self):
        pass


class Discriminator(tf.keras.Model):
    """Class of the Discriminator in the DCGAN.

    Attributes:
      disc_filters
      img_shape
      conv
      leaky_ReLU
      conv_blocks
      flatten
      linear
    """
    def __init__(self, disc_filters, img_shape, momentum, epsilon, w_init, b_init):
        super(Discriminator, self).__init__(name='Discriminator')
        self.disc_filters = disc_filters
        self.img_shape = img_shape
        self.conv = conv_layer(self.disc_filters[0], (5, 5), (2, 2), w_init, b_init, use_bias=True)
        self.leaky_ReLU = leaky_ReLU()
        self.conv_blocks = [ConvBlock(self.disc_filters[i], momentum, epsilon, w_init, b_init) for i in
                            range(1, len(self.disc_filters), 1)]
        self.flatten = flatten()
        self.linear = linear(1, w_init, b_init, use_bias=True)

    def call(self, inputs, training=False, **kwargs):
        inputs = self.conv(inputs)
        inputs = self.leaky_ReLU(inputs)
        for conv_block in self.conv_blocks:
            inputs = conv_block(inputs, training=training)
        inputs = self.flatten(inputs)
        output = self.linear(inputs)
        return output

    def get_config(self):
        pass


class DCGAN:
    """Class of the DCGAN using the Generator and Discriminator.

    Attributes:
      device
      strategy
      z_dim
      global_batchsize
      gen_model
      disc_model
      gen_optimizer
      disc_optimizer
      train_writer
      ckpt
      ckpt_manager
      ckpt_special
      ckpt_manager_special
      global_step
    """
    def __init__(self, device, latent_dim, global_batchsize, projection_factor, gen_filters, disc_filters, gen_lr,
                 disc_lr, beta1, beta2,
                 momentum, epsilon, img_shape, dataset_values, checkpoint=False):

        self.device = device
        self.strategy = tf.distribute.OneDeviceStrategy(self.device)
        self.z_dim = latent_dim
        self.global_batchsize = global_batchsize

        w_init = tf.initializers.TruncatedNormal(stddev=0.02)  # kernel_initializer
        b_init = tf.constant_initializer(0.0)  # bias initializer

        self.gen_model = Generator(gen_filters, projection_factor, img_shape, momentum, epsilon, w_init, b_init)
        self.disc_model = Discriminator(disc_filters, img_shape, momentum, epsilon, w_init, b_init)
        self.gen_optimizer = tf.keras.optimizers.Adam(learning_rate=gen_lr, beta_1=beta1,
                                                      beta_2=beta2)
        self.disc_optimizer = tf.keras.optimizers.Adam(learning_rate=disc_lr, beta_1=beta1,
                                                       beta_2=beta2)
        self.train_writer = tf.summary.create_file_writer(dataset_values['dir_events'] + 'train')

        self.ckpt = tf.train.Checkpoint(step=tf.Variable(0),
                                        generator_optimizer=self.gen_optimizer,
                                        generator_model=self.gen_model,
                                        discriminator_optimizer=self.disc_optimizer,
                                        discriminator_model=self.disc_model)

        self.ckpt_manager = tf.train.CheckpointManager(self.ckpt, dataset_values['dir_checkpoints'], max_to_keep=3)

        self.ckpt_special = tf.train.Checkpoint(step=tf.Variable(0),
                                                generator_optimizer=self.gen_optimizer,
                                                generator_model=self.gen_model,
                                                discriminator_optimizer=self.disc_optimizer,
                                                discriminator_model=self.disc_model)
        self.ckpt_manager_special = tf.train.CheckpointManager(self.ckpt_special,
                                                               dataset_values['dir_checkpoints'] + 'special/',
                                                               max_to_keep=99)

        self.global_step = 0

        if checkpoint:
            latest_ckpt = tf.train.latest_checkpoint(dataset_values['dir_checkpoints'])
            if not latest_ckpt:
                raise Exception('No checkpoints found.')
            self.ckpt.restore(latest_ckpt)
            self.global_step = int(
                latest_ckpt.split('-')[-1])  # .../ckpt-300 returns 300 previously trained totalbatches
            print("Checkpoint of epoch {} restored for {}-dataset.".format(self.global_step, dataset_values['name']))

    def set_global_step(self, value=0, reset=False):
        if reset:
            self.global_step = 0

        self.global_step += value

    @staticmethod
    def compute_loss(labels, predictions):

        cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True,
                                                           reduction=tf.keras.losses.Reduction.NONE)
        return cross_entropy(labels, predictions)

    def disc_loss(self, real_output, fake_output):

        real_loss = self.compute_loss(tf.ones_like(real_output), real_output)
        fake_loss = self.compute_loss(tf.zeros_like(fake_output), fake_output)
        total_loss = real_loss + fake_loss
        total_loss = total_loss / self.global_batchsize
        return total_loss

    def gen_loss(self, fake_output):

        gen_loss = self.compute_loss(tf.ones_like(fake_output), fake_output)
        gen_loss = gen_loss / self.global_batchsize
        return gen_loss

    def train_step(self, real_imgs):

        noise = tf.random.normal(shape=[tf.shape(real_imgs)[0], self.z_dim])

        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            generated_imgs = self.gen_model(noise, training=False)
            real_output = self.disc_model(real_imgs, training=True)
            fake_output = self.disc_model(generated_imgs, training=True)
            d_loss = self.disc_loss(real_output, fake_output)
            g_loss = self.gen_loss(fake_output)

        g_grads = gen_tape.gradient(g_loss, self.gen_model.trainable_variables)
        d_grads = disc_tape.gradient(d_loss, self.disc_model.trainable_variables)

        self.gen_optimizer.apply_gradients(zip(g_grads, self.gen_model.trainable_variables))
        self.disc_optimizer.apply_gradients(zip(d_grads, self.disc_model.trainable_variables))

        with tf.GradientTape() as gen_tape:
            generated_imgs = self.gen_model(noise, training=False)
            fake_output = self.disc_model(generated_imgs, training=True)
            g_loss = self.gen_loss(fake_output)

        g_grads = gen_tape.gradient(g_loss, self.gen_model.trainable_variables)
        self.gen_optimizer.apply_gradients(zip(g_grads, self.gen_model.trainable_variables))

        return g_loss, d_loss

    def gen_step(self, random_latents):
        gen_imgs = self.gen_model(random_latents, training=False)
        return gen_imgs

    @tf.function
    def distribute_train_step(self, dist_dataset):
        per_replica_g_losses, per_replica_d_losses = self.strategy.run(self.train_step, args=(dist_dataset,))
        total_g_loss = self.strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_g_losses, axis=0)
        total_d_loss = self.strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_d_losses, axis=0)

        return total_g_loss, total_d_loss

    @tf.function
    def distribute_gen_step(self, dist_gen_noise):
        per_replica_genimgs = self.strategy.run(self.gen_step, args=(dist_gen_noise,))
        gen_imgs = self.strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_genimgs,
                                        axis=None)
        return gen_imgs
