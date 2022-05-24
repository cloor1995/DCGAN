import os
import logging
import random
import threading
import time
import argparse
import matplotlib.pyplot as plt
import moviepy.video.io.ImageSequenceClip
import numpy as np
import datasets
import sys
from PIL import Image

# Hide Tensorflow Warnings (DCGAN and Tensorflow need to be imported afterwards)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # FATAL
logging.getLogger('tensorflow').setLevel(logging.FATAL)
from dcgan import DCGAN
import tensorflow as tf

# Allow growth so other programs can use the graphics card during training as well
physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

# Default parameters

# General parameters
default_mode = 'demo'  # One of train, generate, demo
default_dataset = 'celeba'  # One of celeba, celeba_hq, comics, simpsons, anime
device = '/device:GPU:0'  # Default GPU used for Tensorflow
num_images = 18  # Number of images to generate for training/generation
num_images_demo = 8  # Number of images to generate for demo
grid_size = (3, 6)  # Layout of image grid for image generation in mode training/generation
random_seed = 12  # Random Seed for image generation
default_epochs = 100  # Total number of epochs for training
pre_trained = False  # If set true, will use checkpoint to continue training
special_checkpoints = False  # If set true, will save special checkpoints that won't be deleted (max: 99)
special_checkpoints_frequency = 10  # Frequency of special checkpoints (in epochs)
generation_frequency = 1000  # Frequency of image generation (in batches)
generate_video = False  # Of set true, will create video of all generated images
threads = 8  # Number of parallel calls for TFRecords (CPU)
start = time.time()  # Time when training started

# Model parameters
momentum = 0.9
epsilon = 1e-5
latent_dim = 100  # Latent variable (usually 100)
disc_lr = 0.0002  # Learn rate of Discriminator
gen_lr = 0.0002  # Learn rate of Generator
disc_filters = [32, 64, 128, 256, 512]  # Filters of Discriminator
gen_filters = [1024, 512, 256, 128, 64]  # Filters of Generator
global_batchsize = 32  # Number of images in one batch
beta1 = 0.5  # Adam optimizer parameters
beta2 = 0.999  # Adam optimizer parameters

# Dictionaries
dataset_values = dict()
gen_losses_epoch = dict()
disc_losses_epoch = dict()
gen_losses_batch = dict()
disc_losses_batch = dict()


def train(gan, epochs):
    """Trains given GAN for chosen amount of epochs. Images and checkpoints
       will be saved on hard drive during training.

    Args:
      gan: GAN that will be trained. Meant for DCGANs,
        but might work for others as well.
      epochs: Total number of epochs given GAN will be trained for.
    """

    noise = tf.random.normal(shape=[num_images, latent_dim], seed=random_seed)  # TODO
    dataset_training = init_dataset()
    dataset_training = tf.distribute.OneDeviceStrategy(device).experimental_distribute_dataset(dataset_training)

    dataset_generator = tf.data.Dataset.from_tensor_slices(noise).batch(num_images)  # TODO
    dataset_generator = tf.distribute.OneDeviceStrategy(device).experimental_distribute_dataset(dataset_generator)

    # Go through entire dataset in each loop
    num_batch = gan.global_step
    for num_epoch in range(epochs):
        print('--------------------------------')
        print('Epoch {}'.format(num_epoch))
        print('--------------------------------')

        gen_loss_epoch, disc_loss_epoch = None, None
        for batch_training in dataset_training:
            gen_loss_batch, disc_loss_batch = gan.distribute_train_step(batch_training)

            gen_losses_batch[num_batch] = float(gen_loss_batch)
            disc_losses_batch[num_batch] = float(disc_loss_batch)
            gen_loss_epoch, disc_loss_epoch = gen_loss_batch, disc_loss_batch

            with gan.train_writer.as_default():
                tf.summary.scalar('generator_loss', gen_loss_batch, step=num_batch)
                tf.summary.scalar('discriminator_loss', disc_loss_batch, step=num_batch)

            # Generate images and graph each generation_frequency batches
            if num_batch % generation_frequency == 0:
                for batch_generator in dataset_generator:
                    generated_image = gan.distribute_gen_step(batch_generator)
                    save_images('batch', num_epoch, num_batch, generated_image.numpy())
                save_graph('batch', num_epoch, num_batch)

            duration = time.time() - start
            print('Epoch: {}, Batch: {}, Generator_loss: {:0.3f}, Discriminator_loss: {:0.3f}, Duration: {:0.3f}s'
                  .format(num_epoch, num_batch, gen_loss_batch, disc_loss_batch, duration))

            num_batch += 1

        gen_losses_epoch[num_epoch] = float(gen_loss_epoch)
        disc_losses_epoch[num_epoch] = float(disc_loss_epoch)

        # Generate images and graph each epoch
        for batch_generator in dataset_generator:
            generated_image = gan.distribute_gen_step(batch_generator)
            save_images('epoch', num_epoch, num_batch, generated_image.numpy())
        save_graph('epoch', num_epoch, num_batch)

        # Create checkpoint for epoch
        gan.ckpt.step.assign(num_epoch + 1)
        gan.ckpt_manager.save()

        #  Create special checkpoint each special_checkpoints_frequency if special_checkpoints is set true
        if special_checkpoints and num_epoch > 0 and (num_epoch + 1) % special_checkpoints_frequency == 0:
            gan.ckpt_special.step.assign(num_epoch + 1)
            gan.ckpt_manager_special.save(checkpoint_number=num_epoch + 1)

    if generate_video:
        save_video()


def generate(gan):
    """Generates images of given GAN. Images will be saved on hard drive.

    Args:
      gan: Trained GAN that will be used for image generation.
    """

    noise = tf.random.normal(shape=[num_images, latent_dim])  # TODO
    dataset_generator = tf.data.Dataset.from_tensor_slices(noise).batch(num_images)  # TODO
    dataset_generator = tf.distribute.OneDeviceStrategy(device).experimental_distribute_dataset(dataset_generator)


    print('')
    print('Generating images...')
    for batch_generator in dataset_generator:
        generated_image = gan.distribute_gen_step(batch_generator)

        print('')
        print('Saving images...')
        save_images('generate', -1, -1, generated_image.numpy())

    print('')
    print('Succesfully generated and saved images for {}-dataset!'.format(dataset_values['name']))


def demo(gans, all_dataset_values):
    """Generates images of given GANs. Images will be saved on hard drive.

    Args:
      gans: All trained GANs that will be used for image generation.
      all_dataset_values: All dicitonaries of datasets for each GAN in gans.
    """

    grid_size = (len(gans), num_images_demo)
    images = None

    print('')
    print('Generating images...')
    from tqdm import trange
    progress = trange(len(gans), desc='Generating images for '+all_dataset_values[0]['name'], leave=True, file=sys.stdout)
    global dataset_values
    for i in progress:
        gan = gans[i]
        dataset_values = all_dataset_values[i]
        progress.set_description("Generating images for " + dataset_values['name'], refresh=True)

        noise = tf.random.normal(shape=[num_images_demo, latent_dim])  # TODO
        dataset_generator = tf.data.Dataset.from_tensor_slices(noise).batch(num_images_demo)  # TODO
        dataset_generator = tf.distribute.OneDeviceStrategy(device).experimental_distribute_dataset(dataset_generator)

        for batch_generator in dataset_generator:
            generated_image = gan.distribute_gen_step(batch_generator)

            width = generated_image.shape[1]
            height = generated_image.shape[2]
            if width < 1024 or height < 1024:
                generated_image = tf.image.resize(generated_image, [1024, 1024])

            if images is None:
                images = generated_image
            else:
                images = np.concatenate((images, generated_image), axis=0)

        if i == len(gans)-1:
            progress.set_description('Generation finished', refresh=True)

    print('')
    print('Saving images...')
    progress = trange(len(gans), desc='Saving images from '+all_dataset_values[0]['name'], leave=True, file=sys.stdout)
    saving = threading.Thread(target=save_images, args=('demo', -1, -1, images, grid_size))
    saving.start()
    for i in progress:
        dataset_values = all_dataset_values[i]
        if saving.is_alive():
            progress.set_description("Saving images from " + dataset_values['name'], refresh=True)
            time.sleep(1)
        else:
            progress.set_description("Saving images from " + dataset_values['name'], refresh=True)
            time.sleep(0.2)

        if i == len(gans)-1:
            progress.set_description('Saving finished', refresh=True)
    saving.join()

    print('')
    print('Succesfully generated and saved images from all specified datasets!')


def init_dataset(overwrite=True):
    """Initializes dataset by serializing folder of images into a TFRecord.

    Args:
      overwrite: If set true, existing TFRecords for dataset will be overwritten.

    Returns:
      TFRecord of dataset
    """

    files = os.listdir(dataset_values['dir_dataset'])
    paths = [dataset_values['dir_dataset'] + file for file in files]
    random.shuffle(paths)

    train_tfrecord = dataset_values['dir_tfrecords'] + 'train.tfrecord'

    def write_fn(img_paths, tfrec_path):
        def preprocess_fn(img_path):
            img_str = tf.io.read_file(img_path)
            img = tf.image.decode_and_crop_jpeg(img_str, dataset_values['data_crop'], channels=3)
            return img

        path_ds = tf.data.Dataset.from_tensor_slices(img_paths)
        image_ds = path_ds.map(preprocess_fn, num_parallel_calls=threads)
        proto_ds = image_ds.map(tf.io.serialize_tensor)
        tfrec = tf.data.experimental.TFRecordWriter(tfrec_path)
        tfrec.write(proto_ds)

    print("Initializing TFRecords for training...")
    if not os.path.isfile(dataset_values['dir_tfrecords'] + train_tfrecord) or overwrite:
        write_fn(paths, train_tfrecord)

    def parse_fn(tfrecord):
        result_img = tf.io.parse_tensor(tfrecord, out_type=tf.uint8)
        result_img = tf.reshape(result_img, dataset_values['img_shape'])
        result_img = tf.cast(result_img, tf.float32)
        result_img = get_data_range(result_img, data_range_in=[0, 255], data_range_out=[-1, 1])
        return result_img

    dataset_training = tf.data.TFRecordDataset(train_tfrecord) \
        .map(map_func=parse_fn, num_parallel_calls=threads) \
        .batch(batch_size=global_batchsize, drop_remainder=True) \
        .prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    return dataset_training


def get_data_range(data, data_range_in=None, data_range_out=None):
    """Converts data range of given data, usually an image.

    Args:
      data: Data that will be converted.
      data_range_in: Input data range. Default is [0, 255].
      data_range_out: Output data range. Default is [-1, 1].

    Returns:
      Converted data
    """

    if data_range_out is None:
        data_range_out = [0, 255]
    if data_range_in is None:
        data_range_in = [-1, 1]

    if data_range_in != data_range_out:
        scale = (np.float32(data_range_out[1]) - np.float32(data_range_out[0])) / (
                np.float32(data_range_in[1]) - np.float32(data_range_in[0]))
        bias = (np.float32(data_range_out[0]) - np.float32(data_range_in[0]) * scale)
        data = data * scale + bias
    return data


def save_images(mode, epoch, batch, images, grid_size=grid_size, quality=95):
    """Saves generated images in a grid layout.

    Args:
      mode: One of epoch, batch, generate or demo. First to only used in training.
      epoch: Number of epochs the GAN used for image generation was trained for.
      batch: Number of batches the GAN used for image generation was trained for.
      images: Images generated by GAN.
      grid_size: Grid layout of images to be saved.
      quality: Quality of image. Default is 95%.

    Returns:
      True if images were saved without errors or exceptions.
    """

    duration = '{:0.3f}'.format(time.time() - start)

    if mode == 'epoch':
        file = dataset_values['dir_gen_images'] + dataset_values['name'] + \
                '_epoch={:02d}_batch={:05d}_genloss={:0.3f}_discloss={:0.3f}_duration={}.jpg' \
                .format(epoch, batch, gen_losses_epoch[epoch], disc_losses_epoch[epoch], duration)
    elif mode == 'batch':
        file = dataset_values['dir_gen_images'] + dataset_values['name'] + \
                '_epoch={:02d}_batch={:05d}_genloss={:0.3f}_discloss={:0.3f}_duration={}.jpg' \
                .format(epoch, batch, gen_losses_batch[batch], disc_losses_batch[batch], duration)
    elif mode == 'generate':
        file = 'results/_generate/' + 'generate_{}.jpg'.format(time.time())
    elif mode == 'demo':
        file = 'results/_demo/' + 'demo_{}.jpg'.format(time.time())

    grid_h, grid_w = grid_size
    img_h, img_w = images.shape[1], images.shape[2]
    grid = np.zeros([grid_h * img_h, grid_w * img_w, 3], dtype=images.dtype)
    for idx in range(images.shape[0]):
        x = (idx % grid_w) * img_w
        y = (idx // grid_w) * img_h
        grid[y: y + img_h, x: x + img_w, :] = images[idx]

    if grid.ndim == 3:
        if grid.shape[2] == 1:
            grid = grid[2]  # grayscale HWC => HW
    image = get_data_range(grid)
    image = np.rint(image).clip(0, 255).astype(np.uint8)
    format = 'RGB' if image.ndim == 3 else 'L'

    image = Image.fromarray(image, format)
    if '.jpg' in file:
        image.save(file, "JPEG", quality=quality, optimize=True)
    else:
        image.save(file)

    return True


def save_video(fps=10):
    """Creates and saves video of all generated images of current dataset.

    Args:
      fps: Number of frames per second in video. Default is 10.

    Returns:
      True if video was saved without errors or exceptions.
    """

    images = [dataset_values['dir_gen_images'] + '/' + img
              for img in os.listdir(dataset_values['dir_gen_images']) if img.endswith(".jpg")]
    video = moviepy.video.io.ImageSequenceClip.ImageSequenceClip(images, fps=fps)
    video.write_videofile(dataset_values['dir_results'] + dataset_values['name'] + '.mp4')

    return True


def save_graph(mode, epoch, batch):
    """Saves current loss functions of generator and discriminator in a graph.
       Can be used to create graph of values from all epochs or all batches.

    Args:
      mode: One of epoch, batch.
      epoch: Number of epochs the GAN used for image generation was trained for.
      batch: Number of batches the GAN used for image generation was trained for.

    Returns:
      True if graph was saved without errors or exceptions.
    """

    duration = '{:0.3f}'.format(time.time() - start)

    file = mode + '_graph_{}_epoch={}_batch={}_duration={}.png'.format(
        dataset_values['name'], epoch, batch, duration)

    graph_range, graph_gen_losses, graph_disc_losses = None, None, None
    if mode == 'epoch':
        graph_range, graph_gen_losses, graph_disc_losses = list(range(epoch)), gen_losses_epoch, disc_losses_epoch
    elif mode == 'batch':
        graph_range, graph_gen_losses, graph_disc_losses = list(range(batch)), gen_losses_batch, disc_losses_batch

    fig = plt.figure(figsize=(15, 7))
    ax = fig.subplots()

    ax.plot(list(graph_gen_losses.keys()), list(graph_gen_losses.values()), '-', label='Generator Loss', color="red")
    ax.plot(list(graph_disc_losses.keys()), list(graph_disc_losses.values()), '-', label='Discriminator Loss', color="blue")

    if mode == 'epoch':
        ax.set_xlabel('Epoch')
        ax.set_title('Generator Loss/Discriminator Loss per Epoch')
    elif mode == 'batch':
        ax.set_xlabel('Batch')
        ax.set_title('Generator Loss/Discriminator Loss per Batch')

    ax.legend()
    plt.savefig(dataset_values['dir_graphs'] + file, bbox_inches='tight', dpi=300)
    plt.close('all')

    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', help='choose one of train, generate or demo', default=default_mode)
    parser.add_argument('--epochs', help='number of epochs used for training', default=default_epochs)
    parser.add_argument('--dataset', help='choose one of celeba, celeba_hq, comics, simpsons or anime',
                        default=default_dataset)
    args = parser.parse_args()

    global dataset_values
    dataset_values = datasets.get_dataset_values(args.dataset)

    if args.mode == 'train':
        gan = DCGAN(device, latent_dim, global_batchsize, dataset_values['projection_factor'], gen_filters,
                    disc_filters, gen_lr, disc_lr, beta1, beta2,
                    momentum, epsilon, dataset_values['img_shape'], dataset_values, checkpoint=pre_trained)
        train(gan, args.epochs)
    elif args.mode == 'generate':
        gan = DCGAN(device, latent_dim, global_batchsize, dataset_values['projection_factor'], gen_filters,
                    disc_filters, gen_lr, disc_lr, beta1, beta2,
                    momentum, epsilon, dataset_values['img_shape'], dataset_values, checkpoint=True)
        generate(gan)
    elif args.mode == 'demo':
        all_dataset_values = datasets.get_dataset_values('all')
        gans = []
        for demo_dataset_values in all_dataset_values:
            dataset_values = demo_dataset_values
            gan = DCGAN(device, latent_dim, global_batchsize, dataset_values['projection_factor'], gen_filters,
                        disc_filters, gen_lr, disc_lr, beta1, beta2,
                        momentum, epsilon, dataset_values['img_shape'], dataset_values, checkpoint=True)
            gans.append(gan)
        demo(gans, all_dataset_values)


if __name__ == '__main__':
    main()
