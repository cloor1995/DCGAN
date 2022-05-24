celeba_name = 'celeba'
celeba_dir_dataset = 'data/celeb_a/'
celeba_dir_results = 'results/celeb_a/'
celeba_dir_tfrecords = 'results/celeb_a/tf_records/'
celeba_dir_checkpoints = 'results/celeb_a/checkpoints/'
celeba_dir_events = 'results/celeb_a/events/'
celeba_dir_gen_images = 'results/celeb_a/generated_images/'
celeba_dir_graphs = 'results/celeb_a/graphs/'
celeba_projection_factor = 4
celeba_img_shape = (128, 128, 3)
celeba_data_crop = [57, 21, 128, 128]  # Cut Dataset to 128x128 resolution
celeba_dataset_values = {'name': celeba_name,
                         'dir_dataset': celeba_dir_dataset,
                         'dir_results': celeba_dir_results,
                         'dir_tfrecords': celeba_dir_tfrecords,
                         'dir_checkpoints': celeba_dir_checkpoints,
                         'dir_events': celeba_dir_events,
                         'dir_gen_images': celeba_dir_gen_images,
                         'dir_graphs': celeba_dir_graphs,
                         'projection_factor': celeba_projection_factor,
                         'img_shape': celeba_img_shape,
                         'data_crop': celeba_data_crop}

celeba_hq_name = 'celeba_hq'
celeba_hq_dir_dataset = 'data/celeb_a_hq/'
celeba_hq_dir_results = 'results/celeb_a_hq/'
celeba_hq_dir_tfrecords = 'results/celeb_a_hq/tf_records/'
celeba_hq_dir_checkpoints = 'results/celeb_a_hq/checkpoints/'
celeba_hq_dir_events = 'results/celeb_a_hq/events/'
celeba_hq_dir_gen_images = 'results/celeb_a_hq/generated_images/'
celeba_hq_dir_graphs = 'results/celeb_a_hq/graphs/'
celeba_hq_projection_factor = 32
celeba_hq_img_shape = (1024, 1024, 3)
celeba_hq_data_crop = [0, 0, 1024, 1024]
celeba_hq_dataset_values = {'name': celeba_hq_name,
                            'dir_dataset': celeba_hq_dir_dataset,
                            'dir_results': celeba_hq_dir_results,
                            'dir_tfrecords': celeba_hq_dir_tfrecords,
                            'dir_checkpoints': celeba_hq_dir_checkpoints,
                            'dir_events': celeba_hq_dir_events,
                            'dir_gen_images': celeba_hq_dir_gen_images,
                            'dir_graphs': celeba_hq_dir_graphs,
                            'projection_factor': celeba_hq_projection_factor,
                            'img_shape': celeba_hq_img_shape,
                            'data_crop': celeba_hq_data_crop}

simpsons_name = 'simpsons'
simpsons_dir_dataset = 'data/simpsons/'
simpsons_dir_results = 'results/simpsons/'
simpsons_dir_tfrecords = 'results/simpsons/tf_records/'
simpsons_dir_checkpoints = 'results/simpsons/checkpoints/'
simpsons_dir_events = 'results/simpsons/events/'
simpsons_dir_gen_images = 'results/simpsons/generated_images/'
simpsons_dir_graphs = 'results/simpsons/graphs/'
simpsons_projection_factor = 8
simpsons_img_shape = (256, 256, 3)
simpsons_data_crop = [0, 0, 256, 256]
simpsons_dataset_values = {'name': simpsons_name,
                           'dir_dataset': simpsons_dir_dataset,
                           'dir_results': simpsons_dir_results,
                           'dir_tfrecords': simpsons_dir_tfrecords,
                           'dir_checkpoints': simpsons_dir_checkpoints,
                           'dir_events': simpsons_dir_events,
                           'dir_gen_images': simpsons_dir_gen_images,
                           'dir_graphs': simpsons_dir_graphs,
                           'projection_factor': simpsons_projection_factor,
                           'img_shape': simpsons_img_shape,
                           'data_crop': simpsons_data_crop}

anime_name = 'anime'
anime_dir_dataset = 'data/anime/'
anime_dir_results = 'results/anime/'
anime_dir_tfrecords = 'results/anime/tf_records/'
anime_dir_checkpoints = 'results/anime/checkpoints/'
anime_dir_events = 'results/anime/events/'
anime_dir_gen_images = 'results/anime/generated_images/'
anime_dir_graphs = 'results/anime/graphs/'
anime_projection_factor = 2
anime_img_shape = (64, 64, 3)
anime_data_crop = [0, 0, 64, 64]
anime_dataset_values = {'name': anime_name,
                        'dir_dataset': anime_dir_dataset,
                        'dir_results': anime_dir_results,
                        'dir_tfrecords': anime_dir_tfrecords,
                        'dir_checkpoints': anime_dir_checkpoints,
                        'dir_events': anime_dir_events,
                        'dir_gen_images': anime_dir_gen_images,
                        'dir_graphs': anime_dir_graphs,
                        'projection_factor': anime_projection_factor,
                        'img_shape': anime_img_shape,
                        'data_crop': anime_data_crop}

comics_name = 'comics'
comics_dir_dataset = 'data/comics/'
comics_dir_results = 'results/comics/'
comics_dir_tfrecords = 'results/comics/tf_records/'
comics_dir_checkpoints = 'results/comics/checkpoints/'
comics_dir_events = 'results/comics/events/'
comics_dir_gen_images = 'results/comics/generated_images/'
comics_dir_graphs = 'results/comics/graphs/'
comics_projection_factor = 16
comics_img_shape = (512, 512, 3)
comics_data_crop = [0, 0, 512, 512]
comics_dataset_values = {'name': comics_name,
                         'dir_dataset': comics_dir_dataset,
                         'dir_results': comics_dir_results,
                         'dir_tfrecords': comics_dir_tfrecords,
                         'dir_checkpoints': comics_dir_checkpoints,
                         'dir_events': comics_dir_events,
                         'dir_gen_images': comics_dir_gen_images,
                         'dir_graphs': comics_dir_graphs,
                         'projection_factor': comics_projection_factor,
                         'img_shape': comics_img_shape,
                         'data_crop': comics_data_crop}

all_dataset_values = [celeba_dataset_values, anime_dataset_values, simpsons_dataset_values,
                      comics_dataset_values, celeba_hq_dataset_values]


def get_dataset_values(dataset):
    """Return dictionary of chosen dataset.

    Args:
      dataset: One of celeba, celeba_hq, comics, simpsons, anime.

    Returns:
      Dictionary of dataset with keys name, dir_dataset, dir_results, dir_tfrecords,
      dir_checkpoints, dir_events, dir_events, dir_gen_images, dir_graphs, projection_factor,
      img_shape, data_crop.
    """

    if dataset == 'celeba':
        return celeba_dataset_values
    elif dataset == 'celeba_hq':
        return celeba_hq_dataset_values
    elif dataset == 'simpsons':
        return simpsons_dataset_values
    elif dataset == 'anime':
        return anime_dataset_values
    elif dataset == 'comics':
        return comics_dataset_values
    elif dataset == 'all':
        return all_dataset_values
    else:
        print('Dataset ' + str(dataset) + ' does not exist, using celeb_a instead.')
        return celeba_dataset_values
