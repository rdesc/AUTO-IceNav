import argparse
import glob
import os

import cv2
from PIL import Image


def sort_images(image_fps):
    return sorted(image_fps, key=lambda x: int(os.path.basename(x).split(sep='-')[-1].split('.')[0]))


def make_gif(image_dir, gif_name):
    image_fps = sort_images(glob.glob(f'{image_dir}/*.png'))
    if not image_fps:
        print('No images found in the specified directory.')
        return
    print('Found {} images'.format(len(image_fps)))

    frames = [Image.open(image) for image in image_fps]
    frame_one = frames[0]
    to_save_path = os.path.join(image_dir, gif_name)
    # https://pillow.readthedocs.io/en/stable/handbook/image-file-formats.html#gif-saving
    frame_one.save(to_save_path, format='GIF', append_images=frames,
                   save_all=True, duration=100, loop=1, optimize=True)

    print('GIF generated successfully at {}'.format(to_save_path))


def make_mp4(image_dir, mp4_name, fps=2):
    image_fps = sort_images(glob.glob(f'{image_dir}/*.png'))
    if not image_fps:
        print('No images found in the specified directory.')
        return
    print('Found {} images'.format(len(image_fps)))

    first_image = cv2.imread(image_fps[0])
    height, width, _ = first_image.shape

    to_save_path = os.path.join(image_dir, mp4_name)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(to_save_path, fourcc, fps, (width, height))

    for image_file in image_fps:
        frame = cv2.imread(image_file)
        out.write(frame)
    out.release()

    print('MP4 generated successfully at {}'.format(to_save_path))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('image_directory', type=str, help='path to directory containing images')
    parser.add_argument('media_name', type=str, help='name of media file, .gif or .mp4')
    parser.add_argument('--fps', type=int, default=2, help='frames per second')
    args = parser.parse_args()

    if args.media_name.endswith('.gif'):
        make_gif(args.image_directory, args.media_name)
    elif args.media_name.endswith('.mp4'):
        make_mp4(args.image_directory, args.media_name, args.fps)
