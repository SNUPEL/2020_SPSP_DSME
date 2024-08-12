import PIL.Image
import numpy as np
import skimage
import moviepy.editor as mpy

from PIL import Image


def save_gif(frames, file_path):
    time_per_step = 0.05
    height, width = frames[0].shape[1], frames[0].shape[2]
    images = np.reshape(np.array(frames), [len(frames), height, width])

    if images.shape[1] != 3:
        images = color_frame_continuous(images)

    big_images = []
    for image in images:
        image = PIL.Image.fromarray(image.astype(np.uint8))
        big_images.append(np.array(image.resize([width * 40, height * 10], PIL.Image.NEAREST)))
    big_images = np.array(big_images)

    make_gif(big_images, file_path, duration=len(big_images) * time_per_step, true_image=True)


def make_gif(images, file_path, duration=2, true_image=False):
    def make_frame(t):
        try:
            x = images[int(len(images) / duration * t)]
        except:
            x = images[-1]

        if true_image:
            return x.astype(np.uint8)
        else:
            return ((x + 1) / 2 * 255).astype(np.uint8)

    txtClip = mpy.TextClip('.', color='white', font="Amiri-Bold", kerning=5, fontsize=10)
    clip = mpy.VideoClip(make_frame, duration=duration)
    clip = mpy.CompositeVideoClip([clip, txtClip])
    clip.duration = duration
    clip.write_gif(file_path, fps=len(images) / duration, verbose=False, logger=None)


def color_frame_continuous(images, dim=2):
    if dim == 2:
        colored_images = np.zeros([len(images), images.shape[1], images.shape[2], 3])
        for k in range(len(images)):
            for i in range(images.shape[1]):
                for j in range(images.shape[2]):
                    if images[k, i, j] == -1.0:
                        colored_images[k, i, j] = [0, 0, 0]
                    else:
                        grey = max(0, 255 - 0.05 * 255 * images[k, i, j])
                        colored_images[k, i, j] = [grey, grey, grey]
    return colored_images