import numpy as np
import os
import pickle
import argparse
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

# size 100*62 (golden ratio)
parser = argparse.ArgumentParser()
parser.add_argument("img_dir", help="image directory", default=None)
parser.add_argument("output_dir", help="image directory", default=None)
parser.add_argument("--out_width", type=int, help="width of output image", default=100)
parser.add_argument("--out_height", type=int, help="height of output image", default=64)

def crop_preserve_aspect(im, out_width=100, out_height= 64):
    w, h = im.size
    if w <= h:
        im = im.rotate(90)
        w, h = im.size

    if w/h < (out_width/out_height):
        div_factor = h / out_height
        new_h = out_height
        new_w = np.ceil(w / div_factor)
    else:
        div_factor = w / out_width
        new_h = np.ceil(h / div_factor)
        new_w = out_width

    im = im.resize((int(new_w), int(new_h)))
    im = im.crop((0, 0, out_width, out_height))

    return im

def main(input_dir, output_dir, out_width, out_height):
    input_list = list()
    for filename in os.listdir(input_dir):
        try:
            im = Image.open(os.path.join(input_dir, filename))
        except:
            print("Cannot parse file: ", filename)
            continue
        w, h = im.size
        im = crop_preserve_aspect(im, out_width, out_height)

        if im is None:
            continue

        im_array = np.array(im)

        if len(im_array.shape) == 2:
            pass
        else:
            im_array = im_array[:, :, :3]
            input_list.append(im_array)
        print("Successfully parse ", filename)

    input_array = np.array(input_list)
    with open(os.path.join(output_dir, "input_array.pickle"), "wb") as f:
        pickle.dump(input_array, f)


if __name__ == '__main__':
    args = parser.parse_args()
    image_directory = args.img_dir
    output_dir = args.output_dir
    out_width = args.out_width
    out_height = args.out_height
    main(image_directory, output_dir, out_width, out_height)









