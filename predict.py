import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from model import make_generator_model

parser = argparse.ArgumentParser()
parser.add_argument("input_array_file", help="image array directory", default=None)
parser.add_argument("model_weight", help="image array directory", default=None)
parser.add_argument("--img_prefix", help="image array directory", default='dna_art')

def predict(model_file, data_file, image_prefix):
    generator = make_generator_model(input_width=16, input_height=22, output_width=200, output_height=128)
    generator.load_weights(model_file)
    input_array = np.load(data_file, allow_pickle=True)
    input_array = np.concatenate([np.expand_dims(i, axis=0) for i in input_array if i.shape == (22, 16)], axis=0)
    input_array = input_array + 0.01
    input_array = (input_array - np.expand_dims(input_array.mean(axis=1), axis=1)) / np.expand_dims(
        input_array.std(axis=1), axis=1)
    input_tensor = tf.constant(input_array)
    n_sample, w, h = input_tensor.shape
    input_tensor = tf.reshape(input_tensor, shape=(n_sample, w * h))
    generated_pic = generator(input_tensor, training=False)
    generated_pic = np.array(generated_pic)
    generated_pic = generated_pic * 127.5 + 127.5
    generated_pic = np.round(generated_pic, 0).astype(np.int32)

    for idx, pic in enumerate(generated_pic):
        plt.imshow(np.flipud(np.rot90(pic)))
        plt.savefig(image_prefix + str(idx) + ".png")

if __name__ == '__main__':
    args = parser.parse_args()
    input_array_file = args.input_array_file
    model_weight = args.model_weight
    img_prefix = args.img_prefix
    predict(model_weight, input_array_file, img_prefix)









