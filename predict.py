import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from model import make_generator_model

parser = argparse.ArgumentParser()
parser.add_argument("input_array", help="image array directory", default=None)

generator = make_generator_model(input_width=16, input_height=22, output_width=200, output_height=128)
generator.load_weights("model_epoch_59.h5")

input_array = np.load('/home/ubuntu/data/pres/final_freq_matrix.npy', allow_pickle=True)
input_array = np.concatenate([np.expand_dims(i, axis=0) for i in input_array if i.shape == (22, 16)], axis=0)
input_array = input_array + 0.01
input_array = (input_array - np.expand_dims(input_array.mean(axis=1), axis=1))/np.expand_dims(input_array.std(axis=1), axis=1)
input_tensor = tf.constant(input_array)
n_sample, w, h = input_tensor.shape
input_tensor = tf.reshape(input_tensor, shape=(n_sample, w*h))
generated_pic = generator(input_tensor, training=False)
generated_pic = np.array(generated_pic)
generated_pic = generated_pic * 127.5 + 127.5
generated_pic = np.round(generated_pic, 0).astype(np.int32)

for idx, pic in enumerate(generated_pic):
    plt.imshow(np.flipud(np.rot90(pic)))
    plt.savefig("dna_art" + str(idx) + ".png")


