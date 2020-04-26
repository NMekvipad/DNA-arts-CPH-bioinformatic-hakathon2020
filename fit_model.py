import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
import argparse
import time
from model import make_generator_model, make_discriminator_model, discriminator_loss, generator_loss

parser = argparse.ArgumentParser()
parser.add_argument("train_data", help="image array directory", default=None)
parser.add_argument("--in_width", type=int, help="width of output image", default=16)
parser.add_argument("--in_height", type=int, help="height of output image", default=22)
parser.add_argument("--out_width", type=int, help="width of output image", default=200)
parser.add_argument("--out_height", type=int, help="height of output image", default=128)

BUFFER_SIZE = 3000
BATCH_SIZE = 24
EPOCHS = 60
noise_dim = 352
num_examples_to_generate = 16


def main(input_array, input_width=16, input_height=22, output_width=100, output_height=64):
    input_array = (input_array - 127.5) / 127.5
    train_dataset = tf.data.Dataset.from_tensor_slices(input_array).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

    generator = make_generator_model(input_width, input_height, output_width, output_height)
    discriminator = make_discriminator_model(output_width, output_height)

    generator_optimizer = tf.keras.optimizers.Adam(1e-4)
    discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

    checkpoint_dir = './training_checkpoints'
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
    checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                     discriminator_optimizer=discriminator_optimizer,
                                     generator=generator,
                                     discriminator=discriminator)

    seed = tf.random.normal([num_examples_to_generate, noise_dim])

    @tf.function
    def train_step(images):
        noise = tf.random.normal([BATCH_SIZE, noise_dim])

        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            generated_images = generator(noise, training=True)

            real_output = discriminator(images, training=True)
            fake_output = discriminator(generated_images, training=True)

            gen_loss = generator_loss(fake_output)
            disc_loss = discriminator_loss(real_output, fake_output)

        gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
        gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

        generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
        discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

    def generate_and_save_images(model, epoch, test_input):
        # Notice `training` is set to False.
        # This is so all layers run in inference mode (batchnorm).
        predictions = model(test_input, training=False)

        fig = plt.figure(figsize=(4, 4))

        for i in range(predictions.shape[0]):
            plt.subplot(4, 4, i + 1)
            plt.imshow(predictions[i, :, :, :3] * 127.5 + 127.5)
            plt.axis('off')

        plt.savefig('image_at_epoch_{:04d}.png'.format(epoch))
        plt.show()

    def train(dataset, epochs):
        for epoch in range(epochs):
            start = time.time()

            for image_batch in dataset:
                train_step(image_batch)

            # Produce images for the GIF as we go
            generate_and_save_images(generator,
                                     epoch + 1,
                                     seed)

            # Save the model every 15 epochs
            if (epoch + 1) % 20 == 0:
                checkpoint.save(file_prefix=checkpoint_prefix)
                generator.save_weights("model_epoch_"+ str(epoch) + ".h5")

            print('Time for epoch {} is {} sec'.format(epoch + 1, time.time() - start))

        # Generate after the final epoch
        generate_and_save_images(generator,
                                 epochs,
                                 seed)

    train(train_dataset, EPOCHS)


if __name__ == '__main__':
    args = parser.parse_args()
    train_data_path = args.train_data
    input_width = args.in_width
    input_height = args.in_height
    output_width = args.out_width
    output_height = args.out_height

    with open(train_data_path, "rb") as f:
        input_array = pickle.load(f)

    main(input_array, input_width, input_height, output_width, output_height)

