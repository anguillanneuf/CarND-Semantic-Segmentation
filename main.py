import imageio
imageio.plugins.ffmpeg.download()

import os.path
import tensorflow as tf
import helper
import warnings
from distutils.version import LooseVersion
import project_tests as tests
from moviepy.editor import VideoFileClip
import numpy as np
import scipy as sp


# Check TensorFlow Version
assert LooseVersion(tf.__version__) >= LooseVersion('1.0'), 'Please use TensorFlow version 1.0 or newer.  You are using {}'.format(tf.__version__)
print('TensorFlow Version: {}'.format(tf.__version__))

# Check for a GPU
if not tf.test.gpu_device_name():
    warnings.warn('No GPU found. Please use a GPU to train your neural network.')
else:
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))


def load_vgg(sess, vgg_path):
    """
    Load Pretrained VGG Model into TensorFlow.
    :param sess: TensorFlow Session
    :param vgg_path: Path to vgg folder, containing "variables/" and "saved_model.pb"
    :return: Tuple of Tensors from VGG model (image_input, keep_prob, layer3_out, layer4_out, layer7_out)
    """
    # TODO: Implement function
    #   Use tf.saved_model.loader.load to load the model and weights

    vgg_tag = 'vgg16'
    vgg_input_tensor_name = 'image_input:0'
    vgg_keep_prob_tensor_name = 'keep_prob:0'
    vgg_layer3_out_tensor_name = 'layer3_out:0'
    vgg_layer4_out_tensor_name = 'layer4_out:0'
    vgg_layer7_out_tensor_name = 'layer7_out:0'


    # load the graph (MetaGraphDef protocol buffer) in session
    tf.saved_model.loader.load(sess, [vgg_tag], vgg_path)

    image_input = sess.graph.get_tensor_by_name(vgg_input_tensor_name)
    keep_prob = sess.graph.get_tensor_by_name(vgg_keep_prob_tensor_name)
    layer3_out = sess.graph.get_tensor_by_name(vgg_layer3_out_tensor_name)
    layer4_out = sess.graph.get_tensor_by_name(vgg_layer4_out_tensor_name)
    layer7_out = sess.graph.get_tensor_by_name(vgg_layer7_out_tensor_name)

    return image_input, keep_prob, layer3_out, layer4_out, layer7_out

tests.test_load_vgg(load_vgg, tf)


def layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes):
    """
    Create the layers for a fully convolutional network.  Build skip-layers using the vgg layers.
    :param vgg_layer7_out: TF Tensor for VGG Layer 3 output
    :param vgg_layer4_out: TF Tensor for VGG Layer 4 output
    :param vgg_layer3_out: TF Tensor for VGG Layer 7 output
    :param num_classes: Number of classes to classify
    :return: The Tensor for the last layer of output
    """
    # TODO: Implement function

    # vgg_layer7_out_1x1 = tf.layers.conv2d(vgg_layer7_out, num_classes, kernel_size=1, strides=1,
    #                                       kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))

    vgg_layer7_out_upsampled = tf.layers.conv2d_transpose(vgg_layer7_out, num_classes, 4, 2, 'same',
                                                          kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))

    vgg_layer4_out_1x1 = tf.layers.conv2d(vgg_layer4_out, num_classes, 1, 1,
                                          kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))

    skip_layer1 = tf.add(vgg_layer7_out_upsampled, vgg_layer4_out_1x1)

    skip_layer1_upsampled = tf.layers.conv2d_transpose(skip_layer1, num_classes, 4, 2, 'same',
                                                       kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))

    vgg_layer3_out_1x1 = tf.layers.conv2d(vgg_layer3_out, num_classes, 1, 1,
                                          kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))

    skip_layer2 = tf.add(skip_layer1_upsampled, vgg_layer3_out_1x1)

    skip_layer2_upsampled = tf.layers.conv2d_transpose(skip_layer2, num_classes, 16, 8, 'same',
                                                       kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))

    return skip_layer2_upsampled

tests.test_layers(layers)


def optimize(nn_last_layer, correct_label, learning_rate, num_classes):
    """
    Build the TensorFLow loss and optimizer operations.
    :param nn_last_layer: TF Tensor of the last layer in the neural network
    :param correct_label: TF Placeholder for the correct label image
    :param learning_rate: TF Placeholder for the learning rate
    :param num_classes: Number of classes to classify
    :return: Tuple of (logits, train_op, cross_entropy_loss)
    """
    # TODO: Implement function

    logits = tf.reshape(nn_last_layer, (-1, num_classes))

    labels = tf.reshape(correct_label, (-1, num_classes))

    cross_entropy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels, name="Softmax"))

    train_op = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy_loss)

    return logits, train_op, cross_entropy_loss

tests.test_optimize(optimize)


def train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, input_image,
             correct_label, keep_prob, learning_rate):
    """
    Train neural network and print out the loss during training.
    :param sess: TF Session
    :param epochs: Number of epochs
    :param batch_size: Batch size
    :param get_batches_fn: Function to get batches of training data.  Call using get_batches_fn(batch_size)
    :param train_op: TF Operation to train the neural network
    :param cross_entropy_loss: TF Tensor for the amount of loss
    :param input_image: TF Placeholder for input images
    :param correct_label: TF Placeholder for label images
    :param keep_prob: TF Placeholder for dropout keep probability
    :param learning_rate: TF Placeholder for learning rate
    """
    # TODO: Implement function


    with sess.as_default():

        epoch_loss = 1e10
        lr = 1e-4
        prob = 0.7

        sess.run(tf.global_variables_initializer())

        for epoch in range(epochs):

            curr_epoch_loss = 0

            for steps, (images, labels) in enumerate(get_batches_fn(batch_size)):

                _, loss = sess.run([train_op, cross_entropy_loss],
                                   feed_dict={input_image: images, correct_label: labels, learning_rate: lr,
                                              keep_prob: prob})

                curr_epoch_loss += loss

            print("Epoch {} Average Loss Per Image: {:.5f}".format(epoch, curr_epoch_loss / (len(images) * float(steps))))

            if curr_epoch_loss > epoch_loss:
                print("Loss not improving! Early stopping")
                break
            else:
                epoch_loss = curr_epoch_loss


tests.test_train_nn(train_nn)


def run():
    num_classes = 2
    image_shape = (160, 576)
    data_dir = './data'
    runs_dir = './runs'
    tests.test_for_kitti_dataset(data_dir)

    # Download pretrained vgg model
    helper.maybe_download_pretrained_vgg(data_dir)

    # OPTIONAL: Train and Inference on the cityscapes dataset instead of the Kitti dataset.
    # You'll need a GPU with at least 10 teraFLOPS to train on.
    #  https://www.cityscapes-dataset.com/

    with tf.Session() as sess:
        # Path to vgg model
        vgg_path = os.path.join(data_dir, 'vgg')
        # Create function to get batches
        get_batches_fn = helper.gen_batch_function(os.path.join(data_dir, 'data_road/training'), image_shape)

        # OPTIONAL: Augment Images for better results
        #  https://datascience.stackexchange.com/questions/5224/how-to-prepare-augment-images-for-neural-network

        # TODO: Build NN using load_vgg, layers, and optimize function
        image_input, keep_prob, layer3_out, layer4_out, layer7_out = load_vgg(sess, vgg_path)

        output_layer = layers(layer3_out, layer4_out, layer7_out, num_classes)

        correct_label = tf.placeholder(dtype=tf.float32, shape=(None, None, None, num_classes))

        learning_rate = tf.placeholder(dtype=tf.float32)

        logits, train_op, cross_entropy_loss = optimize(output_layer, correct_label, learning_rate, num_classes)

        # TODO: Train NN using the train_nn function

        train_nn(sess, 20, 10, get_batches_fn, train_op, cross_entropy_loss, image_input,
                 correct_label, keep_prob, learning_rate)


        # TODO: Save inference data using helper.save_inference_samples
        helper.save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, keep_prob, image_input)

        # saves the graph
        saver = tf.train.Saver()
        saver.save(sess, 'mymodel')

        # OPTIONAL: Apply the trained model to a video
        def process_video(original_image, sess=sess, image_shape=image_shape, logits=logits, keep_prob=keep_prob,
                               image_input=image_input):

            original_image_shape = original_image.shape

            image = sp.misc.imresize(original_image, image_shape)

            im_softmax = sess.run([tf.nn.softmax(logits)], {keep_prob: 1.0, image_input: [image]})
            im_softmax = im_softmax[0][:, 1].reshape(image_shape[0], image_shape[1])
            segmentation = (im_softmax > 0.5).reshape(image_shape[0], image_shape[1], 1)

            # color road green
            mask = np.dot(segmentation, np.array([[0, 255, 0, 127]]))
            mask = sp.misc.toimage(mask, mode="RGBA")

            street_im = sp.misc.toimage(image)
            street_im.paste(mask, box=None, mask=mask)

            return np.array(sp.misc.imresize(street_im, original_image_shape))

        clip1 = VideoFileClip("./data/harder_challenge_video.mp4")
        white_clip = clip1.fl_image(process_video)
        white_clip.write_videofile("./data/segmented_project_video.mp4", audio=False)


if __name__ == '__main__':
    run()
