import numpy as np
import os
import tensorflow as tf
from PIL import Image
import utility as Utility
import argparse

from make_datasets_food101 import Make_datasets_food101

def parser():
    parser = argparse.ArgumentParser(description='train LSGAN')
    parser.add_argument('--batch_size', '-b', type=int, default=20, help='Number of images in each mini-batch')
    parser.add_argument('--log_file_name', '-lf', type=str, default='log180716', help='log file name')
    parser.add_argument('--epoch', '-e', type=int, default=1000, help='epoch')
    parser.add_argument('--dir_name', '-dn', type=str, default='PATH/TO/DATASETS',
                        help='directory name of real data')

    return parser.parse_args()

args = parser()


#global variants
BATCH_SIZE = args.batch_size
LOGFILE_NAME = args.log_file_name
EPOCH = args.epoch
DIR_NAME = args.dir_name
IMG_WIDTH = 64
IMG_HEIGHT = 64
NOISE_UNIT_NUM = 100
NOISE_MEAN = 0.0
NOISE_STDDEV = 1.0
TEST_DATA_SAMPLE = 5 * 5
L2_NORM = 0.001
KEEP_PROB_RATE = 0.5
SEED = 1234
np.random.seed(seed=SEED)
BOARD_DIR_NAME = './tensorboard/' + LOGFILE_NAME

out_image_dir = './out_images_LSGAN' #output image file
out_model_dir = './out_models_LSGAN' #output model file

try:
    os.mkdir(out_image_dir)
    os.mkdir(out_model_dir)
    os.mkdir('./out_images_Debug') #for debug
except:
    pass

make_datasets = Make_datasets_food101(DIR_NAME, IMG_WIDTH, IMG_HEIGHT, SEED, NOISE_MEAN, NOISE_STDDEV, TEST_DATA_SAMPLE,
                                      NOISE_UNIT_NUM)

def leaky_relu(x, alpha):
    return tf.nn.relu(x) - alpha * tf.nn.relu(-x)

def gaussian_noise(input, std): #used at discriminator
    noise = tf.random_normal(shape=tf.shape(input), mean=0.0, stddev=std, dtype=tf.float32, seed=SEED)
    return input + noise

#generator------------------------------------------------------------------
def generator(z, reuse=False):
    with tf.variable_scope('generator', reuse=reuse):

        with tf.name_scope("G_layer1"): #layer1 linear
            wg1 = tf.get_variable('wg1', [NOISE_UNIT_NUM, 512 * 4 * 4], initializer=tf.random_normal_initializer
            (mean=0.0, stddev=0.02, seed=SEED), dtype=tf.float32)
            bg1 = tf.get_variable('bg1', [512 * 4 * 4], initializer=tf.constant_initializer(0.0))
            scaleg1 = tf.get_variable('sg1', [512 * 4 * 4], initializer=tf.constant_initializer(1.0))
            betag1 = tf.get_variable('beg1', [512 * 4 * 4], initializer=tf.constant_initializer(0.0))

            fc1 = tf.matmul(z, wg1, name='G_fc1') + bg1
            #batch normalization
            batch_mean1, batch_var1 = tf.nn.moments(fc1, [0])
            bn1 = tf.nn.batch_normalization(fc1, batch_mean1, batch_var1, betag1, scaleg1 , 0.0001, name='G_BN1')
            #leaky relu
            lR1 = leaky_relu(bn1, alpha=0.2)
            #reshape nx4x4x512 -> [n, 4, 4, 512]
            re1 = tf.reshape(lR1, [-1, 4, 4, 512])

        with tf.name_scope("G_layer2"):
            #layer2 4x4x512 -> 8x8x256
            wg2 = tf.get_variable('wg2', [4, 4, 256, 512], initializer=tf.random_normal_initializer
            (mean=0.0, stddev=0.02, seed=SEED), dtype=tf.float32)
            bg2 = tf.get_variable('bg2', [256], initializer=tf.constant_initializer(0.0))
            scaleg2 = tf.get_variable('sg2', [256], initializer=tf.constant_initializer(1.0))
            betag2 = tf.get_variable('beg2', [256], initializer=tf.constant_initializer(0.0))

            output_shape2 = tf.stack(
                [tf.shape(re1)[0], tf.shape(re1)[1] * 2, tf.shape(re1)[2] * 2, tf.div(tf.shape(re1)[3], tf.constant(2))])
            deconv2 = tf.nn.conv2d_transpose(re1, wg2, output_shape=output_shape2, strides=[1, 2, 2, 1],
                                             padding="SAME") + bg2
            # batch normalization
            batch_mean2, batch_var2 = tf.nn.moments(deconv2, [0, 1, 2])
            bn2 = tf.nn.batch_normalization(deconv2, batch_mean2, batch_var2, betag2, scaleg2, 0.0001, name='G_BN2')
            # leaky relu
            lR2 = leaky_relu(bn2, alpha=0.2)

        with tf.name_scope("G_layer3"): # layer3 8x8x256 -> 16x16x128
            wg3 = tf.get_variable('wg3', [4, 4, 128, 256], initializer=tf.random_normal_initializer
            (mean=0.0, stddev=0.02, seed=SEED), dtype=tf.float32)
            bg3 = tf.get_variable('bg3', [128], initializer=tf.constant_initializer(0.0))
            scaleg3 = tf.get_variable('sg3', [128], initializer=tf.constant_initializer(1.0))
            betag3 = tf.get_variable('beg3', [128], initializer=tf.constant_initializer(0.0))

            output_shape3 = tf.stack(
            [tf.shape(lR2)[0], tf.shape(lR2)[1] * 2, tf.shape(lR2)[2] * 2, tf.div(tf.shape(lR2)[3], tf.constant(2))])
            deconv3 = tf.nn.conv2d_transpose(lR2, wg3, output_shape=output_shape3, strides=[1, 2, 2, 1],
                                             padding="SAME") + bg3
            # batch normalization
            batch_mean3, batch_var3 = tf.nn.moments(deconv3, [0, 1, 2])
            bn3 = tf.nn.batch_normalization(deconv3, batch_mean3, batch_var3, betag3, scaleg3, 0.0001, name='G_BN3')
            # leaky relu
            lR3 = leaky_relu(bn3, alpha=0.2)

        with tf.name_scope("G_layer4"): # layer4 16x16x128 -> 32x32x64
            wg4 = tf.get_variable('wg4', [4, 4, 64, 128], initializer=tf.random_normal_initializer
            (mean=0.0, stddev=0.02, seed=SEED), dtype=tf.float32)
            bg4 = tf.get_variable('bg4', [64], initializer=tf.constant_initializer(0.0))
            scaleg4 = tf.get_variable('sg4', [64], initializer=tf.constant_initializer(1.0))
            betag4 = tf.get_variable('beg4', [64], initializer=tf.constant_initializer(0.0))

            output_shape4 = tf.stack(
                # [tf.shape(lR3)[0], tf.shape(lR3)[1], tf.shape(lR3)[2], tf.shape(lR3)[3]])
            [tf.shape(lR3)[0], tf.shape(lR3)[1] * 2, tf.shape(lR3)[2] * 2, tf.div(tf.shape(lR3)[3], tf.constant(2))])
            deconv4 = tf.nn.conv2d_transpose(lR3, wg4, output_shape=output_shape4, strides=[1, 2, 2, 1],
                                             padding="SAME") + bg4
            # batch normalization
            batch_mean4, batch_var4 = tf.nn.moments(deconv4, [0, 1, 2])
            bn4 = tf.nn.batch_normalization(deconv4, batch_mean4, batch_var4, betag4, scaleg4, 0.0001, name='G_BN4')
            # leaky relu
            lR4 = leaky_relu(bn4, alpha=0.2)

        with tf.name_scope("G_layer5"): # layer5 32x32x648 -> 64x64x3
            wg5 = tf.get_variable('wg5', [4, 4, 3, 64], initializer=tf.random_normal_initializer
            (mean=0.0, stddev=0.02, seed=SEED), dtype=tf.float32)
            bg5 = tf.get_variable('bg5', [3], initializer=tf.constant_initializer(0.0))

            output_shape5 = tf.stack(
                [tf.shape(lR4)[0], tf.shape(lR4)[1] * 2, tf.shape(lR4)[2] * 2, tf.constant(3)])
            deconv5 = tf.nn.conv2d_transpose(lR4, wg5, output_shape=output_shape5, strides=[1, 2, 2, 1],
                                             padding="SAME") + bg5
            # tanh
            tanh5 = tf.nn.tanh(deconv5)

        return tanh5


#discriminator-----------------------------------------------------------------
def discriminator(x, reuse=False):
    with tf.variable_scope('discriminator', reuse=reuse):

        with tf.name_scope("D_layer1"): # layer1 conv1
            wd1 = tf.get_variable('wd1', [3, 3, 3, 32], initializer=tf.random_normal_initializer
            (mean=0.0, stddev=0.02, seed=SEED), dtype=tf.float32)
            bd1 = tf.get_variable('bd1', [32], initializer=tf.constant_initializer(0.0))
            scaled1 = tf.get_variable('sd1', [32], initializer=tf.constant_initializer(1.0))
            betad1 = tf.get_variable('bed1', [32], initializer=tf.constant_initializer(0.0))
            conv1 = tf.nn.conv2d(x, wd1, strides=[1, 1, 1, 1], padding="SAME", name='D_conv1') + bd1
            # batch normalization
            batch_mean1, batch_var1 = tf.nn.moments(conv1, [0, 1, 2])
            bn1 = tf.nn.batch_normalization(conv1, batch_mean1, batch_var1, betad1, scaled1, 0.0001, name='D_BN1')
            #gaussian noise
            gn1 = gaussian_noise(bn1, 0.3)
            # leakyReLU function
            lr1 = leaky_relu(gn1, alpha=0.2)

        with tf.name_scope("D_layer2"): # layer2 conv2
            wd2 = tf.get_variable('wd2', [4, 4, 32, 64], initializer=tf.random_normal_initializer
            (mean=0.0, stddev=0.02, seed=SEED), dtype=tf.float32)
            bd2 = tf.get_variable('bd2', [64], initializer=tf.constant_initializer(0.0))
            scaled2 = tf.get_variable('sd2', [64], initializer=tf.constant_initializer(1.0))
            betad2 = tf.get_variable('bed2', [64], initializer=tf.constant_initializer(0.0))

            conv2 = tf.nn.conv2d(lr1, wd2, strides=[1, 2, 2, 1], padding="SAME", name='D_conv2') + bd2
            # batch normalization
            batch_mean2, batch_var2 = tf.nn.moments(conv2, [0, 1, 2])
            bn2 = tf.nn.batch_normalization(conv2, batch_mean2, batch_var2, betad2, scaled2, 0.0001, name='D_BN2')
            #gaussian noise
            gn2 = gaussian_noise(bn2, 0.3)
            # leakyReLU function
            lr2 = leaky_relu(gn2, alpha=0.2)

        with tf.name_scope("D_layer3"): # layer3  conv3
            wd3 = tf.get_variable('wd3', [4, 4, 64, 128], initializer=tf.random_normal_initializer
            (mean=0.0, stddev=0.02, seed=SEED), dtype=tf.float32)
            bd3 = tf.get_variable('bd3', [128], initializer=tf.constant_initializer(0.0))
            scaled3 = tf.get_variable('sd3', [128], initializer=tf.constant_initializer(1.0))
            betad3 = tf.get_variable('bed3', [128], initializer=tf.constant_initializer(0.0))

            conv3 = tf.nn.conv2d(lr2, wd3, strides=[1, 2, 2, 1], padding="SAME", name='D_conv3') + bd3
            # batch normalization
            batch_mean3, batch_var3 = tf.nn.moments(conv3, [0, 1, 2])
            bn3 = tf.nn.batch_normalization(conv3, batch_mean3, batch_var3, betad3, scaled3, 0.0001, name='D_BN3')
            # gaussian noise
            gn3 = gaussian_noise(bn3, 0.3)
            # leakyReLU function
            lr3 = leaky_relu(gn3, alpha=0.2)

        with tf.name_scope("D_layer4"): # layer4  conv4
            wd4 = tf.get_variable('wd4', [4, 4, 128, 256], initializer=tf.random_normal_initializer
            (mean=0.0, stddev=0.02, seed=SEED), dtype=tf.float32)
            bd4 = tf.get_variable('bd4', [256], initializer=tf.constant_initializer(0.0))
            scaled4 = tf.get_variable('sd4', [256], initializer=tf.constant_initializer(1.0))
            betad4 = tf.get_variable('bed4', [256], initializer=tf.constant_initializer(0.0))

            conv4 = tf.nn.conv2d(lr3, wd4, strides=[1, 2, 2, 1], padding="SAME", name='D_conv4') + bd4
            # batch normalization
            batch_mean4, batch_var4 = tf.nn.moments(conv4, [0, 1, 2])
            bn4 = tf.nn.batch_normalization(conv4, batch_mean4, batch_var4, betad4, scaled4, 0.0001, name='D_BN4')
            # gaussian noise
            gn4 = gaussian_noise(bn4, 0.3)
            # leakyReLU function
            lr4 = leaky_relu(gn4, alpha=0.2)

        with tf.name_scope("D_layer5"): # layer5  conv5
            wd5 = tf.get_variable('wd5', [4, 4, 256, 512], initializer=tf.random_normal_initializer
            (mean=0.0, stddev=0.02, seed=SEED), dtype=tf.float32)
            bd5 = tf.get_variable('bd5', [512], initializer=tf.constant_initializer(0.0))
            scaled5 = tf.get_variable('sd5', [512], initializer=tf.constant_initializer(1.0))
            betad5 = tf.get_variable('bed5', [512], initializer=tf.constant_initializer(0.0))

            conv5 = tf.nn.conv2d(lr4, wd5, strides=[1, 2, 2, 1], padding="SAME", name='D_conv5') + bd5
            # batch normalization
            batch_mean5, batch_var5 = tf.nn.moments(conv5, [0, 1, 2])
            bn5 = tf.nn.batch_normalization(conv5, batch_mean5, batch_var5, betad5, scaled5, 0.0001, name='D_BN5')
            # gaussian noise
            gn5 = gaussian_noise(bn5, 0.3)
            # leakyReLU function
            lr5 = leaky_relu(gn5, alpha=0.2)
            # reshape [n, 4, 4, 512] -> nx4x4x512
            re5 = tf.reshape(lr5, [-1, 4 * 4 * 512])

        with tf.name_scope("D_layer6"): # layer6 linear
            wd6 = tf.get_variable('wd6', [512 * 4 * 4, 1], initializer=tf.random_normal_initializer
            (mean=0.0, stddev=0.02, seed=SEED), dtype=tf.float32)
            bd6 = tf.get_variable('bd6', [1], initializer=tf.constant_initializer(0.0))

            fc6 = tf.matmul(re5, wd6, name='G_fc6') + bd6

        # norm_L2 = tf.nn.l2_loss(wd1) + tf.nn.l2_loss(wd2) + tf.nn.l2_loss(wd3) + tf.nn.l2_loss(wd4) + tf.nn.l2_loss(wd5) \
        #           + tf.nn.l2_loss(wd6)

        # return out_dis, norm_L2
        return fc6


z_ = tf.placeholder(tf.float32, [None, NOISE_UNIT_NUM], name='z_') #noise to generator
x_ = tf.placeholder(tf.float32, [None, 64, 64, 3], name='x_') #image to classifier
d_dis_g_ = tf.placeholder(tf.float32, [None, 1], name='d_dis_g_') #target of discriminator related to generator
d_dis_r_ = tf.placeholder(tf.float32, [None, 1], name='d_dis_r_') #target of discriminator related to real image


# stream around generator
x_gen = generator(z_, reuse=False)

#stream around discriminator
out_dis_g = discriminator(x_gen, reuse=False) #from generator
out_dis_r = discriminator(x_, reuse=True) #real image

with tf.name_scope("loss"):
    loss_dis_g = tf.reduce_mean(tf.square(out_dis_g - d_dis_g_), name='Loss_dis_gen') #loss related to generator
    loss_dis_r = tf.reduce_mean(tf.square(out_dis_r - d_dis_r_), name='Loss_dis_rea') #loss related to real imaeg
    #total loss of discriminator
    loss_dis_total =  loss_dis_g + loss_dis_r

    #total loss of generator
    loss_gen_total = loss_dis_g * 2.0


tf.summary.scalar('loss_dis_total', loss_dis_total)
tf.summary.scalar('loss_gen_total', loss_gen_total)
merged = tf.summary.merge_all()

# t_vars = tf.trainable_variables()
g_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="generator")
d_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="discriminator")
with tf.name_scope("train"):
    train_dis = tf.train.AdamOptimizer(learning_rate=0.0001, beta1=0.5).minimize(loss_dis_total, var_list=d_vars
                                        # var_list=[wd1, wd2, wd3, wd4, wd5, wd6, bd1, bd2, bd3, bd4, bd5, bd6]
                                                                                , name='Adam_dis')
    train_gen = tf.train.AdamOptimizer(learning_rate=0.0001, beta1=0.5).minimize(loss_gen_total, var_list=g_vars
                                        # var_list=[wg1, wg3, wg5, bg1, bg3, bg5, betag2, scaleg2, betag4, scaleg4]
                                                                                , name='Adam_gen')

sess = tf.Session()
sess.run(tf.global_variables_initializer())

summary_writer = tf.summary.FileWriter(BOARD_DIR_NAME, sess.graph)

#training loop
for epoch in range(0, EPOCH):
    sum_loss_gen = np.float32(0)
    sum_loss_dis = np.float32(0)
    sum_loss_dis_r = np.float32(0)
    sum_loss_dis_g = np.float32(0)


    len_data = make_datasets.make_data_for_1_epoch()

    for i in range(0, len_data, BATCH_SIZE):
        img_batch = make_datasets.get_data_for_1_batch(i, BATCH_SIZE)
        z = make_datasets.make_random_z_with_norm(NOISE_MEAN, NOISE_STDDEV, len(img_batch), NOISE_UNIT_NUM)
        tar_g_1 = make_datasets.make_target_1_0(1.0, len(img_batch))
        tar_g_0 = make_datasets.make_target_1_0(0.0, len(img_batch))

        #train discriminator
        sess.run(train_dis, feed_dict={z_:z, x_: img_batch, d_dis_g_: tar_g_0, d_dis_r_: tar_g_1})

        #train generator
        sess.run(train_gen, feed_dict={z_:z, d_dis_g_: tar_g_1})

        loss_gen_total_ = sess.run(loss_gen_total, feed_dict={z_: z, d_dis_g_: tar_g_1})

        loss_dis_total_, loss_dis_r_, loss_dis_g_ = sess.run([loss_dis_total, loss_dis_r, loss_dis_g],
                                                feed_dict={z_:z, x_: img_batch, d_dis_g_: tar_g_0, d_dis_r_: tar_g_1})

        #for tensorboard
        merged_ = sess.run(merged, feed_dict={z_:z, x_: img_batch, d_dis_g_: tar_g_0, d_dis_r_: tar_g_1})

        summary_writer.add_summary(merged_, epoch)

        sum_loss_gen += loss_gen_total_
        sum_loss_dis += loss_dis_total_
        sum_loss_dis_r += loss_dis_r_
        sum_loss_dis_g += loss_dis_g_

    print("----------------------------------------------------------------------")
    print("epoch = {:}, Generator Total Loss = {:.4f}, Discriminator Total Loss = {:.4f}".format(
        epoch, sum_loss_gen / len_data, sum_loss_dis / len_data))
    print("Discriminator Real Loss = {:.4f}, Discriminator Generated Loss = {:.4f}".format(
        sum_loss_dis_r / len_data, sum_loss_dis_g / len_data))

    if epoch % 10 == 0:
        z_test = make_datasets.initial_noise
        gen_images = sess.run(x_gen, feed_dict={z_:z_test})
        Utility.make_output_img(gen_images, int(TEST_DATA_SAMPLE ** 0.5) ,out_image_dir, epoch, LOGFILE_NAME)


