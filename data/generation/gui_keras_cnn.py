import tensorflow as tf
import tensorflow.keras as keras


number_layers = 5
filter_sizes = [[5,5],[2,2]]*(number_layers//2) + [[5,5]]*(number_layers%2)
padding = ["VALID", "SAME"]*(number_layers//2) + ["VALID"]*(number_layers%2)
strides = [[1, 1]] * number_layers
pooling_in_layer = [1] * number_layers
activations = ["relu"]*number_layers
image_height,image_width,number_channels=32,32,3
num_filters=1024
activations_dict = {"relu":tf.nn.relu, "tanh":tf.nn.tanh}
sigmaw = 1.41
sigmab = 0
weight_initializer = keras.initializers.VarianceScaling(scale=sigmaw**2, mode='fan_in', distribution='normal', seed=None)
bias_initializer = keras.initializers.RandomNormal(stddev=sigmab)

intermediate_pooling_layer = [keras.layers.AvgPool2D(pool_size=2, padding='same')]
pooling_layer = [keras.layers.GlobalAveragePooling2D()]

model = keras.Sequential(
            sum([
                [keras.layers.Conv2D(input_shape=(image_height,image_width,number_channels) if index==0 else (None,), \
                    filters=num_filters, \
                    kernel_size=filter_size, \
                    padding=padding, \
                    strides=strides, \
                    activation=activations_dict[activation],
                data_format='channels_last',
                kernel_initializer=weight_initializer,
                bias_initializer=bias_initializer,)] +
                 (intermediate_pooling_layer if have_pooling else [])
                for index,(filter_size,padding,strides,have_pooling,activation) in enumerate(zip(filter_sizes,padding,strides,pooling_in_layer,activations))
            ],[])
            #+ pooling_layer
            #+ [ keras.layers.Flatten() ]
            #+ [
            #    # keras.layers.Dense(1,activation=tf.nn.sigmoid,)
            #    keras.layers.Dense(1,#activation=tf.nn.sigmoid,)
            #    kernel_initializer=weight_initializer,
            #    bias_initializer=bias_initializer,)
            #    ])
            )


if __name__ == "__main__":
    x = tf.random.normal([1, 32, 32, 3])
    with tf.compat.v1.Session() as sess:
        y = model(x)
    print(y.shape)
