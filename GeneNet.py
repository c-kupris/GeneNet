# imports
import csv
import keras  # version 3.4.0

# requires tensorflow 2.17.0rc0

#  *Create a CSV file dataset.*
#  *The dataset should contain the following columns:*
#  *- Gene name*
#  *- Gene description*
#  *- Is the gene methylated? Is the gene acetylated? *
#  *- mRNA? Signaling molecules? *
#  *- Gene location*
#  *- Gene expression

#  Open relevant CSV file to write to.
open_csv = csv.reader(open('gene_net_dataset.csv', 'w'))

#  Write data to the open file.

#  Save the CSV file.

#  Close the CSV file.

# parameters
filters = 50
kernel_size = (3, 3, 3)
strides = (1, 1, 1)

# Create the layers for the GeneNet model.

# Input layer.
input_layer = keras.layers.InputLayer(shape=(100000, 100000, 100000, 100000))

# Layer one.
conv_layer_one = keras.layers.Conv3D(filters=filters,
                                     kernel_size=kernel_size,
                                     strides=strides)
batch_norm_layer_one = keras.layers.BatchNormalization()
avg_pool_layer_one = keras.layers.AveragePooling3D(pool_size=(2, 2, 2))

# Layer two.
conv_layer_two = keras.layers.Conv3D(filters=filters,
                                     kernel_size=kernel_size,
                                     strides=strides)
batch_norm_layer_two = keras.layers.BatchNormalization()
avg_pool_layer_two = keras.layers.AveragePooling3D(pool_size=(2, 2, 2))

# Layer three.
conv_layer_three = keras.layers.Conv3D(filters=filters,
                                       kernel_size=kernel_size,
                                       strides=strides)
batch_norm_layer_three = keras.layers.BatchNormalization()
avg_pool_layer_three = keras.layers.AveragePooling3D(pool_size=(2, 2, 2))

# Layer four.
conv_layer_four = keras.layers.Conv3D(filters=filters,
                                      kernel_size=kernel_size,
                                      strides=strides)
batch_norm_layer_four = keras.layers.BatchNormalization()
avg_pool_layer_four = keras.layers.AveragePooling3D(pool_size=(2, 2, 2))

# Layer five.
conv_layer_five = keras.layers.Conv3D(filters=filters,
                                      kernel_size=kernel_size,
                                      strides=strides)
batch_norm_layer_five = keras.layers.BatchNormalization()
avg_pool_layer_five = keras.layers.AveragePooling3D(pool_size=(2, 2, 2))

# Layer six.
conv_layer_six = keras.layers.Conv3D(filters=filters,
                                     kernel_size=kernel_size,
                                     strides=strides)
batch_norm_layer_six = keras.layers.BatchNormalization()
avg_pool_layer_six = keras.layers.AveragePooling3D(pool_size=(2, 2, 2))

# Layer seven.
conv_layer_seven = keras.layers.Conv3D(filters=filters,
                                       kernel_size=kernel_size,
                                       strides=strides)
batch_norm_layer_seven = keras.layers.BatchNormalization()
avg_pool_layer_seven = keras.layers.AveragePooling3D(pool_size=(2, 2, 2))

# Layer eight.
conv_layer_eight = keras.layers.Conv3D(filters=filters,
                                       kernel_size=kernel_size,
                                       strides=strides)
batch_norm_layer_eight = keras.layers.BatchNormalization()
avg_pool_layer_eight = keras.layers.AveragePooling3D(pool_size=(2, 2, 2))

# Layer nine.
conv_layer_nine = keras.layers.Conv3D(filters=filters,
                                      kernel_size=kernel_size,
                                      strides=strides)
batch_norm_layer_nine = keras.layers.BatchNormalization()
avg_pool_layer_nine = keras.layers.AveragePooling3D(pool_size=(2, 2, 2))

# Layer ten.
conv_layer_ten = keras.layers.Conv3D(filters=filters,
                                     kernel_size=kernel_size,
                                     strides=strides)
batch_norm_layer_ten = keras.layers.BatchNormalization()
avg_pool_layer_ten = keras.layers.AveragePooling3D(pool_size=(2, 2, 2))

# Layer eleven.
conv_layer_eleven = keras.layers.Conv3D(filters=filters,
                                        kernel_size=kernel_size,
                                        strides=strides)
batch_norm_layer_eleven = keras.layers.BatchNormalization()
avg_pool_layer_eleven = keras.layers.AveragePooling3D(pool_size=(2, 2, 2))

# Layer twelve.
conv_layer_twelve = keras.layers.Conv3D(filters=filters,
                                        kernel_size=kernel_size,
                                        strides=strides)
batch_norm_layer_twelve = keras.layers.BatchNormalization()
avg_pool_layer_twelve = keras.layers.AveragePooling3D(pool_size=(2, 2, 2))

model = keras.Sequential(layers=[input_layer,
                                 conv_layer_one, batch_norm_layer_one, avg_pool_layer_one,
                                 conv_layer_two, batch_norm_layer_two, avg_pool_layer_two,
                                 conv_layer_three, batch_norm_layer_three, avg_pool_layer_three,
                                 conv_layer_four, batch_norm_layer_four, avg_pool_layer_four,
                                 conv_layer_five, batch_norm_layer_five, avg_pool_layer_five,
                                 conv_layer_six, batch_norm_layer_six, avg_pool_layer_six,
                                 conv_layer_seven, batch_norm_layer_seven, avg_pool_layer_seven,
                                 conv_layer_eight, batch_norm_layer_eight, avg_pool_layer_eight,
                                 conv_layer_nine, batch_norm_layer_nine, avg_pool_layer_nine,
                                 conv_layer_ten, batch_norm_layer_ten, avg_pool_layer_ten,
                                 conv_layer_eleven, batch_norm_layer_eleven, avg_pool_layer_eleven,
                                 conv_layer_twelve, batch_norm_layer_twelve, avg_pool_layer_twelve])

# Summarize the model.
model.summary()
