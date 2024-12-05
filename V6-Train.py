import argparse
import os
import itertools
import shutil
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow import keras
import matplotlib.pyplot as plt 
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import RandomFlip, RandomRotation, RandomZoom, Rescaling
from tensorflow.keras.regularizers import l2
from keras import models,losses
from keras import layers
import datetime
from tqdm import tqdm
import signal
import sys

# Define the argument parser
parser = argparse.ArgumentParser(description='CCT Model with RL Code')
parser.add_argument('--custom_dataset_path', type=str, help='Path to the custom dataset folder')
parser.add_argument('--filename', type=str, help='Name of the folder')

args = parser.parse_args()

# Get the custom dataset path and filename from command-line arguments
custom_dataset_path = args.custom_dataset_path
filename = args.filename

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.set_logical_device_configuration(
            gpus[0],
            [tf.config.LogicalDeviceConfiguration(memory_limit=9216)]
        )
        logical_gpus = tf.config.list_logical_devices('GPU')
        # print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        pass
        # print(e)

# Configuration parameters
num_epochs = 200
positional_emb = True
conv_layers = 2
projection_dim = 128
num_heads = 3
transformer_units = [projection_dim, projection_dim]
transformer_layers = 2
stochastic_depth_rate = 0.1
learning_rate = 0.04
weight_decay = 0.005
batch_size = 64
image_size = (120, 120)
num_classes = 2  # Number of classes in the custom dataset
input_shape = (120, 120, 3)

# Load custom dataset
train_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    custom_dataset_path,
    image_size=image_size,
    batch_size=batch_size,
    labels="inferred",
    label_mode="categorical",  # Use categorical labels for multi-class classification
    class_names=["NORMAL", "GLAUCOMA"],  # List your class names here
    seed=1337,
    validation_split=0.2,
    subset="training"
)
val_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    custom_dataset_path,
    image_size=image_size,
    batch_size=batch_size,
    labels="inferred",
    label_mode="categorical",
    class_names=["NORMAL", "GLAUCOMA"],
    seed=1337,
    validation_split=0.2,
    subset="validation"
)

# Data augmentation
data_augmentation = keras.Sequential(
    [
        layers.Rescaling(scale=1.0 / 255),
        layers.RandomCrop(image_size[0], image_size[1]),
        layers.RandomFlip("horizontal"),
        RandomFlip("horizontal_and_vertical"),
        RandomRotation(0.1),
        RandomZoom(0.1),
    ],
    name="data_augmentation",
)

# # Function to extract images and labels from the dataset
# def get_images_and_labels(dataset):
#     images = []
#     labels = []
#     for img, lbl in dataset:
#         images.append(img.numpy())
#         labels.append(lbl.numpy())
#     return np.concatenate(images), np.concatenate(labels)

# # Extract images and labels from the datasets
# x_train, y_train = get_images_and_labels(train_dataset)
# x_val, y_val = get_images_and_labels(val_dataset)

# # Normalize the images
# x_train = x_train.astype("float32") / 255.0
# x_val = x_val.astype("float32") / 255.0

# # If using binary labels for binary classification
# y_train = y_train[:, 1]
# y_val = y_val[:, 1]

# x_train = x_train.shuffle().batch(batch_size = batch_size)
# x_val = x_val.batch(batch_size = batch_size)

# print(f"x_train shape: {x_train.shape}, y_train shape: {y_train.shape}")
# print(f"x_val shape: {x_val.shape}, y_val shape: {y_val.shape}")


# TOKENIZER

class CCTTokenizer(layers.Layer):
    def __init__(
        self,
        kernel_size=3,
        stride=1,
        padding=1,
        pooling_kernel_size=3,
        pooling_stride=2,
        num_conv_layers=conv_layers,
        num_output_channels=[64, 128],
        positional_emb=positional_emb,
        *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.pooling_kernel_size = pooling_kernel_size
        self.pooling_stride = pooling_stride
        self.num_conv_layers = num_conv_layers
        self.num_output_channels = num_output_channels
        self.positional_emb = positional_emb
        self.conv_model = keras.Sequential()
        for i in range(num_conv_layers):
            self.conv_model.add(
                layers.Conv2D(
                    num_output_channels[i],
                    kernel_size,
                    stride,
                    padding="valid",
                    use_bias=False,
                    activation="relu",
                    kernel_initializer="he_normal",
                )
            )
            self.conv_model.add(layers.ZeroPadding2D(padding))
            self.conv_model.add(
                layers.MaxPooling2D(pooling_kernel_size, pooling_stride, "same")
            )
        self.positional_emb = positional_emb

    def call(self, images, training = None):
        outputs = self.conv_model(images)
        reshaped = tf.reshape(
            outputs,
            (
                -1,
                outputs.shape[1] * outputs.shape[2],
                outputs.shape[-1],
            ),
        )
        return reshaped
    def get_config(self):
        config = super().get_config()
        config.update({
            "kernel_size": self.kernel_size,
            "stride": self.stride,
            "padding": self.padding,
            "pooling_kernel_size": self.pooling_kernel_size,
            "pooling_stride": self.pooling_stride,
            "num_conv_layers": self.num_conv_layers,
            "num_output_channels": self.num_output_channels,
            "positional_emb": self.positional_emb,
        })
        return config
# positional_emb

class PositionEmbedding(keras.layers.Layer):
    def __init__(
        self,
        sequence_length,
        initializer="glorot_uniform",
        *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        if sequence_length is None:
            raise ValueError("`sequence_length` must be an Integer, received `None`.")
        self.sequence_length = int(sequence_length)
        self.initializer = keras.initializers.get(initializer)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "sequence_length": self.sequence_length,
                "initializer": keras.initializers.serialize(self.initializer),
            }
        )
        return config

    def build(self, input_shape):
        feature_size = input_shape[-1]
        self.position_embeddings = self.add_weight(
            name="embeddings",
            shape=[self.sequence_length, feature_size],
            initializer=self.initializer,
            trainable=True,
        )
        super().build(input_shape)

    def call(self, inputs, start_index=0,training = None):
        shape = tf.shape(inputs)
        feature_length = shape[-1]
        sequence_length = shape[-2]
        position_embeddings = tf.convert_to_tensor(self.position_embeddings)
        position_embeddings = tf.slice(
            position_embeddings,
            (start_index, 0),
            (sequence_length, feature_length),
        )
        return tf.broadcast_to(position_embeddings, shape)

    def compute_output_shape(self, input_shape):
        return input_shape

# sequence pooling

class SequencePooling(layers.Layer):
    def __init__(self,*args, **kwargs):
        super().__init__(*args, **kwargs)
        self.attention = layers.Dense(1)

    def call(self, x, training = None ):
        attention_weights = tf.keras.activations.softmax(self.attention(x), axis=1)
        attention_weights = tf.transpose(attention_weights, perm=(0, 2, 1))
        weighted_representation = tf.matmul(attention_weights, x)
        return tf.squeeze(weighted_representation, -2)

# stocastic depth 

class StochasticDepth(layers.Layer):
    def __init__(self, drop_prop,*args, **kwargs):
        super().__init__(*args, **kwargs)
        self.drop_prob = drop_prop
        self.seed_generator = tf.random.Generator.from_seed(1337)

    def call(self, x, training=None):
        if training:
            keep_prob = 1 - self.drop_prob
            shape = (tf.shape(x)[0],) + (1,) * (len(x.shape) - 1)
            random_tensor = keep_prob + self.seed_generator.uniform(
                shape, 0, 1)
            random_tensor = tf.floor(random_tensor)
            return (x / keep_prob) * random_tensor
        return x
    
    def get_config(self):
        config = super().get_config()
        config.update({'drop_prob': self.drop_prob})
        return config

# MLP for transformers

def mlp(x, hidden_units, dropout_rate):
    for units in hidden_units:
        x = layers.Dense(units, activation=tf.keras.activations.gelu,kernel_regularizer=l2(0.001))(x)
        x = layers.Dropout(dropout_rate)(x)
    return x

# Define CCT model
def create_cct_model(
    image_size=image_size,
    input_shape=input_shape,
    num_heads=num_heads,
    projection_dim=projection_dim,
    transformer_units=transformer_units,
):
    inputs = layers.Input(input_shape)

    # Augment data.
    augmented = data_augmentation(inputs)

    # Encode patches.
    cct_tokenizer = CCTTokenizer()
    encoded_patches = cct_tokenizer(augmented)

    # Apply positional embedding.
    if positional_emb:
        sequence_length = encoded_patches.shape[1]
        encoded_patches += PositionEmbedding(sequence_length=sequence_length)(
            encoded_patches
        )

    # Calculate Stochastic Depth probabilities.
    dpr = [x for x in np.linspace(0, stochastic_depth_rate, transformer_layers)]

    # Create multiple layers of the Transformer block.
    for i in range(transformer_layers):
        # Layer normalization 1.
        x1 = layers.LayerNormalization(epsilon=1e-5)(encoded_patches)

        # Create a multi-head attention layer.
        attention_output = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=projection_dim, dropout=0.1
        )(x1, x1)

        # Skip connection 1.
        attention_output = StochasticDepth(dpr[i])(attention_output)
        x2 = layers.Add()([attention_output, encoded_patches])

        # Layer normalization 2.
        x3 = layers.LayerNormalization(epsilon=1e-5)(x2)

        # MLP.
        x3 = mlp(x3, hidden_units=transformer_units, dropout_rate=0.1)

        # Skip connection 2.
        x3 = StochasticDepth(dpr[i])(x3)
        encoded_patches = layers.Add()([x3, x2])

    # Apply sequence pooling.
    representation = layers.LayerNormalization(epsilon=1e-5)(encoded_patches)
    weighted_representation = SequencePooling()(representation)

    # Classify outputs.
    # logits = layers.Dense(num_classes)(weighted_representation)
    logits = layers.Dense(256)(weighted_representation)

    # Create the Keras model.
    model = keras.Model(inputs=inputs, outputs=logits)
    return model

class ActorCritic(tf.keras.Model):
    def __init__(self, state_dim, action_dim, hidden_dim):
        super(ActorCritic, self).__init__()
        self.actor = models.Sequential([
            layers.Dense(hidden_dim, activation='relu'),
            layers.Dropout(0.01),
            layers.Dense(hidden_dim, activation='relu'),
            layers.Dropout(0.01),
            layers.Dense(action_dim, activation='softmax')
        ])
        self.critic = models.Sequential([
            layers.Dense(hidden_dim, activation='relu'),
            layers.Dropout(0.01),
            layers.Dense(hidden_dim, activation='relu'),
            layers.Dropout(0.01),
            layers.Dense(2)
        ])
    
    def call(self, inputs):
        policy = self.actor(inputs)
        inputs = tf.cast(inputs, tf.float32)
        inputs2 = tf.concat([policy, inputs], axis=1)
        value = self.critic(inputs2)
        return policy, value


class IntegratedModel(tf.keras.Model):
    def __init__(self, cvt, actor_critic):
        super(IntegratedModel, self).__init__()
        self.cvt = cvt
        self.actor_critic = actor_critic
    
    def call(self, inputs):
        embeddings = self.cvt(inputs)
        policy, value = self.actor_critic(embeddings)
        return policy, value
    
@tf.function
def compute_loss(logits, labels, values):
    value_loss = losses.MeanSquaredError()(values, labels)
    policy_loss = losses.CategoricalCrossentropy(from_logits=True)(logits, labels) 
    return value_loss + policy_loss


@tf.function
def train_step(model,optimizer,inputs,labels):
    with tf.GradientTape() as tape:
        policy, value = model(inputs)
        loss = compute_loss(policy,labels,value)
    grads = tape.gradient(loss,model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return loss,policy

import tensorflow as tf

class F1Score(tf.keras.metrics.Metric):
    def __init__(self, num_classes, name='f1_score', **kwargs):
        super(F1Score, self).__init__(name=name, **kwargs)
        self.num_classes = num_classes
        self.precision = [tf.keras.metrics.Precision() for _ in range(num_classes)]
        self.recall = [tf.keras.metrics.Recall() for _ in range(num_classes)]

    def update_state(self, y_true, y_pred, sample_weight=None):
        for i in range(self.num_classes):
            self.precision[i].update_state(y_true[:, i], y_pred[:, i])
            self.recall[i].update_state(y_true[:, i], y_pred[:, i])

    def result(self):
        precision_values = [precision.result() for precision in self.precision]
        recall_values = [recall.result() for recall in self.recall]
        
        precision = tf.reduce_mean(precision_values)
        recall = tf.reduce_mean(recall_values)
        
        f1_score = (2 * precision * recall) / (precision + recall + 1e-10)
        return f1_score

    def reset_states(self):
        for precision in self.precision:
            precision.reset_states()
        for recall in self.recall:
            recall.reset_states()


import csv

def train_model(model, train_dataset, val_dataset, epochs, optimizer, csv_file_path):
    with open(csv_file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Epoch', 'Loss', 'Train Accuracy', 'Validation Accuracy', 'Precision', 'Recall', 'F1 Score', 'TP', 'TN', 'FP', 'FN'])
        
        for epoch in range(epochs):
            epoch_loss_avg = tf.keras.metrics.Mean()
            train_accuracy = tf.keras.metrics.CategoricalAccuracy()
            val_accuracy = tf.keras.metrics.CategoricalAccuracy()
            precision = tf.keras.metrics.Precision()
            recall = tf.keras.metrics.Recall()
            f1_score = F1Score(num_classes=num_classes)
            tp = tf.keras.metrics.TruePositives()
            tn = tf.keras.metrics.TrueNegatives()
            fp = tf.keras.metrics.FalsePositives()
            fn = tf.keras.metrics.FalseNegatives()

            progress_bar = tqdm(train_dataset, desc=rf'Epoch {epoch+1}/{epochs}')

            for step, (x_batch_train, y_batch_train) in enumerate(progress_bar):
                loss, policy = train_step(model, optimizer, x_batch_train, y_batch_train)
                epoch_loss_avg.update_state(loss)
                train_accuracy.update_state(y_batch_train, policy)
                progress_bar.set_postfix(loss=epoch_loss_avg.result().numpy(), train_accuracy=train_accuracy.result().numpy())

            # Validation step
            for x_batch_val, y_batch_val in val_dataset:
                val_policy, _ = model(x_batch_val)
                val_accuracy.update_state(y_batch_val, val_policy)
                precision.update_state(y_batch_val, val_policy)
                recall.update_state(y_batch_val, val_policy)
                f1_score.update_state(y_batch_val, val_policy)
                tp.update_state(y_batch_val, val_policy)
                tn.update_state(y_batch_val, val_policy)
                fp.update_state(y_batch_val, val_policy)
                fn.update_state(y_batch_val, val_policy)

            # Print metrics
            # print(f'Epoch {epoch+1}, Loss: {epoch_loss_avg.result().numpy()}, '
            #       f'Train Accuracy: {train_accuracy.result().numpy() * 100:.2f}%, '
            #       f'Validation Accuracy: {val_accuracy.result().numpy() * 100:.2f}%, '
            #       f'Precision: {precision.result().numpy()}, '
            #       f'Recall: {recall.result().numpy()}, '
            #       f'F1 Score: {f1_score.result().numpy()}, '
            #       f'TP: {tp.result().numpy()}, '
            #       f'TN: {tn.result().numpy()}, '
            #       f'FP: {fp.result().numpy()}, '
            #       f'FN: {fn.result().numpy()}')
            
            # Write metrics to CSV file
            writer.writerow([
                epoch + 1,
                epoch_loss_avg.result().numpy(),
                train_accuracy.result().numpy() * 100,
                val_accuracy.result().numpy() * 100,
                precision.result().numpy(),
                recall.result().numpy(),
                f1_score.result().numpy(),
                tp.result().numpy(),
                tn.result().numpy(),
                fp.result().numpy(),
                fn.result().numpy()
            ])

import signal
import sys

if __name__ == '__main__':
    # Create CCT model
    cvt = create_cct_model()
    
    # Create ActorCritic model
    actor_critic = ActorCritic(state_dim=projection_dim, action_dim=num_classes, hidden_dim=128)
    
    # Integrated model
    model = IntegratedModel(cvt, actor_critic)
    
    # Model summary
    # cvt.summary()
    
    # Optimizer
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)

    weights_folder = r"C:\Sambhav\Manipal\Year 2\3rd Semester\Glaucoma Detection\CCT\AyushRLModel\Results"
    wwf = weights_folder + "\\" + filename 

    try:
        # Train model
        train_model(model, train_dataset, val_dataset, epochs=num_epochs, optimizer=optimizer, csv_file_path=f"C:\\Sambhav\\Manipal\\Year 2\\3rd Semester\\Glaucoma Detection\\CCT\\AyushRLModel\\Results\\{filename}\\Metrics_{filename}.csv")
    except KeyboardInterrupt:
        print('Training interrupted by Ctrl+C')
        # Save the model weights
        # model.save_weights(os.path.join(weights_folder, f"{filename}_weights.keras"))
        tf.saved_model.save(model, os.path.join(wwf, f"{filename}_saved_model"))
        
    # Save the model in TensorFlow SavedModel format
    # model.save_weights(os.path.join(weights_folder, f"{filename}_weights.keras"))
  # Save weights separately
    tf.saved_model.save(model, os.path.join(wwf, f"{filename}_saved_model"))  # Save the entire model

# Define the signal handler to save the model weights on interruption
def signal_handler(sig, frame):
    print('You pressed Ctrl+C!')
    # Save the model weights
    # model.save_weights(os.path.join(weights_folder, f"{filename}_weights.keras"))
    tf.saved_model.save(model, os.path.join(wwf, f"{filename}_saved_model"))
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)