import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Input
import numpy as np
import tensorflow_addons as tfa

def Unet(
        num_dim,
        time_embedding,
        input_embedding_dims,
        stride,
        kernel,
        block_depth,
        widths,
        attentions,
        pad=((2,1),(0,0),(4,3)),
        use_1D=False,
):
    act = tf.keras.activations.swish
    
    def ResidualBlock(width, attention):
        def forward(x):
            x , n = x
            input_width = x.shape[2] if use_1D else x.shape[4]
            if input_width == width:
                residual = x
            else:
                if use_1D:
                    residual = layers.Conv1D(width, kernel_size=1)(x)
                else:
                    residual = layers.Conv3D(width, kernel_size=1)(x)

            n = layers.Dense(width)(n)
            x = act(x)
            if use_1D:
                x = layers.Conv1D(width, kernel_size=kernel, padding="same")(x)
            else:
                x = layers.Conv3D(width, kernel_size=kernel, padding="same")(x)
            x = layers.Add()([x, n])
            x = act(x)
            if use_1D:
                x = layers.Conv1D(width, kernel_size=kernel, padding="same")(x)
            else:
                x = layers.Conv3D(width, kernel_size=kernel, padding="same")(x)
            x = layers.Add()([residual, x])

            if attention:
                residual = x
                if use_1D:                    
                    x = tf.keras.layers.LayerNormalization(epsilon=1e-6, center=False, scale=False)(x)
                    x = layers.MultiHeadAttention(
                        num_heads=1, key_dim=width, attention_axes=(1)
                    )(x, x)
                else:
                    x = tfa.layers.GroupNormalization(groups=4, center=False, scale=False)(x)
                    x = layers.MultiHeadAttention(
                        num_heads=1, key_dim=width, attention_axes=(1, 2, 3)
                    )(x, x)

                x = layers.Add()([residual, x])
            return x
        return forward

    def DownBlock(block_depth, width, attention):
        def forward(x):
            x, n, skips = x
            for _ in range(block_depth):
                x = ResidualBlock(width, attention)([x,n])
                skips.append(x)        
            if use_1D:
                x = layers.AveragePooling1D(pool_size=stride)(x)
            else:
                x = layers.AveragePooling3D(pool_size=stride)(x)

            return x
        return forward

    def smart_crop_1d(x, target_length):
        """Inteligently crop x to target_length, preserving the center"""
        current_length = tf.shape(x)[1]
        diff = current_length - target_length
        
        def crop_tensor():
            start = diff // 2
            return x[:, start:start + target_length, :]
        
        def return_as_is():
            return x
        
        return tf.cond(tf.greater(diff, 0), crop_tensor, return_as_is)

    def smart_pad_1d(x, target_length):
        """Intelligently pad x to target_length"""
        current_length = tf.shape(x)[1]
        diff = target_length - current_length
        
        def pad_tensor():
            pad_left = diff // 2
            pad_right = diff - pad_left
            return tf.pad(x, [[0, 0], [pad_left, pad_right], [0, 0]])
        
        def return_as_is():
            return x
        
        return tf.cond(tf.greater(diff, 0), pad_tensor, return_as_is)

    def match_dimensions_1d(x, target):
        """Make x match target's sequence length"""
        x_len = tf.shape(x)[1]
        target_len = tf.shape(target)[1]
        
        # First pad if necessary, then crop if necessary
        x = smart_pad_1d(x, target_len)
        x = smart_crop_1d(x, target_len)
        
        return x

    def smart_crop_3d(x, target_shape):
        """Intelligently crop 3D tensor to target shape"""
        current_shape = tf.shape(x)
        
        result = x
        for axis in range(1, 4):  # spatial dimensions
            current_dim = current_shape[axis]
            target_dim = target_shape[axis]
            diff = current_dim - target_dim
            
            def crop_axis():
                start = diff // 2
                slices = [slice(None)] * 5
                slices[axis] = slice(start, start + target_dim)
                return result[tuple(slices)]
            
            def keep_axis():
                return result
            
            result = tf.cond(tf.greater(diff, 0), crop_axis, keep_axis)
            
        return result

    def smart_pad_3d(x, target_shape):
        """Intelligently pad 3D tensor to target shape"""
        current_shape = tf.shape(x)
        
        result = x
        for axis in range(1, 4):  # spatial dimensions
            current_dim = current_shape[axis]
            target_dim = target_shape[axis]
            diff = target_dim - current_dim
            
            def pad_axis():
                pad_left = diff // 2
                pad_right = diff - pad_left
                pad_spec = [[0, 0] for _ in range(5)]
                pad_spec[axis] = [pad_left, pad_right]
                return tf.pad(result, pad_spec)
            
            def keep_axis():
                return result
            
            result = tf.cond(tf.greater(diff, 0), pad_axis, keep_axis)
            
        return result

    def match_dimensions_3d(x, target):
        """Make x match target's spatial dimensions"""
        target_shape = tf.shape(target)
        
        # First pad if necessary, then crop if necessary
        x = smart_pad_3d(x, target_shape)
        x = smart_crop_3d(x, target_shape)
        
        return x

    def UpBlock(block_depth, width, attention):
        def forward(inp):
            x, n, skips = inp
            
            if use_1D:
                x = layers.UpSampling1D(size=stride)(x)
            else:
                x = layers.UpSampling3D(size=stride)(x)
            
            # Get skip connection and match dimensions
            skip = skips.pop()
            if use_1D:
                x = layers.Lambda(lambda inputs: match_dimensions_1d(inputs[0], inputs[1]))([x, skip])
            else:
                x = layers.Lambda(lambda inputs: match_dimensions_3d(inputs[0], inputs[1]))([x, skip])
            
            # Concatenate matched tensors
            x = layers.Concatenate(axis=-1)([x, skip])

            for _ in range(block_depth):
                x = ResidualBlock(width, attention)([x, n])
            return x
        return forward

    # Build the network
    inputs = keras.Input(num_dim)
    
    # Store original input shape for final output matching
    if use_1D:
        original_length = num_dim[0] if isinstance(num_dim[0], int) else None
        x = layers.Conv1D(input_embedding_dims, kernel_size=1)(inputs)
        n = layers.Reshape((1, time_embedding.shape[-1]))(time_embedding)
    else:
        inputs_padded = layers.ZeroPadding3D(pad)(inputs)
        x = layers.Conv3D(input_embedding_dims, kernel_size=1)(inputs_padded)
        n = layers.Reshape((1, 1, 1, time_embedding.shape[-1]))(time_embedding)
    
    # Encoder
    skips = []
    for width, attention in zip(widths[:-1], attentions[:-1]):
        x = DownBlock(block_depth, width, attention)([x, n, skips])

    # Bottleneck
    for _ in range(block_depth):
        x = ResidualBlock(widths[-1], attentions[-1])([x, n])

    # Decoder
    for width, attention in zip(widths[-2::-1], attentions[-2::-1]):
        x = UpBlock(block_depth, width, attention)([x, n, skips])

    # Output layer
    if use_1D:
        x = layers.Conv1D(1, kernel_size=1, kernel_initializer="zeros")(x)
        # Final dimension matching to ensure output matches input exactly
        outputs = layers.Lambda(lambda inp: match_dimensions_1d(inp[0], inp[1]))([x, inputs])
    else:
        outputs = layers.Conv3D(1, kernel_size=1, kernel_initializer="zeros")(x)
        outputs = layers.Cropping3D(pad)(outputs)

    return inputs, outputs

def Resnet(
        inputs,
        end_dim,
        time_embedding,
        num_embed,
        num_layer = 3,
        mlp_dim=128,
        activation='leakyrelu'
):
    act = layers.LeakyReLU(alpha=0.01)

    def resnet_dense(input_layer, hidden_size):
        layer, time = input_layer
        
        residual = layers.Dense(hidden_size)(layer)
        embed = layers.Dense(2*hidden_size)(time)
        scale, shift = tf.split(embed, 2, -1)
        
        x = act(layer)
        x = layers.Dense(hidden_size)(x)
        x = act((1.0 + scale) * x + shift)
        x = layers.Dropout(.1)(x)
        x = layers.Dense(hidden_size)(x)
        x = layers.Add()([x, residual])
        return x

    embed = act(layers.Dense(mlp_dim)(time_embedding))
    
    layer = layers.Dense(mlp_dim)(inputs)
    for i in range(num_layer-1):
        layer = resnet_dense([layer, embed], mlp_dim)

    outputs = layers.Dense(end_dim)(layer)

    return outputs