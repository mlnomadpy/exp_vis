# ==============================================================================
# models.py -- Model Architectures
# ==============================================================================
import typing as tp
import jax
import jax.numpy as jnp
from flax import nnx
from functools import partial
from nmn.nnx.yatconv import YatConv
from nmn.nnx.yatconv_transpose import YatConvTranspose
from nmn.nnx.nmn import YatNMN

Array = jax.Array

class YatConvBlock(nnx.Module):
    def __init__(self, in_channels: int, out_channels: int, *, dropout_rate: float, pool: bool, rngs: nnx.Rngs):
        self.yat_conv = YatConv(in_channels, out_channels, kernel_size=(3, 3), use_bias=False, padding='SAME', rngs=rngs)
        self.lin_conv = nnx.Conv(out_channels, out_channels, kernel_size=(3, 3), use_bias=False, padding='SAME', rngs=rngs)
        self.dropout = nnx.Dropout(rate=dropout_rate, rngs=rngs)
        self.needs_projection = in_channels != out_channels
        if self.needs_projection:
            self.residual_proj = nnx.Conv(in_channels, out_channels, kernel_size=(1, 1), use_bias=False, padding='SAME', rngs=rngs)
        self.pool = pool
        if self.pool:
            self.avg_pool = partial(nnx.avg_pool, window_shape=(2, 2), strides=(2, 2))
    def __call__(self, x, training: bool):
        residual = x
        y = self.yat_conv(x)
        proj = self.lin_conv(y)
        y = proj
        if self.needs_projection:
            residual = self.residual_proj(residual)
        y = y + residual
        y = self.dropout(y, deterministic=not training)
        if self.pool:
            y = self.avg_pool(y)
        return y

class YatResBlock(nnx.Module):
    def __init__(self, in_channels: int, out_channels: int, *, stride: int = 1, dropout_rate: float = 0.2, rngs: nnx.Rngs):
        self.yat_conv = YatConv(in_channels, out_channels, kernel_size=(3, 3), strides=(stride, stride), use_bias=False, padding='SAME', rngs=rngs)
        self.lin_conv = nnx.Conv(out_channels, out_channels, kernel_size=(3, 3), use_bias=False, padding='SAME', rngs=rngs)
        self.dropout = nnx.Dropout(rate=dropout_rate, rngs=rngs)
        self.needs_projection = in_channels != out_channels or stride != 1
        if self.needs_projection:
            self.residual_proj = nnx.Conv(in_channels, out_channels, kernel_size=(1, 1), strides=(stride, stride), use_bias=False, padding='SAME', rngs=rngs)
    
    def __call__(self, x, training: bool):
        residual = x
        y = self.yat_conv(x)
        y = self.lin_conv(y)
        y = self.dropout(y, deterministic=not training)

        if self.needs_projection:
            residual = self.residual_proj(residual)
        y = y + residual
        return y
# Resnet-18/2
class YatCNN(nnx.Module): 
    def __init__(self, *, num_classes: int, input_channels: int, rngs: nnx.Rngs):
        # Initial convolution
        self.conv1 = nnx.Conv(input_channels, 64, kernel_size=(3, 3), strides=(2, 2), use_bias=False, padding='SAME', rngs=rngs)
        
        # ResNet-18 structure: 4 groups of 2 blocks each
        # Group 1: 64 channels, no stride reduction
        self.layer1_block1 = YatResBlock(64, 64, rngs=rngs)
        # self.layer1_block2 = YatResBlock(64, 64, rngs=rngs)
        
        # Group 2: 128 channels, stride=2 for first block
        self.layer2_block1 = YatResBlock(64, 128, stride=2, rngs=rngs)
        # self.layer2_block2 = YatResBlock(128, 128, rngs=rngs)
        
        # Group 3: 256 channels, stride=2 for first block
        self.layer3_block1 = YatResBlock(128, 256, stride=2, rngs=rngs)
        # self.layer3_block2 = YatResBlock(256, 256, rngs=rngs)
        
        # Group 4: 512 channels, stride=2 for first block
        self.layer4_block1 = YatResBlock(256, 512, stride=2, rngs=rngs)
        # self.layer4_block2 = YatResBlock(512, 512, rngs=rngs)
        
        # Global average pooling and final classification layer
        self.avg_pool = partial(nnx.avg_pool, window_shape=(1, 1), strides=(1, 1))
        self.out_linear = YatNMN(512, num_classes, use_bias=False, rngs=rngs)
        self.rngs = rngs
    
    def __call__(self, x, training: bool = False, return_activations_for_layer: tp.Optional[str] = None, apply_masking: bool = False, mask_ratio: float = 0.75):
        x = self.conv1(x)
        if return_activations_for_layer == 'conv1': return x
        
        # Layer 1
        x = self.layer1_block1(x, training=training)
        # x = self.layer1_block2(x, training=training)
        if return_activations_for_layer == 'layer1': return x
        
        # Layer 2
        x = self.layer2_block1(x, training=training)
        # x = self.layer2_block2(x, training=training)
        if return_activations_for_layer == 'layer2': return x
        
        # Layer 3
        x = self.layer3_block1(x, training=training)
        # x = self.layer3_block2(x, training=training)
        if return_activations_for_layer == 'layer3': return x
        
        # Layer 4
        x = self.layer4_block1(x, training=training)
        # x = self.layer4_block2(x, training=training)
        if return_activations_for_layer == 'layer4': return x
        
        # Global average pooling
        representation = jnp.mean(x, axis=(1, 2))
        if return_activations_for_layer == 'representation': return representation
        
        # Final classification
        x = self.out_linear(representation)
        return x

# Keep the old YatCNN for backward compatibility
class YatCNN_old(nnx.Module):
    def __init__(self, *, num_classes: int, input_channels: int, rngs: nnx.Rngs):
        self.stem = YatConv(input_channels, 64, kernel_size=(3, 3), strides=(1, 1), use_bias=False, padding='SAME', rngs=rngs)

        self.block1 = YatConvBlock(64, 64, dropout_rate=0.2, pool=False, rngs=rngs)
        self.block2 = YatConvBlock(64, 128, dropout_rate=0.2, pool=True, rngs=rngs)
        self.block3 = YatConvBlock(128, 256, dropout_rate=0.2, pool=True, rngs=rngs)
        self.block4 = YatConvBlock(256, 512, dropout_rate=0.2, pool=True, rngs=rngs)
        self.block5 = YatConvBlock(512, 512, dropout_rate=0.2, pool=False, rngs=rngs)
        self.out_linear = YatNMN(512, num_classes, use_bias=False, rngs=rngs)
        self.rngs = rngs
    def __call__(self, x, training: bool = False, return_activations_for_layer: tp.Optional[str] = None, apply_masking: bool = False, mask_ratio: float = 0.75):
        x = self.stem(x)
        x = self.block1(x, training=training)
        if return_activations_for_layer == 'block1': return x
        x = self.block2(x, training=training)
        if return_activations_for_layer == 'block2': return x
        x = self.block3(x, training=training)
        if return_activations_for_layer == 'block3': return x
        x = self.block4(x, training=training)
        if return_activations_for_layer == 'block4': return x
        x = self.block5(x, training=training)
        if return_activations_for_layer == 'bottleneck': return x
        representation = jnp.mean(x, axis=(1, 2))
        if return_activations_for_layer == 'representation': return representation
        x = self.out_linear(representation)
        return x

class Decoder(nnx.Module):
    def __init__(self, *, output_channels: int, rngs: nnx.Rngs):
        self.deconv1 = YatConvTranspose(1024, 256, kernel_size=(3, 3), strides=(2, 2), padding='SAME', rngs=rngs)
        self.deconv2 = YatConvTranspose(256, 128, kernel_size=(3, 3), strides=(2, 2), padding='SAME', rngs=rngs)
        self.deconv3 = YatConvTranspose(128, 64, kernel_size=(3, 3), strides=(2, 2), padding='SAME', rngs=rngs)
        self.deconv4 = YatConvTranspose(64, 32, kernel_size=(3, 3), strides=(2, 2), padding='SAME', rngs=rngs)
        self.final_conv = nnx.Conv(32, output_channels, kernel_size=(3, 3), use_bias=False, padding='SAME', rngs=rngs)
    def __call__(self, x, training: bool = False):
        x = self.deconv1(x)
        x = self.deconv2(x)
        x = self.deconv3(x)
        x = self.deconv4(x)
        x = self.final_conv(x)
        x = jax.nn.sigmoid(x)
        return x

class ConvAutoencoder(nnx.Module):
    def __init__(self, *, num_classes: int, input_channels: int, rngs: nnx.Rngs):
        self.encoder = YatCNN(num_classes=num_classes, input_channels=input_channels, rngs=rngs)
        self.decoder = Decoder(output_channels=input_channels, rngs=rngs)
    def __call__(self, x, training: bool = False):
        bottleneck = self.encoder(
            x,
            training=training,
            return_activations_for_layer='bottleneck',
            apply_masking=False
        )
        reconstructed_image = self.decoder(bottleneck, training=training)
        return reconstructed_image 