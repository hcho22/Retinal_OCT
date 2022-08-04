import tensorflow as tf
from tensorflow.keras import layers

# Distorts the color distributions of images
class RandomColorAffine(layers.Layer):
    def __init__(self, brightness=0, jitter = 0, **kwargs):
        super().__init__(**kwargs)
        
        self.brightness = brightness
        self.jitter = jitter
        
    def get_config(self):
        config = super().get_config()
        config.update({"brightness": self.brightness, "jitter": self.jitter})
        return config
    
    def call(self, images, training = True):
        if training:
            batch_size = tf.shape(images)[0]
            
            # Same for all colors
            brightness_scales = 1 + tf.random.uniform(
            (batch_size, 1,1,1), minval = self.brightness, maxval = self.brightness
            )
            
            # Different for all colors
            jitter_matrices = tf.random.uniform(
            (batch_size, 1, 3, 3), minval = self.jitter, maxval = self.jitter
            )
            
            color_transforms = (
            tf.eye(3, batch_shape = [batch_size, 1]) * brightness_scales + jitter_matrices
            )
            
            images = tf.clip_by_value(tf.matmul(images, color_transforms), 0, 1)
        return images

# Image augmentation module
def get_augmenter(min_area, brightness, jitter):
    zoom_factor = 1.0 - math.sqrt(min_area)
    
    return keras.Sequential(
    [
        keras.Input(shape=(image_size, image_size, image_channels)),
        layers.Rescaling(1 / 255),
        layers.RandomFlip("horizontal"),
        layers.RandomTranslation(zoom_factor / 2, zoom_factor / 2),
        layers.RandomZoom((-zoom_factor, 0.0), (-zoom_factor, 0.0)),
        RandomColorAffine(brightness, jitter)
    ])