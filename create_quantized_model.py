import tensorflow as tf
import numpy as np
import tensorflow_model_optimization as tfmot
from network import compile_model, load_or_create_model

def apply_quantization(layer):
    if layer.name not in ['residual-block-14-bn', 'value-dense-bn', 'value-bn', 'value', 'mlh-dense-bn', 'mlh-bn'] :
        return tfmot.quantization.keras.quantize_annotate_layer(layer)
    #if isinstance(layer, tf.keras.layers.Dense) or isinstance(layer, tf.keras.layers.Conv2D):
    #    if layer.name != "value":
    return layer

def create_quantization_aware_model(base_model):
    # Use `tf.keras.models.clone_model` to apply `apply_quantization_to_dense`
    # to the layers of the model.
    annotated_model = tf.keras.models.clone_model(
        base_model,
        clone_function=apply_quantization,
    )
    # Now that the Dense layers are annotated,
    # `quantize_apply` actually makes the model quantization aware.
    quant_aware_model = tfmot.quantization.keras.quantize_apply(annotated_model)
    compile_model(quant_aware_model, prefix="quant_")
    quant_aware_model.summary()
    return quant_aware_model

if __name__ == "__main__":
    name = "zero_13.3"

    model = load_or_create_model(f"{name}.h5")
    qmodel = create_quantization_aware_model(model)
    qmodel.save(f"{name}_q.h5")
