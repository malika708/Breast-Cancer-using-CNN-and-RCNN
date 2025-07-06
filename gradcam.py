import tensorflow as tf
import numpy as np
import cv2
from keras.models import Model

def gradcam_heatmap(model, image, layer_name):
    """Generate GradCAM heatmap for the given image."""
    # Make sure the model is built and has inputs/outputs defined
    _ = model.predict(image)

    # Use model.inputs and model.outputs for better compatibility
    try:
        model_input = model.inputs[0]
        layer_output = model.get_layer(layer_name).output
        model_output = model.outputs[0]
    except Exception as e:
        # If still having issues, try rebuilding the model graph explicitly
        model(image)
        model_input = model.inputs[0]
        layer_output = model.get_layer(layer_name).output
        model_output = model.outputs[0]

    grad_model = Model(inputs=model_input, outputs=[layer_output, model_output])

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(image)
        loss = predictions[:, np.argmax(predictions[0])]
    output = conv_outputs[0]
    grads = tape.gradient(loss, conv_outputs)
    if grads is None:
        # If gradients cannot be computed, return a zero heatmap and print a warning
        print("Warning: Gradients could not be computed for GradCAM. Returning zero heatmap.")
        cam = np.zeros(output.shape[0:2], dtype=np.float32)
        heatmap = cam
    else:
        grads = grads[0]
        weights = tf.reduce_mean(grads, axis=(0, 1))
        cam = np.zeros(output.shape[0:2], dtype=np.float32)
        for i, w in enumerate(weights):
            cam += w * output[:, :, i]
        cam = np.maximum(cam, 0)
        cam = cv2.resize(cam, (image.shape[1], image.shape[2]))
        heatmap = (cam - cam.min()) / (cam.max() - cam.min() + 1e-10)  # Added small epsilon to prevent division by zero
    return heatmap