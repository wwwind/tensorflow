import numpy as np
import tensorflow as tf

import os
import logging
from shutil import rmtree
import tempfile


class SimpleModel:
    def __init__(self):
        self.model_name = "mnist"
        self.train_data = None
        self.test_data = None
        self.calib_data = None
        self.num_calib = 1000
        # (data preprocessing) Normalize the input image so that
        # each pixel value is between 0 to 1.
        self.pre_process = lambda x: x / 255.0

        # Different temp paths
        self.model_dir = None
        self.savedModel_dir = None
        self.keras_model_dir = None
        self.tflite_dir = None
        self.tflite_model_FP32 = None
        self.tflite_model_INT8 = None
        self.tflite_model_INT16 = None

        self._temp_path = os.path.join(tempfile.gettempdir(), self.model_name)

        # Load dataset
        self._load_data()

        # Set temp paths
        self._set_path()

        self._set_logging()

    def _set_logging(self):
        logging.getLogger("tensorflow").setLevel(logging.DEBUG)

    def _mkdir(self, *name_path: str):
        dir_path = os.path.join(*name_path)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        return dir_path

    def _load_data(self):
        # Load MNIST dataset
        mnist = tf.keras.datasets.mnist

        # _data: (images, labels)
        self.train_data, self.test_data = mnist.load_data()
        self.calib_data = self.pre_process(
            self.train_data[0][0 : self.num_calib].astype(np.float32)
        )

    def _set_path(self):
        # setup relative path
        self.model_dir = self._mkdir(self._temp_path, self.model_name)
        self.savedModel_dir = self._mkdir(
            self.model_dir, "{:s}_saved_model".format(self.model_name)
        )
        self.tflite_dir = self._mkdir(
            self.model_dir, "{:s}_tflite_model".format(self.model_name)
        )
        self.tflite_model_FP32 = os.path.join(
            self.tflite_dir, "{:s}_FP32.tflite".format(self.model_name)
        )
        self.tflite_model_INT8 = os.path.join(
            self.tflite_dir, "{:s}_INT8.tflite".format(self.model_name)
        )
        self.tflite_model_INT16 = os.path.join(
            self.tflite_dir, "{:s}_INT16.tflite".format(self.model_name)
        )

    def train(self):
        # Define the model architecture
        model = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=(28, 28)),
                tf.keras.layers.Reshape(target_shape=(28, 28, 1)),
                tf.keras.layers.Conv2D(
                    filters=12, kernel_size=(3, 3), activation=tf.nn.relu
                ),
                tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(10, activation=tf.nn.softmax),
            ]
        )
        model.summary()

        train_images = self.pre_process(self.train_data[0])
        train_labels = self.train_data[1]
        test_images = self.pre_process(self.test_data[0])
        test_labels = self.test_data[1]
        # Train the digit classification model
        model.compile(
            optimizer="adam",
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
        model.fit(
            train_images,
            train_labels,
            epochs=1,
            validation_data=(test_images, test_labels),
        )
        # dump SavedModel
        model.save(str(self.savedModel_dir))

        return self

    def eval(self, tflite_model_path: str):
        interpreter = tf.lite.Interpreter(model_path=str(tflite_model_path))
        interpreter.allocate_tensors()

        input_index = interpreter.get_input_details()[0]["index"]
        output_index = interpreter.get_output_details()[0]["index"]

        # (data preprocessing) Normalize the input image so that
        # each pixel value is between 0 to 1.
        test_images = self.pre_process(self.test_data[0])
        test_labels = self.test_data[1]
        # Run predictions on every image in the "test" dataset.
        prediction_digits = []
        for test_image in test_images:
            # Pre-processing: add batch dimension and convert to float32 to match with
            # the model's input data format.
            test_image = np.expand_dims(test_image, axis=0).astype(np.float32)
            interpreter.set_tensor(input_index, test_image)

            # Run inference.
            interpreter.invoke()

            # Post-processing: remove batch dimension and find the digit with highest
            # probability.
            output = interpreter.tensor(output_index)
            digit = np.argmax(output()[0])
            prediction_digits.append(digit)

        # Compare prediction results with ground truth labels to calculate accuracy.
        accurate_count = 0
        for index, _ in enumerate(prediction_digits):
            if prediction_digits[index] == test_labels[index]:
                accurate_count += 1
        accuracy = accurate_count * 1.0 / len(prediction_digits)

        return accuracy

    def _get_calib_data_func(self):
        def representative_data_gen():
            for input_value in self.calib_data:
                input_value = np.expand_dims(input_value, axis=0).astype(np.float32)
                yield [input_value]

        return representative_data_gen

    def clean(self):
        """Clean all temporary files"""
        rmtree(self.model_dir, ignore_errors=True)

    def gen_tflite(self, enable_mlir=False, allow_float_16x8=False, save_format="tf"):
        """generate .tflite model for FP32, INT8, INT16"""
        self.converter = tf.lite.TFLiteConverter.from_saved_model(self.savedModel_dir)

        # save FP32 tflite
        tflite_model = self.converter.convert()
        open(self.tflite_model_FP32, "wb").write(tflite_model)

        self.converter.optimizations = [tf.lite.Optimize.DEFAULT]
        self.converter.experimental_new_converter = enable_mlir
        self.converter._experimental_new_quantizer = True
        self.converter.representative_dataset = self._get_calib_data_func()

        # save INT8 tflite
        self.converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        tflite_model_INT8 = self.converter.convert()
        open(self.tflite_model_INT8, "wb").write(tflite_model_INT8)

        # save INT16 tflite
        self.converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS_ACTIVATIONS_INT16_WEIGHTS_INT8
        ]

        tflite_model_INT16 = self.converter.convert()
        open(self.tflite_model_INT16, "wb").write(tflite_model_INT16)


if __name__ == "__main__":
    temp = SimpleModel()
    temp.train().gen_tflite(True)

    print("FP32 model eval results: {:f}".format(temp.eval(temp.tflite_model_FP32)))
    print("INT8 model eval results: {:f}".format(temp.eval(temp.tflite_model_INT8)))
    print("INT16 model eval results: {:f}".format(temp.eval(temp.tflite_model_INT16)))

    print("Created model is in {}".format(temp.savedModel_dir))
    print("Converted models are in {}".format(temp.tflite_dir))
    # temp.clean()
