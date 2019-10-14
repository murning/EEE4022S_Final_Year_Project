import numpy as np
import pandas as pd
import librosa
import data_cnn_format as cnn
import gccphat
import rotate
import final_models
import utility_methods
from audio_recorder import audio_recorder
from motor import Motor
import sys
import time


class Predict:
    """
    Class for making a prediction based on real hardware
    """

    def __init__(self, directory, model):

        self.current_position = 0
        self.iteration = 0
        self.directory = directory
        self.predictions = []
        self.model = model
        self.rotation = 0
        self.mic_centre = np.array([1.5, 1.5])
        self.prediction = 0
        self.audio = audio_recorder(self.directory)
        self.motor = Motor()
        self.current_model = self.get_model()  # init
        self.motor_offset = 0
        self.relative_prediction = 0
        self.results = []
        self.initial_adjust = False

    def store_prediction(self, doa_list):
        """
        convert relative prediction to home coordinates
        """

        true_doas = [utility_methods.cylindrical(self.current_position + doa_list[0]),
                     utility_methods.cylindrical(self.current_position + doa_list[1])]

        self.predictions.append(true_doas)

    def get_model(self):

        model = None
        if self.model == "gcc_cnn":
            model = final_models.gcc_cnn()
        elif self.model == "gcc_dsp":
            model = final_models.gcc_dsp()
        else:
            print("Error -> No file found")
        return model

    def record(self):

        self.motor.red_led_on()
        self.audio.record(self.iteration)
        self.motor.red_led_off()

    def load_audio(self):
        """
        Reads wav file based on csv values, resamples audio to 8000hz, fixes length to 1 second
        :return: numpy array of stereo audio, DOA from file
        """
        df = pd.read_csv("{dir}/iteration_{iter}.csv".format(dir=self.directory, iter=self.iteration),
                         usecols=[1, 2])

        wav_name = df.iloc[0][0]
        filename = "{wav_name}".format(wav_name=wav_name)

        y, sr = librosa.load(filename, mono=False)

        y_8k = librosa.resample(y, sr, 8000)

        o_env = librosa.onset.onset_strength(y_8k[0], sr=8000)

        peaks = librosa.util.peak_pick(o_env, 3, 3, 3, 5, 0.25, 5)

        times = librosa.frames_to_time(np.arange(len(o_env)),
                                       sr=8000, hop_length=512)

        peak_times = times[peaks]

        time = 0
        for i in range(1, len(peak_times) + 1):
            if 3 - peak_times[-i] >= 0.75:
                time = peak_times[-i] - 0.25
                break

        sample = librosa.time_to_samples(np.array([time]), sr=8000)

        sliced_y = np.array([y_8k[0][sample[0]:], y_8k[1][sample[0]:]])

        y_out = librosa.util.fix_length(sliced_y, 8000)

        return y_out

    def format_gcc_cnn(self):
        """
            Format the stereo Audio file for input to the gcc_phat CNN
            :return: data formatted for gcc_phat CNN, DOA read from file
            """
        result_x = self.load_audio()

        signal = result_x[0]
        reference_signal = result_x[1]
        _, raw_gcc_vector = gccphat.gcc_phat(signal=signal, reference_signal=reference_signal, fs=8000)

        cross_correlation_vector = cnn.reshape_x_for_cnn(cnn.normalize_x_data(np.array([raw_gcc_vector])))

        return cross_correlation_vector

    def format_gcc_dsp(self):
        """
        Format stereo audio file for gcc_dsp model
        :return: signal, reference signal and doa_from_file
        """
        result_x = self.load_audio()

        return result_x

    def load_and_process_audio(self):
        """
        Wrapping loading and processing of models into a single function
        """
        output_vector = None
        if self.model == "gcc_cnn":
            output_vector = self.format_gcc_cnn()
        elif self.model == "gcc_dsp":
            output_vector = self.format_gcc_dsp()
        else:
            print("Error -> No file found")

        return output_vector

    def compute_rotation(self):
        """
        compute rotation based on current and prior predictions
        :return:
        """

        if self.predictions[self.iteration][0] == 90.0 or self.predictions[self.iteration][0] == 270.0:
            self.rotation = 20
            self.initial_adjust = True
            return

        if self.iteration == 0:
            self.rotation = rotate.get_90_deg_rotation(self.predictions[self.iteration])
        elif self.iteration == 1:
            self.rotation = rotate.get_45_deg_rotation(self.predictions, self.current_position)
        elif self.iteration >= 2:
            self.rotation = rotate.get_fine_rotation(self.iteration)

    def update_position(self):
        """
        update current position of microphone based on rotation
        :param rotation:
        :return:
        """
        self.current_position = utility_methods.cylindrical(self.current_position + self.rotation)
        self.motor.rotate(np.abs(self.rotation), "cw" if self.rotation < 0 else "ccw")

    def compute_rotation_to_prediction(self):

        self.rotation = self.prediction - self.current_position - 90
        # self.rotation = ((self.prediction - self.current_position) + 180) % 360 - 180 -90

    def reset(self):
        """
        This method resets the prediction, iteration, position and rotation values to initial state. Rotates The
        motor back to 0 degrees

        :return:
        """

        self.rotation = -(self.prediction - 90)
        self.update_position()

        self.iteration = 0
        self.predictions = []
        self.prediction = 0
        self.change_motor_offset(-self.motor_offset)

    def change_motor_offset(self, offset):
        self.motor_offset = offset

        self.motor.rotate(np.abs(self.motor_offset), "ccw" if self.motor_offset < 0 else "cw")

    def save_results(self):

        results_row = [self.prediction, self.motor_offset, self.prediction - self.motor_offset, self.iteration]

        self.results.append(results_row)

        #

    def run(self):

        while self.iteration < 6:

            self.record()

            print("Recording Successful")

            vector = self.load_and_process_audio()

            doa_list = self.current_model.predict(vector)

            print("Model Prediction: {list}".format(list=doa_list))

            self.store_prediction(doa_list)

            print("Prediction List: {list}".format(list=self.predictions))

            val = utility_methods.check_if_twice(self.predictions, self.iteration)

            if val is not None:
                self.prediction = val
                self.compute_rotation_to_prediction()
                self.update_position()
                return self.prediction

            self.compute_rotation()

            print("Rotation: {rotation}".format(rotation=self.rotation))

            self.update_position()
            print("Current Position: {position}".format(position=self.current_position))

            self.iteration += 1

        self.prediction = utility_methods.get_mean_prediction(prediction_list=self.predictions)
        self.compute_rotation_to_prediction()
        self.update_position()
        self.motor.green_led()
        return self.prediction

    def evaluate(self):

        for i in np.arange(0, 360, 45):
            self.change_motor_offset(i)
            self.run()

            self.save_results()
            self.reset()

        df = pd.DataFrame(self.results, columns=['prediction', 'offset', 'prediction - offset', 'Number of iterations'])
        df.to_csv("{dir}/iteration_{model}.csv".format(dir="evaluation_results", model=self.model))


if __name__ == '__main__':

    model_name = sys.argv[1]
    model = Predict(directory="test", model=model_name)

    while 1:

        input("Press Enter to continue...")
        prediction = model.run()

        print("Final Prediction: {prediction}".format(prediction=prediction))

        x = input("Press Enter to rerun or x to exit")

        if x == "x":
            break

        model.reset()


    model.motor.clean_up()
