import pyaudio
import wave
from tqdm import tqdm
import uuid
import pandas as pd
from warning_mute_alsa import noalsaerr

class audio_recorder:

    def __init__(self, directory):
        self.resolution = pyaudio.paInt16
        self.channels = 2
        self.fs = 44100
        self.buffer_size = 4096  # 2^12 samples for buffer
        self.recording_length_seconds = 3
        self.device_index = 2  # device index found by p.get_device_info_by_index(ii)
        self.directory = directory

    def record(self, iteration):
        print("recording...")

        with noalsaerr():
            audio = pyaudio.PyAudio()
        stream = audio.open(format=self.resolution, rate=self.fs, channels=self.channels,
                            input_device_index=self.device_index, input=True, frames_per_buffer=self.buffer_size)
        frames = []

        # loop through stream and append audio chunks to frame array
        for ii in tqdm(range(0, int((self.fs / self.buffer_size) * self.recording_length_seconds))):
            data = stream.read(self.buffer_size)
            frames.append(data)

        # stop the stream, close it, and terminate the pyaudio instantiation
        stream.stop_stream()
        stream.close()
        audio.terminate()

        wav_name = "{dir}/{wav_name}_{iteration}.wav".format(dir=self.directory, wav_name=uuid.uuid4().hex,
                                                               iteration=iteration)

        output = [wav_name, iteration]

        df = pd.DataFrame([output], columns=['stereo wav', 'Rotation Iteration'])
        df.to_csv("{dir}/iteration_{iter}.csv".format(dir=self.directory, iter=iteration))

        # save the audio frames as .wav file
        wavfile = wave.open(wav_name, 'wb')
        wavfile.setnchannels(self.channels)
        wavfile.setsampwidth(audio.get_sample_size(self.resolution))
        wavfile.setframerate(self.fs)
        wavfile.writeframes(b''.join(frames))
        wavfile.close()



