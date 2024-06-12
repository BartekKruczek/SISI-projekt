import pyaudio
import numpy as np
import librosa
import numpy as np
import sounddevice
import time
import pyaudio
import wave

from multilingual_kws.embedding import input_data
import tensorflow as tf

model = tf.keras.models.load_model("/home/piotrek/Documents/personal/STUDIA_DS/semestr_1/PP/multilingual_kws/off_5shot")

def make_prediction(audio: np.array):
    
    settings = input_data.standard_microspeech_model_settings(label_count=1)
    spectrogram = np.expand_dims(input_data.to_micro_spectrogram(settings, audio), axis=0)
    
    # fetch softmax predictions from the finetuned model:
    # (class 0: silence/background noise, class 1: unknown keyword, class 2: target)

    predictions = model.predict(spectrogram,verbose=None)
    categorical_predictions = np.argmax(predictions, axis=1)
    
    return predictions, categorical_predictions

def main():
    CHUNK = 1600
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 16000
    WAVE_OUTPUT_FILENAME = "output.wav"
    
    p = pyaudio.PyAudio()

    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)

    try:
        print("* recording")

        frames = []

        # for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        while True:
            data = stream.read(CHUNK)
            data = np.frombuffer(data, dtype=np.int16) / np.iinfo(np.int16).max
            frames.append(data)
            
            if len(frames) > 10:
                del frames[0]

            if len(frames) > 0:
                infer_second = np.concatenate(frames)
                if infer_second.shape[0] < 16000:
                    infer_second = np.concatenate([infer_second, np.zeros(16000-infer_second.shape[0])])
                
                predictions, categorical_predictions = make_prediction(infer_second)
                print(predictions)
        
    finally:
        print("* done recording")
        stream.stop_stream()
        stream.close()
        p.terminate()
        
        
main()