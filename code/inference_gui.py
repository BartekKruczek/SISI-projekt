import pyaudio
import numpy as np
import matplotlib.pyplot as plt
import sounddevice
from multilingual_kws.embedding import input_data
import tensorflow as tf

model_path = "/home/piotrek/Documents/personal/STUDIA_DS/semestr_1/PP/multilingual_kws/off_5shot"
model = tf.keras.models.load_model(model_path)

def make_prediction(audio: np.array):
    settings = input_data.standard_microspeech_model_settings(label_count=1)
    spectrogram = np.expand_dims(input_data.to_micro_spectrogram(settings, audio), axis=0)
    
    predictions = model.predict(spectrogram, verbose=None)
    categorical_predictions = np.argmax(predictions, axis=1)
    
    return predictions, categorical_predictions

def main():   
    CHUNK = 1600
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 16000
    
    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)

    plt.ion()
    fig, ax = plt.subplots()
    x = np.arange(0, 100)
    y = np.zeros(100)
    line, = ax.plot(x, y)

    ax.set_ylim(0, 1)
    ax.set_xlim(0, 100)
    ax.set_xlabel('Time (frames)')
    ax.set_ylabel('Prediction')
    ax.set_title('Real-time Prediction')

    try:
        print("* recording")
        frames = []

        while True:
            data = stream.read(CHUNK)
            data = np.frombuffer(data, dtype=np.int16) / np.iinfo(np.int16).max
            frames.append(data)
            
            if len(frames) > 10:
                del frames[0]

            if len(frames) > 0:
                infer_second = np.concatenate(frames)
                if infer_second.shape[0] < 16000:
                    infer_second = np.concatenate([infer_second, np.zeros(16000 - infer_second.shape[0])])
                
                # fetch softmax predictions from the finetuned model:
                # (class 0: silence/background noise, class 1: unknown keyword, class 2: target)
                predictions, categorical_predictions = make_prediction(infer_second)
                print(f"silence/background noise {predictions[0][0]}, unknown keyword:  {predictions[0][1]}, target:  {predictions[0][2]}")
                
                y = np.roll(y, -1)
                y[-1] = predictions[0][2]
                line.set_ydata(y)
                ax.draw_artist(ax.patch)
                ax.draw_artist(line)
                fig.canvas.flush_events()
                plt.pause(0.01)
        
    finally:
        print("* done recording")
        stream.stop_stream()
        stream.close()
        p.terminate()
        plt.ioff()
        plt.show()
      
# print(sounddevice.query_devices())
main()
