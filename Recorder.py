import pyaudio
import wave

# Record in chunks of 1024 samples
chunk = 1024
# 16 bits per sample
sample_format = pyaudio.paInt16
chanels = 1
# Record at 48000 samples per second
smpl_rt = 48000
filename = "temp.wav"

stream = None
frames = []


def init_recorder():
    return pyaudio.PyAudio()


def record(audio):
    global stream
    global frames
    # Create an interface to PortAudio
    stream = audio.open(format=sample_format, channels=chanels,
                     rate=smpl_rt, input=True,
                     frames_per_buffer=chunk)
    print('Recording...')
    for i in range(0, int(smpl_rt / chunk * 6)):
        data = stream.read(chunk)
        frames.append(data)

    # Stop and close the stream
    stream.stop_stream()
    stream.close()
    # Terminate - PortAudio interface
    # audio.terminate()

    print('Done !!! ')
    stop(audio)
    return stream


def stop(audio):
    global stream
    global frames
    # Save the recorded data in a .wav format
    sf = wave.open(filename, 'wb')
    sf.setnchannels(chanels)
    sf.setsampwidth(audio.get_sample_size(sample_format))
    sf.setframerate(smpl_rt)
    sf.writeframes(b''.join(frames))
    sf.close()

    # Reset frames
    frames.clear()
    return filename
