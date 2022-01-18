import tkinter
from googletrans import Translator
import ModelLoader
import Recorder

# LOADING MODEL, RECORDER AND TRANSLATOR #
model = ModelLoader.load_model()
audio = Recorder.init_recorder()
translator = Translator()

# VARIABLES
sentence = "Oración: "
translated = "Traducción: "


def record():
    Recorder.record(audio)
    translate()


def write_prediction(prediction):
    # use global variable
    global sentence
    global sentence_label
    # configure
    text = sentence + prediction
    sentence_label.config(text=text)


def write_translation(translation):
    # use global variable
    global translated
    global translate_label
    # configure
    text = translated + translation
    translate_label.config(text=text)


def translate():
    filename = Recorder.filename
    prediction = ModelLoader.predict(filename, model)
    prediction = ' '.join(prediction)
    write_prediction(prediction)
    translation = translator.translate(prediction, src='es', dest='en')
    write_translation(translation.text)


# GUI #
gui = tkinter.Tk(className='Traductor')
# set window size
gui.geometry("700x500")
# record button
pixelVirtual = tkinter.PhotoImage(width=1, height=1)
recordBtn = tkinter.Button(
    gui,
    text="Grabar",
    image=pixelVirtual,
    width=150,
    height=40,
    compound="center",
    command=record)
# Pack buttons
recordBtn.place(relx=.5, rely=.4, anchor=tkinter.CENTER)

sentence_label = tkinter.Label(gui, text="Oración: ", font='Helvetica 16')
sentence_label.place(relx=.5, rely=.52, anchor=tkinter.CENTER)
translate_label = tkinter.Label(gui, text="Traducción: ", font='Helvetica 16 bold')
translate_label.place(relx=.5, rely=.6, anchor=tkinter.CENTER)

gui.mainloop()
