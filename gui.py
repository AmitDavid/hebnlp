from copyreg import pickle
import PySimpleGUI as sg
import os
from pilmoji import Pilmoji
from PIL import Image, ImageFont
import pickle
import threading


def save_emoji_as_img(text):
    my_string = text

    with Image.new('RGB', (400, 80), (0, 0, 0)) as image:
        font = ImageFont.truetype('arial.ttf', 40)

        with Pilmoji(image) as pilmoji:
            pilmoji.text((10, 10), my_string, (255, 255, 255), font)

            im1 = image.save("gui\emojiImg.png", "PNG")
# ------------------------------------------------------------------------

def load_model(file_name):
    infile = open(file_name,'rb')
    model = pickle.load(infile)
    infile.close()
    return model

# ------------------------------------------------------------------------
def model_prediction(window, sentence, file_name):
    path = "models\\" + file_name
    # load_model
    model = load_model(path)

    #predict
    predictions, raw_outputs = model.predict([sentence])
    print(predictions)
    # show result in GUI
    window['-TemporarlyOutput-'].update(str(predictions.tolist()))

    # important important 
    # str_emoji = " 😁 💯"
    # save_emoji_as_img(str_emoji)
    # window['-OUTPUT-'].update('gui\emojiImg.png')
    # window.write_event_value('-THREAD DONE-', '')

# ------------------------------------------------------------------------
def clean_choices(window):
    window['-INPUT-'].update("")
    window['-OUTPUT-'].update("")
    window['-MODEL-'].update(default_model)
    window['-TemporarlyOutput-'].update('')

    
# ------------------------------------------------------------------------
def print_invalid_input(message, window):
    window['-TemporarlyOutput-'].update(message)

# ------------------------------------------------------------------------

def initialize_layout():
    layout = [[sg.Push(), sg.Image('gui/logo.png'), sg.Push()],
        [sg.Push(), sg.OptionMenu(models_list, default_value=default_model, key='-MODEL-'), sg.Push()],
                [sg.Push(), sg.Text('הכנס טקסט ולסיום לחץ אנטר', font='Arial 18', text_color='white'), sg.Push()],
                [sg.Push(), sg.InputText(justification='r', font='Arial 18', key='-INPUT-', expand_y=True)],
                [sg.Push(), sg.Image(key='-OUTPUT-'), sg.Push()],
                [sg.Push(), sg.Text(key='-TemporarlyOutput-',font='Arial 18'), sg.Push()],
                [sg.Button('! עוד אחד', font='Arial 16'), sg.Push()]]
    
    return layout

# ------------------------------------------------------------------------


if __name__ == '__main__':

    models_list = os.listdir('models')
    default_model = "בחר מודל"

    sg.theme('Black')
    
    layout = initialize_layout()

    window = sg.Window('התאמת אימוגי לטקסט בעברית', layout, return_keyboard_events=True, finalize=True)
    while True:
        event, values = window.read()

        if event == sg.WIN_CLOSED:
            break

        if event == '! עוד אחד':
            clean_choices(window)

        # todo: האם \r הוא תמיד אנטר בכל מערכת.
        if event == "\r":

            # missing model
            if (len(values['-MODEL-']) == 0) or (values['-MODEL-']==default_model):
                print_invalid_input("אנא הכנס מודל", window)

            # missing sentence
            elif len(values['-INPUT-'].strip()) == 0:
                print_invalid_input("אנא הכנס טקסט", window)

            # great, now predict!
            else:
                model_prediction(window, values['-INPUT-'], values['-MODEL-'])

    window.close()

