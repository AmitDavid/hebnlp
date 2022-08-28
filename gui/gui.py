from copyreg import pickle
from faulthandler import disable
from operator import truediv
import PySimpleGUI as sg
import os
from pilmoji import Pilmoji
from PIL import Image, ImageFont
import pickle
from threading import Thread, Lock


def save_emoji_as_img(text):
    my_string = text

    with Image.new('RGB', (150, 150), (232, 223, 220)) as image:
        font = ImageFont.truetype('arial.ttf', 40)

        with Pilmoji(image) as pilmoji:
            pilmoji.text((30, 30), my_string, (0, 0, 0), font, emoji_scale_factor=2.5)

            im1 = image.save("gui\emojiImg.png", "PNG")
# ------------------------------------------------------------------------


def load_model(file_name):
    infile = open(file_name, 'rb')
    model = pickle.load(infile)
    infile.close()
    return model

def load_dict(file_name):
    infile = open(file_name, 'rb')
    dicts = pickle.load(infile)
    infile.close()
    return dicts
# ------------------------------------------------------------------------
def model_prediction(window, sentence, file_name):
    path = "models\\" + file_name
    # load_model
    model = load_model(path)

    # predict
    predictions, raw_outputs = model.predict([sentence])
    print(predictions)
    # show result in GUI
    # window['-TemporarlyOutput-'].update(str(predictions.tolist()))

    path_dict = "models\\" + file_name + '_dict'
    emoji_dict = load_dict(path_dict)
    str_emoji = emoji_dict[predictions[0]]
    save_emoji_as_img(str_emoji)
    window['-OUTPUT-'].update('gui\emojiImg.png')

# ------------------------------------------------------------------------


def clean_choices(window):
    window['-INPUT-'].update("")
    window['-OUTPUT-'].update("")
    window['-TemporarlyOutput-'].update('')


# ------------------------------------------------------------------------
def print_invalid_input(message, window):
    window['-TemporarlyOutput-'].update(message)

# ------------------------------------------------------------------------


def initialize_layout():
    layout = [[sg.Push(), sg.Image('gui/logo.png'), sg.Push()],
              [sg.Push(), sg.OptionMenu(
                  models_list, default_value=models_list[0], key='-MODEL-'), sg.Push()],
              [sg.Push(), sg.Text('הכנס טקסט ולסיום לחץ אנטר',
                                  font='Arial 18'), sg.Push()],
              [sg.Push(), sg.InputText(justification='r',
                                       font='Arial 18', key='-INPUT-', expand_y=True)],
              [sg.Column([[sg.Image(key='-OUTPUT-')]], justification='center')],
              [sg.Column([[sg.Text(key='-TemporarlyOutput-', font='Arial 18')]], justification='center')],
              [sg.Column([[sg.Image(key='-LOADINGANIMATION-', visible=False)]], justification='center')],
              [sg.Button('! עוד אחד', font='Arial 16', key = "reset_Button"), sg.Push()]]

    return layout


# ------------------------------------------------------------------------

if __name__ == '__main__':

    models_list = os.listdir('models')
    default_model = "בחר מודל"

    # Add your new theme colors and settings
    sg.LOOK_AND_FEEL_TABLE['MyCreatedTheme'] = sg.LOOK_AND_FEEL_TABLE['Material2']
    sg.LOOK_AND_FEEL_TABLE['MyCreatedTheme']['BACKGROUND'] = '#E8DfDC'

    # Switch to use your newly created theme
    sg.theme('MyCreatedTheme')

    # # Call a popup to show what the theme looks like
    # sg.popup_get_text('This how the MyNewTheme is created')

    layout = initialize_layout()

    window = sg.Window('התאמת אימוגי לטקסט בעברית', layout,
                       return_keyboard_events=True, finalize=True)
                       
    event, values = window.read()
    prediction_thread_new = Thread(target = model_prediction, daemon=True, args=(window, values['-INPUT-'], values['-MODEL-']))
    prediction_thread = None
    while True:
        event, values = window.read(timeout=10)
        if prediction_thread is not None and prediction_thread.is_alive():
            window['-LOADINGANIMATION-'].update(visible=True)
            window["-LOADINGANIMATION-"].UpdateAnimation("gui/Adobe_cat.gif", time_between_frames=40)
            window['-TemporarlyOutput-'].update(visible=False)
            window['reset_Button'].update(disabled=True)

        else:
            window['-LOADINGANIMATION-'].update(visible=False)
            window['-TemporarlyOutput-'].update(visible=True)
            window['reset_Button'].update(disabled=False)


        if event == sg.WIN_CLOSED:
            break

        if event == 'reset_Button':
            clean_choices(window)

        # todo: האם \r הוא תמיד אנטר בכל מערכת.
        if event == "\r":

            # missing sentence
            if len(values['-INPUT-'].strip()) == 0:
                print_invalid_input("אנא הכנס טקסט", window)

            # great, now predict!
            else:
                prediction_thread = prediction_thread_new
                prediction_thread.start()
                prediction_thread_new = Thread(target = model_prediction, daemon=True, args=(window, values['-INPUT-'], values['-MODEL-']))


    window.close()
