from copyreg import pickle
import PySimpleGUI as sg
import os
from pilmoji import Pilmoji
from PIL import Image, ImageFont
import pickle
from threading import Thread
import pandas as pd

GUI_TEXT = {
    "ProgramTitle": "התאמת אימוגי לטקסט בעברית",
    "InputText": "הכנס טקסט ולסיום לחץ אנטר",
    "NoInput": "אנא הכנס טקסט",
    "DefaultModel": "בחר מודל",
    "TryAgain": "!עוד אחד",
}

PATH_TO_RESULT_IMG = "gui\ResultEmoji.png"
PATH_TO_LOADING_ANIMATION = "gui\loadingCat.gif"


def save_emoji_as_img(text):
    '''
    Save emoji as image, so it can be displayed in the GUI
    '''
    with Image.new('RGB', (150, 150), (232, 223, 220)) as image:
        font = ImageFont.truetype('arial.ttf', 40)

        with Pilmoji(image) as pilmoji:
            pilmoji.text((30, 30), text, (0, 0, 0), font, emoji_scale_factor=2.5)
            image.save(PATH_TO_RESULT_IMG, "PNG")


def load_variable_from_file(file_name):
    '''
    Load variable from file
    '''
    with open(file_name, 'rb') as file:
        return pickle.load(file)


def perdict_emoji(count_vect, tf_transformer, model, sentence):
    '''
    Predict emoji for the given sentence
    '''
    train_df = pd.DataFrame({'text': [sentence]})
    X_sentence_counts = count_vect.transform(train_df.text)
    X_test_tfidf = tf_transformer.transform(X_sentence_counts)
    predictions = model.predict(X_test_tfidf)
    print(predictions)
    return predictions


def model_prediction(window, sentence, file_name):
    '''
    Predict emoji for the given sentence
    '''
    # Load model, count_vect and tf_transformer
    path_to_file = f'models\{file_name}'

    path_to_model = f'{path_to_file}_model'
    path_to_count_vect = f'{path_to_file}_count_vect'
    path_to_tf_transformer = f'{path_to_file}_tf_transformer'
    if os.path.isfile(path_to_model) and os.path.isfile(path_to_count_vect) and os.path.isfile(path_to_tf_transformer):
        model = load_variable_from_file(path_to_model)
        count_vect = load_variable_from_file(path_to_count_vect)
        tf_transformer = load_variable_from_file(path_to_tf_transformer)
    else:
        print("Model not found")
        return

    # Predict
    predictions = perdict_emoji(count_vect, tf_transformer, model, sentence)

    # Convert perdiction to emoji
    emoji_dict = load_variable_from_file(f'{path_to_file}_dict')
    str_emoji = emoji_dict[predictions[0]]

    # Convert emoji to image
    save_emoji_as_img(str_emoji)
    window['Output'].update(PATH_TO_RESULT_IMG)


def clean_choices(window):
    '''
    Clean choices in the GUI
    '''
    window['Input'].update("")
    window['Output'].update("")
    window['TemporarlyOutput'].update("")


def initialize_layout_and_theme():
    '''
    Initialize the layout for the GUI
    '''
    # Change theme background color
    sg.LOOK_AND_FEEL_TABLE['MyTheme'] = sg.LOOK_AND_FEEL_TABLE['Material2']
    sg.LOOK_AND_FEEL_TABLE['MyTheme']['BACKGROUND'] = '#E8DfDC'

    # Switch to use new theme
    sg.theme('MyTheme')

    return [
        [sg.Push(), sg.Image('gui/logo.png'), sg.Push()],
        [sg.Push(), sg.OptionMenu(key='Model', values=models_list, default_value=GUI_TEXT['DefaultModel']), sg.Push()],
        [sg.Push(), sg.Text(key='InputText', text=GUI_TEXT['InputText'], font='Arial 18'), sg.Push()],
        [sg.Push(), sg.InputText(key='InputBox', justification='r', font='Arial 18', expand_y=True)],
        [sg.Column([[sg.Image(key='Output')]], justification='center')],
        [sg.Column([[sg.Text(key='TemporarlyOutput', font='Arial 18')]], justification='center')],
        [sg.Column([[sg.Image(key='LoadingAnimation', visible=False)]], justification='center')],
        [sg.Button(key="ResetButton", button_text=GUI_TEXT['TryAgain'], font='Arial 16'), sg.Push()]
    ]


def start_loading_animation(window):
    '''
    Start loading animation, disable the button and clear old results
    '''
    window['LoadingAnimation'].update(visible=True)
    window["LoadingAnimation"].UpdateAnimation(PATH_TO_LOADING_ANIMATION, time_between_frames=60)
    window['TemporarlyOutput'].update(visible=False)
    window['ResetButton'].update(disabled=True)
    window['Output'].update()


def stop_loading_animation(window):
    '''
    Stop loading animation, enable the button
    '''
    window['LoadingAnimation'].update(visible=False)
    window['TemporarlyOutput'].update(visible=True)
    window['ResetButton'].update(disabled=False)

if __name__ == '__main__':
    # Keep only files that end with '_model' from the models folder
    # The names for the list will not contain '_model'
    models_list = [file[:-6] for file in os.listdir('models') if file.endswith('_model')]
    if models_list == []:
        print("No models found")
        exit()

    layout = initialize_layout_and_theme()
    window = sg.Window(GUI_TEXT['ProgramTitle'], layout, return_keyboard_events=True, finalize=True)
    event, values = window.read()

    # Create thread for prediction
    prediction_thread = None
    prediction_thread_new = Thread(target=model_prediction, daemon=True, args=(window, values['InputBox'], values['Model']))
    while True:
        event, values = window.read(timeout=10)
        if event == sg.WIN_CLOSED:
            break

        if prediction_thread is not None and prediction_thread.is_alive():
            start_loading_animation(window)

        else:  # Prediction has finished, stop animation and enable button
            stop_loading_animation(window)

        if event == 'ResetButton':
            clean_choices(window)

        # If user pressed enter predict
        if event == '\r':
            # If input is empty, prompt user to enter text
            if len(values['InputBox'].strip()) == 0:
                window['TemporarlyOutput'].update(GUI_TEXT['NoInput'])

            else:  # great, now predict!
                prediction_thread = prediction_thread_new
                prediction_thread.start()
                prediction_thread_new = Thread(target=model_prediction, daemon=True, args=(window, values['InputBox'], values['Model']))

    window.close()
