import PySimpleGUI as sg

from pilmoji import Pilmoji
from PIL import Image, ImageFont


def save_emoji_as_img(text):
    my_string = text

    with Image.new('RGB', (400, 80), (0, 0, 0)) as image:
        font = ImageFont.truetype('arial.ttf', 50)

        with Pilmoji(image) as pilmoji:
            pilmoji.text((10, 10), my_string, (0, 0, 0), font)

            im1 = image.save("emoji_img.png", "PNG")
# ------------------------------------------------------------------------


sg.theme('Black')
layout = [[sg.Push(), sg.Image('gui/logo.png'), sg.Push()],
          [sg.Push(), sg.Text('הכנס טקסט ולסיום לחץ אנטר', font='Arial 18', text_color='white'), sg.Push()],
          [sg.Push(), sg.InputText(justification='r', font='Arial 18', key='-INPUT-', expand_y=True)],
          [sg.Push(), sg.Image(key='-OUTPUT-'), sg.Push()],
          [sg.Button('! עוד אחד', font='Arial 16'), sg.Push()]]


window = sg.Window('התאמת אימוגי לטקסט בעברית', layout, return_keyboard_events=True, finalize=True)
while True:
    event, values = window.read()
    if event == sg.WIN_CLOSED:
        break
    if event == '! עוד אחד':
        window['-INPUT-'].update("")
        window['-OUTPUT-'].update("")
        # todo: האם \r הוא תמיד אנטר בכל מערכת.
    if event == "\r":
        str_emoji = "        😁 💯"
        save_emoji_as_img(str_emoji)

        window['-OUTPUT-'].update('gui/emojiImg.png')

window.close()

# Hello, world! 👋 Here are some emojis: 🎨 🌊 😎
# I also support Discord emoji: <:rooThink:596576798351949847>

# print('You entered ', values[0])
