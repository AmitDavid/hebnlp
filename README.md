# Simlon סמלון
An AI that predict the prefred emoji for your **Hebrew** text!

# How to run
1) Clone git to your compter
2) Download project dependencies from 'requirements.txt' file (using 'pip install -r requirements.txt')
3) From the folder "HEBNLP", run the command "gui/gui.py"
4) Choose model from the list (We recommand LogisticRegression 😃)
5) Write your hebrew text in the text box
6) Press 'enter' to predict emoji
7) You can try diffrent models and diffrent texts to see the diffrence between them

## How to create models
Run "models\createModel.py" with your preferred model and parameters by changing 'MODEL_TITLES' and get_model_from_title

## How to get more youtube comments
Run "data\createCSV.py", you can add more channels by changing 'LIST_OF_CHANNEL_IDS' and adding there more channel IDs
