from simpletransformers.classification import ClassificationModel, ClassificationArgs
import pandas as pd
import logging
import pickle
import re
from sklearn.model_selection import train_test_split
import numpy as np

EMOJI_SUBGROUPS = {
    '😃': {'💯','👑','🏆','😀', '😃', '😄', '😁', '😅', '🙂', '🙃', '😉', '😊', '😇', '😋', '😛', '🤗', '🤭', '🤩', '😺', '😸', '🤠', '🥳', '😎', '🤓', '🧐', '🤤', '😌', },
    '🤣': {'🤣', '😂', '😆', '🤪', '😝', '😹'},
    '🥰': {'🥰', '😍', '😘', '😗', '☺', '😚', '😙', '😻', '😽', },
    '💗': {'💋', '💌', '💘', '💝', '💖', '💗', '💓', '💞', '💕', '💟', '❣', '❤', '🧡', '💛', '💚', '💙', '💜', '🤎', '🖤', '🤍', },
    '🙁': {'🤦‍','😦', '😧', '😨', '😰', '😥', '😢', '😭', '😩', '😫', '😟', '💔', '🙁', '😿', '😖', '😣', '😞', '😓', '☹', '🥺', '😕', '🤒', '🤕', '🤮', '🤧', '😔', '😪', },
    '😱': {'🙀', '😯', '😱', '😮', '😲', '😳', '😵', '🤯', },
    '😾': {'😾', '😤', '😡', '😠', '🤬', '👿', '😒'},
    '👍':{'👍','👌','💪','🤟','✌','🙌','👏',}
}



logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.WARNING)

LIST_OF_PATH_TO_CSV = {
    "Kan11":"data/Kan11_comments.csv",
    # "Kan11News":"data/Kan11News_comments.csv",
    "ScienceDavidson": "data/ScienceDavidson_comments.csv",
}


def reads_csv(path):
    # read from csv
    df = pd.read_csv(path, names=["text", "emoji"], encoding='utf-8')

    # replace &quot; with ""
    df['text'] = df['text'].apply(lambda x: re.sub('&quot;', '"', x))

    # reverse text
    # todo_1: problem with english..............
    df['text'] = df['text'].apply(lambda x: x[::-1])

    return df


if __name__ == '__main__':

    # load data from csv
    df_davison = reads_csv(LIST_OF_PATH_TO_CSV["ScienceDavidson"])
    df_Kan11 = reads_csv(LIST_OF_PATH_TO_CSV["Kan11"])
    # df_kan11News = reads_csv(LIST_OF_PATH_TO_CSV["Kan11News"])

    # concat data and convert emojis to unique numbers
    all_data = pd.concat([df_davison, df_Kan11])
    all_data_2 = pd.DataFrame()
    moving_rows = pd.DataFrame()
    for emoji in EMOJI_SUBGROUPS:
        for emoji_to_replace in EMOJI_SUBGROUPS[emoji]:
            regex = emoji_to_replace + '.*'
            all_data['emoji'] = all_data['emoji'].str.replace(regex, emoji, regex=True)


    #emojis_option = '😃🤣🥰💗🙁😱😾👍'
    for index, row in all_data.iterrows():
        if row['emoji'] in EMOJI_SUBGROUPS:
            all_data_2 = all_data_2.append(row)
    
    print(all_data_2)
    all_data = all_data_2
    
    all_data['labels'] = all_data.groupby(["emoji"]).ngroup()

    emoji_dict = {}
    for index, row in all_data.iterrows():
        emoji_dict[int(row["labels"])] = row["emoji"]

    # save model
    file_name = 'models\\model_29.8_2_dict'
    outfile = open(file_name, 'wb')
    pickle.dump(emoji_dict, outfile)
    outfile.close()

    all_data = all_data.drop('emoji', axis=1)

   # smaller data frame - to delete later
    num_of_classes = max(all_data['labels']) + 1
    print(num_of_classes)
    # all_data = all_data[all_data['labels']< num_of_classes]

    # Preparing train data and eval data
    X_train, X_test, y_train, y_test = train_test_split(
        all_data['text'], all_data['labels'], test_size=0.2, random_state=42)
    train_df = pd.DataFrame({'text': X_train, 'labels': y_train})
    eval_df = pd.DataFrame({'text': X_test, 'labels': y_test})

    # Optional model configuration
    # todo3 : I added output_dir="output_dir"
    model_args = ClassificationArgs(
        num_train_epochs=1, output_dir="outputs", overwrite_output_dir=True)

    # todo2 : use_cuda=False why??? doesnt work otherwise
    # Create a ClassificationModel
    model = ClassificationModel(
        "roberta", "roberta-base", args=model_args, use_cuda=False, num_labels=num_of_classes)

    # Train the model
    model.train_model(train_df)

    # Evaluate the model
    # todo: 4 - need to save these parameters too!!
    result, model_outputs, wrong_predictions = model.eval_model(eval_df)

    # save model
    file_name = 'models\\model_29.8_2'
    outfile = open(file_name, 'wb')
    pickle.dump(model, outfile)
    outfile.close()

    # Make predictions with the model
    # predictions, raw_outputs = model.predict(["Sam was a Wizard"])
