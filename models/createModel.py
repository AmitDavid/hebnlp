import os
import pickle
import re
from datetime import datetime

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, confusion_matrix,
                             precision_recall_fscore_support)
# sklearn classifiers
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestCentroid
from sklearn import tree
from sklearn.neural_network import MLPClassifier, MLPRegressor

ENGLISH_LETTERS = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']

SUBGROUP_LIMIT = 100000000000
TEXT_LEN_LIMIT = 100000000000
FILE_NAME_DATE_FORMAT = "%m-%d_%H-%M"
TEST_SIZE = 0.2

EMOJI_SUBGROUPS = {
    '😃': {'👍', '👌', '🦾', '💪', '🤟', '🖖', '✌', '🙌', '👏', '🙂', '🙃', '🤤', '😌', '😎', '🤠','🥂', '🥳', '✨', '🍻', '🎶', '💐', '🎆', '🎊', '🎉', '🏅', '🔥', '💯', '👑', '🏆', '😀', '😃', '😄', '😁', '😅', '😉', '😊', '😇', '😋', '😛', '🤗', '🤭', '🤩', '😺', '😸', '🤓', '🧐', },
    '🤣': {'🤣', '😂', '😆', '🤪', '😝', '😹'},
    '🥰': {'🌈', '🙏', '💋', '💌', '💘', '💝', '💖', '💗', '💓', '💞', '💕', '💟', '❣', '❤', '🧡', '💛', '💚', '💙', '💜', '🤎', '🖤', '🤍', '🥰', '😍', '😘', '😗', '☺', '😚', '😙', '😻', '😽', },
    '😢': {'🤦‍', '😦', '😧', '😨', '😰', '😥', '😢', '😭', '😩', '😫', '😟', '💔', '🙁', '😿', '😖', '😣', '😞', '😓', '☹', '🥺', '😕', '🤒', '🤕', '🤧', '😔', '😪'},
    '😱': {'🙀', '😯', '😱', '😮', '😲', '😳', '😵', '🤯', },
    '😡': {'👎','🤮', '😾', '😤', '😡', '😠', '🤬', '👿', '😒', '🖕'},
}

MODEL_TITLES = ['LogisticRegression', 'RandomForestClassifier', 'AdaBoostClassifier', 'NaiveBayes', 'NearestNeighbors', 'DecisionTreeClassifier', 'NeuralNetwork']


def get_classifier(title):
    '''
    Get the model from the title
    '''
    if title == 'LogisticRegression':
        return LogisticRegression(class_weight='balanced')
    elif title == 'RandomForestClassifier':
        return RandomForestClassifier(n_estimators=20)
    elif title == 'AdaBoostClassifier':
        return AdaBoostClassifier(n_estimators=50, learning_rate=1)
    elif title == 'NaiveBayes':
        return MultinomialNB()
    elif title == 'NearestNeighbors':
        return NearestCentroid()
    elif title == 'DecisionTreeClassifier':
        return tree.DecisionTreeClassifier()
    elif title == 'NeuralNetwork':
        return MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)
    else:
        print(f"Unknown model title: {title}")

    return None


def is_non_english(text):
    '''
    Check if there is an hebrew letter in the first 'letters_limit' letters of the string
    Use 'letters_limit=None' to check entire string
    '''
    text = text.lower()
    for i, letter in enumerate(text):

        if letter in ENGLISH_LETTERS:
            return False

    return True

def limit_emojis_subgroup_size(all_data, label_to_emoji_dict, subgroup_limit):
    '''
    Limit the size of each emoji subgroup to 'subgroup_limit'
    Return dataframe with each label showing no more than 'subgroup_limit' times
    '''
    limited_df = pd.DataFrame()
    for label in label_to_emoji_dict:
        label_df = all_data[all_data['labels'] == label]
        if (label_df.shape[0] > subgroup_limit):
            label_df = label_df.sample(n=subgroup_limit, random_state=1)

        limited_df = pd.concat([limited_df, label_df])

    return limited_df


def evaluate(y_true, y_pred, labels, model_name):
    '''
    Print the evaluation metrics of the model
    '''
    string_to_print = f'Statistics for {model_name}'
    string_to_print += "\nNumber of samples per label (Test semple size):\n"

    statistics_df = pd.DataFrame(
        get_label_and_emoji_dicts()[0], index=['emoji'])

    # Get sum
    sum_of_labels = pd.Series(y_true).value_counts()
    sum_of_labels = sum_of_labels.to_frame('sum')
    statistics_df = pd.concat([statistics_df, sum_of_labels.T])

    # Get mean
    mean_of_labels = pd.Series(y_true).value_counts(normalize=True)
    mean_of_labels = mean_of_labels.to_frame('mean')
    statistics_df = pd.concat([statistics_df, mean_of_labels.T])

    string_to_print += str(statistics_df.T)

    string_to_print += "\n\nmacro:"
    string_to_print += str(precision_recall_fscore_support(y_true, y_pred, average='macro', warn_for=tuple()))

    string_to_print += "\n\nmicro:"
    string_to_print += str(precision_recall_fscore_support(y_true, y_pred, average='micro', warn_for=tuple()))

    string_to_print += "\n\nconfusion matrix:\n"
    string_to_print += str(confusion_matrix(y_true, y_pred, labels=labels))

    string_to_print += "\n\naccuracy:"
    string_to_print += str(accuracy_score(y_true, y_pred))

    print(string_to_print)

    # Save output to file
    with open(f'{model_name}_evaluation.txt', 'w', encoding='utf8') as file:
        file.write(string_to_print)


def reads_csv(path):
    '''
    Read csv file and return a dataframe
    '''
    # Read from csv
    df = pd.read_csv(path, names=["text", "emoji"], encoding='utf-8')

    # Replace &quot; with "
    df['text'] = df['text'].apply(lambda x: re.sub('&quot;', '"', x))

    df_rows = pd.DataFrame()
    for index, row in df.iterrows():
        row_text = row['text']
        emoji = row['emoji']
        if (len(row['text']) < TEXT_LEN_LIMIT) and (is_non_english(row_text)):
            result = ''.join([i for i in row_text if not i.isdigit()])
            dff = pd.DataFrame([[result, emoji]], columns=["text", "emoji"])
            df_rows = pd.concat([df_rows, dff])

    return df_rows


def get_label_and_emoji_dicts(file_path=None):
    '''
    Get the dictionarys that maps each label to an emoji and vice versa
    When 'file_name != None', save only label to emoji dictionary to the given file path
    Return 2 dictionaries: label_to_emoji and emoji_to_label
    '''
    label_to_emoji_dict = {}
    emoji_to_label_dict = {}
    for index, emoji in enumerate(EMOJI_SUBGROUPS):
        label_to_emoji_dict[index] = emoji
        emoji_to_label_dict[emoji] = index

    if file_path is not None:
        save_variable_to_file(f'{file_path}_dict', label_to_emoji_dict)

    return label_to_emoji_dict, emoji_to_label_dict


def prepare_data(file_path):
    '''
    Read and prepare the data for training
    Return dataframe with text and labels
    '''
    print("Preparing data:")
    # Get list of all csv files in the 'data' directory
    csv_files_list = [file for file in os.listdir(
        'data') if file.endswith('.csv')]

    # Read all csv files and concatenate them into one dataframe
    print('\tLoading csv files')
    all_data = pd.DataFrame()
    for channel in csv_files_list:
        all_data = pd.concat([all_data, reads_csv(f'data\{channel}')])

    # Replace emoji with it's equivalence class label and remove any emoji modifiers (skin tone, gender etc..)
    print('\tReplacing emoji with it\'s equivalence class label')
    for emoji in EMOJI_SUBGROUPS:
        for emoji_to_replace in EMOJI_SUBGROUPS[emoji]:
            regex = emoji_to_replace + '.*'
            all_data['emoji'] = all_data['emoji'].str.replace(
                regex, emoji, regex=True)

    # Remove emojies that are not in our emoji dictionary
    print('\tRemoving emojies that are not in our emoji dictionary')
    df_to_model = all_data[all_data['emoji'].isin(EMOJI_SUBGROUPS)].copy()
    del all_data

    # Replace emoji with labels. Each emoji will convert to unique label. Save dictionary to a file
    print('\tAdding labels to the dataframe')
    emoji_to_label_dict = get_label_and_emoji_dicts(file_path)[1]
    df_to_model['emoji'] = df_to_model['emoji'].map(emoji_to_label_dict)

    # Rename 'emoji' column to 'labels'
    df_to_model.rename(columns={'emoji': 'labels'}, inplace=True)

    df_to_model = df_to_model.drop_duplicates(subset='text', keep='first')

    return df_to_model


def save_variable_to_file(file_path, variable):
    '''
    Save variable to file
    '''
    with open(file_path, 'wb') as f:
        pickle.dump(variable, f)


def create_correct_incorrect_csvs(X_test, y_test):
    '''
    Save the correct and incorrect predictions to csv files
    '''
    df = pd.DataFrame(X_test.to_numpy(), columns=["text"])
    labels_to_emoji_dict = get_label_and_emoji_dicts()[0]

    df["actual"] = [labels_to_emoji_dict[actual_emoji]
                    for actual_emoji in y_test.to_numpy()]
    df["predicted"] = [labels_to_emoji_dict[predicted_emoji]
                       for predicted_emoji in test_predicted]

    incorrect = df[df["actual"] != df["predicted"]]
    incorrect.to_csv("incorrect.csv")

    correct = df[df["actual"] == df["predicted"]]
    correct.to_csv("correct.csv")


if __name__ == '__main__':
    # Try different models
    for title in MODEL_TITLES:
        now = datetime.now()
        file_path = f'models\{now.strftime(FILE_NAME_DATE_FORMAT)}_{title}'
        print(f'File path is \'{file_path}\'')

        all_data = prepare_data(file_path)

        # Limit the size of each emoji subgroup to 'SUBGROUP_LIMIT'
        print(f'\tLimiting the size of each emoji subgroup to {SUBGROUP_LIMIT}')
        label_to_emoji_dict = get_label_and_emoji_dicts(file_path)[0]
        df_to_model = limit_emojis_subgroup_size(
            all_data, label_to_emoji_dict, SUBGROUP_LIMIT)

        # Preparing train data and eval data
        X_train, X_test, y_train, y_test = train_test_split(df_to_model['text'], df_to_model['labels'], test_size=TEST_SIZE, random_state=42)
        train_df = pd.DataFrame({'text': X_train, 'labels': y_train})
        test_df = pd.DataFrame({'text': X_test, 'labels': y_test})

        count_vect=CountVectorizer(ngram_range=(1, 2))
        X_train_counts = count_vect.fit_transform(train_df.text)
        X_test_counts = count_vect.transform(test_df.text)

        tf_transformer = TfidfTransformer(use_idf=True).fit(X_train_counts)
        X_train_tf = tf_transformer.transform(X_train_counts)
        X_test_tfidf = tf_transformer.transform(X_test_counts)

        print("Creating classifier")

        model = get_classifier(title)
        model = model.fit(X_train_tf, train_df.labels)

        print("Predict on test data")
        test_predicted = model.predict(X_test_tfidf)

        inverse_dict = {count_vect.vocabulary_[w]: w for w in count_vect.vocabulary_.keys()}

        # Write wrong\right classification to csv and print the evaluation
        create_correct_incorrect_csvs(X_test, y_test)
        evaluate(y_test.to_numpy(), test_predicted, model.classes_, file_path)

        save_variable_to_file(file_path + '_model', model)
        save_variable_to_file(file_path + '_count_vect', count_vect)
        save_variable_to_file(file_path + '_tf_transformer', tf_transformer)
