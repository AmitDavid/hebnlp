import pandas as pd
import pickle
import re
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import AdaBoostClassifier

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import train_test_split
import numpy as np
import os
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from datetime import datetime


SUBGROUP_LIMIT = 100000
EMOJI_SUBGROUPS = {
    '😃': {'🥂','🥳','✨','🍻','🎶','💐','🎆','🎊','🎉','🏅','🔥','💯','👑','🏆','😀', '😃', '😄', '😁', '😅', '😉', '😊', '😇', '😋', '😛', '🤗', '🤭', '🤩', '😺', '😸', '🤓', '🧐', },
    '🤣': {'🤣', '😂', '😆', '🤪', '😝', '😹'},
    '💗': {'🌈','🙏','💋', '💌', '💘', '💝', '💖', '💗', '💓', '💞', '💕', '💟', '❣', '❤', '🧡', '💛', '💚', '💙', '💜', '🤎', '🖤', '🤍','🥰', '😍', '😘', '😗', '☺', '😚', '😙', '😻', '😽', },
    '😢': {'🤦‍','😦', '😧', '😨', '😰', '😥', '😢', '😭', '😩', '😫', '😟', '💔', '🙁', '😿', '😖', '😣', '😞', '😓', '☹', '🥺', '😕', '🤒', '🤕', '🤧', '😔', '😪', '👎' },
    '😱': {'🙀', '😯', '😱', '😮', '😲', '😳', '😵', '🤯', },
    '😾': {'🤮','😾', '😤', '😡', '😠', '🤬', '👿', '😒','🖕'},
    '👍': {'👍','👌','🦾','💪','🤟','🖖','✌','🙌','👏','🙂', '🙃','🤤', '😌','😎', '🤠'},
}

def even_emojis(all_data, emoji_dict):
    df_emoji_list = []
    num_labels = len(emoji_dict)
    for label in range(num_labels):
        df = all_data[all_data['labels'] == label]
        if (df.shape[0] > SUBGROUP_LIMIT):
            df = df.sample(n=SUBGROUP_LIMIT)

        print(str(emoji_dict[label]) + ':  ' +str(df.shape[0]))
        df_emoji_list+=[df]
    
    return pd.concat(df_emoji_list)


def evaluate(y_true, y_pred, labels, model_name):
    '''
    Print the evaluation metrics of the model
    '''
    print(model_name)

    print("macro:")
    print(precision_recall_fscore_support(y_true, y_pred, average='macro', warn_for=tuple()))

    print("\nmicro:")
    print(precision_recall_fscore_support(y_true, y_pred, average='micro', warn_for=tuple()))

    print("\nconfusion matrix:")
    print(confusion_matrix(y_true, y_pred, labels=labels))

    print("\naccuracy:")
    print(accuracy_score(y_true, y_pred))


def reads_csv(path):
    '''
    Read csv file and return a dataframe
    '''
    # read from csv
    df = pd.read_csv(path, names=["text", "emoji"], encoding='utf-8')

    # replace &quot; with "
    df['text'] = df['text'].apply(lambda x: re.sub('&quot;', '"', x))

    return df


def prepare_data():
    '''
    Read and prepare the data for training
    '''
    # load data from csv
    data_list = os.listdir('data')

    df = []
    for channel in data_list:
        if channel.lower().endswith(".csv"):
            print(f'Loading {channel}')
            df += [reads_csv(f'data\\{channel}')]

    # concat data and convert emojis to unique numbers(labels)
    all_data = pd.concat(df)
    all_data_2 = pd.DataFrame()
    for emoji in EMOJI_SUBGROUPS:
        for emoji_to_replace in EMOJI_SUBGROUPS[emoji]:
            regex = emoji_to_replace + '.*'
            all_data['emoji'] = all_data['emoji'].str.replace(regex, emoji, regex=True)

    for index, row in all_data.iterrows():
        if row['emoji'] in EMOJI_SUBGROUPS:
            all_data_2 = all_data_2.append(row)

    all_data = all_data_2

    all_data['labels'] = all_data.groupby(["emoji"]).ngroup()

    all_data = all_data.drop_duplicates()
    return all_data


def create_emoji_label_dict(file_name, all_data):
    emoji_dict = {}
    for index, row in all_data.iterrows():
        emoji_dict[int(row["labels"])] = row["emoji"]

    # save dictionary
    file_name = file_name + '_dict'
    outfile = open(file_name, 'wb')
    pickle.dump(emoji_dict, outfile)
    outfile.close()

    return emoji_dict


def save_var_to_file(file_name, var):
    outfile = open(file_name, 'wb')
    pickle.dump(var, outfile)
    outfile.close()


def create_correct_incorrect_excels(X_test, y_test):
    df = pd.DataFrame(X_test.to_numpy(), columns=["text"])
    df["actual"] = [emoji_dict[tes] for tes in y_test.to_numpy()]
    df["predicted"] = [emoji_dict[pre] for pre in test_predicted]
    incorrect = df[df["actual"] != df["predicted"]]
    correct = df[df["actual"] == df["predicted"]]
    incorrect.to_excel("incorrect.xlsx")
    correct.to_excel("correct.xlsx")


if __name__ == '__main__':

    all_data = prepare_data()
    print("data is ready")

    title = 'AdaBoost'
    #title = 'logistic_regression'
    #title = 'Random_Forest'
    now = datetime.now()
    file_name = f'models\\{now.strftime("%d-%m_%H-%M")}_{title}'
    print(f"file name is {file_name}")

    # create dictionary (label->emoji)
    emoji_dict = create_emoji_label_dict(file_name, all_data)
    print("emoji_dict is ready")

    all_data = all_data.drop('emoji', axis=1)

    all_data = even_emojis(all_data, emoji_dict)
    #Preparing train data and eval data
    X_train, X_test, y_train, y_test = train_test_split(
        all_data['text'], all_data['labels'], test_size=0.2, random_state=42)
    train_df = pd.DataFrame({'text': X_train, 'labels': y_train})
    test_df = pd.DataFrame({'text': X_test, 'labels': y_test})

    
    count_vect = CountVectorizer(ngram_range=(1, 2))
    X_train_counts = count_vect.fit_transform(train_df.text)
    X_test_counts = count_vect.transform(test_df.text)

    tf_transformer = TfidfTransformer(use_idf=False).fit(X_train_counts)
    X_train_tf = tf_transformer.transform(X_train_counts)
    X_test_tfidf = tf_transformer.transform(X_test_counts)


    #Logistic Regression Classifier
    #model=LogisticRegression().fit(X_train_tf, train_df.labels)
    #model = RandomForestClassifier(n_estimators=20).fit(X_train_tf, train_df.labels)

    abc = AdaBoostClassifier(n_estimators=50, learning_rate=1)
    model = abc.fit(X_train_tf, y_train)
    print("finished: creating classifier")

    test_predicted = model.predict(X_test_tfidf)

    print("finished: predict the test set")

    inverse_dict = {count_vect.vocabulary_[w]: w for w in count_vect.vocabulary_.keys()}

    # write wrong\right classification to excel
    create_correct_incorrect_excels(X_test, y_test)

    y_test.to_numpy()

    print(emoji_dict)

    evaluate(y_test, test_predicted, model.classes_, file_name)

    save_var_to_file(file_name + '_model', model)
    save_var_to_file(file_name + '_count_vect', count_vect)
    save_var_to_file(file_name + '_tf_transformer', tf_transformer)

