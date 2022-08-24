from simpletransformers.classification import ClassificationModel, ClassificationArgs
import pandas as pd
import logging
import pickle
import re
from sklearn.model_selection import train_test_split


logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.WARNING)

LIST_OF_PATH_TO_CSV= {
    "Kan11":r"C:\Users\User\Desktop\study\hebnlp\hebnlp\data\Kan11_comments.csv",
   "Kan11News":r"C:\Users\User\Desktop\study\hebnlp\hebnlp\data\Kan11News_comments.csv",
    "ScienceDavidson":r"C:\Users\User\Desktop\study\hebnlp\hebnlp\data\ScienceDavidson_comments.csv",
}

def reads_csv(path):
    # read from csv
    df = pd.read_csv(path,names=["text", "emoji"], encoding='utf-8' )

    # replace &quot; with ""
    df['text'] = df['text'].apply(lambda x:re.sub('&quot;', '"', x))

    # reverse text
    # todo_1: problem with english..............
    df['text'] = df['text'].apply(lambda x:x[::-1])

    return df


if __name__ == '__main__':

    # load data from csv
    df_davison = reads_csv(LIST_OF_PATH_TO_CSV["ScienceDavidson"])
    df_Kan11 = reads_csv(LIST_OF_PATH_TO_CSV["Kan11"])
    df_kan11News = reads_csv(LIST_OF_PATH_TO_CSV["Kan11News"])

    # concat data and convert emojis to unique numbers 
    all_data = pd.concat([df_davison, df_Kan11, df_kan11News])
    all_data['labels'] = all_data.groupby(["emoji"]).ngroup()

   #### smaller data frame - to delete later
    num_of_classes = 20
    all_data = all_data[all_data['labels']< num_of_classes]


    # Preparing train data and eval data
    X_train, X_test, y_train, y_test = train_test_split(all_data['text'], all_data['labels'], test_size=0.2, random_state=42)
    train_df = pd.DataFrame({'text':X_train, 'labels':y_train})
    eval_df = pd.DataFrame({'text':X_test, 'labels':y_test})
   
   
    # Optional model configuration
    # todo3 : I added output_dir="output_dir"
    model_args = ClassificationArgs(num_train_epochs=1, output_dir="outputs", overwrite_output_dir=True)

    # todo2 : use_cuda=False why??? doesnt work otherwise
    # Create a ClassificationModel
    model = ClassificationModel("roberta", "roberta-base", args=model_args,num_labels=num_of_classes, use_cuda=False)

    # Train the model
    model.train_model(train_df)

    # Evaluate the model
    # todo: 4 - need to save these parameters too!! 
    result, model_outputs, wrong_predictions = model.eval_model(eval_df)

    #save model
    file_name = 'models\model_23.8'
    outfile = open(file_name, 'wb')
    pickle.dump(model, outfile)
    outfile.close()

    # Make predictions with the model
    # predictions, raw_outputs = model.predict(["Sam was a Wizard"])
