from googleapiclient.discovery import build
import pandas
import emoji
import re

YOUTUBE_API_KEY = "AIzaSyCgJDfT5eyRi6M9u893npJRC7gQ6Ql_YQU"
HEBREW_LETTERS = ['א', 'ב', 'ג', 'ד', 'ה', 'ו', 'ז', 'ח', 'ט', 'י', 'כ', 'ל', 'מ', 'נ', 'ס', 'ע', 'פ', 'צ', 'ק', 'ר', 'ש', 'ת']

LIST_OF_CHANNEL_IDS = {
    "UCDJ6HHS5wkNaSREumdJRYhg": "Kan11",
    "UC_HwfTAcjBESKZRJq6BTCpg": "Kan11News",
    "UCP926AKuhsPeSAnqf8avq7w": "ScienceDavidson",
}


def is_heb(text, letters_limit=20):
    '''
    Check if there is an hebrew letter in the first 'letters_limit' letters of the string
    Use 'letters_limit=None' to check entire string
    '''
    for i, letter in enumerate(text):
        if (letters_limit is not None) and (i >= letters_limit):
            return False

        if letter in HEBREW_LETTERS:
            return True

    return False


def senitize_text(text):
    '''
    Clean text from unwanted characters
    Return list of emojis in text
    '''
    # Remove characters using regex
    text = re.sub('\,', '', text)
    text = re.sub('\.', '', text)
    text = re.sub('\"', '', text)
    text = re.sub('\'', '', text)

    # Convert tabs to spaces
    text = text.replace('\t', ' ')

    # Remove excess whitespaces using regex
    text = re.sub('\s+', ' ', text)
    text.strip()

    # Remove non hebrew letters using regex
    text = re.sub('[^א-ת]', '', text)

    # Remove HTML tags using regex
    text = re.sub('<.*?>', '', text)

    return text


def split_emojis_from_text(text):
    '''
    Split text from emojis
    Return text and list of emojis
    '''
    # Get all emojis in text
    emojis_list = emoji.emoji_list(text)
    # Keep only emoji from response
    emoji_text = [emoji.emoji for emoji in emojis_list]

    # Remove emojis from text
    text = emoji.replace_emoji(text)

    return text, emojis_list


def get_comments_from_video(youtube, video_ID, max_emojis_in_text=10):
    '''
    Return list of comments from video ID
    Returns DF with columns: text, emoji
    Will return only top comments (No comments in comment threads)
    '''
    df = pandas.DataFrame(columns=['text', 'emoji'])

    try:
        res = youtube.commentThreads().list(
            part="snippet",
            videoId=video_ID,
            maxResults=10000,
            order="orderUnspecified"
        ).execute()

        items = res["items"]

        for item in items:
            text = item["snippet"]["topLevelComment"]["snippet"]["textDisplay"]

            # Skip non-hebrew text
            if is_heb(text):
                # Senitize text
                text = senitize_text(text)

                # Create row for each emoji in text is there are less than 'max_emojis_in_text' emojis in text
                text, emojis_list = split_emojis_from_text(text)
                if len(emojis_list) <= max_emojis_in_text:
                    for emoji in emojis_list:
                        df = pandas.concat([df, {'text': text, 'emoji': emoji}])

    except Exception as e:
        print(f'HttpError: {e}')
        return None

    return df


def get_comments_from_videos(youtube, video_IDs_list):
    '''
    Return list of comments from all the videos in  video_IDs_list
    Will return only top comments (No comments in comment threads)
    '''
    df = pandas.DataFrame(columns=['text', 'emoji'])

    number_of_videos = len(video_IDs_list)
    progress_percentage = 0
    for i, video_id in enumerate(video_IDs_list):
        # Show progress
        new_progress_percentage = int((i / number_of_videos) * 100)
        if new_progress_percentage > progress_percentage:
            print(f'{new_progress_percentage}%')
            progress_percentage = new_progress_percentage

        # Get video comments
        df = pandas.concat([df, get_comments_from_video(youtube, video_id)])


def get_channel_videos(youtube, channel_ID):
    '''
    Return list of video IDs from channel ID
    '''
    res = youtube.channels().list(id=channel_ID,
                                  part='contentDetails').execute()

    playlist_id = res['items'][0]['contentDetails']['relatedPlaylists']['uploads']

    videos = []
    next_page_token = None

    while 1:
        res = youtube.playlistItems().list(playlistId=playlist_id,
                                           part='snippet',
                                           maxResults=5000,
                                           pageToken=next_page_token).execute()
        videos += res['items']
        next_page_token = res.get('nextPageToken')

        if next_page_token is None:
            break

    videos_id_list = []
    for video in videos:
        videos_id_list.append(video['snippet']['resourceId']["videoId"])

    return videos_id_list


def create_CSV_from_DT(df, output_file_name):
    '''
    Create CSV from dataframe
    file name is output_file_name
    '''
    output_file_name += '.csv'
    df.to_csv(output_file_name, index=False, header=False, encoding='utf-8')


if __name__ == '__main__':
    # Init API
    youtube = build('youtube', 'v3', developerKey=YOUTUBE_API_KEY)

    list_of_comments = []
    for channel_id in LIST_OF_CHANNEL_IDS:
        print(f'Getting comments from {LIST_OF_CHANNEL_IDS[channel_id]}')

        # Get videos from channel
        print('Getting video IDs list')
        videos_id_list = get_channel_videos(youtube, channel_id)

        # Get comments from videos
        print('Getting comments from videos')
        list_of_comments += get_comments_from_videos(youtube, videos_id_list)

        # Create CSV from dataframe
        print('Creating CSV from dataframe')
        create_CSV_from_DT(list_of_comments, LIST_OF_CHANNEL_IDS[channel_id])

    print("Done")
