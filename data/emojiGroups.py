import emoji
import requests
from bs4 import BeautifulSoup

# scraping emoji groups from AltCodeUnicode.com
ALTCODEUNICODE_URL = "https://altcodeunicode.com/emoji-group-smileys-emotion/"

if __name__ == '__main__':
    # Get html and parse using BeautifulSoup
    request = requests.get(ALTCODEUNICODE_URL)
    page_soup = BeautifulSoup(request.text, "html")

    # Find all emoji group names
    categories = page_soup.find_all("h2")[1:]
    categories_name_list = []
    for category in categories:
        title = category.string
        title = title.split(':')[1].strip()
        categories_name_list.append(title)

    # Find emojis in each group
    emojies_list = []
    tables = page_soup.find_all("tbody")
    for table in tables:
        group_emojies_list = []
        for row in table:
            extracted_emoji = emoji.distinct_emoji_list(str(row))
            if (len(extracted_emoji) > 0) and (len(extracted_emoji[0]) == 1):
                emoji_code = extracted_emoji[0]
                group_emojies_list.append(emoji_code)

        if len(group_emojies_list) > 0:
            emojies_list.append(group_emojies_list)

    # Print results
    print(emojies_list)
    print(categories_name_list)
