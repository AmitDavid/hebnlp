# scraping emoji groups from AltCodeUnicode.com

import emoji
import requests
from bs4 import BeautifulSoup
import pickle



categories_lst = []
r = requests.get("https://altcodeunicode.com/emoji-group-smileys-emotion/")
page_soup = BeautifulSoup(r.text, "lxml")
categories_title = page_soup.find_all("h2")
categories_title = categories_title[1:]
for subgroup in categories_title:
    group_title = subgroup.string
    group_title = group_title.split(':')[1].strip()
    categories_lst.append(group_title)


groups_lst = []
tabels = page_soup.find_all("tbody")


for table in tabels:
    table_lst = []
    for row in table:
        extracted_emoji = emoji.distinct_emoji_list(str(row))
        if len(extracted_emoji) != 0:
            if len(extracted_emoji[0]) == 1:
                emoji_code = extracted_emoji[0]
                table_lst.append(emoji_code)
    
    if len(table_lst) != 0 :
        groups_lst.append(table_lst)



print(groups_lst)

print(categories_lst)



    #save groups_lst
    # file_name = 'emoji_subgroups'
    # outfile = open(file_name, 'wb')
    # pickle.dump(groups_lst, outfile)
    # outfile.close()


    #save groups corresponding titles
    # file_name = 'group_titles'
    # outfile = open(file_name, 'wb')
    # pickle.dump(categories_title, outfile)
    # outfile.close()