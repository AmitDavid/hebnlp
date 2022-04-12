from bs4 import BeautifulSoup
import requests

xmlDict = {}

r = requests.get("https://thinkil.co.il/texts-sitemap.xml")
xml = r.text

soup = BeautifulSoup(xml)
sitemapTags = soup.find_all("sitemap")

print("The number of sitemaps are {0}".format(len(sitemapTags)))

for sitemap in sitemapTags:
    xmlDict[sitemap.findNext("loc").text] = sitemap.findNext("lastmod").text

print(xmlDict)