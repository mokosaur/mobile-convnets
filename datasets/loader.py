from bs4 import BeautifulSoup
import requests
import re
from urllib.request import urlopen, Request
import os
import json


def get_soup(url, header):
    return BeautifulSoup(urlopen(Request(url, headers=header)), 'html.parser')


def download_images(dirname, query):
    image_type = "ActiOn"
    query = query.split()
    query = '+'.join(query)
    url = "https://www.google.co.in/search?q=" + query + "&source=lnms&tbm=isch"
    print(url)
    header = {
        'User-Agent': "Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/43.0.2357.134 Safari/537.36"
    }
    soup = get_soup(url, header)

    actual_images = []  # contains the link for Large original images, type of  image
    for a in soup.find_all("div", {"class": "rg_meta"}):
        link, Type = json.loads(a.text)["ou"], json.loads(a.text)["ity"]
        actual_images.append((link, Type))

    print("there are total", len(actual_images), "images")

    if not os.path.exists(dirname):
        os.mkdir(dirname)
    dirname = os.path.join(dirname, query.split()[0])

    if not os.path.exists(dirname):
        os.mkdir(dirname)
    ###print images
    for i, (img, Type) in enumerate(actual_images):
        try:
            req = Request(img, headers=header)
            raw_img = urlopen(req).read()

            cntr = len([i for i in os.listdir(dirname) if image_type in i]) + 1
            print(cntr)
            if len(Type) == 0:
                f = open(os.path.join(dirname, image_type + "_" + str(cntr) + ".jpg"), 'wb')
            else:
                f = open(os.path.join(dirname, image_type + "_" + str(cntr) + "." + Type), 'wb')

            f.write(raw_img)
            f.close()
        except Exception as e:
            print("could not load : " + img)
            print(e)


def load(dataset_name, classes):
    pass


# download_images("cladonia", "cladonia fimbriata")
# download_images("cladonia", "cladonia chlorophaea")
# download_images("cladonia", "cladonia arbuscula")
# download_images("cladonia", "cladonia rangiferina")
# download_images("cladonia", "cladonia uncialis")
# download_images("cladonia", "cladonia gracilis")
# download_images("cladonia", "cladonia pyxidata")
# download_images("cladonia", "cladonia squamosa")
# download_images("cladonia", "cladonia coniocraea")
# download_images("cladonia", "cladonia subulata")
# download_images("cladonia", "cladonia furcata")
# download_images("cladonia", "cladonia rangiformis")
# download_images("moss", "sphagnum palustre")
# download_images("moss", "sphagnum fuscum")
# download_images("moss", "sphagnum fimbriatum")
# download_images("moss", "sphagnum flexuosum")
# download_images("moss", "polytrichum piliferum")
# download_images("moss", "polytrichum commune")
# download_images("moss", "polytrichum juniperinum")
# download_images("moss", "bryum weigelii")
# download_images("moss", "dicranum scoparium")
# download_images("moss", "splachnum sphaericum")

# download_images("lichen", "xanthoria elegans")
# download_images("lichen", "xanthoria parietina")
# download_images("lichen", "cetraria islandica")
# download_images("lichen", "cetraria nivalis")
# download_images("lichen", "cetraria tilesii")
# download_images("lichen", "cetraria aculeata")
# download_images("lichen", "alectoria ochroleuca")
# download_images("lichen", "alectoria sarmentosa")