# Criado por: André Luiz Santos
# Data: 10/02/2020
# Função: buscar o metadata da imagem.
from PIL import Image
from PIL.ExifTags import TAGS
from PIL.ExifTags import GPSTAGS


def __version__():
    return 'metadata version: v0.2'


def get_exif_all(image_path):
    ret = {}
    i = Image.open(image_path)
    info = i._getexif()
    for tag, value in info.items():
        decoded = TAGS.get(tag, tag)
        ret[decoded] = value
    return ret


def get_geotagging(exif):
    if not exif:
        raise ValueError("No EXIF metadata found")

    geotagging = {}
    for (idx, tag) in TAGS.items():
        if tag == 'GPSInfo':
            if idx not in exif:
                raise ValueError("No EXIF geotagging found")

            for (key, val) in GPSTAGS.items():
                if key in exif[idx]:
                    geotagging[val] = exif[idx][key]

    return geotagging


def get_labeled_exif(exif):
    labeled = {}
    for (key, val) in exif.items():
        labeled[TAGS.get(key)] = val

    return labeled


def get_exif(image_path):
    image = Image.open(image_path)
    image.verify()
    return image._getexif()


def get_gps(image_path):
    exif = get_exif(image_path)
    if exif is not None:
        geotags = get_geotagging(exif)
        if 'GPSAltitude' in geotags:
            alt = geotags['GPSAltitude']
            lat = str(geotags['GPSLatitude'])
            long = str(geotags['GPSLongitude'])

            return alt, lat, long
    return 500


def get_datetime(image_path):
    exif_tags = get_exif_all(image_path)
    dt_original = exif_tags['DateTimeOriginal']
    return dt_original


def segment_estimation(h=0):
    default = 500
    low = 1000
    medium = 2000
    medium_high = 2800
    high = 3000

    if 75 < h <= 150:
        segment = high
    elif 40 < h <= 75:
        segment = medium_high
    elif 25 < h <= 40:
        segment = medium
    elif 5 < h <= 25:
        segment = low
    else:
        segment = default

    return segment

