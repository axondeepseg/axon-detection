import requests
from PIL import Image
from io import BytesIO



url = 'https://raw.githubusercontent.com/axondeepseg/data_axondeepseg_sem/master/sub-rat2/micr/sub-rat2_sample-data5_SEM.png'
def loadImage():
    response = requests.get(url)

    if response.status_code == 200:
        img = Image.open(BytesIO(response.content))
        img.save('downloaded_image.png')
    else:
        print(f"Failed to retrieve the image. Status code: {response.status_code}")

