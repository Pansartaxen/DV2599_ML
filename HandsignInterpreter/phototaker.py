from PIL import Image
from numpy import asarray

def png_to_array(path):
    png = Image.open(path)
    png_array = asarray(png)

    print(png_array[0][0])
    print(png_array.shape)

if __name__ == "__main__":
    png_to_array('HandsignInterpreter/test.png')