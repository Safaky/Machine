from PIL import Image

image = Image.open('myimage.jpg')
image.show()

print(image.mode)
print(image.size)


print(image.palette)
#Yukarısı elimizdeki görselin özelliklerini veriyor

image = Image.open('myimage.jpg')
image.save('new_image.png')
#jpg oldu png

image = Image.open('myimage.jpg')
new_image = image.resize((400, 400))
new_image.save('image_400.jpg')

print(image.size)
print(new_image.size)
#yeniden boyutlandırdı findera kaydetti

image = Image.open('myimage.jpg')
box = (200, 300, 700, 600)
cropped_image = image.crop(box)
cropped_image.save('cropped_image.jpg')


print(cropped_image.size)
#burda görseli kırptı



image = Image.open('myimage.jpg')

image_rot_90 = image.rotate(90)
image_rot_90.save('image_rot_90.jpg')

image_rot_180 = image.rotate(180)
image_rot_180.save('image_rot_180.jpg')
#görüntü dönderme

image = Image.open('myimage.jpg')

image_flip = image.transpose(Image.FLIP_LEFT_RIGHT)
image_flip.save('image_flip.jpg')
#aynalama

image = Image.open('myimage.jpg')

greyscale_image = image.convert('L')
greyscale_image.save('greyscale_image.jpg')

print(image.mode)
print(greyscale_image.mode)
#rgb den gri tonlamaya geçiş

image = Image.open('myimage.jpg')

red, green, blue = image.split()

print(image.mode)
print(red.mode)
print(green.mode)
print(blue.mode)

new_image = Image.merge("RGB", (green, red, blue))
new_image.save('new_image.jpg')

print(new_image.mode)
#Splitting and Merging Bands

from PIL import Image, ImageEnhance

image = Image.open('myimage.jpg')

contrast = ImageEnhance.Contrast(image)
contrast.enhance(1.5).save('contrast.jpg')
