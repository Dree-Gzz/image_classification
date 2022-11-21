from deepface import DeepFace


# import matplotlib.pyplot as plt
# import matplotlib.image as mpimg


# def showImage(imagePath):
#     plt.imshow(mpimg.imread(imagePath))
#     plt.show()
#     print(username, " image displayed")


# Compares 2 images and concludes if they match or not
def facialVerification(imagePathA, imagePathB):
    result = DeepFace.verify(img1_path=imagePathA, img2_path=imagePathB)
    print(result)
    print(username, ' images match? --- ', result['verified'])


# Compares an image with multiple images and concludes if they match or not
def facialRecognition(imagePath, imageDirectoryPath):
    result = DeepFace.find(img_path=imagePath, db_path=imageDirectoryPath)
    print(result)


def detectFace(imagePath):
    result = DeepFace.detectFace(img_path=imagePath, target_size=(224, 224))
    print(result)


# Analyses an image and determines the image's metrics
def facialAnalysis(imagePath):
    result = DeepFace.analyze(img_path=imagePath, actions=('age', 'gender', 'race', 'emotion'))

    print(username, ' Gender : ', result['gender'])
    print(username, ' Age : ', result['age'])
    print(username, ' Race : ', result['race'])
    print(username, ' Emotion : ', result['emotion'])

    # showImage(imagePath)


def realTimeRecognition(imageDirectoryPath):
    DeepFace.stream(db_path=imageDirectoryPath)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    username = input("Enter username for image classification : ") + '\'s'
    realTimeRecognition("C:/Users/Dan Mwangi/Pictures/Camera Roll")
    # facialAnalysis("C:/Users/jmska/Downloads/image3.jpeg")
    # detectFace("C:/Users/jmska/Downloads/images/group.jpg")
    # facialRecognition('C:/Users/jmska/Downloads/images/4.jpg', 'C:/Users/jmska/Downloads/images')
