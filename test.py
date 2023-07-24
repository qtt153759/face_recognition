# from face_recognition import FaceRecognition


import matplotlib.pyplot as plt
from face_recognition import FaceRecognition
import matplotlib.image as mpimg

img_path = "data/testImg/tu-vu-dang-van-lam-ve-nuoc-quang-hai-co-the-that-bai-o-sau-pau-fc-vi-khong-nghe-loi-hlv-park-hang-seo-4.jpg"
db_path="data/dataset"

# find
dfs = FaceRecognition.find(
    img_path=img_path, db_path=db_path
)
for df in dfs:
    print(df)

# extract faces
face_objs = FaceRecognition.extract_faces(
    img_path=img_path
)
rows, columns=2,2
for face_obj,df in zip(face_objs,dfs):
    fig = plt.figure(figsize=(10, 7))
    fig.add_subplot(rows, columns, 1)
    face = face_obj["face"]
    plt.imshow(face)
    plt.axis('off')
    plt.title("First")
      

    top_three_img_paths = df["identity"].values[:3]
    print(top_three_img_paths)
    # Adds a subplot at the 2nd position
    fig.add_subplot(rows, columns, 2)
    # showing image
    plt.imshow(mpimg.imread(top_three_img_paths[0]))
    plt.axis('off')
    plt.title("Second")
      
    # Adds a subplot at the 3rd position
    fig.add_subplot(rows, columns, 3)
      
    # showing image
    plt.imshow(mpimg.imread(top_three_img_paths[1]))
    plt.axis('off')
    plt.title("Third")
      
    # Adds a subplot at the 4th position
    fig.add_subplot(rows, columns, 4)
      
    # showing image
    plt.imshow(mpimg.imread(top_three_img_paths[2]))
    plt.axis('off')
    plt.title("Fourth")
    plt.show()
