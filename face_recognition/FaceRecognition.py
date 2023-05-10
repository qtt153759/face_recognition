# common dependencies
import os
from os import path
import warnings
import time
import pickle
import logging

# 3rd party dependencies
import numpy as np
import pandas as pd
from tqdm import tqdm
import cv2
import tensorflow as tf
from deprecated import deprecated

# package dependencies
from face_recognition.basemodels import (
    VGGFace,

)
from face_recognition.commons import functions, realtime, distance as dst

# -----------------------------------
# configurations for dependencies

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
tf_version = int(tf.__version__.split(".", maxsplit=1)[0])
if tf_version == 2:
    tf.get_logger().setLevel(logging.ERROR)
# -----------------------------------


def build_model():

    global model_obj

   
    return VGGFace.loadModel()




def find(
    img_path,
    db_path,
    # model_name="VGG-Face",
    enforce_detection=True,
    # detector_backend="opencv",
    isSkip=False,
    align=True,
    normalization="base",
    silent=False,
):
    tic = time.time()

    # -------------------------------
    if os.path.isdir(db_path) is not True:
        raise ValueError("Passed db_path does not exist!")

    target_size = (224, 224)

    # ---------------------------------------

    file_name = f"representations_VGG.pkl"
    file_name = file_name.replace("-", "_").lower()

    if path.exists(db_path + "/" + file_name):

        if not silent:
            print(
                f"WARNING: Representations for images in {db_path} folder were previously stored"
                + f" in {file_name}. If you added new instances after the creation, then please "
                + "delete this file and call find function again. It will create it again."
            )

        with open(f"{db_path}/{file_name}", "rb") as f:
            representations = pickle.load(f)

        if not silent:
            print("There are ", len(representations), " representations found in ", file_name)

    else:  # create representation.pkl from scratch
        employees = []

        for r, _, f in os.walk(db_path):
            for file in f:
                if (
                    (".jpg" in file.lower())
                    or (".jpeg" in file.lower())
                    or (".png" in file.lower())
                ):
                    exact_path = r + "/" + file
                    employees.append(exact_path)

        if len(employees) == 0:
            raise ValueError(
                "There is no image in ",
                db_path,
                " folder! Validate .jpg or .png files exist in this path.",
            )

        # ------------------------
        # find representations for db images

        representations = []

        # for employee in employees:
        pbar = tqdm(
            range(0, len(employees)),
            desc="Finding representations",
            disable=silent,
        )
        for index in pbar:
            employee = employees[index]

            img_objs = functions.extract_faces(
                img=employee,
                target_size=target_size,
                isSkip=isSkip,
                grayscale=False,
                enforce_detection=enforce_detection,
                align=align,
            )

            for img_content, _, _ in img_objs:
                embedding_obj = represent(
                    img_path=img_content,
                    # model_name=model_name,
                    enforce_detection=enforce_detection,
                    isSkip=True,
                    align=align,
                    normalization=normalization,
                )

                img_representation = embedding_obj[0]["embedding"]

                instance = []
                instance.append(employee)
                instance.append(img_representation)
                representations.append(instance)

        # -------------------------------

        with open(f"{db_path}/{file_name}", "wb") as f:
            pickle.dump(representations, f)

        if not silent:
            print(
                f"Representations stored in {db_path}/{file_name} file."
                + "Please delete this file when you add new identities in your database."
            )

    # ----------------------------
    # now, we got representations for facial database
    df = pd.DataFrame(representations, columns=["identity", "VGG_representation"])

    # img path might have more than once face
    target_objs = functions.extract_faces(
        img=img_path,
        target_size=target_size,
        isSkip=isSkip,
        grayscale=False,
        enforce_detection=enforce_detection,
        align=align,
    )

    resp_obj = []

    for target_img, target_region, _ in target_objs:
        target_embedding_obj = represent(
            img_path=target_img,
            # model_name=model_name,
            enforce_detection=enforce_detection,
            isSkip=True,
            align=align,
            normalization=normalization,
        )

        target_representation = target_embedding_obj[0]["embedding"]

        result_df = df.copy()  # df will be filtered in each img
        result_df["source_x"] = target_region["x"]
        result_df["source_y"] = target_region["y"]
        result_df["source_w"] = target_region["w"]
        result_df["source_h"] = target_region["h"]

        distances = []
        for index, instance in df.iterrows():
            source_representation = instance["VGG_representation"]

            
            distance = dst.findCosineDistance(source_representation, target_representation)
            
            distances.append(distance)

            # ---------------------------

        result_df["VGG_cosine"] = distances

        threshold = dst.findThreshold()
        result_df = result_df.drop(columns=["VGG_representation"])
        result_df = result_df[result_df["VGG_cosine"] <= threshold]
        result_df = result_df.sort_values(
            by=["VGG_cosine"], ascending=True
        ).reset_index(drop=True)

        resp_obj.append(result_df)

    # -----------------------------------

    toc = time.time()

    if not silent:
        print("find similarity took ", toc - tic, " seconds")

    return resp_obj


def represent(
    img_path,
    # model_name="VGG-Face",
    enforce_detection=True,
    isSkip=False,
    align=True,
    normalization="base",
):

    resp_objs = []

    model = build_model()

    # ---------------------------------
    # we have run pre-process in verification. so, this can be skipped if it is coming from verify.
    target_size = (224, 224)
    if isSkip ==False:
        img_objs = functions.extract_faces(
            img=img_path,
            target_size=target_size,
            isSkip=isSkip,
            grayscale=False,
            enforce_detection=enforce_detection,
            align=align,
        )
    else:  # skip
        if isinstance(img_path, str):
            img = functions.load_image(img_path)
        elif type(img_path).__module__ == np.__name__:
            img = img_path.copy()
        else:
            raise ValueError(f"unexpected type for img_path - {type(img_path)}")
        # --------------------------------
        if len(img.shape) == 4:
            img = img[0]  # e.g. (1, 224, 224, 3) to (224, 224, 3)
        if len(img.shape) == 3:
            img = cv2.resize(img, target_size)
            img = np.expand_dims(img, axis=0)
        # --------------------------------
        img_region = [0, 0, img.shape[1], img.shape[0]]
        img_objs = [(img, img_region, 0)]
    # ---------------------------------

    for img, region, confidence in img_objs:
        # custom normalization
        img = functions.normalize_input(img=img, normalization=normalization)

        # represent
        if "keras" in str(type(model)):
            # new tf versions show progress bar and it is annoying
            embedding = model.predict(img, verbose=0)[0].tolist()
        else:
            # SFace and Dlib are not keras models and no verbose arguments
            embedding = model.predict(img)[0].tolist()

        resp_obj = {}
        resp_obj["embedding"] = embedding
        resp_obj["facial_area"] = region
        resp_obj["face_confidence"] = confidence
        resp_objs.append(resp_obj)

    return resp_objs


def stream(
    db_path="",
    # model_name="VGG-Face",
    # detector_backend="opencv",
    isSkip=False,
    # enable_face_analysis=True,
    source=0,
    time_threshold=1,
    frame_threshold=1,
):

    if time_threshold < 1:
        raise ValueError(
            "time_threshold must be greater than the value 1 but you passed " + str(time_threshold)
        )

    if frame_threshold < 1:
        raise ValueError(
            "frame_threshold must be greater than the value 1 but you passed "
            + str(frame_threshold)
        )

    realtime.analysis(
        db_path,
        # model_name,
        # detector_backend,
        isSkip=isSkip,
        source=source,
        time_threshold=time_threshold,
        frame_threshold=frame_threshold,
    )


def extract_faces(
    img_path,
    target_size=(224, 224),
    # detector_backend="opencv",
    isSkip=False,
    enforce_detection=True,
    align=True,
    grayscale=False,
):

    

    resp_objs = []
    img_objs = functions.extract_faces(
        img=img_path,
        target_size=target_size,
        isSkip=isSkip,
        grayscale=grayscale,
        enforce_detection=enforce_detection,
        align=align,
    )

    for img, region, confidence in img_objs:
        resp_obj = {}

        # discard expanded dimension
        if len(img.shape) == 4:
            img = img[0]

        resp_obj["face"] = img[:, :, ::-1]
        resp_obj["facial_area"] = region
        resp_obj["confidence"] = confidence
        resp_objs.append(resp_obj)

    return resp_objs


# ---------------------------
# deprecated functions



# ---------------------------
# main

functions.initialize_folder()


def cli():
    """
    command line interface function will be offered in this block
    """
    import fire

    fire.Fire()
