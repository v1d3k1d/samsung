import io
import os
import cv2

import numpy as np
import streamlit as st

from PIL import Image
from ultralytics import YOLO
from shapely.geometry import Polygon
from matplotlib.colors import to_hex


def trim_whitespace(image_path):
    with Image.open(image_path) as image:
        bbox = image.getbbox()
        image = image.crop(bbox)
        image.save(image_path)


def predict_image():
    if not os.path.exists(f"crop/{uploaded_file.name}"):
        os.mkdir(f"crop/{uploaded_file.name}", 0o666)

    results = model.predict(source=img, save=False, project="images", name=f"{uploaded_file.name}")
    masks=results[0].masks.xy

    i = 0
    for mask in masks:
        i += 1
        polygon_coords = mask
        polygon = Polygon(polygon_coords)
        image = cv2.imread(f"images/{uploaded_file.name}")
        mask = np.zeros_like(image[:, :, 0], dtype=np.uint8)
        cv2.fillPoly(mask, [np.array(polygon_coords, dtype=np.int32)], color=255)

        cropped_image = cv2.bitwise_and(image, image, mask=mask)
        cropped_image_bgra = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2BGRA)
        cropped_image_bgra[mask == 0] = (0, 0, 0, 0)

        path = f"crop/{uploaded_file.name}"
        cv2.imwrite(os.path.join(path , f"{i}.png"), cropped_image_bgra)
        trim_whitespace(os.path.join(path , f"{i}.png"))


def get_most_common_color(file):
    image = cv2.imread(file)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    height, width, _ = np.shape(image)
    colors = np.reshape(image, (height * width, 3))
    colors = colors[~np.all(colors == [0, 0, 0], axis=1)]
    colors = colors[~np.all(colors == [255, 255, 255], axis=1)]

    unique_colors, counts = np.unique(colors, axis=0, return_counts=True)
    most_common_color = unique_colors[counts.argmax()]

    return most_common_color


def get_all_colors():
    all_colors_rgb = []
    folder_path = f"crop/{uploaded_file.name}"
    files_and_folders = os.listdir(folder_path)

    files = [f for f in files_and_folders if os.path.isfile(os.path.join(folder_path, f))]
    for file in files:
        full_file_path = os.path.join(folder_path, file)
        color = get_most_common_color(full_file_path)
        all_colors_rgb.append(color)

    unique_all_colors_rgb = np.unique(all_colors_rgb, axis=0)
    unique_all_colors_rgb.sort()

    for i in range(1, len(unique_all_colors_rgb)-1):
        color_distance = np.sqrt(np.sum((np.array(unique_all_colors_rgb[i-1]) - np.array(unique_all_colors_rgb[i]))**2))
        if color_distance < 5:
            unique_all_colors_rgb = np.delete(unique_all_colors_rgb, i-1, axis=0)
    return unique_all_colors_rgb


def load_image():
    uploaded_file = st.file_uploader(label='Выберите изображение для распознавания')
    if uploaded_file is not None:
        image_data = uploaded_file.getvalue()
        st.image(image_data)
        return Image.open(io.BytesIO(image_data))
    else:
        return None


def create_folders_and_file():
    if not os.path.exists(f"crop"):
        os.mkdir(f"crop", 0o666)
    if not os.path.exists(f"images"):
        os.mkdir(f"images", 0o666)

    path = "images/"
    img.save(os.path.join(path , uploaded_file.name))


def rgb_to_str(colors):
    temp_arr = []
    for color in colors:
        temp_arr.append(f"R:{color[0]} G:{color[1]} B:{color[2]}")
    return temp_arr


def rgb_to_hex(colors):
    temp_arr = []
    for color in colors:
        hex_color = to_hex(tuple(v/255. for v in color))
        temp_arr.append(hex_color)
    return temp_arr


def print_colors(colors_str, colors_hex):
    for i in range(0, len(colors_str)):
        title = f'<p style="font-family:sans-serif; color:White; font-size:25px;">{colors_str[i]}</p>'
        st.markdown(title, unsafe_allow_html=True)
        
        color_picker_design = f"""
            <div style="
            height:50px; 
            width:180px; 
            background-color:{colors_hex[i]}; 
            border-radius: 5px;
            color: #003366; 
            border: solid 1px gray;
            margin: 0px 0px 15px;
            "></div>
            """
        st.markdown(color_picker_design, unsafe_allow_html=True)
        
        #color = st.color_picker('Цвет: ', f'{colors_hex[i]}', label_visibility="collapsed")


model = YOLO('best.pt')

st.title('Распознование доминирующего цвета на одежде')

uploaded_file = st.file_uploader(label='Выберите изображение для распознавания')
if uploaded_file is not None:
    image_data = uploaded_file.getvalue()
    st.image(image_data)
    img = Image.open(io.BytesIO(image_data))
    create_folders_and_file()
else:
    None

result = st.button('Распознать изображение')
if result:
    predict_image()
    all_colors = get_all_colors()

    all_colors_str = rgb_to_str(all_colors)
    all_colors_hex = rgb_to_hex(all_colors)

    print_colors(all_colors_str, all_colors_hex)