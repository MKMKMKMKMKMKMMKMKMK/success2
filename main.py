import streamlit as st
from PIL import Image
import io
from rembg import remove
import cv2
import numpy as np
def image_to_dot_art(image, pixel_size=10):
    # 画像を縮小してピクセル化する
    small_image = image.resize(
        (image.width // pixel_size, image.height // pixel_size), 
        Image.NEAREST
    )
    
    # ピクセル化された画像を元のサイズにリサイズする
    dot_art_image = small_image.resize(
        (image.width, image.height), 
        Image.NEAREST
    )
    
    return dot_art_image

def remove_background(image):
    # 画像の背景を除去する
    image_data = io.BytesIO()
    image.save(image_data, format='PNG')
    image_data = image_data.getvalue()
    output_data = remove(image_data)
    return Image.open(io.BytesIO(output_data))
def convert_image(image):
    # OpenCVを使って画像を処理
    img = np.array(image.convert('RGB'))
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    
    # マスク画像を生成 (白色以外も幅広くカバー)
    lower_white = np.array([200, 200, 200], dtype=np.uint8)
    upper_white = np.array([255, 255, 255], dtype=np.uint8)
    mask = cv2.inRange(img, lower_white, upper_white)
    
    # 元画像をBGRA形式に変換
    dst = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
    
    # マスク画像をもとに、白色部分を透明化
    dst[:, :, 3] = np.where(mask == 255, 0, 255)
    
    return dst

# Streamlitアプリケーションのタイトル
st.title('画像処理アプリ')

# ドット絵変換セクション
st.header("ドット絵変換")
uploaded_file_dot = st.file_uploader("ドット絵変換用の画像をアップロードしてください", type=['png', 'jpg', 'jpeg'], key="dot_upload")
if uploaded_file_dot is not None:
    image_dot = Image.open(uploaded_file_dot)

    # ドット絵に変換するピクセルサイズの選択
    pixel_size = st.slider('ピクセルサイズを選択してください', min_value=1, max_value=30, value=10, key="dot_pixel_size")

    # ドット絵に変換
    dot_art_image = image_to_dot_art(image_dot, pixel_size)

    # 画像を表示
    st.image(dot_art_image, caption='変換後のドット絵', use_column_width=True)

    # 画像を保存するオプション
    if st.button('ドット絵を保存'):
        dot_art_image.save('dot_art_image.png')
        st.success('ドット絵を保存しました！')

# 背景除去セクション
st.header("背景除去ツール")
uploaded_file_bg = st.file_uploader("背景除去用の画像をアップロードしてください", type=["png", "jpg", "jpeg"], key="bg_upload")
if uploaded_file_bg is not None:
    image_bg = Image.open(uploaded_file_bg)
    # st.image(image_bg, caption='アップロードされた画像', use_column_width=True)

    if st.button('背景を除去'):
        image_bg_removed = remove_background(image_bg)
        st.image(image_bg_removed, caption='背景除去後の画像', use_column_width=True)

        # 背景除去後の画像を保存するオプション
        if st.button('背景除去画像を保存'):
            image_bg_removed.save('background_removed_image.png')
            st.success('背景除去画像を保存しました！')

# 画像の合成セクション
st.header("白色部分を透明化")

uploaded_file = st.file_uploader("画像ファイルをアップロードしてください。", type=['png', 'jpg', 'jpeg'],key="transparent_upload")

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='アップロードされた画像', use_column_width=True)
    
    # 画像処理関数を呼び出し
    result_image = convert_image(image)
    
    # 結果の表示
    st.image(result_image, caption='透明化処理後の画像', channels="BGR", use_column_width=True)
    st.write("※ 見た目は同じように見えますが、画像の白色部分が透明化されました。")
    # 結果を保存するオプション
    save_button = st.button("画像を保存")
    if save_button:
        result_image_pil = Image.fromarray(cv2.cvtColor(result_image, cv2.COLOR_BGRA2RGBA))
        result_image_pil.save("processed_image.png")
        st.success("画像を保存しました。")

st.header("サイズリサイザー")

uploaded_file = st.file_uploader("画像ファイルをアップロードしてください。", type=['png', 'jpg', 'jpeg'],key = "resize_upload")

if uploaded_file is not None:
        # PILを使用して画像を開く
        img = Image.open(uploaded_file)
        
        # オリジナル画像を表示
        st.image(img, caption='オリジナル画像', use_column_width=True)

        # 画像をリサイズ
        resized_image = img.resize((480, 360))

        # リサイズした画像を表示
        st.image(resized_image, caption='リサイズ後の画像', use_column_width=True)

        # リサイズした画像を保存（オプションで実行）
        if st.button("画像を保存"):
            resized_image.save("resized_image.png")
            st.success("画像が保存されました！")
        
        