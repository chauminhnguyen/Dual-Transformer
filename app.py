import os
import pipeline
import streamlit as st


def main():
    st.set_page_config(layout="wide")

    st.image(os.path.join('Images','Banner No2.png'), use_column_width  = True)
    st.markdown("<h1 style='text-align: center; color: black;'>Writing Vietnamese poem from image</h1>", unsafe_allow_html=True)
    
    img_file_buffer = st.file_uploader('Upload your image here',type=['jpg','jpeg','png'])
    if img_file_buffer is not None:
        from PIL import Image
        col1, col2 = st.columns(2)
        image = Image.open(img_file_buffer).convert("RGB")
        with col1:
            st.image(image)

        poem = pipeline.main(image)
        print(poem)
        poem = poem.split(' [SEP] ')[-1].replace('\n', '<br />')

        with col2:
            st.markdown("<h4 style='text-align: center; color: black; line-height: 2;'>" + poem + "</h4>", unsafe_allow_html=True)


if __name__ == '__main__':
    main()