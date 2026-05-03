#! bin/bash


# --format=gif: 强制输出 GIF
# -ql: 低画质 (Quality Low) 渲染速度快，调试用
# -qh: 高画质 (Quality High) 最终展示用
manim -ql --format=gif scene.py AttentionMechanism



# run app
# streamlit run app.py