import streamlit as st 
from pytubefix import YouTube
from faster_whisper import WhisperModel
import os 


video_id=st.text_input("Enter the video id")

process_btn=st.button("Process")

if process_btn:
    if video_id.strip():
        try: 
            video_url=f"https://www.youtube.com/watch?v={video_id}"
            yt=YouTube(video_url)
            audio_file_path=yt.streams.get_audio_only().download(output_path=".",filename="downloaded_audio.m4a")

            model=WhisperModel("base",device="cpu",compute_type="int8")

            segments,info=model.transcribe(audio_file_path,beam_size=5)

            full_text=" ".join([segment.text for segment in segments])

            st.write(full_text)

            os.remove(audio_file_path)
        except Exception as e:
            st.error(f"An error occurred : {e} ")

