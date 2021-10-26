import streamlit as st
from hydralit import HydraHeadApp
from itertools import chain
import time

class SpeechToTextApp(HydraHeadApp):
    
    def __init__(self, title = "", **kwargs):
        self.__dict__.update(kwargs)
        self.title = title
        
    def run(self):
        
        try:
            st.title("Speech To Text")
            st.subheader("App to transcribe available audio voice to text")
            st.markdown('<br><br>',unsafe_allow_html=True)
            
            self.display_app_header(self.title, True)
            
            with st.sidebar:
                slider_val = st.slider("Max Duration", min_value=1, max_value=240)
                st.info(f"Duration scope: {slider_val}s")
                
                ascoustic_model_option = st.selectbox("Which acoustic model?", 
                            ('Model 50k', 'Model 130k'))
                st.info(f"You selected:{ascoustic_model_option}")
                
                lm_option = st.selectbox("Which language model?", 
                            ('CTC', 'CTC + 4-gram'))
                st.info(f"You selected:{lm_option}")
            
            _, col2, _ = st.columns((1,8,1))
            audio_file = col2.file_uploader("Upload audio", type=['wav','mp3'])             
            if audio_file is not None:
                col2.audio(audio_file)
            
            trans_btn = col2.button("Transcribe")
            if trans_btn:
                col2.markdown('**Processing**...')
                
            transcript_result = col2.text_area("Text Transcript", height=300)

            if transcript_result is not None:
                data_down = transcript_result.strip()
            
                current_time = time.strftime("%H%M%S-%d%M%y")
                file_name = "transcript_" + str(current_time)
                col2.download_button(label="Save transcript",
                                    data=data_down,
                                    file_name=f'{file_name}.txt',
                                    mime='text/csv')


        except Exception as e:
            st.image("./resources/failure.png",width=100,)
            st.error('An error has occurred, someone will be punished for your inconvenience, we humbly request you try again.')
            st.error('Error details: {}'.format(e))
            
    def display_app_header(self, main_txt,is_sidebar = False):
        """
        function to display major headers at user interface
        ----------
        main_txt: str -> the major text to be displayed
        sub_txt: str -> the minor text to be displayed 
        is_sidebar: bool -> check if its side panel or major panel
        """

        html_temp = f"""
        <h2 style = "color:#F74369; text_align:center; font-weight: bold;"> {main_txt} </h2>
        </div>
        """
        if is_sidebar:
            st.sidebar.markdown(html_temp, unsafe_allow_html = True)
        else: 
            st.markdown(html_temp, unsafe_allow_html = True)
