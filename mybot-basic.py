#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Basic chatbot design --- for your own modifications
"""
from pathlib import Path

#ARTEM KAPUSTIN T0322864


#######################################################
# ALL IMPORTS
#######################################################
from PIL import Image
import numpy as np
import tkinter as tk
from tkinter import filedialog
from tensorflow import keras
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.stem import WordNetLemmatizer
from nltk.sem import Expression
from nltk.inference import ResolutionProver
import pandas
import aiml
from vosk import Model, KaldiRecognizer
import pyaudio
import pyttsx3
from TTS.utils.synthesizer import Synthesizer
from TTS.utils.manage import ModelManager

#######################################################
# Initialise VOSK model and microphone
#######################################################

model = Model(r"vosk-model-en-us-0.22-lgraph\vosk-model-en-us-0.22-lgraph")
recognizer = KaldiRecognizer(model, 44100)

mic = pyaudio.PyAudio()
stream = mic.open(input_device_index=1, format=pyaudio.paInt16, channels=1, rate=44100, input=True,
                  frames_per_buffer=1024)
stream.start_stream()

#######################################################
#  Initialise NLTK Inference
#######################################################

read_expr = Expression.fromstring


#######################################################
#  Initialise Knowledgebase.
#######################################################
def check_kb_integrity(kb):
    for expr1 in kb:
        for expr2 in kb:
            if expr1 != expr2:
                result = ResolutionProver().prove(expr1, [expr2])
                if result:
                    return False
    return True


kb = []
data = pandas.read_csv('logical-kb.csv', header=None)
[kb.append(read_expr(row.lower())) for row in data[0]]
if not check_kb_integrity(kb):
    print("Error: Contradiction found in the knowledge base.")
    exit()

#######################################################
#  Initialise AIML agent
#######################################################

# Create a Kernel object. No string encoding (all I/O is unicode)
kern = aiml.Kernel()
kern.setTextEncoding(None)
kern.bootstrap(learnFiles="mybot-basic.xml")

#######################################################
# Initialise lemmatizer and TF-IDF vectorizer
#######################################################

qa_data = pd.read_csv('QA Pairs.csv', delimiter=';')
lemmatizer = WordNetLemmatizer()


#######################################################
# Lemmatize Function
#######################################################
def lemmatize_text(text):
    return ' '.join([lemmatizer.lemmatize(w) for w in text.split()])


qa_data['question_lemmatized'] = qa_data['question'].apply(lemmatize_text)
tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(qa_data['question_lemmatized'])

#######################################################
# Select Image
#######################################################

loaded_model = keras.models.load_model('my_model_w_hyperparameters5.h5')
loaded_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])


#######################################################
# Main loops
#######################################################

def handle_audio_input():
    while True:
        data = stream.read(4000, exception_on_overflow=False)
        if len(data) == 0:
            break
        if recognizer.AcceptWaveform(data):
            text = recognizer.Result()
            print(text)
            voice_command = text[14:-3]
            print("Your command: ", voice_command)
            responseAgent = 'aiml'
            # activate selected response agent
            if responseAgent == 'aiml':
                answer = kern.respond(voice_command)
            # post-process the answer for commands
            if answer:
                if answer[0] == '#':
                    params = answer[1:].split('$')
                    cmd = int(params[0])
                    if cmd == 0:
                        print(params[1])
                        generate_speech(params[1])
                        break
                    elif cmd == 31:  # if input pattern is "I know that * is *"
                        object, subject = params[1].lower().split(' is ')
                        expr = read_expr(subject + '(' + object + ')')
                        check_contradiction = ResolutionProver().prove(expr, kb)
                        if check_contradiction:
                            print("Error: Contradiction found in the knowledge base.")
                            generate_speech("Error: Contradiction found in the knowledge base.")
                        else:
                            kb.append(expr)
                            print('OK, I will remember that', object, 'is', subject)
                            generate_speech(input=f'OK, I will remember that {object} is {subject}')
                    elif cmd == 32:  # if the input pattern is "check that * is *"
                        object, subject = params[1].lower().split(' is ')
                        expr = read_expr(subject + '(' + object + ')')
                        prover = ResolutionProver()
                        answer = ResolutionProver().prove(expr, kb,
                                                          verbose=False)  # checking on correct, incorrect and unknown answer in kb
                        if answer:
                            print('Correct.')
                            generate_speech('Correct.')
                        else:
                            expr_text = str(expr)
                            if "-" in expr_text:
                                expr_text = expr_text.replace("-", "")
                                expr_text = read_expr(expr_text)
                                answer = ResolutionProver().prove(expr_text, kb)
                                if answer:
                                    print("Incorrect")
                                    generate_speech("Incorrect")
                                else:
                                    print("Sorry, I don't know")
                                    generate_speech("Sorry, I don't know")
                            else:
                                expr_text = read_expr(expr_text)
                                answer2 = ResolutionProver().prove(expr_text, kb)
                                if answer2:
                                    print("Incorrect")
                                    generate_speech("Incorrect")
                                else:
                                    print("Sorry, I don't know")
                                    generate_speech("Sorry, I don't know")
                    elif cmd == 33:  # if the input pattern is "check that * and * are plugged into *"
                        objects = params[1].lower().split(' and ')
                        subject = objects[-1]
                        objects = objects[:-1]
                        expr = None
                        for obj in objects:
                            obj_expr = read_expr('plugs(' + obj.strip() + ', ' + subject.strip() + ')')
                            if expr is None:
                                expr = obj_expr
                            else:
                                #using disjunction for all expression to get correct or incorrect answer
                                expr = expr | obj_expr
                        prover = ResolutionProver()
                        answer = prover.prove(expr, kb, verbose=False)
                        if answer:
                            print('Correct.')
                        else:
                            expr_text = str(expr)
                            if "-" in expr_text:
                                expr_text = expr_text.replace("-", "")
                                expr_text = read_expr(expr_text)
                                answer = ResolutionProver().prove(expr_text, kb)
                                if answer:
                                    print("Incorrect")
                                else:
                                    print("Sorry, I don't know")
                            else:
                                expr_text = read_expr(expr_text)
                                answer2 = ResolutionProver().prove(expr_text, kb)
                                if answer2:
                                    print("Incorrect")
                                else:
                                    print("Sorry, I don't know")
                    elif cmd == 99: # if user input is general question and it is not found in aiml xml statements
                        user_input_lemmatized = lemmatize_text(voice_command)
                        user_input_tfidf = tfidf_vectorizer.transform([user_input_lemmatized])
                        #check similarity with user input and QA pairs.csv
                        similarity = cosine_similarity(user_input_tfidf, tfidf_matrix)
                        idx = similarity.argmax()
                        #if similarity less than 40, then didn't get the users question.
                        if similarity[0, idx] < 0.40:
                            print("Sorry, I didn't get that, please try again")
                            generate_speech("Sorry, I didn't get that, please try again")
                        else:
                            #getting the most similar question from QA pairs.csv
                            matched_answer = qa_data.loc[idx, 'answer']
                            print("CSV Answer:", matched_answer)
                            generate_speech(matched_answer)
                    elif cmd == 34: #getting user image and getting the class using model
                        print("Select the image, input the path to the image: ")
                        image_path = input("> ")
                        image = Image.open(image_path)
                        image = image.convert("RGB")
                        image = image.resize((128, 128))
                        image_array = np.array(image) / 255.0
                        image_array = np.expand_dims(image_array, axis=0)
                        predictions = loaded_model.predict(image_array)
                        predicted_class_name = None
                        predicted_class_index = np.argmax(predictions)
                        if predicted_class_index == 0:
                            predicted_class_name = "Processor (CPU)"
                        elif predicted_class_index == 1:
                            predicted_class_name = "Graphic Card (GPU)"
                        elif predicted_class_index == 2:
                            predicted_class_name = "Hard disk drive (HDD)"
                        elif predicted_class_index == 3:
                            predicted_class_name = "Motherboard"
                        elif predicted_class_index == 4:
                            predicted_class_name = "Random Access Memory (RAM)"
                        print("This is a", predicted_class_name)
                        generate_speech(("This is a", predicted_class_name))
                else:
                    print(answer)
                    generate_speech(answer)
            else:
                print("No response from the AIML agent.")
                generate_speech("No response from the AIML agent.")


def handle_user_input():
    while True:
        try:
            userInput = input("> ")
            # pre-process user input and determine response agent (if needed)
            responseAgent = 'aiml'
            # activate selected response agent
            if responseAgent == 'aiml':
                answer = kern.respond(userInput)
            # post-process the answer for commands
            if answer[0] == '#':
                params = answer[1:].split('$')
                cmd = int(params[0])
                if cmd == 0:
                    print(params[1])
                    break
                elif cmd == 31:  # if input pattern is "I know that * is *"
                    object, subject = params[1].lower().split(' is ')
                    expr = read_expr(subject + '(' + object + ')')
                    check_contradiction = ResolutionProver().prove(expr, kb)
                    # if contradiction found, then print error
                    if check_contradiction:
                        print("Error: Contradiction found in the knowledge base.")
                    else:
                        kb.append(expr)
                        print('OK, I will remember that', object, 'is', subject)
                elif cmd == 32:
                    object, subject = params[1].lower().split(' is ')
                    expr = read_expr(subject + '(' + object + ')')
                    prover = ResolutionProver()
                    answer = ResolutionProver().prove(expr, kb, verbose=False)
                    if answer:
                        print('Correct.')
                    else:
                        expr_text = str(expr)
                        if "-" in expr_text:
                            expr_text = expr_text.replace("-", "")
                            expr_text = read_expr(expr_text)
                            answer = ResolutionProver().prove(expr_text, kb)
                            if answer:
                                print("Incorrect")
                            else:
                                print("Sorry, I don't know")
                        else:
                            expr_text = read_expr(expr_text)
                            answer2 = ResolutionProver().prove(expr_text, kb)
                            if answer2:
                                print("Incorrect")
                            else:
                                print("Sorry, I don't know")
                elif cmd == 33:
                    objects = params[1].lower().split(' and ')
                    subject = objects[-1]
                    objects = objects[:-1]
                    expr = None
                    for obj in objects:
                        obj_expr = read_expr('plugs(' + obj.strip() + ', ' + subject.strip() + ')')
                        if expr is None:
                            expr = obj_expr
                        else:
                            expr = expr | obj_expr
                    prover = ResolutionProver()
                    answer = prover.prove(expr, kb, verbose=False)
                    if answer:
                        print('Correct.')
                    else:
                        expr_text = str(expr)
                        if "-" in expr_text:
                            expr_text = expr_text.replace("-", "")
                            expr_text = read_expr(expr_text)
                            answer = ResolutionProver().prove(expr_text, kb)
                            if answer:
                                print("Incorrect")
                            else:
                                print("Sorry, I don't know")
                        else:
                            expr_text = read_expr(expr_text)
                            answer2 = ResolutionProver().prove(expr_text, kb)
                            if answer2:
                                print("Incorrect")
                            else:
                                print("Sorry, I don't know")

                elif cmd == 99:
                    user_input_lemmatized = lemmatize_text(userInput)
                    user_input_tfidf = tfidf_vectorizer.transform([user_input_lemmatized])
                    similarity = cosine_similarity(user_input_tfidf, tfidf_matrix)
                    idx = similarity.argmax()
                    if similarity[0, idx] < 0.40:
                        print("Sorry, I didn't get that, please try again")
                    else:
                        matched_answer = qa_data.loc[idx, 'answer']
                        print("CSV Answer:", matched_answer)
                elif cmd == 34:
                    print("Select the image, input the path to the image: ")
                    image_path = input("> ")
                    image = Image.open(image_path)
                    image = image.convert("RGB")
                    image = image.resize((128, 128))
                    image_array = np.array(image) / 255.0
                    image_array = np.expand_dims(image_array, axis=0)
                    predictions = loaded_model.predict(image_array)
                    predicted_class_name = None
                    predicted_class_index = np.argmax(predictions)
                    if predicted_class_index == 0:
                        predicted_class_name = "Processor (CPU)"
                    elif predicted_class_index == 1:
                        predicted_class_name = "Graphic Card (GPU)"
                    elif predicted_class_index == 2:
                        predicted_class_name = "Hard disk drive (HDD)"
                    elif predicted_class_index == 3:
                        predicted_class_name = "Motherboard"
                    elif predicted_class_index == 4:
                        predicted_class_name = "Random Access Memory (RAM)"
                    print("This is a", predicted_class_name)


            else:
                print(answer)

        except (KeyboardInterrupt, EOFError) as e:
            print("Bye!")
            break


######################################
# Text To Speech function
######################################
def generate_speech(input):
    # Initialize the text-to-speech engine
    if "the" in input:
        input = input.replace("the", "")
    engine = pyttsx3.init()
    voices = engine.getProperty('voices')
    engine.setProperty('voice', voices[2].id)

    text = input
    # Convert text to speech
    engine.say(text)
    # Play the converted speech
    engine.runAndWait()

#tested another model Coqui TTS for text to speech, it works absolutely amazing and has human-like voice, however processing of one phrase takes 5-7 seconds.


##############################################################################################################
# Tested Coqui TTS Model, it has good human-like voice, but 5-6 seconds of processing is too long
##############################################################################################################

# def generate_speech(input):
#     path = r"C:\Users\Артем\AICourseWork\venv\Lib\site-packages\TTS\.models.json"
#     model_manager = ModelManager(path)
#     model_path, config_path, model_item = model_manager.download_model("tts_models/en/ljspeech/tacotron2-DDC")
#     voc_path, voc_config_path, _ = model_manager.download_model(model_item["default_vocoder"])
#     synt = Synthesizer(
#         tts_checkpoint=model_path, tts_config_path=config_path,
#         vocoder_checkpoint=voc_path, vocoder_config=voc_config_path
#     )
#     text = input
#
#     outputs = synt.tts(text)
#     synt.save_wav(outputs, "audio.wav")


#######################################################
# Welcome user
#######################################################
print("Welcome to this chat bot")
print("Choose the text or voice input, by typing 1 or 2")
choice = input("Type in 1 or 2: ")
# started user input, to choose text or voice based chatbot
if choice == "1":
    print("You choose text-base chatbot. Please ask questions from me!")
    handle_user_input()
elif choice == "2":
    generate_speech(input="You choose voice recognition model. Please ask questions from me!")
    handle_audio_input()

else:
    print("incorrect input, try again")
