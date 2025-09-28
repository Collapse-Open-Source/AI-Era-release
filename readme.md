# Exorcist
Exorcist is an Open Source AI tool for comment moderation, which has been created by Collapse Open Source Team. We spread it under BSD-2 License, so you can feel free to use it anywhere and in any case you want

## Response type:
It returns 1 if there are swearing in the comment and returns 0 if there are not 

## How to start
download dependencies (note: firstly download a model from google drive): `pip install torch transformers pandas scikit-learn`  

CLI-mode:  
`python3 ./ai_CLI_starter.py`  

Web app mode:  
`python3 ./app.py`

Teach AI and check some stuff:  
`python3 ./mod.py`

## Model itself
Model doesn't fit GitHub requirements (it's weight is more than 100MB). So it is placed on Google Drive as a tar archieve: `https://drive.google.com/file/d/17lSIUb8o4MLc_Vc-U30XjlXoAQZqKIOI/view?usp=sharing` or `https://drive.google.com/drive/folders/1ltVonls7YebHUZn67Pw8FozCsHoyF5V2?usp=sharing`
