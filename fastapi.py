import pandas as pd
import os
import tree_sitter
from tree_sitter import Language, Parser
import codecs
import shutil
import base64
from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel
from typing import List
import uvicorn
from model import predict


#Настраиваем пул наших языков
Language.build_library(
  # Store the library in the `build` directory
  'build/my-languages.so',

  # Include one or more languages
  [
 #   'tree-sitter-python',
    'tree-sitter-c-sharp'
  ]
)

#Настраиваем парсер для C#
parser = Parser()
CSHARP_LANGUAGE = Language('build/my-languages.so', 'c_sharp')
parser.set_language(CSHARP_LANGUAGE)

#Функция для считывания содержимого
def file_inner(path):
    with codecs.open(path, 'r', 'utf-8') as file:
        code = file.read()
    return code

#Функция выделения метода из файла
def parsing(code):
    snippets = []
    #Строим дерево по нашему файлу
    tree = parser.parse(bytes(code, "utf8"))
    #Прыгаем по дереву
    root_node = tree.root_node
    for node in root_node.children:
        if node.type == "namespace_declaration":
            for node in node.children:
                if node.type == 'declaration_list':
                    for node in node.children:
                        if node.type == 'class_declaration':
                            for node in node.children:
                                if node.type == 'declaration_list':
                                    for node in node.children:
                                        if node.type == 'method_declaration':
                                            snippets.append(code[node.start_byte: node.end_byte])
    return snippets


def find_vulnarabilities_in_file(content):
    methods = parsing(file_inner(content))
    try:
        result = predict(model=model, tokenizer=tokenizer, methods=methods, do_linelevel_preds = True)
    except:
        predictions = {"Error": "Ошибка сканирования файла"}
        os.remove(content)
        return predictions
    else:
        os.remove(content)
        return predictions


class Data(BaseModel):
    content: str
    name: str

class Item(BaseModel):
    accept: List[Data]

app = FastAPI()

@app.get("/")
async def hello():
    return {"message": "Hello world"}

@app.post("/uploadfile")
async def create_upload_file(file: UploadFile = File(...)):
    try:
        with open(file.filename, "wb") as f:
            shutil.copyfileobj(file.file, f)
    except Exception:
        return {"message":"ERROR uploading file"}
    finally:
        file.file.close()
    res_preds = find_vulnarabilities_in_file(file.filename)
    f_name = file.filename
    os.remove(file.filename)
    return {f_name : res_preds}


if __name__ == "__main__":
    uvicorn.run("fastapi_app:app", host="0.0.0.0", reload = True)
