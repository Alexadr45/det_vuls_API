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
from model import predict, model, tokenizer, parser, file_inner, parsing


def find_vulnarabilities_in_file(content):
    methods = parsing(file_inner(content))
    try:
        result = predict(model, tokenizer, methods=methods, do_linelevel_preds = True)
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
    res_preds = find_vulnarabilities_in_file(model, tokenizer)
    f_name = file.filename
    os.remove(file.filename)
    return {f_name : res_preds}


if __name__ == "__main__":
    uvicorn.run("fastapi_app:app", host="0.0.0.0", reload = True)
