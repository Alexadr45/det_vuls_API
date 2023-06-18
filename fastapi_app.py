import torch
# import os
# import json
from tree_sitter import Language, Parser
import shutil
from fastapi import FastAPI, File, UploadFile
import uvicorn
from peft import PeftModel, PeftConfig
from transformers import AutoTokenizer, RobertaForSequenceClassification, \
    set_seed, RobertaConfig
from model import predict, file_inner, cleaner1, parser, obfuscate, Model


parser = Parser()
CSHARP_LANGUAGE = Language('build/my-languages.so', 'c_sharp')
parser.set_language(CSHARP_LANGUAGE)

base = 'microsoft/unixcoder-base'
model_id = "Model"

tokenizer = AutoTokenizer.from_pretrained(base)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
n_gpu = torch.cuda.device_count()
set_seed(n_gpu)

config = RobertaConfig.from_pretrained(base)
config.num_labels = 1
model = RobertaForSequenceClassification\
    .from_pretrained(base,
                     config=config,
                     ignore_mismatched_sizes=True).to(device)

model = Model(model, config, tokenizer)
model.to(device)

config = PeftConfig.from_pretrained(model_id)
model = PeftModel.from_pretrained(model, model_id)


def find_vulnarabilities_in_file(content, model, tokenizer, device):
    methods = obfuscate(parser, cleaner1(file_inner(content)))
    try:
        predictions = predict(model, tokenizer, methods, device,
                              do_linelevel_preds=True)
    except Exception:
        predictions = {"Error": "Ошибка сканирования файла"}
        # os.remove(content)
        return predictions
    else:
        # os.remove(content)
        return predictions


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
        return {"message": "ERROR uploading file"}
    finally:
        file.file.close()
    res_preds = find_vulnarabilities_in_file(file.filename,
                                             model,
                                             tokenizer,
                                             device)
    f_name = file.filename
    # os.remove(file.filename)
    result = [{f_name: res_preds}]
    # result = json.dumps(result)
    return result


if __name__ == "__main__":
    uvicorn.run("fastapi_app:app", host="0.0.0.0", reload=False)
