import fastapi
import httpx
import torch
from tree_sitter import Language, Parser
from peft import PeftModel, PeftConfig
from transformers import AutoTokenizer, RobertaForSequenceClassification, \
    set_seed, RobertaConfig
from model import predict, file_inner, parser, obfuscate, Model, add_line_delimiter, cleaner1
from fastapi.testclient import TestClient
from fastapi_app import app


client = TestClient(app)

base = 'microsoft/unixcoder-base'
model_id = "saved_models"

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

# Настраиваем парсер для C#
parser = Parser()
CSHARP_LANGUAGE = Language('build/my-languages.so', 'c_sharp')
parser.set_language(CSHARP_LANGUAGE)


def test_read_main():
    response = client.get('/')
    assert response.status_code == 200
    assert response.json() == {"message": "Hello world"}


def test_file_with_vuls():
    string_data = file_inner("data/file_with_vuls.cs")
    string_data = obfuscate(parser, string_data)
    result = predict(model, tokenizer, string_data, device)
    assert result != 'Уязвимости не найдены'


def test_without_vuls():
    string_data = file_inner("data/file_without_vuls.cs")
    string_data = obfuscate(parser, string_data)
    result = predict(model, tokenizer, string_data, device)
    assert result == 'Уязвимости не найдены'


def test_file_inner():
    string_data = file_inner("data/file_without_vuls.cs")
    assert string_data.type() == str


def test_cleaner1():
    string_data = cleaner1('2 + 2 = 4 /* comment')
    assert string_data == '2 + 2 = 4'


def test_add_line_delimiter():
    string_data = add_line_delimiter('{var_66 = var_BB;return var_66;}')
    assert string_data == '{\nvar_66 = var_BB;\nreturn var_66;\n}\n'
