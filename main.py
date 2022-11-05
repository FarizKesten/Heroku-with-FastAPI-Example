from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing_extensions import Literal
from helper import read_config
from ml.pipeline import run_pipeline
from pandas import DataFrame
import pathlib


main_path = pathlib.Path(__file__).parent.resolve()

app = FastAPI()


class InputData(BaseModel):
    age: int #1
    workclass : Literal['State-gov', 'Self-emp-not-inc', 'Private', 'Federal-gov', '?',
                        'Local-gov', 'Self-emp-inc', 'Without-pay', 'Never-worked'] #9
    fnlgt: int #1
    education: Literal['Bachelors', 'HS-grad', '11th', 'Masters', '9th', 'Some-college',
                       'Assoc-acdm', 'Assoc-voc', '7th-8th', 'Doctorate', 'Prof-school',
                       '5th-6th', '10th', '1st-4th', 'Preschool', '12th'] #16
    marital_status: Literal['Never-married', 'Married-civ-spouse', 'Divorced',
                            'Married-spouse-absent', 'Separated', 'Married-AF-spouse',
                            'Widowed'] = Field(alias="marital-status") #7
    occupation: Literal['Adm-clerical', 'Exec-managerial', 'Handlers-cleaners',
                        'Prof-specialty', 'Other-service', 'Sales', 'Craft-repair',
                        'Transport-moving', 'Farming-fishing', 'Machine-op-inspct',
                        'Tech-support', '?', 'Protective-serv', 'Armed-Forces',
                        'Priv-house-serv'] #15
    relationship: Literal['Not-in-family', 'Husband', 'Wife', 'Own-child', 'Unmarried',
                          'Other-relative'] #6
    race: Literal['White', 'Black', 'Asian-Pac-Islander', 'Amer-Indian-Eskimo','Other'] #5
    sex : Literal['Male', 'Female'] #2
    hours_per_week: int = Field(alias="hours-per-week") #1
    native_country: Literal['United-States', 'Cuba', 'Jamaica', 'India', '?', 'Mexico',
                           'South', 'Puerto-Rico', 'Honduras', 'England', 'Canada', 'Germany',
                           'Iran', 'Philippines', 'Italy', 'Poland', 'Columbia', 'Cambodia',
                           'Thailand', 'Ecuador', 'Laos', 'Taiwan', 'Haiti', 'Portugal',
                           'Dominican-Republic', 'El-Salvador', 'France', 'Guatemala',
                           'China', 'Japan', 'Yugoslavia', 'Peru',
                           'Outlying-US(Guam-USVI-etc)', 'Scotland', 'Trinadad&Tobago',
                           'Greece', 'Nicaragua', 'Vietnam', 'Hong', 'Ireland', 'Hungary',
                           'Holand-Netherlands'] = Field(alias="native-country") #42

    # allow conversion to Field Name for some variables with hyphens
    class Config:
        allow_population_by_field_name = True
        schema_extra = {
            "example": {
                'age': 40,
                'workclass': 'Private',
                'fnlgt': 140000,
                'education': 'Doctorate',
                'marital-status': 'Never-married',
                'occupation': 'Prof-specialty',
                'relationship': 'Not-in-family',
                'race': 'White',
                'sex': 'Male',
                'hours-per-week': 60,
                'native-country': 'United-States'
            }
        }


@app.get("/")
async def say_hello():
    return {"greeting": "Welcome!!"}

@app.post("/")
async def inference(input_data: InputData):
    input_data = input_data.dict()

    # Convert all underscores to hyphens in the input data
    input_data = {key.replace('_', '-'): value for key, value in input_data.items()}

    df = DataFrame(data=input_data.values(), index=input_data.keys()).T
    _, pred = run_pipeline(df, label=None, class_name=True)
    return {"prediction": pred[0]}