from fastapi import FastAPI
import uvicorn
from pydantic import BaseModel
from model import create_model

app = FastAPI()


class Input(BaseModel):
    """
    Input features validation for the ML model
    """
    text: str

@app.post('/predict')
def predict(json: Input):
    text = json.text
    model = create_model()
    categories = ["Ham", "Spam"]
    return (categories[int(model.predict([text])[0])])

if __name__ == '__main__':
    # Run server using given host and port
    uvicorn.run(app, host='127.0.0.1', port=80)