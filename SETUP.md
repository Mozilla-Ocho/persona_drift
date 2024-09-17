# Modified Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
python -m ipykernel install --user --name=drift --display-name="drift"
cp .env.example .env
# Edit .env file to fill in your OpenAI API key
```
