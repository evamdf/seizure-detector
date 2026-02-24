How to run:

Navigate to src

```bash
    python src/main.py                   # run everything
    python src/main.py --part classifier # train model
    python src/main.py --part midi       # generate the MIDI files only
    python src/main.py --part stream     # stream demo only (this will create and train model as well if it isn't found)
```



To set up python venv:
```bash
python3 -m venv venv
```
```bash
source venv/bin/activate && pip install -r requirements.txt
```

### TO DOs

- Finish this documentation.....
- Add arg for declaring a variable command line 
- Add instructions on how to play the MIDI files (and maybe a little demo/screen recording with noise?)
- Probs a nice restructure











