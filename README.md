# Seizure Detector, Music Maker 🎵

This project experiments using real EEG data that captures brain activity during different states (healthy with eyes open and closed, interictal (in between seizures) and ictal (during seizure)).

The first part of the project is a seizure monitoring/alerting system. It simulates receiving EEG data in real time by moving a sliding window over the signal at a fixed rate. At each window, a set of features are extracted from the signal (normalised band powers, spectral entropy and RMS amplitude) and are fed to a simple classifier to make a prediction about if a seizure is currently happening. If there are enough consecutive seizure predictions, it will print an alert message. This can be seen in action using the `stream-demo` flag, which shows one non-ictal and one ictal segment being 'streamed'.

The second part uses the same feature extraction method to represent one segment from each of the five EEG classes. These features are then mapped to different musical elements, which are in turn used to generate simple songs (MIDI tracks). How each song sounds directly relates to the qualities of different EEG data types. 


## Setup & Execution

Setting up a python venv makes it easy to install requirements:

```bash
python3 -m venv venv
```
```bash
source venv/bin/activate && pip install -r requirements.txt
```

### Running the project

```bash
cd src
```
To run the whole project:
```bash
python main.py
```

To run only a specific part of the project:
```bash           
python main.py --part model         # Only train/load model
python main.py --part midi          # Only generate the MIDI files 
python main.py --part stream-demo   # Just do the stream demo (this will create and train model as well if it isn't found)
```

An example of the detector output during an ictal EEG signal:

<img width="397" height="327" alt="detector-example" src="https://github.com/user-attachments/assets/d61ae5a5-684d-4dda-8d82-f28367bc200b" />


## Listening to the MIDI files

To listen to the EEG songs, you can use any sort of media player or web based tool compatible with MIDI. Download the `.mid` files (in `/midi-output`).  

[Midiano](https://app.midiano.com/) is a super easy web app you can use to listen to these. 

Or, you can listen to this recording of me playing two of the songs in Midiano:

https://github.com/user-attachments/assets/0f5d1dea-292e-4e84-98a0-7c019664b00b

Please note, the files are named according to the EEG class they were made from:

| Folder | Description |
| :--- | :--- |
| Z | Surface EEG, eyes open, healthy |
| O | Surface EEG, eyes closed, healthy |
| N | Interictal - seizure free interval |
| F | Intericta - seizure free interval |
| S | Ictal - during seizure |

## The Data 

The data in this directory was first analyzed in:

Andrzejak RG, Lehnertz K, Rieke C, Mormann F, David P, Elger CE (2001) Indications of nonlinear deterministic and finite dimensional structures in time series of brain electrical activity: Dependence on recording region and brain state, Phys. Rev. E, 64, 061907.

https://www.upf.edu/documents/229517819/232450661/Andrzejak-PhysicalReviewE2001.pdf/0e9a54b8-8993-b400-743e-4d64fa29fb63


## TO DOs

- Add ability to declare variables in command line
- Future: try out replacing sliding window/streaming with Python queue and threading?








