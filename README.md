## Drowsiness Detection

Drowsiness detection is a safety technology that can prevent accidents that are caused by drivers who fell asleep while driving.

The objective of this project is to build a drowsiness detection system that will detect that a person’s eyes are closed for a few seconds. This system will alert the driver when drowsiness is detected.

## Functions

This project notifies you if your face is not visible in the frame for a fixed amount of time. It also notifies if your eyes are closed for a fixed amount of time. The alert is in the form of a Beep Sound. 

## Setup

Cloning the Repository

```
$ git clone https://github.com/Rishabh7896/DrowsinessDetection
$ cd DrowsinessDetection
```

Creating Virtual Environment

```
$ python -m venv myenv
```

Activating the Environment

```
$ myenv\Scripts\activate
```

Installing Dependencies

```
$ pip install -r requirements.txt
```

Training the Model

```
$ python model.py
```

Running the project for prediction

```
$ python predict.py
```
