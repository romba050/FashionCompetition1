# Kaggle Competition
Decide between:
* [Furniture Competition](https://www.kaggle.com/c/imaterialist-challenge-furniture-2018#Timeline)
* [Fashion Competition](https://www.kaggle.com/c/imaterialist-challenge-fashion-2018#)

### LSDA Brainstorm, aspects to improve:
/

# read-in
reading in data faster with different function
selecting data pints at random

# preprocessing:

grayscaling
SIFT
resizing to different sizes (200x200?, 100x100?, 10x10?)
different resizing methods: crop, seamcarver, …
PCA (first resizing, then feature selection via PCA?)

# ML prediction
right now: knn, logistic regression
convolutional neural networks (for part 2 of the competition?)

# Assignment for next meeting:
edit:
download_furniture.py
line 33:
“for item in images:”
-> “for item in images[::200]:”
run this on train.json

-> “for item in images[::12]:”
run this on validation.json
validation [::12]

-> write furniture_<your_name>.py containing your variation of the code on the data subset.
We'll compare accuracies on thursday. Ideally, upload your script to github so we can compare them on thursday.

# Next meetings:
10. mai, Thursday 13:00 outside of Biocenter, back entrance

# Github Storage
Github is limited to 1 GB per repository and 100 MB per file.
For details see:
https://help.github.com/articles/working-with-large-files/

