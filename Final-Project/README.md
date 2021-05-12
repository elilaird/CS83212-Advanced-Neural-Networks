# Final Project, Adaptive Focus Learning

* Clay Harper, Eli Laird

In this directory, you can find `requirements.txt` to create an environment with
the required packages to run our code.  You can also use `ml_final_proj.yml` to
create from `.yml`, but Eli had issues getting it to work on his machine due to
some obscure error.  

* `learn_what_you_dont_know_focus.pdf`
  * Where our NeurIPS/Nature, best paper-worthy paper resides.
  * "Best paper I've ever read" - Geoffrey Hinton, Yann LeCun
    -Probably

* `FocusTraining`
The majority of the code for this project in terms of implementation for *Adaptive
  Focus Learning* can be found in the directory `FocusTraining`.  It is a python
  module that can be imported into a project (this is what we did in the notebooks).
  * `focus_training.py`
    * The module includes `focus_training.py` where the functions for the modified
      training procedure can be found.  The main class, `FocusLearning` can be found on
      line 196.  Doing modifications with our work, we simply subclassed this and
      wrote custom functions for the reversed procedure defined in our paper.  For
      changing hyperparameters for training, please see the `train` function where
      different gradient penalties can be added.  
  * `metrics.py`
    * This script contains code for different metrics to track during training
    and can simply be passed to the `FocusTraining`.  For the purposes of this
    project, we only explored accuracy because CIFAR-10 is perfectly balanced.
  * `utils.py`
    * This scipt contains utility functions for handling history objects that we
    create during the training process to track metrics.  Essentially, the functions
    just help with saving/loading and getting the worst/best performers at each
    epoch.

* `*.ipynb`
  * We used jupyter notebooks to train our models for visual feedback (we are
    proud of our progressbars!).  To speed up training/coding, we used some of
    the same notebooks and just modified the hyperparameters.  These notebooks
    aren't too clean, but the majority of the code we use is in `FocusTraining`
    and the code is well documented there.
* `*.R`
  * We used R for plotting for our paper.  The code to generate our figures can
  be found in `FinalProjectAnalysis.R`.
* `models`
  * Directory containing the trained models from our analysis.
* `results`
  * Directory containing performance metrics as `.csv` files for reading in with
  R.
* `trained_model_histories`
  * Directory containing the histories during training for all models.


We wanted to make the implementation as modular as possible if we wanted to
continue this project in the future, so subclassing was important for us to make
possible.  Given more time/space in the paper, we would've tried different
gradient penalties, adjust k for the k worst classes, and also explore changing
the function `time_to_focus` in the `StartToFocus` class.  Instead of focusing
each epoch, we could have changed this to be every x number of epochs or based
on the values in the validation loss/whatever metric we were tracking.  
