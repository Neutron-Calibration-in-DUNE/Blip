### Installation
----------------
First you will need some dependencies.
The best way to proceed is to create a conda environment dedicated to the API using the packages defined in ``environment.yml``:
```bash
conda env create -f environment.yml
conda activate blip
```
You can optionally add the flag ``-n <name>`` to specify a name for the environment.

From here you can run:
```bash
pip install .
```
which should install the API for you.

### Basic Usage
---------------


### Datasets
------------

  <!-- One immediate way of addressing this is to come up with a common format for expressing datasets.  We can do this by imposing a set of constraints on how a dataset should be expressed in memory, which for now is done by creating a compressed numpy file (.npz) with the following minimal set of arrays:

```python

events = np.random.normal(0,1,1000) # random array of values
classes = np.ones((1000,1))
weights = np.ones((1000,1))
class_weights = np.ones((1000,1))

# dictionary containing meta data
event_meta = {
    "who_created":  "none",
    "when_created": "end_of_time",
    "where_created":"the_void",
    "num_events":   len(events), 
    "features":     {"x": 0},
    "classes":      {"y": 0},
    "sample_weights":{"w": 0},
    "class_weights":{"c": 0},
}

np.savez(
    "compressed_file.npz",
    meta=event_meta,
    event_features=features,
    event_classes=classes,
    event_sample_weights=weights,
    event_class_weights=class_weights,
)
```
Here we have a set of arrays containing **features** (events), **classes** (classes), **sample_weights** (weights) and **class_weights** (class_weights), as well as a dictionary **meta** which contains information about who/when/where the dataset was created, as well as the number of events and a set of dictionaries describing the various items in the arrays.  Everything but the **sample_weights** and **class_weights** items are required in the meta dictionary, which will be checked whenever a dataset is loaded from an .npz file. -->