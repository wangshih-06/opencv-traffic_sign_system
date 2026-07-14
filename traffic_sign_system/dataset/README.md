# Dataset directory

The GTSRB image dataset is intentionally excluded from Git because the local
copy contains tens of thousands of generated/downloaded image files.

Expected layouts are supported by the loaders, including:

```text
dataset/
??? train/
?   ??? Train/<ClassId>/*.png
?   ??? Test/*.png
?   ??? Train.csv
?   ??? Test.csv
??? test/
    ??? Test/*.png
```

Download or copy GTSRB locally before running feature construction, training,
evaluation, or prediction commands. Dataset files remain untracked by design.
