# Route

The following sections detail some general notes for working with `route`, including setup, and
various dependency requirements.

## Prerequisites

We will be using [OpenStreetMap](https://www.openstreetmap.org/) (OSM) data, and visualizing maps nicely in the browser.
To do so, we will need additional dependencies installed. In a virtual environment, such as
[miniconda](https://docs.conda.io/en/latest/miniconda.html#linux-installers)), run the following:

```bash
# we recommend creating a new conda environment given hw3 has unique dependencies
conda create --name cs221-hw3 python=3.12
conda activate cs221-hw3
```

```bash
pip install -r requirements.txt
```

This command should work out of the box for all platforms (Linux, Mac OS, Windows).

Note: If you have issues with installing the `osmium` package (or other packages), the error messages can be quite long. However,
they usually list near the beginning the reason for the issue. Common solutions might be having to manually install the `cmake` and `boost` dependencies first:

- On macOS, please do this with homebrew: (1) install [homebrew](https://brew.sh/) here (if you do not already have it), and (2) `brew install cmake` and `brew install boost`
- On Windows, please use review these [instructions for downloading cmake](https://cmake.org/download/) and these [instructions for downloading boost](https://www.geeksforgeeks.org/how-to-install-c-boost-libraries-on-windows/)

If you're getting "module not found" errors after downloading all the requirements, try using `python3` instead of `python` on the command line.

If you are facing other issues, we encourage you to come to Office Hours, or post on Ed. Please start early so you can troubleshoot any issues with ample time!

## Visualizing the Map

To visualize a particular map, you can use the following:

```bash
python visualization.py --path-file None # or "python3 ..." for the following

# You can customize the map and the landmarks
python visualization.py --map-file data/stanford.pbf --landmark-file data/stanford-landmarks.json --path-file None

# Visualize a particular solution path (requires running `grader.py` on question 1b/2c first!)
python visualization.py --path-file path.json
```
