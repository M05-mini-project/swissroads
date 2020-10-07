import os
from . import database, data_exploration, baseline, analysis

# We don't know where the file data.csv will be installed
# on the user filesystem, we need to ask package management
# where it is and load it from there.  We do this here.
import pkg_resources

DATAFOLDER = pkg_resources.resource_filename(__name__, "/swissroads_images")
print("DATAFOLDER : " + DATAFOLDER)


def main():

    ### if does not exist create a folder output/
    output_path = r"./output"
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    ### first load the dataset
    # the output of the function is the generation of a file called "images_data.npz" saved into the current execution folder
    # once done a first time, the next line can be commented for future executions"
    database.load_data(DATAFOLDER)

    ### second step to explore the different categories into our dataset
    df = data_exploration.main()

    ### third step to generate our baseline model
    baseline_acc_tr, baseline_acc_te = baseline.main(df)

    ### fourth step to model Neural network
    analysis.main(df, 10)
