import os
import database
import data_exploration
import baseline
import analysis

### if does not exist create a folder output/
output_path = r"./output"
if not os.path.exists(output_path):
    os.makedirs(output_path)

### first load the dataset
# the output of the function is the generation of a file called "images_data.npz" saved into the current execution folder
# once done a first time, the next line can be cmmented for future executions"
# database.load_data(
#    "swissroads_images"
# )  # <--- comment this line if data already loaded into "images_data.npz"

### second step to explore the different categories into our dataset
df = data_exploration.main()

### third step to generate our baseline model
# baseline_acc_tr, baseline_acc_te = baseline.main(df)

### fourth step to model Neural network
analysis.main(df, 10)
