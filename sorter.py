import os
import shutil

# Create the "cats" and "dogs" folders


# Iterate over the files in the "data" folder
for filename in os.listdir("cats-and-dogs"):
    if filename.startswith("d"):  # Dog image
        shutil.move("cats-and-dogs/" + filename, "cats-and-dogs/dogs/" + filename)