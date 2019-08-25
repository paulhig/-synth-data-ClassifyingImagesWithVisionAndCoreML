# set up the data
import turicreate as tc

#import images from directory
data = tc.image_analysis.load_images("data/sData/trainData", with_path=True)

#label the from filenames
data["label"] = data["path"].apply(lambda path: "cup" if "cup" in path else ( "glass" if "glass" in path else "other"))

# save data to an sframe
data.save("cupz-glaz-otherz.sframe")

# Explore the SArray in an interactive GUI. Opens a new app window.
data.explore()
