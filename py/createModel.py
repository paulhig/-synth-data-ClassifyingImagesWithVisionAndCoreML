import turicreate as tc

#load the data
data = tc.SFrame("cupz-glaz-otherz.sframe")

#split data into 80% training, 20% testing
train_data, test_data = data.random_split(0.8)

#Create the model
model = tc.image_classifier.create(train_data, target="label")

#Save predictions to an SArray
predicts = model.predict(test_data)

#evaluate the model
metrics = model.evaluate(test_data)

#display metrics for accuracy
print(metrics["accuracy"])

#display confusion matrix
print(metrics["confusion_matrix"])

#save the Turi Create model
model.save("TuriCupzGlazOtherzClassifier.model")

#export the model to Core ML
model.export_coreml("CupzGlazOtherzClassifier.mlmodel")
