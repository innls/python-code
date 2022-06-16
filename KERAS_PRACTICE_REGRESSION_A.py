#
#    CHAPTER 4.3  EXAMPLE.  PREDICTION OF MEDIAN HOME PRICE
#    FROM 13 OTHER ATTRIBUTES USING DATA IN THE ARCHIVED
#    BOSTON HOUSING DATASET 
#

# Import Needed Libraries And Assign Aliases
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
import matplotlib.pyplot as plt

# Boston Housing Data Set From UCI Machine Learning Repository (No Longer Stored There)
# It Contains 506 Records With 14 Attributes From 
# Harrison, D. and Rubinfeld, D.L. `Hedonic prices and the demand for clean air', J. Environ. Economics & Management, vol.5, 81-102, 1978.

# Loading Function Has Randomly Sampled Attribute 14
# And Put The Data In Training And Test "_targets" Arrays.
# Attributes 1 - 13 Are In The "_data" Arrays.

from tensorflow.keras.datasets import boston_housing
(training_data, training_targets), (test_data, test_targets) = (
boston_housing.load_data())

tf.keras.backend.set_floatx('float64')         # Force 64-bit Numerical Computation On Backend

print(training_targets.shape, training_data.shape)

attrib_subset =   range(training_data.shape[1])

training_data_filtered = training_data[ :training_data.shape[0] ,  attrib_subset ]
test_data_filtered = test_data[ :test_data.shape[0], attrib_subset ]

# Filtered Training And Test Data Sets
training_data = training_data_filtered
test_data = test_data_filtered

# Confirm Filtered Dimensions
print(training_targets.shape, training_data.shape)

NumUnits = 64

# Build The Model
def build_model():
  model = tf.keras.Sequential([
  layers.Dense(NumUnits, activation="relu",name="layer1"),    # Rectifying Unit
  layers.Dense(NumUnits, activation="relu",name="layer2"),    # Rectifying Unit
  layers.Dense(1, activation=None, name="output")     # Linear Unit
  ])
  model.compile(optimizer="rmsprop", loss="mse", metrics=["mae"])
  return model

# Remove Mean From Training Data
training_data_mean = training_data.mean(axis=0)
training_data -= training_data_mean

# Calculate Standard Deviation Of Zero-Mean Training Data Set
std = training_data.std(axis=0)
training_data /= std             

# Test Data Parameters Must Be Isolated From Training Data.
test_data -= training_data_mean        # Remove Mean From Test Data Set
test_data /= std              # Normalize Test Data  Using Training Data Standard Deviation
  
numFolds = 4
num_validation_samples = tf.cast( len(training_data) / numFolds , tf.int32)
num_epochs = 700

all_scores = []
all_MAE_histories = []
all_LOSS_histories = []
all_VAL_LOSS_histories = []

print(num_validation_samples,training_data.size,training_targets.size)

for k in range(numFolds):
  print(f"Processing fold #{k+1}")
  val_data = training_data[k * num_validation_samples: (k + 1) * num_validation_samples]
  val_targets = training_targets[k * num_validation_samples: (k + 1) * num_validation_samples]
  
  # k-th Fold Training Excerpt
  fold_training_data =  np.concatenate(       [training_data[     : k * num_validation_samples],
                                                     training_data[(k + 1) * num_validation_samples:]],axis=0)
  # k-th Fold Target Excerpt
  fold_training_targets =  np.concatenate( [training_targets[            : k * num_validation_samples],
                                                     training_targets[(k + 1) * num_validation_samples:]],axis=0)
  
  model = build_model()   # Model Input Dimension Set By Training Data Shape (For This Model)
  history =  model.fit(fold_training_data, fold_training_targets,
                       validation_data = (val_data, val_targets),
                       epochs=num_epochs, batch_size=64, verbose=0)
  #validation_mse, validation_mae = model.evaluate(validation_data, validation_targets, verbose=0)
  #print(history.history.keys())   # print(history.params)
  all_MAE_histories.append( history.history["val_mae"] ) 
  all_LOSS_histories.append( history.history["loss"] )
  all_VAL_LOSS_histories.append( history.history["val_loss"] )
  
  print( history.history.keys() )
  
#for att in dir(all_MAE_histories):
#    print (att, getattr(all_MAE_histories,att))

epoch_skip = 100
avg_MAE_history = [ np.mean([ x[i] for x in all_MAE_histories]) for i in range(epoch_skip,num_epochs,1) ] 
avg_LOSS_history = [ np.mean([ x[j] for x in all_LOSS_histories]) for j in range(epoch_skip,num_epochs,1) ] 
avg_VAL_LOSS_history = [ np.mean([ x[k] for x in all_VAL_LOSS_histories]) for k in range(epoch_skip,num_epochs,1) ] 

#plt.plot( range(epoch_skip,len(avg_MAE_history)), avg_MAE_history[epoch_skip:],"y",label="Validation MAE" )
plt.plot( range(epoch_skip,len(avg_LOSS_history)), avg_LOSS_history[epoch_skip:], "ro", label="training" )
plt.plot( range(epoch_skip,len(avg_VAL_LOSS_history)), avg_VAL_LOSS_history[epoch_skip:] ,"b-", label="validation")
plt.title("Training And Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("LOSS")
plt.legend()
plt.show()

# GENERATE MODEL FROM COMPLETE TRAINING DATA SET USING "OPTIMAL" FITTING PARAMETERS
model = build_model()   
model.fit(training_data, training_targets, epochs=220, batch_size=64, verbose=0)
test_mse_score, test_mae_score = model.evaluate(test_data,test_targets)
print(test_mae_score,np.sqrt(test_mse_score))
