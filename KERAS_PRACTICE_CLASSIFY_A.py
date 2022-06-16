#                                  BINARY CLASSIFICATION EXAMPLE

# Import Needed Libraries And Assign Aliases
import h5py
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
import matplotlib.pyplot as plt

tf.keras.backend.set_floatx('float64')         # Force 64-bit Numerical Computation On Backend

NumUnits = 16

# Build The Model
def build_model():
  model = tf.keras.Sequential([
  layers.Dense(NumUnits, activation="relu",name="layer1"),    # Rectifying Unit
  layers.Dense(NumUnits, activation="relu",name="layer2"),    # Rectifying Unit
  layers.Dense(1, activation="sigmoid", name="output")    
  ])  
  model.compile(optimizer="rmsprop",
                loss="binary_crossentropy", metrics=["BinaryAccuracy"])
  return model

# LOAD THE LINKED RING DATA FROM AN HDF5 FORMATTED FILE
f = h5py.File('RingData.mat','r')
ds = f['data']
full_data = np.array(ds['value'])   # CONVERT THE 'value' FIELD OF THE DATASET OBJECT TO A NUMPY ARRAY

f.flush()
f.close()

print(full_data.shape)   # FOUR COLUMNS OF DATA :  X,Y, Z COORDS AND THE LABEL ( 0 OR 1 ).   Number Of Rows Is Multiple of 4. 

randomized_idx = np.random.permutation( len(full_data) )   

all_data = full_data[randomized_idx, ]          # SHUFFLE THE ROWS OF INCOMING DATA

#  RESERVE 25% OF SHUFFLED DATA FOR TESTING
num_test_samples =    tf.cast(  len(full_data)/4  , tf.int64)   
test_data =      all_data[  : num_test_samples   , 0:2]
test_targets = all_data[ : num_test_samples , full_data.shape[1]-1]

# USE 25% OF REMAINING SHUFFLED DATA FOR VALIDATION
num_validation_samples = tf.cast( (len(all_data)-num_test_samples) / 4 , tf.int64)
val_data = all_data[ num_test_samples : num_test_samples + num_validation_samples, 0:2]
val_targets = all_data[ num_test_samples : num_test_samples + num_validation_samples, full_data.shape[1]-1]

# USE REMAINING SHUFFLED DATA FOR TRAINING
training_data =      all_data[ num_test_samples + num_validation_samples : , 0:2]
training_targets = all_data[ num_test_samples + num_validation_samples : , full_data.shape[1]-1]

num_epochs = 700
  
model = build_model()   # Model Input Dimension Set By Training Data Shape (For This Model)
history =  model.fit(training_data, training_targets,
                       validation_data = (val_data, val_targets),
                       epochs=num_epochs, batch_size=128, verbose=0)

print( history.history.keys() )

all_LOSS_histories = []
all_VAL_LOSS_histories = []

avg_LOSS_history = history.history["loss"] 
avg_VAL_LOSS_history = history.history["val_loss"] 

epoch_skip = 20

plt.plot( range(epoch_skip,len(avg_LOSS_history)), avg_LOSS_history[epoch_skip:], "ro", label="training" )
plt.plot( range(epoch_skip,len(avg_VAL_LOSS_history)), avg_VAL_LOSS_history[epoch_skip:] ,"b-", label="validation")
plt.title("Training And Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("LOSS")
plt.legend()
plt.show()

results = model.evaluate(  test_data, test_targets, batch_size=64 )
print(results)





