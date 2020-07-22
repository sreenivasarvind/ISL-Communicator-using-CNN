from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
model= load_model('/Users/appleapple/Documents/MajorProject/LearnLeapAnalyseISL-master/LearnLeapAnalyseISL/my_model1.h5')
top_layer = model.layers[0]
plt.imshow(top_layer.get_weights()[0][:,:,:,0].squeeze(),cmap='gray')