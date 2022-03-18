from tensorflow import keras

def createNN(inputShape,hiddenLayers,nNodes,output):
    Input=keras.Input(inputShape)

    for i in range(hiddenLayers):
        layer=keras.layers.Dense()

    