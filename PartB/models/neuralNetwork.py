from tensorflow import keras


def neuralBlock(layer,nNodes,alpha):
    x=keras.layers.Dense(nNodes,kernel_regularizer=keras.regularizers.l1(alpha))(layer)
    x=keras.layers.LeakyReLU(alpha=0.1)(x)
    return x
    

def createNN(inputShape,hiddenLayers,nNodes,alpha):
    input=keras.Input(inputShape)
    x=neuralBlock(input,nNodes[0],alpha)
    for i in range(1,hiddenLayers):
        x=neuralBlock(x,nNodes[i],alpha)
    # dr=keras.layers.Dropout(0.5)(x)
    bnorm=keras.layers.BatchNormalization()(x)
    output=neuralBlock(bnorm,1,alpha)
    model=keras.Model(input,output)
    return model
    
def trainNn(X,y,optimizer,loss,metrics):
    model=createNN((X.shape[1]),6,[64,128,128,128,64,32],1e-1)
    model.compile(optimizer=optimizer,loss=loss,metrics=metrics)
    print(X)
    print(y)
    es=keras.callbacks.EarlyStopping(monitor='val_loss',patience=5,verbose=0,restore_best_weights=True)
    history=model.fit(X,y,epochs=40,validation_split=0.2,callbacks=[es],batch_size=16)
    return {'model':model,'history':history}


    