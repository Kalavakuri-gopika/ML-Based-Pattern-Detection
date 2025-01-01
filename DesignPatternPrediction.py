
from tkinter import messagebox
from tkinter import *
from tkinter import simpledialog
import tkinter
from tkinter import filedialog
import matplotlib.pyplot as plt
import numpy as np
from tkinter.filedialog import askopenfilename
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from keras.callbacks import ModelCheckpoint 
from sklearn.model_selection import train_test_split
import re
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from string import punctuation
from nltk.corpus import stopwords
from sklearn.preprocessing import StandardScaler
import nltk
import pickle
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
import seaborn as sns

main = tkinter.Tk()
main.title("ML-Based Software Design Pattern Detection in Source Code Using Neural Network") #designing main screen
main.geometry("1300x1200")

global filename
global X_train, X_test, y_train, y_test
global X, Y, dataset
global ann_model
global tfidf_vectorizer, scaler

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

labels = ['AbstractFactory', 'Adapter', 'FactoryMethod', 'Visitor']

def getID(name):
    index = 0
    for i in range(len(labels)):
        if labels[i] == name:
            index = i
            break
    return index    

def cleanPost(doc):
    tokens = doc.split()
    table = str.maketrans('', '', punctuation)
    tokens = [w.translate(table) for w in tokens]
    tokens = [word for word in tokens if word.isalpha()]
    tokens = [w for w in tokens if not w in stop_words]
    tokens = [word for word in tokens if len(word) > 1]
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    tokens = ' '.join(tokens)
    return tokens

def uploadDataset(): #function to upload tweeter profile
    global filename
    filename = filedialog.askdirectory(initialdir=".")
    text.delete('1.0', END)
    text.insert(END,filename+" loaded\n\n")
    text.insert(END,"Gang of 4 Design Patterns Found in Dataset : "+str(labels))
    
def preprocessDataset():
    text.delete('1.0', END)
    global filename, dataset, tfidf_vectorizer
    if os.path.exists("model/features.csv"):
        dataset = pd.read_csv("model/features.csv")
        with open('model/vector.txt', 'rb') as file:
            tfidf_vectorizer = pickle.load(file)
        file.close()
    else:
        X = []
        Y = []
        for root, dirs, directory in os.walk(filename):
            for j in range(len(directory)):
                name = os.path.basename(root)
                with open(root+"/"+directory[j],"rb") as file:
                    content = file.read()
                file.close()
                content = content.decode()
                content = cleanPost(content)
                label = getID(name)
                X.append(content)
                Y.append(label)
        tfidf_vectorizer = TfidfVectorizer(stop_words=stop_words, use_idf=True, smooth_idf=False, norm=None, decode_error='replace', max_features=300)
        tfidf = tfidf_vectorizer.fit_transform(X).toarray()        
        dataset = pd.DataFrame(tfidf, columns=tfidf_vectorizer.get_feature_names())        
        dataset['class_label'] = Y
        dataset.to_csv("model/features.csv", index = False)
        with open('model/vector.txt', 'wb') as file:
            pickle.dump(tfidf_vectorizer, file)
        file.close()
    text.insert(END,"Dataset Preprocessing Task Completed\n\n")
    text.insert(END,"Each word in the code is conveted into numeric vector by replacing with its average frequency\n\n")
    text.insert(END,str(dataset)+"\n\n")
    text.insert(END,"First row contains Code word names and remaining rows contains average frequency of those words\n\n")
    text.update_idletasks()
    unique, count = np.unique(dataset['class_label'], return_counts = True)
    height = count
    bars = labels
    y_pos = np.arange(len(bars))
    plt.bar(y_pos, height)
    plt.xticks(y_pos, bars)
    plt.title("Design Patterns found in dataset")
    plt.xlabel("Design Pattern Name")
    plt.ylabel("Count")
    plt.show()

def trainTestSplit():
    global X_train, X_test, y_train, y_test, scaler
    text.delete('1.0', END)
    global filename, dataset, tfidf_vectorizer, X, Y, scaler
    Y = dataset['class_label'].ravel()
    dataset = dataset.values
    X = dataset[:,0:dataset.shape[1]-1]
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    X = X[indices]
    Y = Y[indices]
    Y = to_categorical(Y)
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
    text.insert(END,"Total records found in dataset        : "+str(X.shape[0])+"\n")
    text.insert(END,"80% dataset records used for training : "+str(X_train.shape[0])+"\n")
    text.insert(END,"20% dataset records used for testing  : "+str(X_test.shape[0])+"\n")
                

def calculateMetrics(algorithm, predict, target):
    acc = accuracy_score(target,predict)*100
    p = precision_score(target,predict,average='macro') * 100
    r = recall_score(target,predict,average='macro') * 100
    f = f1_score(target,predict,average='macro') * 100
    text.insert(END,algorithm+" Precision  : "+str(p)+"\n")
    text.insert(END,algorithm+" Recall     : "+str(r)+"\n")
    text.insert(END,algorithm+" F1-Score   : "+str(f)+"\n")
    text.insert(END,algorithm+" Accuracy   : "+str(acc)+"\n\n")
    text.update_idletasks()
    LABELS = ['AbstractFactory', 'Adapter', 'FactoryMethod', 'Visitor']
    conf_matrix = confusion_matrix(target, predict) 
    plt.figure(figsize =(6, 6)) 
    ax = sns.heatmap(conf_matrix, xticklabels = LABELS, yticklabels = LABELS, annot = True, cmap="viridis" ,fmt ="g");
    ax.set_ylim([0,4])
    plt.title(algorithm+" Confusion matrix") 
    plt.ylabel('True class') 
    plt.xlabel('Predicted class') 
    plt.show()

def runANN():
    text.delete('1.0', END)
    global X_train, X_test, y_train, y_test, ann_model
    ann_model = Sequential()
    ann_model.add(Dense(512, input_shape=(X_train.shape[1],)))
    ann_model.add(Activation('relu'))
    ann_model.add(Dropout(0.3))
    ann_model.add(Dense(512))
    ann_model.add(Activation('relu'))
    ann_model.add(Dropout(0.3))
    ann_model.add(Dense(y_train.shape[1]))
    ann_model.add(Activation('softmax'))
    ann_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    if os.path.exists("model/model_weights.hdf5") == False:
        model_check_point = ModelCheckpoint(filepath='model/model_weights.hdf5', verbose = 1, save_best_only = True)
        hist = ann_model.fit(X_train, y_train, batch_size = 16, epochs = 30, validation_data=(X_test, y_test), callbacks=[model_check_point], verbose=1)
        f = open('model/history.pckl', 'wb')
        pickle.dump(hist.history, f)
        f.close()    
    else:
        ann_model.load_weights("model/model_weights.hdf5")
    predict = ann_model.predict(X_test)
    predict = np.argmax(predict, axis=1)
    y_test = np.argmax(y_test, axis=1)
    calculateMetrics("Artificial Neural Network", predict, y_test)

def graph():
    f = open('model/history.pckl', 'rb')
    data = pickle.load(f)
    f.close()
    accuracy = data['val_accuracy']
    loss = data['val_loss']

    plt.figure(figsize=(10,6))
    plt.grid(True)
    plt.xlabel('EPOCH')
    plt.ylabel('Accuracy/Loss Values')
    plt.plot(accuracy, 'ro-', color = 'green')
    plt.plot(loss, 'ro-', color = 'blue')
    plt.legend(['ANN Accuracy', 'ANN Loss'], loc='upper left')
    plt.title('ANN Accuracy & Loss Graph')
    plt.show()


def designPatternDetection():
    text.delete('1.0', END)
    global ann_model, tfidf_vectorizer, scaler
    filename = filedialog.askopenfilename(initialdir="testCodes")
    with open(filename,"rb") as file:
        content = file.read()
    file.close()
    data = content.decode()
    content = cleanPost(data)

    temp = []
    temp.append(content)
    temp = tfidf_vectorizer.transform(temp).toarray()
    temp = scaler.transform(temp)
    predict = ann_model.predict(temp)
    predict = np.argmax(predict)
    text.insert(END,"Uploaded File Name : "+os.path.basename(filename)+"\n\n")
    text.insert(END,"Detected Design Pattern = "+labels[predict])


font = ('times', 16, 'bold')
title = Label(main, text='ML-Based Software Design Pattern Detection in Source Code Using Neural Network')
title.config(bg='darkviolet', fg='gold')  
title.config(font=font)           
title.config(height=3, width=120)       
title.place(x=0,y=5)

font1 = ('times', 12, 'bold')
text=Text(main,height=20,width=150)
scroll=Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=50,y=120)
text.config(font=font1)


font1 = ('times', 12, 'bold')
uploadButton = Button(main, text="Upload Source Code Dataset", command=uploadDataset)
uploadButton.place(x=50,y=550)
uploadButton.config(font=font1)  

processButton = Button(main, text="Preprocess & Normalize Dataset", command=preprocessDataset)
processButton.place(x=290,y=550)
processButton.config(font=font1) 

splitButton = Button(main, text="Dataset Train & Test Split", command=trainTestSplit)
splitButton.place(x=570,y=550)
splitButton.config(font=font1) 

annButton = Button(main, text="Run ANN Algorithm", command=runANN)
annButton.place(x=800,y=550)
annButton.config(font=font1)

graphButton = Button(main, text="ANN Training Graph", command=graph)
graphButton.place(x=50,y=600)
graphButton.config(font=font1)

predictButton = Button(main, text="Design Pattern Detection from Test Code", command=designPatternDetection)
predictButton.place(x=290,y=600)
predictButton.config(font=font1)


main.config(bg='sea green')
main.mainloop()
error in this code change above ouputs are not coming
