import tkinter as tk
from tkinter import ttk
from ttkbootstrap import Style
from PIL import Image,ImageDraw,ImageOps
import numpy as np
from keras.models import load_model 

class Whiteboard:
    def __init__(self,master):
        self.master = master
        self.master.title("Whiteboard")
        self.master.resizable(True,True)

        self.style = Style(theme='pulse')
        self.canvasWidth = 560
        self.canvasHeight = 560
        self.canvas = tk.Canvas(self.master,width=self.canvasWidth,height=self.canvasHeight,bg='white')
        self.canvas.pack()

        self.buttonFrame = ttk.Frame(self.master)
        self.buttonFrame.pack(side='top',pady=10)

        buttonConfig = {
            "blue":("info.TButton","Predict",lambda:self.imageMaker()),
            
            "clear":("light.TButton","Clear",lambda:self.clearCanvas())
        }

        for color,(style,name,command) in buttonConfig.items():
            ttk.Button(self.buttonFrame,text=name,command=command,style=style).pack(side='left',padx=5,pady=5)

        self.resultLabel = ttk.Label(self.buttonFrame,text="Prediction Ready",font = ("Helvetica",14,"bold"))

        self.drawColor = 'black'
        self.lineWidth = 25
        self.oldX,self.oldY = None,None

        self.image = Image.new('RGB',(self.canvasWidth,self.canvasHeight),"white")
        self.drawCTX = ImageDraw.Draw(self.image)

        self.canvas.bind("<Button-1>",self.startLine)
        self.canvas.bind("<B1-Motion>",self.drawLine)

        try:
            # self.model = load_model("identifyNumbers.keras")
            self.model = load_model("cnnIdentify.keras")

        except Exception as e:
            print(f"Error loading model: {e}")
            self.model = None

    def startLine(self,event):
        self.oldX,self.oldY = event.x,event.y

    def drawLine(self,event):
        if self.oldX and self.oldY:
            self.canvas.create_line(self.oldX,self.oldY,event.x,event.y, width = self.lineWidth,
                                    fill = self.drawColor,capstyle = tk.ROUND,smooth = tk.TRUE)
            self.drawCTX.line([self.oldX,self.oldY,event.x,event.y],fill=self.drawColor,width=self.lineWidth,joint = "round")
            self.oldX,self.oldY = event.x,event.y


    def clearCanvas(self):
        self.canvas.delete("all")
        self.image = Image.new('RGB',(self.canvasWidth,self.canvasHeight),"white")
        self.drawCTX = ImageDraw.Draw(self.image)
        self.resultLabel.config(text = "Prediction Ready")

    def imageMaker(self):
        if self.model is None:
            self.resultLabel.config(text="Error:Model not loaded")
            return
        
        grayImg = self.image.convert('L')
        invertedImg = ImageOps.invert(grayImg)
        resizedImg = invertedImg.resize((28,28),Image.Resampling.LANCZOS)
        # resizedImg.save("whatANNsees.png")
        # imgArray = np.asarray(resizedImg)/255.0 # not required
        # networkInput = imgArray.reshape(1,784)
        imgArray = np.asarray(resizedImg).reshape(1,28,28,1).astype('float32')/255.0
        
        prediction = self.model.predict(imgArray) # type: ignore
        print(np.argmax(prediction))
        self.resultLabel.config(text=f"Prediction: {np.argmax(prediction)}")
        

if __name__ == "__main__":
    root = tk.Tk()
    whiteboard = Whiteboard(root)
    root.mainloop()




