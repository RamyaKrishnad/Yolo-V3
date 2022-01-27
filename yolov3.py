{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {},
   "outputs": [],
   "source": [
    "import cv2\n",
    "import numpy as np\n",
    "cap = cv2.VideoCapture(0)\n",
    "whT = 320\n",
    "confThreshold = 0.5\n",
    "nmsThreshold = 0.3"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {},
   "outputs": [],
   "source": [
    "classesFile = 'coco.names'\n",
    "classNames = []\n",
    "with open(classesFile,'rt') as f:\n",
    "    classNames = f.read().rstrip('n').split('n')\n",
    "#print(classNames)\n",
    "#print(len(classNames))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {},
   "outputs": [],
   "source": [
    "modelConfiguration = 'yolov3.cfg'\n",
    "modelWeights = 'yolov3.weights'\n",
    "net = cv2.dnn.readNetFromDarknet(modelConfiguration, modelWeights)\n",
    "net.setPreferableBackend(cv2.dnn.DNN_BACKEND_DEFAULT)\n",
    "net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 25,
   "metadata": {},
   "outputs": [],
   "source": [
    "def findObjects(outputs,img):\n",
    "    hT, wT, cT = img.shape\n",
    "    bbox = []\n",
    "    classIds = []\n",
    "    confs = []\n",
    "    for output in outputs:\n",
    "        for det in output:\n",
    "            scores = det[5:]\n",
    "            classId = np.argmax(scores)\n",
    "            confidence = scores[classId]\n",
    "            if confidence > confThreshold:\n",
    "                w,h = int(det[2]* wT), int(det[3]*hT)\n",
    "                x,y = int((det[0]*wT)-w/2), int((det[1]*hT)-h/2)\n",
    "                bbox.append([x,y,w,h])\n",
    "                classIds.append(classId)\n",
    "                confs.append(float(confidence))\n",
    "\n",
    "\n",
    "    #print(len(bbox))\n",
    "    indices = cv2.dnn.NMSBoxes(bbox, confs,confThreshold,nmsThreshold)\n",
    "\n",
    "    for i in indices:\n",
    "        i = i[0]\n",
    "        box = bbox[i]\n",
    "        x,y,w,h = box[0], box[1], box[2], box[3]\n",
    "        cv2.rectangle(img, (x,y),(x+w,y+h),(255,0,255),2)\n",
    "        cv2.putText(img,f'{classNames[classIds[i]].upper()} {int(confs[i]*100)}%',\n",
    "                    (x,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6,(255,0,255),2)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "while True:\n",
    "    success, img = cap.read()\n",
    "    blob = cv2.dnn.blobFromImage(img, 1/255,(whT,whT),[0,0,0],crop=False)\n",
    "    net.setInput(blob)\n",
    "\n",
    "    layerNames = net.getLayerNames()\n",
    "    outputNames = [layerNames[i[0]-1] for i in net.getUnconnectedOutLayers()]\n",
    "\n",
    "    outputs = net.forward(outputNames)\n",
    "    findObjects(outputs,img)\n",
    "\n",
    "    cv2.imshow('Image', img)\n",
    "    cv2.waitKey(1)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    " "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.8.5"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
