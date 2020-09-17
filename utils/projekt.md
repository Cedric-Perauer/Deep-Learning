

\documentclass[11pt,ngerman ]{article}


\usepackage[ngerman]{babel}
\usepackage[utf8]{inputenc}
\usepackage{amsmath}
\usepackage{graphicx}
\usepackage[colorinlistoftodos]{todonotes}
\usepackage[legalpaper, margin=1in]{geometry}
\usepackage{natbib}
\usepackage[sorting=none]{biblatex}
\addbibresource{refs.bib}
\bibliographystyle{unsrt}

\title{Projektdokumentation Perception}

\author{
  Cedric Perauer\\
  \texttt{757652}
  \and
  Yannik Mezger\\
  \texttt{1000}
  
  \and
  Michael Nützel\\
  \texttt{1000}
  
  \and
  Peter Frank\\
  \texttt{1000}
  
  \and
  Jonas Steinbrenner\\
  \texttt{1000}
}



\date{Sommersemester 2020}

\begin{document}
\maketitle

\begin{abstract}

\end{abstract}


\newpage
\section{Kamera}
\label{sec:examples}

Im Rahmen der Projektarbeit im Sommersemester 2019 wurde bereits die Verwendung klassischer Computer Vision Algorithmen wie Harris Corner Detection, Blob Detection oder Threshold basierte Kantenfilter (Otsu) untersucht. Diese Verfahren sind zwar deutlich schneller als Neuronale Netze, aber leider nur in sehr beschränkten Bedingungen stabil und erfordern deutlich mehr Handarbeit als die gängigen CNN Verfahren. Da außerdem in allen Bereichen der Robotik (nicht nur im Autonomen Fahren) an Deep Learning basierten Verfahren geforscht wird, können wir von den Ergebnissen dieser weitreichenden Arbeiten und Entwicklungen profitieren die sich seit dem Durchbruch durch die Architektur Alex Net \cite{krizhevsky2012imagenet} im Jahre 2012 ergeben haben. 

\subsection{Single Class Object Detection mit YOLOv3}
Für den ersten Ansatz der Object Detection wurde der open source Object Detector des MIT/Delft Teams \cite{strobel2020accurate} verwendet. Die Implementierung basiert auf der Pytorch Implementierung von Ultralytics, welche eine der bekanntesten open source Implementierungen von YOLOv3 \cite{redmon2018yolov3} darstellt. Im weiteren Verlauf wird die MIT/Delt Implementierung auf Grund verschiedener Nachteile (Laufzeit, single class) ersetzt werden, diese sind allerdings erst bei genauerer Betrachtung aufgetreten und im Rahmen des Gesamtkonzepts nicht relevant. \newline

Betrachtung der Metriken \newline

\begin{center}
\begin{tabular}{ c c }
 Recall & 88.74 \%  \\ 
 mAP @IOU 0.5 & 88.64 \%  \\  
 Precision & 86.84 \%     
\end{tabular}
\end{center}

Für eine Erklärung der Metriken wird auf die Dokumentation der Projektarbeit im Sommersemester 2019 hingewiesen.
Durch die Verwendung von Transfer Learning kann die Anzahl an Trainingsdaten deutlich reduziert werden, in dem das CNN auf Millionen von Daten vortrainiert wird. In diesem Fall wird der COCO Datensatz \cite{lin2014microsoft} von Microsoft verwendet. Dadurch kann eine sehr gute Performance mit nur 7730 Frames erreicht werden, wovon 6957 zum Training und 773 zum Testing eingesetzt werden. Es wurden außerdem keine Checkpoints unter Verwendung eines Validation Sets abgespeichert, weshalb  kein early stopping eingesetzt wurde. Die Architektur wurde mit Hilfe des bekannten ADAM \cite{kingma2019method} Optimierungsalgorithmus unter Verwendung einer initialen learning rate von 0.0001 trainiert. Es wurde kein Learning Rate Decay eingesetzt oder Annealing der Learning Rate eingesetzt. 
Es sollte angemerkt werden, dass die oben genannten Werte nicht  unter der Verwendung von Data Augmentation enstanden sind. Data Augmentation verwendet Techniken welche die Varianz des Trainigsdatensatzes durch die Veränderung bereits vorhandener Daten erhöht, dazu gehören u.a. Rotation oder Spiegeln des Bildes und der Label sowie die Veränderung der Helligkeit, Exposure, Sättigung, als auch Cropping basierte Verfahren wie Mosaic Augmentation.  

\begin{figure}[h!]
\centering
\includegraphics[height=8cm]{detection.png}
\caption{Object Detector Output}
\end{figure}

In Abbildung 1 kann der Output eines Frames im Testset gesehen werden. Die grünen Boudning Boxen stellen die Ground Truth und die roten die Prediction des CNNs dar. Die Erkennung von "False Positives" am linken Rand kann durch fehlende Label erklärt werden. Dies erklärt auch die niedrigere Precision im Vergleich zum Recall während des Trainings. Die meisten Cones können korrekt wieder erkannt werden, allerdings entstehen False Positives auf Grund fehlender Label. Deswegen gibt es im Vergleich mehr False Positives und weniger False Negatives, wodurch die Precision im Vergleich zum Recall niedriger ist. 

\subsection{Keypoint Detector}

Um die Position des Cones genauer bestimmen zu können, werden die Outputs des Object Detectors verwendet und als Input in ein Keypoint Extractor Network genutzt. Dieses ResNet basierte CNN bestimmt 7 markante Punkte am Cone die dann zur präzisen Bestimmung der 2D Position genutzt werden kann. Die Architektur basiert ebenfalls auf dem MIT/Delft Open Source Ansatz mit verschiedenen Anpassungen. 

\begin{figure}[h!]
\centering
\includegraphics[height=4cm]{rektnet.png}
\caption{Keypoint Extractor Network}
\end{figure}

Wie in Abbildung 2 zu sehen, handelt es sich um eine ResNet Architektur \cite{he2016deep}, welche mit same Padding arbeitet um eine Output Heatmap mit selben Höhen und Breiten Dimensionen wie dem Input Bild zu erzeugen. Diese Heatmap wird dann verwendet um entsprechend die Koordinaten der einzelnen Keypoints zu extrahieren. Nach Ansatz des Papers \cite{he2019bag} wurde der Input 7x7 Filter auf Grund von Laufzeit Vorteilen durch einen 3x3 Convolutional Filter ersetzt. Nach Ansatz der ResNet-C Architektur im Paper wurde der 7x7 Filter erst mit 3 3x3 Filtern ersetzt. Eine empirische Herangehensweise zeigte aber, dass 1 3x3 Filter im Rahmen der Performance keine Nachteile bringt.  
Die Verwendung von größeren 7x7 Filtern war in der Zeit der ersten ResNet Architektur noch gängig, heutzutage werden aber meist kleinere 3x3 oder 1x1 Filter (z.B. Mobilenet) verwendet.   \newline

Dies bietet den Vorteil, dass keine Fully Connected Layer einegesetzt werden muss. Convolutional Layer sind besser in der Feature Extraction als Fully Connected Layer, weshalb auch in der bekannten YOLO Architektur im Laufe der Zeit die Fully Connected Layer der ersten Version durch Convolutional Layer ersetzt worden sind. Durch Parameter Sharing wird außerdem die Anzahl der Parameter reduziert und daher die Konvergenz des Optimierungsproblem verbessert. \newline
Als Loss Funktion wird von den Autoren ein Term aus zwei Teilen verwendet.

\par
$L_{total} = L_{mse} + \gamma_{horz} * (2 - V_{12} * V_{34} - V_{34} * V_{56}) + \gamma_{vert} * (4 - V_{01} * V_{13} - V_{13} * V_{35} - V_{02} * V_{24} - V_{24} * V_{46})  $ 

Dieser besteht einmal aus dem L2-Abstand (Pythagoras) zweier Punkte und einem zusätzlichen Term, welcher die Charakterstiken der Hütchen berücksichtigt. Da die Keypoints an der Seite des Cones Kollinear sind, müssen die Vektorprodukte der Einheitsvektoren zwischen den Punkten  1 ergeben wenn die Vektoren Parallel sind. Aus Abbildung 2 kann man so erkennen, dass im Optimalfall dieser Term nahe 0 liegt. In der Praxis hat dieser Term allerdings einen sehr geringen Einfluss der bei einzelnen Frames im Bereich weniger Pixel lag und über den gesamten Trainigssatz bei entsprechenden Hyperparametern nicht erkennbar war. Nach Vorschlag des Papers wurden 
$\gamma_{horz} = 0.055 $ und $\gamma_{vert} = 0.038 $ gewählt.  

\newpage
Es wurde ein Trainigsloop implementiert um die optimalen Hyperparameter zu finden. Nach dem gängigen Vorbild von \cite{bergstra2012random} wurde dabei eine Random Hyperparameter Search mit einer Logarithmus Skala verwendet. Dabei wurden sowohl die Learning Rate als auch die Decay Rate des Learning Rate Schedulers als Variablen formuliert. Die besten Parameter eines Trainigsdurchlaufs wurde durch den Validation Error gekennzeichnet und mit den besten Ergebniss bis zum aktuellen Zeitpunkt verglichen. Wenn der Validation Error unter diesem Wert lag wurde ein Checkpoint gespeichert. Im besten Fall lag dieser Wert bei 0.22 (gerundet auf 2 Nachkommastellen) und der L2-Abstand zwischen Prediction und Ground Truth bei durchschnittlich ca. 57. Dieser Wert entspricht dem Pixel Abstand aller 7 Punkte addiert. Dabei wurden die ca. 3300 Bilder des MIT Datensatzes verwendet und ein Train/Test/Validation Split von 0.8/0.1/0.1 eingesetzt. 
Funktionen und eine Klasse wurden zum tracken der schlechtesten Ergebnisse basierend auf dem L2-Abstand zwischen Prediction und Ground Truth angelegt. Diese wurde jeweils im Falle des niedrigsten Validation Errors aufgerufen. Beispiel Ergebnisse sind in der unteren Abbildung zu sehen.

\begin{figure}[h!]
\centering
\begin{minipage}{.5\textwidth}
  \centering
  \includegraphics[width=.6\linewidth]{cone_good.PNG}
  \captionof{}{\\ Durchschnittlicher Cone Detection Error}
  \label{fig:test1}
\end{minipage}%
\begin{minipage}{.5\textwidth}
  \centering
  \includegraphics[width=.6\linewidth]{cone_bad.PNG}
  \captionof{}{\\ Negativ Beispiel mit hohem Error}
  \label{fig:test2}
\end{minipage}
\end{figure}

Wie in der Abbildung zu sehen, liegt der durchschnittliche Prediction Error sehr nah an der Ground Truth. Hohe Fehler enstehen vor allem durch kleine Bounding Box Inputs und hohe Kontrastunterschiede (sehr helle Cones durch Sonneneinstrahlung), von denen allerdings wenige im aktuell gelabelten Datensatz vorhanden sind. Außerdem stellen liegende Cones ein Problem dar, diese treten im Normalfall zwar nicht auf und könnten auch nur über den Object Detector verarbeitet werden, sollten aber im Optimalfall auch vom Keypoint Detector erkannt werden. Durch Data Augmentation könnte dies in nachfolgenden Arbeiten gelingen.  \newline

Um die Performance zu verbessern sollten daher mehr Label erstellt werden. Im gegebenen Datensatz haben wir ca. 3300 gelabelte Beispiele, laut \cite{dhall2019real} verwendet AMZ Racing aus Zürich dafür ca. 16000 Bilder. Da das Labeling ein sehr aufwendiger Prozess sein kann, wurde hierfür ein Semi Automatisches Label Tool basierend auf dem Labelme Tool erstellt. 

\begin{figure}[h!]
\centering
\includegraphics[height=10cm]{labeling.PNG}
\caption{Semi Automatisches Labeling Tool}
\end{figure}

\newpage
Abbildung 3 zeigt das Interface des Labeling Tool. Im Vergleich zum Standard Labelme Tool wurden 2 Buttons eingefügt, einmal Image Predict dass eine Keypoint Prediction auf Basis der oben beschriebenen Architektur ausführt und dann die Korrektur durch den Labeler erlaubt. Selbst wenn die Label im Falle von schwierigen Cases nicht korrekt prädiziert werden erlaubt das durch die Projektion der Punkte eine Beschleunigung des Label Prozesses um Faktor 2-3, da die Punkte nicht immer extra ausgewählt und eingefügt werden müssen. Für Geräte mit Cuda Support gibt es außerdem die Möglichkeit die Grafikkarte zu nutzen um z.B. bei Laptops durch die vergleichsweise bessere Leistungseffizienz der GPU die Akkulaufzeit zu erhöhen und eine etwas schnellere Nutzung zu gewährleisten. 

\subsection{Python Proof of Concept 2D Pipeline}

Um das Verständnis für die Pipeline und die Vorgehensweise festzulegen, wurde eine 2D Pipeline in Python aufgebaut die allerdings auf Grund der Art der Implementierung nicht Echtzeitfähig läuft. Dabei werden zuerst Bounding Boxes für Cones detektiert und dann diese Boxen als Input in ein Keypoint Detector Netz gegeben. Was dabei beachtet werden muss sind die Outputs des Detectors relativ zur Bildgröße um die Position der Punkte korrekt zu erhalten. Der Ouput ist in Abbildung 4 zu sehen.  

\begin{figure}[h!]
\centering
\includegraphics[height=8cm]{poc_output.PNG}
\caption{Output 2D Pipeline}
\end{figure}


\subsection{Depth Estimation mit 3D Ansatz}

\subsection{Basler Kamera mit ROS}

\subsection{Testaufbau Kamera}

\subsection{Ausblick Kamera}


\newpage
\section{Lidar}

\subsection{Segmentierung}

\subsection{KDTree Clustering mit adaptiver Distance threshold}

\subsection{"Tracken" der Cones im aktuellen Zeitschritt}



Im Fall von Überlappenden Cones (Cones die direkt hintereinander stehen), kann es zu größeren Abständen einzelner Punkte in einem Cluster kommen. Diese Abstände können nicht immer durch die distance threshold erfasst werden, so dass es zur doppelten Erfassung einzelner Cones kommen kann. Das bedeutet, dass ein Cluster als 2 Cluster wahrgenommen wird und doppelt abgelegt wird. 

\begin{figure}[h!]
\centering
  \makebox[\textwidth]{\includegraphics[width=\paperwidth]{lidar_cap.PNG}}
  \caption{C++ Lidarcode}
\end{figure}


Um dies zu verhindern werden die Cones im aktuellen Zeitschritt in einem C++ vector gespeichert um neue Cones mit bereits existierenden Cones abzugleichen. Der abgleich basiert auf dem Vergleich von x/y Koordinaten der Cluster Centroide. Die Centriode werden durch den median der jeweiligen Koordinaten ermittelt. Wenn 2 Centroide innerhalb eines Cone Durchmessers liegen, wird das neue Cluster verworfen. 
Es kann also als eine Art Non Max Suppression angesehen werden. 
Dieser Vector wird außerdem die Grundlage für die Rostopic darstellen, welche die aktuellen Cone Positionen des Lidars ablegt.
\newline

\textit{Erste Experimente außerhalb des Rahmens der Projektarbeit im neuen Formula Student Driverless Simulator der beim FS online event im August eingesetzt wurde zeigen, dass diese Funktion wahrscheinlich nicht mehr notwendig ist}

\subsection{Erkennung der Ziellinie}

Im Rahmen der Arbeit mit dem FSSIM Simulator von AMZ Racing, können verschiedene Streckenszenarien simuliert werden. Dabei wurde vor allem mit dem Acceleration track gearbeitet. 

\begin{figure}[h!]
\centering
\includegraphics[height=4cm]{acceleration.PNG}
\caption{Acceleration Layout im Simulator}
\end{figure}


Die Start- und Ziellinie sind jeweils durch 2 hohe Cones gekennzeichnet. Mit einer Hashmap (eine von vielen Möglichkeiten) wird die Anzahl der großen Cones in jedem Zeitschritt getrackt, wenn man auf die Ziellinie zufährt "sieht" man die großen Cones und kann diese registrieren. Nach passieren der Ziellinie werden diese nicht mehr wahrgenommen. Mit der Information dass die Cones im vorherigen Zeitschritt noch registriert wurden kann nun ein Stopp Signal gesendet werden. Als zusätzliches Signal wird dabei die Position des Fahrzeugs benötigt um das Stopp Signal nicht direkt nach dem Start auszulösen. In diesem Fall wurde eine if Bedingung mit zurückgelegter Strecke > 3m genutzt, es ist also keine genaue Information über die Fahrzeugposition notwendig. 
\newline
Das ganze kann genutzt werden um ein Stop Signal über eine rostopic des typs \texttt{std\char`_msgs} zu senden. Der typ der rostopic ist hierbei nicht genau druchdacht worden, aber im Rahmen der eigentlichen Funktion auch nicht relevant. \newline Es wird in Zukunft also eine bool message verschickt werden, da es sich dann nur um eine 1 bit Information handelt. Der Aufwand um dies anzupassen ist simpel und beschränkt sich auf wenige Zeilen Code. 

\begin{figure}[h!]
\centering
\includegraphics[height=8cm]{ros_lidar.PNG}
\caption{ROS Architektur Simulator}
\end{figure}

In der Abbildung .. ist der \texttt{rqt\char`_graph} des Simulators zu sehen. Die Lidar Node ist dabei die \texttt{ground\char`_segmentation} node. Diese registriert die entsprechenden Cone Cluster und kann ein stop signal (rostopic  \texttt{stop\char`_signal}) senden. Diese rostopic wird nur im Falle einer registrierten Ziellinie geschickt. Die topic wird dann von der Control Node \texttt{/control} empfangen und entsprechend verarbeitet um das Fahrzeug dadurch bis zum Stillstand abzubremsen. 



\subsection{Validierung im Simulator}

Um die Lidar Perception zu validieren wurde die Ground Truth Position der Cones aus dem Config Dokument für den Acceleration Track verwendet. Da die kleinen Cones alle 5 Meter und in einer Reihe positioniert sind, muss in diesem Fall nicht die Ground Truth Position aus dem file ausgelesen werden. Abhängig von der aktuellen Fahrzeugposition können die nächsten 10 - 15 Meter in der Simulation betrachtet werden um die Ground Truth Position zu generieren. Z.b. wenn das Fahrzeug bei 39 Meter globaler Position ist, wissen wir dass die nächsten 6 Cones bei 40, 45 und 50 Meter stehen. Dies kann einfach durch Division der Fahrzeugkoordinate in Längsrichtung mit 5 und Interpolierung realisiert werden. In jedem Run werden dabei x und y Fehler zur Ground Truth berechnet. X und Y Position der Cones werden durch x/y median Wert der Punkte in einem Cluster gekennzeichnet. Zusätzlich wird die Fahrzeugposition in der Welt und die Verbauposition des Lidars auf dem Fahrzeug verwendet um die Lidar Outputs in globale Koordinaten umzurechnen. Um dies so genau wie möglich zu gestalten, werden die rostopic der Fahrzeugposition und die Outputs des Lidars mit Hilfe der \texttt{message\char`_filters::Synchronizer} API synchronisiert. Dies versichert, dass Lidar Punktwolke und Fahrzeugposition nei gemeinsamer Verarbeitung in der Callback Funktion so gut wie möglich synchronisiert sind. 


\begin{figure}[h!]
\centering
\includegraphics[height=3cm]{lidar_distance.PNG}
\caption{Error zwischen Ground Truth und Perception Output des Lidar Clustering}
\end{figure}

Wie in Abbildung ... zu sehen ist, beträgt der Lokalisierungsfehler im Simulator im Durchschnitt wenige wenige centimeter. 
Es sollte angemerkt werden, dass dabei nur der x und y fehler separat berechnet wurden. Wenn wir annehmen, dass maximaler x und y Fehler zum selben Zeitpunkt auftreten (was nicht der Fall sein sollte), würde der absolute Lokalisierungsfehler nach L2 Distance 9,2 cm betragen. 

\subsection{Ausblick Lidar}

\newpage

\printbibliography

\end{document}
