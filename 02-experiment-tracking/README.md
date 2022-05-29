This folder contains the solutions to the problems provided for in the second week of MLOPS ZoomCamp. The lectures to the second session provided at the link 
https://github.com/DataTalksClub/mlops-zoomcamp/tree/main/02-experiment-tracking
The homework questions are also given there. This repository contains only the answers, more precisely for the programming problems where the given programmes (Questions 2, 3, 5 and 6) are modified to 
to obtain the solution. The first question was to check the version. I have used mlflow 1.26.0. The 3rd question was to launch the tracking server with sqlite as the backend store and 
the folder 'artifacts' as the artifact store instead of the default folder mlruns. This can be done using the following command

'''mlflow ui --backend-store-uri sqlite:///<path to folder where database>/<database name> --default-artifact-root file:///<path to folder where the file resides>/<folder name>
