For phase 1:
1)For running the code, open it in any IDE such as VSC. Your working directory should be the Phase1 folder

2)change path to data folder as required and number of images as well

3)Run the code by using the following command
    python3 Wrapper.py
    

For Phase 2:
    Copy Train, Val and Test(Make sure the names of the folders are this only)Folders, to Phase2/Data/
    cd Phase2/Code
    For supervised training run 
    python3 Train.py --ModelType sup

    For unsupervised training run
    python3 Train.py --ModelType Unsup

    For supervised testing run
    python3 Test.py --ModelType sup

    For unsupervised testing run
    python3 Test.py --ModelType Unsup
